from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from nla_code_interp.qwen_models import (
    LoraSettings,
    QwenActivationVerbalizer,
    target_transform_from_checkpoint_state,
)
from scripts.run_qwen_nla_loop import resolve_checkpoint_mode
from scripts.train_ar import ActivationArtifact, TargetTransform
from scripts.train_qwen_joint_nla import (
    GeneratedAnchorExample,
    build_generated_anchor_examples,
    build_manifest_payload,
    make_joint_ar_collate_fn,
    metric_row,
)


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 2
    eos_token = "<eos>"

    def __call__(
        self,
        texts: list[str],
        *,
        padding: bool,
        truncation: bool,
        max_length: int,
        return_tensors: str,
        return_attention_mask: bool,
    ) -> dict[str, torch.Tensor]:
        del padding, return_tensors, return_attention_mask
        encoded = []
        for text in texts:
            ids = [max(3, min(29, ord(char) % 30)) for char in text]
            if truncation:
                ids = ids[:max_length]
            encoded.append(ids or [self.eos_token_id])
        width = max(len(ids) for ids in encoded)
        input_ids = torch.zeros((len(encoded), width), dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for row_index, ids in enumerate(encoded):
            input_ids[row_index, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            attention_mask[row_index, : len(ids)] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class FakeQwenCausalLM(nn.Module):
    def __init__(self, *, hidden_size: int = 5, vocab_size: int = 31) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size, vocab_size=vocab_size)
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.output = nn.Linear(hidden_size, vocab_size)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embedding

    def forward(
        self,
        *,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        output_hidden_states: bool = False,
        use_cache: bool = False,
    ):
        del attention_mask, output_hidden_states, use_cache
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("input_ids or inputs_embeds is required")
            inputs_embeds = self.embedding(input_ids)
        hidden = torch.tanh(inputs_embeds)
        logits = self.output(hidden)
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss(ignore_index=-100)(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
            )
        return SimpleNamespace(logits=logits, loss=loss, hidden_states=(hidden,))


def fake_artifact(num_examples: int = 2, activation_dim: int = 3) -> ActivationArtifact:
    activations = torch.arange(num_examples * activation_dim, dtype=torch.float32).reshape(
        num_examples,
        activation_dim,
    )
    metadata_rows = [
        {
            "activation_index": index,
            "example_id": f"ex_{index}",
            "reference_description": f"reference explanation {index}",
            "prompt": f"prompt {index}",
            "code": f"def f_{index}(): return {index}",
            "split": "train",
            "language": "python",
            "transformation_type": "original",
        }
        for index in range(num_examples)
    ]
    return ActivationArtifact(
        activation_dir=Path("fake"),
        activations=activations,
        metadata_rows=metadata_rows,
        manifest={},
    )


def test_joint_ar_batch_construction() -> None:
    tokenizer = TinyTokenizer()
    examples = [
        GeneratedAnchorExample(
            generated_text="generated",
            anchor_text="reference",
            target=torch.ones(3),
            metadata={"example_id": "ex_0"},
        )
    ]

    batch = make_joint_ar_collate_fn(tokenizer, max_length=8)(examples)

    assert batch["generated_input_ids"].shape[0] == 1
    assert batch["anchor_input_ids"].shape[0] == 1
    assert batch["targets"].shape == (1, 3)
    assert batch["metadata"][0]["example_id"] == "ex_0"


def test_qwen_av_loss_is_scalar() -> None:
    model = QwenActivationVerbalizer(
        qwen_model=FakeQwenCausalLM(hidden_size=5, vocab_size=31),
        activation_dim=3,
    )
    input_ids = torch.tensor([[3, 4, 0], [5, 6, 7]])
    attention_mask = torch.tensor([[1, 1, 0], [1, 1, 1]])

    outputs = model(
        activations=torch.zeros((2, 3)),
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=input_ids,
    )

    assert outputs.loss is not None
    assert outputs.loss.ndim == 0
    assert outputs.logits.shape == (2, 4, 31)


def test_ar_generated_text_dataset_uses_generated_text() -> None:
    artifact = fake_artifact(num_examples=2, activation_dim=3)
    generated_rows = [
        {"generated_text": "generated explanation 0"},
        {"generated_text": "generated explanation 1"},
    ]

    examples = build_generated_anchor_examples(
        artifact=artifact,
        generated_rows=generated_rows,
        transformed_targets=torch.zeros((2, 3)),
        target_text_field="reference_description",
        fallback_text_fields=["prompt", "code"],
    )

    assert examples[0].generated_text == "generated explanation 0"
    assert examples[0].anchor_text == "reference explanation 0"
    assert examples[0].metadata["qwen_joint_generated_text"] == "generated explanation 0"


def test_target_transform_checkpoint_state_roundtrip() -> None:
    targets = torch.tensor([[1.0, 2.0], [3.0, 8.0], [5.0, 14.0]])
    transform = TargetTransform.fit("standardize", targets)
    restored = target_transform_from_checkpoint_state(
        transform.state_dict_for_checkpoint()
    )

    transformed = restored.transform(targets)
    roundtrip = restored.inverse_transform(transformed)

    assert restored.name == "standardize"
    assert torch.allclose(roundtrip, targets)


def test_joint_metric_row_schema() -> None:
    row = metric_row(
        epoch=1,
        av_train_loss=2.0,
        ar_generated_train_mse=0.3,
        ar_anchor_train_mse=0.4,
        validation_metrics={
            "validation_nla_fve": 0.1,
            "validation_nla_mse": 0.2,
            "validation_nla_rmse": 0.5,
            "validation_cosine_mean": 0.9,
            "validation_mean_baseline_mse": 0.6,
            "validation_zero_baseline_fve": -1.0,
            "validation_shuffled_baseline_fve": -0.2,
        },
        is_best=True,
    )

    assert set(row) == {
        "epoch",
        "av_train_loss",
        "ar_generated_train_mse",
        "ar_anchor_train_mse",
        "validation_nla_fve",
        "validation_nla_mse",
        "validation_nla_rmse",
        "validation_cosine_mean",
        "validation_mean_baseline_mse",
        "validation_zero_baseline_fve",
        "validation_shuffled_baseline_fve",
        "is_best",
    }
    assert row["is_best"] is True


def test_joint_checkpoint_manifest_schema() -> None:
    artifact = fake_artifact(num_examples=2, activation_dim=3)
    args = {
        "activation_dir": "train_dir",
        "validation_activation_dir": "validation_dir",
        "output_dir": "out_dir",
        "model_name_or_path": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "target_text_field": "reference_description",
        "fallback_text_fields": "prompt,code",
        "av_loss_weight": 1.0,
        "ar_generated_loss_weight": 1.0,
        "ar_anchor_loss_weight": 0.25,
    }

    manifest = build_manifest_payload(
        args=args,
        train_artifact=artifact,
        validation_artifact=artifact,
        target_transform=TargetTransform.fit("center", artifact.activations),
        lora_settings=LoraSettings(r=8, alpha=16, dropout=0.05),
        best_epoch=2,
        best_metrics={"validation_nla_fve": 0.25},
        output_files={"qwen_av_adapter": "qwen_av_adapter"},
    )

    assert manifest["schema_version"] == "phase10d_qwen_joint_nla_v1"
    assert manifest["model_name_or_path"] == args["model_name_or_path"]
    assert manifest["target_transform"]["name"] == "center"
    assert manifest["loss_weights"]["ar_anchor_loss_weight"] == pytest.approx(0.25)


def test_run_qwen_nla_loop_joint_checkpoint_mode() -> None:
    args = SimpleNamespace(
        qwen_av_checkpoint_dir=None,
        qwen_ar_checkpoint_dir=None,
        joint_checkpoint_dir="joint_dir",
    )

    assert resolve_checkpoint_mode(args) == "joint"

    split_args = SimpleNamespace(
        qwen_av_checkpoint_dir="av_dir",
        qwen_ar_checkpoint_dir="ar_dir",
        joint_checkpoint_dir=None,
    )
    assert resolve_checkpoint_mode(split_args) == "split"

    with pytest.raises(ValueError, match="either --joint_checkpoint_dir"):
        resolve_checkpoint_mode(
            SimpleNamespace(
                qwen_av_checkpoint_dir="av_dir",
                qwen_ar_checkpoint_dir=None,
                joint_checkpoint_dir="joint_dir",
            )
        )
