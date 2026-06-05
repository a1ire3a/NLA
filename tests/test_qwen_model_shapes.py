from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from nla_code_interp.qwen_models import (
    LoraSettings,
    QwenActivationReconstructor,
    QwenActivationVerbalizer,
    final_non_padding_pool,
    qwen_checkpoint_metadata,
)
from scripts.train_ar import TargetTransform
from scripts.train_qwen_ar import build_examples as build_ar_examples
from scripts.train_qwen_av import build_examples as build_av_examples


class FakeQwenCausalLM(nn.Module):
    def __init__(self, *, hidden_size: int = 5, vocab_size: int = 19) -> None:
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


def fake_artifact_rows() -> list[dict]:
    return [
        {
            "activation_index": 0,
            "example_id": "ex_0",
            "reference_description": "add two numbers",
            "prompt": "prompt text",
            "code": "def add(a, b): return a + b",
            "split": "train",
            "language": "python",
            "transformation_type": "original",
        }
    ]


def test_qwen_av_activation_projection_shape() -> None:
    model = QwenActivationVerbalizer(
        qwen_model=FakeQwenCausalLM(hidden_size=5),
        activation_dim=3,
    )

    projected = model.project_activation(torch.zeros((2, 3)))

    assert projected.shape == (2, 5)


def test_qwen_av_pseudo_token_forward_shape() -> None:
    model = QwenActivationVerbalizer(
        qwen_model=FakeQwenCausalLM(hidden_size=5, vocab_size=19),
        activation_dim=3,
    )
    input_ids = torch.tensor([[1, 2, 0], [3, 4, 5]])
    attention_mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
    activations = torch.zeros((2, 3))

    outputs = model(
        activations=activations,
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=input_ids,
    )

    assert outputs.logits.shape == (2, 4, 19)
    assert outputs.loss is not None


def test_qwen_ar_projection_output_shape() -> None:
    model = QwenActivationReconstructor(
        qwen_model=FakeQwenCausalLM(hidden_size=5),
        activation_dim=3,
    )
    input_ids = torch.tensor([[1, 2, 0], [3, 4, 5]])
    attention_mask = torch.tensor([[1, 1, 0], [1, 1, 1]])

    reconstructed = model(input_ids=input_ids, attention_mask=attention_mask)

    assert reconstructed.shape == (2, 3)


def test_final_non_padding_pool_selects_last_unpadded_token() -> None:
    hidden = torch.arange(2 * 3 * 2, dtype=torch.float32).reshape(2, 3, 2)
    attention_mask = torch.tensor([[1, 1, 0], [1, 1, 1]])

    pooled = final_non_padding_pool(hidden, attention_mask)

    assert torch.equal(pooled[0], hidden[0, 1])
    assert torch.equal(pooled[1], hidden[1, 2])


def test_target_transform_standardize_roundtrip() -> None:
    targets = torch.tensor([[1.0, 2.0], [3.0, 6.0], [5.0, 10.0]])
    transform = TargetTransform.fit("standardize", targets)

    transformed = transform.transform(targets)
    restored = transform.inverse_transform(transformed)

    assert torch.allclose(restored, targets)


def test_qwen_text_fallback_logic_for_ar_and_av() -> None:
    class Artifact:
        activation_dir = "fake"
        activations = torch.zeros((1, 2))
        metadata_rows = fake_artifact_rows()
        manifest = {}

        @property
        def num_examples(self) -> int:
            return 1

        @property
        def activation_dim(self) -> int:
            return 2

    ar_examples = build_ar_examples(
        artifact=Artifact(),
        transformed_targets=torch.zeros((1, 2)),
        text_field="reference_description",
        fallback_text_fields=["prompt", "code"],
    )
    av_examples = build_av_examples(
        artifact=Artifact(),
        target_text_field="reference_description",
        fallback_text_fields=["prompt", "code"],
    )

    assert ar_examples[0].text == "add two numbers"
    assert ar_examples[0].metadata["qwen_ar_text_field"] == "reference_description"
    assert av_examples[0].target_text == "add two numbers"
    assert av_examples[0].metadata["qwen_av_target_field"] == "reference_description"


def test_qwen_checkpoint_metadata_schema() -> None:
    metadata = qwen_checkpoint_metadata(
        component="qwen_ar",
        model_name_or_path="Qwen/Qwen2.5-Coder-0.5B-Instruct",
        activation_dim=1536,
        dtype="bfloat16",
        lora_settings=LoraSettings(r=8, alpha=16, dropout=0.05),
        extra_config={"text_field": "reference_description"},
    )

    assert metadata["component"] == "qwen_ar"
    assert metadata["activation_dim"] == 1536
    assert metadata["lora"]["enabled"] is True
    assert metadata["lora"]["r"] == 8
    assert metadata["text_field"] == "reference_description"


def test_qwen_checkpoint_metadata_rejects_unknown_component() -> None:
    with pytest.raises(ValueError, match="Unsupported Qwen component"):
        qwen_checkpoint_metadata(
            component="unknown",
            model_name_or_path="qwen",
            activation_dim=2,
            dtype="float32",
            lora_settings=LoraSettings(r=0, alpha=16, dropout=0.0),
        )
