from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from nla_code_interp.models import (
    ActivationVerbalizer,
    build_activation_lm_labels,
    prepend_activation_attention_mask,
    prepend_activation_embedding,
)
from scripts.train_av import select_target_text


class FakeCausalLM(nn.Module):
    def __init__(self, *, embedding_dim: int = 5, vocab_size: int = 17) -> None:
        super().__init__()
        self.config = SimpleNamespace(n_embd=embedding_dim, vocab_size=vocab_size)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.output = nn.Linear(embedding_dim, vocab_size)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embedding

    def forward(
        self,
        *,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ):
        del attention_mask
        logits = self.output(inputs_embeds)
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss(ignore_index=-100)(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
            )
        return SimpleNamespace(logits=logits, loss=loss)


def test_activation_projection_shape() -> None:
    model = ActivationVerbalizer(
        text_model_name_or_path="fake",
        activation_dim=3,
        language_model=FakeCausalLM(embedding_dim=5),
        lm_embedding_dim=5,
    )

    projected = model.project_activation(torch.zeros((2, 3)))

    assert projected.shape == (2, 5)


def test_activation_pseudo_token_concatenation_shape() -> None:
    token_embeddings = torch.zeros((2, 4, 5))
    activation_embedding = torch.ones((2, 5))

    combined = prepend_activation_embedding(
        token_embeddings=token_embeddings,
        activation_embedding=activation_embedding,
    )

    assert combined.shape == (2, 5, 5)
    assert torch.equal(combined[:, 0, :], activation_embedding)


def test_attention_and_label_prepending_masks_pseudo_and_padding() -> None:
    input_ids = torch.tensor([[1, 2, 0], [3, 4, 5]])
    attention_mask = torch.tensor([[1, 1, 0], [1, 1, 1]])

    extended_attention = prepend_activation_attention_mask(attention_mask)
    labels = build_activation_lm_labels(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    assert extended_attention.shape == (2, 4)
    assert torch.equal(extended_attention[:, 0], torch.ones(2, dtype=torch.long))
    assert labels.shape == (2, 4)
    assert torch.equal(labels[:, 0], torch.full((2,), -100))
    assert labels[0, 3].item() == -100
    assert labels[1, 3].item() == 5


def test_fake_av_forward_returns_logits_and_loss() -> None:
    model = ActivationVerbalizer(
        text_model_name_or_path="fake",
        activation_dim=3,
        language_model=FakeCausalLM(embedding_dim=5, vocab_size=17),
        lm_embedding_dim=5,
    )
    activations = torch.zeros((2, 3))
    input_ids = torch.tensor([[1, 2, 0], [3, 4, 5]])
    attention_mask = torch.tensor([[1, 1, 0], [1, 1, 1]])

    outputs = model(
        activations=activations,
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=input_ids,
    )

    assert outputs.logits.shape == (2, 4, 17)
    assert outputs.loss is not None


def test_target_text_selection_and_deterministic_fallback() -> None:
    row = {
        "example_id": "ex_0",
        "language": "python",
        "transformation_type": "original",
        "reference_description": "",
        "code": "def add(a, b): return a + b",
        "prompt": "Explain this code",
    }

    text, field = select_target_text(
        row,
        target_text_field="reference_description",
        fallback_text_fields=["code", "prompt"],
    )

    assert text == "def add(a, b): return a + b"
    assert field == "code"

    empty_row = {
        "example_id": "ex_1",
        "language": "python",
        "transformation_type": "original",
    }
    fallback_text, fallback_field = select_target_text(
        empty_row,
        target_text_field="reference_description",
        fallback_text_fields=["code", "prompt"],
    )

    assert fallback_field == "deterministic_fallback"
    assert "Describe the python function" in fallback_text


def test_invalid_pseudo_token_shapes_raise() -> None:
    with pytest.raises(ValueError, match="Hidden dimension mismatch"):
        prepend_activation_embedding(
            token_embeddings=torch.zeros((2, 3, 4)),
            activation_embedding=torch.zeros((2, 5)),
        )
