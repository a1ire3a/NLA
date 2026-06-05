from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from nla_code_interp.models import (
    TextActivationReconstructor,
    count_trainable_parameters,
    mean_pool_last_hidden_state,
)


class FakeTextModel(nn.Module):
    def __init__(self, *, hidden_size: int = 4, vocab_size: int = 32) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.embedding = nn.Embedding(vocab_size, hidden_size)

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ):
        del attention_mask
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


def test_masked_mean_pooling_ignores_padding() -> None:
    hidden_state = torch.tensor(
        [
            [[1.0, 1.0], [3.0, 3.0], [99.0, 99.0]],
            [[2.0, 0.0], [4.0, 2.0], [6.0, 4.0]],
        ]
    )
    attention_mask = torch.tensor([[1, 1, 0], [1, 1, 1]])

    pooled = mean_pool_last_hidden_state(hidden_state, attention_mask)

    expected = torch.tensor([[2.0, 2.0], [4.0, 2.0]])
    assert torch.allclose(pooled, expected)


def test_masked_mean_pooling_all_padding_error() -> None:
    hidden_state = torch.zeros((1, 2, 3))
    attention_mask = torch.tensor([[0, 0]])

    with pytest.raises(ValueError, match="at least one non-padding token"):
        mean_pool_last_hidden_state(hidden_state, attention_mask)


def test_cls_pooling_returns_first_token_representation() -> None:
    fake_text_model = FakeTextModel(hidden_size=3)
    model = TextActivationReconstructor(
        text_model_name_or_path="fake",
        activation_dim=3,
        pooling="cls",
        freeze_text_model=False,
        text_model=fake_text_model,
        text_hidden_dim=3,
    )
    model.projection = nn.Identity()
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    attention_mask = torch.ones_like(input_ids)

    output = model(input_ids=input_ids, attention_mask=attention_mask)
    expected = fake_text_model(input_ids=input_ids).last_hidden_state[:, 0, :]

    assert torch.allclose(output, expected)


def test_reconstructor_output_shape_with_fake_text_model() -> None:
    model = TextActivationReconstructor(
        text_model_name_or_path="fake",
        activation_dim=7,
        pooling="mean",
        projection_hidden_dim=5,
        dropout=0.0,
        freeze_text_model=True,
        text_model=FakeTextModel(hidden_size=4),
        text_hidden_dim=4,
    )
    input_ids = torch.tensor([[1, 2, 0], [3, 4, 5]])
    attention_mask = torch.tensor([[1, 1, 0], [1, 1, 1]])

    output = model(input_ids=input_ids, attention_mask=attention_mask)

    assert output.shape == (2, 7)


def test_freezing_disables_text_model_gradients() -> None:
    fake_text_model = FakeTextModel(hidden_size=4)
    model = TextActivationReconstructor(
        text_model_name_or_path="fake",
        activation_dim=2,
        freeze_text_model=True,
        text_model=fake_text_model,
        text_hidden_dim=4,
    )

    assert all(not parameter.requires_grad for parameter in model.text_model.parameters())
    assert count_trainable_parameters(model.projection) > 0
    assert count_trainable_parameters(model) == count_trainable_parameters(model.projection)
