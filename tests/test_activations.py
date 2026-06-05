from __future__ import annotations

import pytest
import torch

from nla_code_interp.activations import (
    select_final_non_padding_indices,
    select_token_activations,
    summarize_activation,
)


def test_select_final_non_padding_indices() -> None:
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 0],
            [1, 0, 1, 0],
            [0, 0, 1, 1],
        ]
    )

    indices = select_final_non_padding_indices(attention_mask)

    assert torch.equal(indices, torch.tensor([2, 2, 3]))


def test_select_final_non_padding_indices_all_padding_error() -> None:
    attention_mask = torch.tensor([[1, 0], [0, 0]])

    with pytest.raises(ValueError, match="at least one non-padding token"):
        select_final_non_padding_indices(attention_mask)


def test_select_token_activations_batched() -> None:
    hidden_state = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],
        ]
    )
    token_indices = torch.tensor([2, 0])

    activations = select_token_activations(hidden_state, token_indices)

    expected = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    assert torch.equal(activations, expected)


def test_select_token_activations_invalid_index_error() -> None:
    hidden_state = torch.zeros((2, 3, 4))
    token_indices = torch.tensor([1, 3])

    with pytest.raises(ValueError, match="out of range"):
        select_token_activations(hidden_state, token_indices)


def test_summarize_activation_keys() -> None:
    activation = torch.tensor([[1.0, 2.0, 3.0]])

    summary = summarize_activation(activation)

    assert set(summary) == {"mean", "std", "min", "max", "l2_norm"}
    assert summary["mean"] == pytest.approx(2.0)
    expected_l2 = torch.tensor([1.0, 2.0, 3.0]).norm().item()
    assert summary["l2_norm"] == pytest.approx(expected_l2)
