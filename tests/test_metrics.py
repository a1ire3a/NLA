from __future__ import annotations

import torch

from nla_code_interp.metrics import fraction_variance_explained


def test_fraction_variance_explained_perfect_reconstruction() -> None:
    original = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    reconstructed = original.clone()
    score = fraction_variance_explained(original, reconstructed)
    assert torch.isclose(score, torch.tensor(1.0))
