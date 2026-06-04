"""Activation extraction utilities.

The implementation will use Hugging Face Transformers with `output_hidden_states=True`.
"""

from __future__ import annotations


def select_final_non_padding_index(attention_mask):
    """Return final non-padding token index for each sequence.

    Placeholder for the implementation step.
    """
    raise NotImplementedError("Implemented in the activation extraction step.")
