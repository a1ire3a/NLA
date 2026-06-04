"""Dataset preparation utilities for code-level NLA experiments."""

from __future__ import annotations


def normalize_code_snippet(code: str) -> str:
    """Normalize a code snippet for prompt construction.

    This intentionally starts conservative. More transformations will be added later.
    """
    return code.strip()
