"""Prompt templates for code-semantics activation extraction."""


def code_explanation_prompt_v1(code: str) -> str:
    """Return the initial standardized prompt for function-level code understanding."""
    return (
        "Read the following function and prepare to explain what it does.\n\n"
        "<code>\n"
        f"{code}\n"
        "</code>\n\n"
        "Explanation:"
    )
