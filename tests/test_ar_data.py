from __future__ import annotations

from scripts.train_ar import (
    build_text_examples,
    select_text_for_row,
    split_train_validation_indices,
)


def test_text_fallback_uses_reference_description_first() -> None:
    row = {
        "example_id": "ex_0",
        "reference_description": " reference text ",
        "prompt": "prompt text",
        "code": "code text",
    }

    text, selected_field = select_text_for_row(
        row,
        text_field="reference_description",
        fallback_text_fields=["prompt", "code"],
    )

    assert text == "reference text"
    assert selected_field == "reference_description"


def test_text_fallback_uses_prompt_then_code() -> None:
    prompt_row = {
        "example_id": "ex_1",
        "reference_description": None,
        "prompt": "prompt text",
        "code": "code text",
    }
    code_row = {
        "example_id": "ex_2",
        "reference_description": "",
        "prompt": " ",
        "code": "code text",
    }

    prompt_text, prompt_field = select_text_for_row(
        prompt_row,
        text_field="reference_description",
        fallback_text_fields=["prompt", "code"],
    )
    code_text, code_field = select_text_for_row(
        code_row,
        text_field="reference_description",
        fallback_text_fields=["prompt", "code"],
    )

    assert prompt_text == "prompt text"
    assert prompt_field == "prompt"
    assert code_text == "code text"
    assert code_field == "code"


def test_build_text_examples_records_selected_field() -> None:
    rows = [
        {
            "example_id": "ex_0",
            "reference_description": "",
            "prompt": "prompt text",
            "code": "code text",
        }
    ]

    examples = build_text_examples(
        rows,
        text_field="reference_description",
        fallback_text_fields=["prompt", "code"],
    )

    assert examples[0].text == "prompt text"
    assert examples[0].metadata["ar_text_field"] == "prompt"
    assert examples[0].metadata["ar_source_index"] == 0


def test_deterministic_train_validation_split_is_stable() -> None:
    rows = [{"split": "pilot", "example_id": f"ex_{index}"} for index in range(10)]

    train_1, validation_1, strategy_1 = split_train_validation_indices(
        rows,
        validation_fraction=0.2,
        seed=42,
    )
    train_2, validation_2, strategy_2 = split_train_validation_indices(
        rows,
        validation_fraction=0.2,
        seed=42,
    )

    assert strategy_1 == "deterministic_random_split"
    assert strategy_2 == "deterministic_random_split"
    assert train_1 == train_2
    assert validation_1 == validation_2
    assert len(train_1) == 8
    assert len(validation_1) == 2
    assert set(train_1).isdisjoint(validation_1)


def test_explicit_train_validation_split_is_respected() -> None:
    rows = [
        {"split": "train", "example_id": "ex_0"},
        {"split": "validation", "example_id": "ex_1"},
        {"split": "train", "example_id": "ex_2"},
    ]

    train, validation, strategy = split_train_validation_indices(
        rows,
        validation_fraction=0.2,
        seed=42,
    )

    assert strategy == "metadata_split"
    assert train == [0, 2]
    assert validation == [1]
