"""Guard unit tests.

These bypass HTTP entirely. The guards are the part most likely to be edited later
and least likely to be exercised by the API tests if those only check status codes,
so each one gets a direct assertion on the number it measures and the message it
gives.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from app import limits


# --- state ---------------------------------------------------------------------


def test_state_under_the_limit_passes() -> None:
    limits.check_state("hello", 50_000)  # must not raise


def test_state_over_the_limit_names_both_numbers() -> None:
    with pytest.raises(HTTPException) as excinfo:
        limits.check_state("x" * 101, 100)

    assert excinfo.value.status_code == 413
    detail = str(excinfo.value.detail)
    # Both the measured size and the cap: the caller can tell whether to trim or
    # to ask for a different deployment.
    assert "101" in detail and "100" in detail


def test_empty_string_state_is_below_the_limit() -> None:
    # Not this layer's call to make: an empty state is a degenerate question, not an
    # oversized one, and inventing a rule here would be a surprise.
    limits.check_state("", 100)


# --- question sets --------------------------------------------------------------


def choice(qid: str, count: int):
    from app.schemas import ChoiceQuestion

    return ChoiceQuestion(
        type="choice",
        instructions="Which?",
        criteria={f"opt{i}": f"Option {i}" for i in range(count)},
    )


def score(qid: str, count: int):
    from app.schemas import ScoreQuestion

    return ScoreQuestion(
        type="score",
        instructions="How severe?",
        criteria=[f"level{i}" for i in range(count)],
    )


def test_choice_option_limit_is_enforced() -> None:
    with pytest.raises(HTTPException) as excinfo:
        limits.check_option_weights(
            {"q1": choice("q1", 33)},
            max_choice_options=32,
            max_score_levels=32,
            max_total_options=512,
        )

    assert excinfo.value.status_code == 413


def test_score_level_limit_is_enforced_separately() -> None:
    with pytest.raises(HTTPException) as excinfo:
        limits.check_option_weights(
            {"s1": score("s1", 40)},
            max_choice_options=32,
            max_score_levels=32,
            max_total_options=512,
        )

    assert excinfo.value.status_code == 413
    assert "level" in str(excinfo.value.detail).lower()


def test_total_option_limit_catches_an_individually_fine_set() -> None:
    # 10 questions x 3 options each is under both per-question limits but over the
    # total. The total is the real constraint, so this is the case that matters.
    questions = {f"q{i}": choice(f"q{i}", 3) for i in range(10)}

    with pytest.raises(HTTPException) as excinfo:
        limits.check_option_weights(
            questions, max_choice_options=32, max_score_levels=32, max_total_options=16
        )

    assert excinfo.value.status_code == 413


def test_a_set_at_exactly_the_limit_passes() -> None:
    # Off-by-one guard: rejecting the limit itself would make the documented
    # maximum a lie.
    limits.check_option_weights(
        {"q1": choice("q1", 32)},
        max_choice_options=32,
        max_score_levels=32,
        max_total_options=512,
    )


# --- token budgets ---------------------------------------------------------------


def test_budget_under_the_cap_passes_through() -> None:
    assert limits.clamp_budget("max_len", 512, 8192) == 512


def test_budget_at_the_cap_passes_through() -> None:
    assert limits.clamp_budget("max_len", 8192, 8192) == 8192


def test_budget_over_the_cap_is_refused_with_both_numbers() -> None:
    with pytest.raises(HTTPException) as excinfo:
        limits.clamp_budget("max_len", 8193, 8192)

    assert excinfo.value.status_code == 422
    detail = str(excinfo.value.detail)
    assert "8193" in detail and "8192" in detail


def test_absent_budget_stays_absent() -> None:
    # None means "use the checkpoint's default", which must survive as None rather
    # than collapsing to 0 or the cap.
    assert limits.clamp_budget("max_len", None, 8192) is None


def test_bool_is_not_an_integer_budget() -> None:
    # bool is an int subclass, so this would otherwise pass and then be handed to
    # the tokenizer as a window of 1.
    with pytest.raises(HTTPException):
        limits.clamp_budget("max_len", True, 8192)


# --- surrogates ------------------------------------------------------------------


def test_lone_surrogate_is_detected() -> None:
    payload = {"state": "bad \ud800 text"}

    assert limits.has_lone_surrogate(payload) is True


def test_well_formed_non_ascii_is_not_flagged() -> None:
    payload = {"state": "नमस्ते दुनिया — ünïcödé"}

    assert limits.has_lone_surrogate(payload) is False


def test_surrogate_inside_a_nested_structure_is_found() -> None:
    payload = {"state": {"subject": "ok", "body": ["fine", "\udfff"]}}

    assert limits.has_lone_surrogate(payload) is True
