"""Request models for the profile admin endpoints.

These are documentation that is also enforced -- with one caveat specific to this
module. The admin handlers read the body off the raw ``Request``, exactly as the
decision endpoints do, so the byte cap can refuse an oversized body before it is
parsed into objects. FastAPI infers request schemas from handler signatures, so with
a raw-``Request`` signature it infers nothing, and these models are attached to the
operations by hand in ``app.openapi``.

That makes the enforcement asymmetric, and it is worth being explicit about it:
``create`` and ``update`` do validate ``questions`` through
``app.profiles.QUESTIONS_ADAPTER`` -- the same discriminated union
``app.schemas.PredictRequest`` uses -- so a malformed question is a real 422, not
merely an undocumented shape. What these models add on top is the declared shape of
the surrounding fields, so a console or a generated client has something to build
against. They are kept honest by a test that asserts every field here appears in the
served document.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from app.schemas import Question


class ProfileRouting(BaseModel):
    """Checkpoint pins. A pinned field is authoritative over the request."""

    model_config = ConfigDict(extra="forbid")

    model: Optional[str] = None
    task: Optional[str] = None
    lang: Optional[str] = Field(
        default=None, description="Pin a language hint such as 'de' or 'pt-BR'."
    )


class ProfileBudgets(BaseModel):
    """Token-budget defaults applied unless a request overrides them."""

    model_config = ConfigDict(extra="forbid")

    max_len: Optional[int] = Field(default=None, ge=1)
    head_max_len: Optional[int] = Field(default=None, ge=1)


class ProfileCreateRequest(BaseModel):
    """Create a profile. On success its endpoint exists."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(
        min_length=1,
        max_length=64,
        description=(
            "URL-safe identifier, lowercase letters, digits, '-' or '_'. Becomes the "
            "path segment of the minted endpoint."
        ),
    )
    name: str = Field(min_length=1, max_length=200)
    description: Optional[str] = Field(default=None, max_length=2000)
    questions: Dict[str, Question] = Field(min_length=1)
    routing: ProfileRouting = Field(default_factory=ProfileRouting)
    budgets: ProfileBudgets = Field(default_factory=ProfileBudgets)


class ProfileUpdateRequest(BaseModel):
    """Update a profile.

    Changing ``questions`` retires the profile's calibration rather than leaving it
    in place: temperatures fitted to one question set mean nothing on another, and a
    stale threshold that still looked active is the failure this service cannot
    afford.
    """

    model_config = ConfigDict(extra="forbid")

    name: Optional[str] = Field(default=None, min_length=1, max_length=200)
    description: Optional[str] = Field(default=None, max_length=2000)
    questions: Optional[Dict[str, Question]] = Field(default=None, min_length=1)
    routing: Optional[ProfileRouting] = None
    budgets: Optional[ProfileBudgets] = None


class LabelledExample(BaseModel):
    """One ground-truth row.

    ``expected`` is keyed by question id and holds what a person would write: an
    option label, a level name or index, or a bool for ``noul``. It is expanded to the
    one-hot vector the fitter needs, so no positional convention leaks into the file
    format.
    """

    model_config = ConfigDict(extra="forbid")

    state: Any = Field(description="The text or document that was decided on.")
    expected: Dict[str, Any] = Field(
        min_length=1, description="Question id -> correct label, level or boolean."
    )
    questions: Optional[Dict[str, Question]] = Field(
        default=None,
        description="Optional per-row question override; omit to use the profile's own.",
    )
    tags: Optional[List[str]] = None


class ExamplesRequest(BaseModel):
    """Replace a profile's labelled examples wholesale.

    Replace rather than append: a fit is over the whole set, so a partial append
    changes what the next calibration means, and mixing two generations of labels
    silently is how a fit ends up describing neither.
    """

    model_config = ConfigDict(extra="forbid")

    examples: List[LabelledExample] = Field(min_length=1)


class CalibrationFitRequest(BaseModel):
    """Fit temperatures and an abstention threshold from the stored examples."""

    model_config = ConfigDict(extra="forbid")

    target_error: Optional[float] = Field(
        default=None,
        gt=0.0,
        lt=1.0,
        description=(
            "Error rate the fitted threshold targets. 0.10 means: answer unless doing "
            "so would exceed a 10% error rate."
        ),
    )


class ActivateRequest(BaseModel):
    """Adopt a fitted threshold for this profile.

    Takes no fields. Activation is refused when the artifact is stale, and refused
    when another profile already holds the checkpoint's temperature -- so the request
    needs nothing to say and cannot be argued with.
    """

    model_config = ConfigDict(extra="forbid")
