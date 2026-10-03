"""Profiles: named question sets that mint their own endpoint.

A profile is a vetted decision specification -- a question set, optional routing
pins, optional budget defaults -- stored on disk and registered as a real route at
runtime. ``POST /v1/profiles/{id}/predict`` then takes only a ``state``, because the
questions are the profile's.

Two design decisions carry the weight here.

**Questions are validated with the same models the fixed endpoint uses.** A profile
is not a second, looser question dialect; importing the discriminated union from
``app.schemas`` means a question that the fixed endpoint would reject at request
time is rejected at creation time instead, where the operator can see why.

**The fingerprint is over the canonical question set, not the stored JSON.** It is
what decides whether a calibration still applies, so it must not change when
someone reorders keys or the serialiser changes its spacing. ``sort_keys`` plus
``separators`` gives a stable byte sequence, and SHA-256 over it is a value that
can be compared across restarts and across machines.

Persistence is one JSON file per profile, replaced atomically. The service runs a
single worker, so there is no concurrent writer to lose an interleaved update to,
and a file per profile stays inspectable and diffable -- which matters more here
than query speed would, since a profile is a thing a human argues over.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import HTTPException
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)

from app import decision, limits
from app.schemas import Question

# A profile id becomes a path segment and a Prometheus label value, so it is
# restricted to characters that cannot escape a directory or a URL path. `..` is
# excluded by the leading-character rule as much as by the pattern.
ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")

# The same discriminated union the fixed endpoint validates against. A profile is
# not a second, looser question dialect.
QUESTIONS_ADAPTER: TypeAdapter = TypeAdapter(Dict[str, Question])


class ProfileError(HTTPException):
    """A profile operation failed in a way the caller should see and can fix."""


def bad_request(detail: str) -> HTTPException:
    return HTTPException(status_code=422, detail=detail)


class Routing(BaseModel):
    """Optional checkpoint pins.

    A pinned field is authoritative: a caller may repeat the same value, and
    anything else is refused rather than silently overridden. Letting a request
    choose the checkpoint a profile was calibrated against would make the
    calibration meaningless while still appearing to apply.
    """

    model_config = ConfigDict(extra="forbid")

    model: Optional[str] = Field(
        default=None,
        description="Pin a checkpoint: 'english', 'multilingual', 'typed-decisions', or a Hub id.",
    )
    task: Optional[str] = Field(
        default=None, description="Force a checkpoint by workflow name."
    )
    lang: Optional[str] = Field(
        default=None, description="Pin a language hint such as 'de' or 'pt-BR'."
    )

    def pinned(self) -> Dict[str, Any]:
        return {k: v for k, v in self.model_dump().items() if v is not None}


class Budgets(BaseModel):
    """Optional token-budget defaults for this profile.

    A profile default applies unless the request says otherwise, so a caller who
    needs a wider window on one call can still have it. Both are still capped by
    the server's ``token_budget`` at request time.
    """

    model_config = ConfigDict(extra="forbid")

    max_len: Optional[int] = Field(default=None, ge=1)
    head_max_len: Optional[int] = Field(default=None, ge=1)


def canonical_questions(questions: Dict[str, Any]) -> Dict[str, Any]:
    """Normalise questions to the exact shape both inference and the fingerprint use.

    The list form of ``criteria`` is expanded to a mapping here, so a profile whose
    options were written as bare labels fingerprints identically to the same profile
    written as label/description pairs. They are the same question to the model, and
    a calibration must not be invalidated by a change of spelling.
    """
    return {qid: question.as_laya() for qid, question in questions.items()}


def questions_fingerprint(questions: Dict[str, Any]) -> str:
    """A stable digest of a question set.

    SHA-256 over canonical JSON: keys sorted, no insignificant whitespace. Two
    fingerprints being equal is the definition of "these question sets are the same
    question set", which is what decides whether a calibration artifact may be
    applied.
    """
    canonical = json.dumps(
        questions, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class Profile(BaseModel):
    """A stored, validated question set with its routing pins."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="URL-safe identifier; becomes the endpoint path segment.")
    name: str = Field(min_length=1, max_length=200, description="Human label for the console.")
    description: Optional[str] = Field(default=None, max_length=2000)
    questions: Dict[str, Question] = Field(
        min_length=1, description="Question id -> question definition."
    )
    routing: Routing = Field(default_factory=Routing)
    budgets: Budgets = Field(default_factory=Budgets)
    questions_fingerprint: str = Field(
        description="Digest of the canonical question set. Recomputed on every write."
    )
    created_at: str
    updated_at: str

    @field_validator("id")
    @classmethod
    def id_must_be_safe(cls, value: str) -> str:
        if not ID_PATTERN.match(value):
            raise ValueError(
                "id must be 1-64 characters of lowercase letters, digits, '-' or '_', "
                "starting with a letter or digit"
            )
        return value

    @model_validator(mode="after")
    def fingerprint_must_match(self) -> "Profile":
        """Refuse a stored profile whose fingerprint disagrees with its questions.

        The fingerprint gates calibration, so a mismatch means the gate is reading a
        stale digest and would apply an artifact to questions it was not fitted on.
        Recomputing it here would hide the corruption instead of refusing it.
        """
        actual = questions_fingerprint(canonical_questions(self.questions))
        if actual != self.questions_fingerprint:
            raise ValueError(
                "questions_fingerprint does not match the question set; the stored "
                "profile is inconsistent and calibration cannot be trusted against it"
            )
        return self

    def check_limits(self, cfg: Any) -> None:
        """Refuse a profile that would fail every request it is asked to answer.

        The limits are per-deployment, so a profile valid on one host can be invalid
        on another with tighter caps. Checking at write time turns a profile that
        cannot work into a 422 naming the limit, rather than a profile that exists
        and 413s forever.
        """
        limits.check_question_count(self.questions, cfg.limits.questions)
        limits.check_option_weights(
            self.questions,
            max_choice_options=cfg.limits.choice_options,
            max_score_levels=cfg.limits.score_levels,
            max_total_options=cfg.limits.total_options,
        )
        for name in ("max_len", "head_max_len"):
            value = getattr(self.budgets, name)
            try:
                limits.clamp_budget(name, value, cfg.limits.token_budget)
            except HTTPException as exc:
                raise bad_request(
                    f"profile default {exc.detail} (profile {self.id!r})"
                ) from None

    def as_laya_questions(self) -> Dict[str, Dict[str, Any]]:
        return canonical_questions(self.questions)

    def option_total(self) -> int:
        return sum(question.option_count for question in self.questions.values())

    def summary(self) -> Dict[str, Any]:
        """What the console list needs, without the question bodies."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "routing": self.routing.pinned(),
            "budgets": self.budgets.model_dump(exclude_none=True),
            "question_ids": sorted(self.questions),
            "option_total": self.option_total(),
            "questions_fingerprint": self.questions_fingerprint,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "endpoint": f"/v1/profiles/{self.id}/predict",
        }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_atomic(path: Path, payload: Dict[str, Any]) -> None:
    """Replace a file's contents without ever exposing a partial write.

    A reader -- the next process, or the console mid-refresh -- must see either the
    old file or the new one. Writing in place would leave a truncated file if the
    process died between the truncate and the write, and a profile store that can
    deserialize a half-written profile is worse than one that reports a missing
    file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    except BaseException:
        # The rename never happened, so the temporary file is still ours to remove.
        try:
            os.unlink(temp_name)
        except OSError:  # pragma: no cover - the replace already consumed it
            pass
        raise


class ProfileStore:
    """Profiles, their labelled examples, and their calibration artifacts on disk.

    Layout::

        data/profiles/<id>.json          the profile
        data/profiles/<id>.examples.jsonl labelled rows, one per line
        data/profiles/<id>.calibration.json  the fitted artifact

    One directory, so deleting a profile is deleting a directory and there is no
    index to fall out of sync with the files.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.profiles_dir = self.root / "profiles"

    # -- paths ------------------------------------------------------------------

    def profile_path(self, profile_id: str) -> Path:
        # Validated before it reaches here, but re-checked because this is the
        # function that turns an id into filesystem and URL structure.
        if not ID_PATTERN.match(profile_id):
            raise bad_request(f"invalid profile id {profile_id!r}")
        return self.profiles_dir / f"{profile_id}.json"

    def examples_path(self, profile_id: str) -> Path:
        return self.profiles_dir / f"{profile_id}.examples.jsonl"

    def calibration_path(self, profile_id: str) -> Path:
        return self.profiles_dir / f"{profile_id}.calibration.json"

    # -- reads ------------------------------------------------------------------

    def list(self) -> List[Profile]:
        if not self.profiles_dir.is_dir():
            return []
        found: List[Profile] = []
        for path in sorted(self.profiles_dir.glob("*.json")):
            # `.calibration.json` shares the glob; only a profile document has an id.
            if path.name.endswith(".calibration.json"):
                continue
            try:
                found.append(self._read(path))
            except (ValidationError, json.JSONDecodeError, OSError):
                # A file that will not parse is skipped rather than fatal: one
                # corrupt profile should not make the other thirty unreachable.
                continue
        return sorted(found, key=lambda profile: profile.id)

    @property
    def count(self) -> int:
        """How many profiles exist.

        Counts files rather than parsing them, so it stays cheap enough to call from
        ``/health`` on every scrape. A profile that would fail validation is counted
        here and then skipped by ``list``, which is the right way round: the cap exists
        to bound the number of minted endpoints on disk, not to bless what is in them.
        """
        if not self.profiles_dir.is_dir():
            return 0
        return sum(
            1
            for path in self.profiles_dir.glob("*.json")
            if not path.name.endswith(".calibration.json")
        )

    def get(self, profile_id: str) -> Profile:
        path = self.profile_path(profile_id)
        if not path.is_file():
            raise HTTPException(
                status_code=404, detail=f"no profile with id {profile_id!r}"
            )
        try:
            return self._read(path)
        except ValidationError as exc:
            # The route exists on disk but does not describe a usable profile, so
            # it is not registered and must not be answerable.
            raise HTTPException(
                status_code=500,
                detail=f"profile {profile_id!r} is stored in an invalid state",
            ) from exc
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=500, detail=f"profile {profile_id!r} is not valid JSON"
            ) from exc

    def _read(self, path: Path) -> Profile:
        return Profile.model_validate(json.loads(path.read_text(encoding="utf-8")))

    # -- writes -----------------------------------------------------------------

    def create(self, profile_id: str, body: Dict[str, Any], cfg: Any) -> Profile:
        path = self.profile_path(profile_id)
        if path.exists():
            raise HTTPException(
                status_code=409, detail=f"profile {profile_id!r} already exists"
            )
        stamp = _now()
        profile = self._build({**body, "id": profile_id, "created_at": stamp, "updated_at": stamp})
        profile.check_limits(cfg)
        write_atomic(path, profile.model_dump(mode="json"))
        return profile

    def update(self, profile_id: str, body: Dict[str, Any], cfg: Any) -> Profile:
        existing = self.get(profile_id)
        merged = {
            **existing.model_dump(mode="json"),
            **body,
            "id": profile_id,
            "updated_at": _now(),
        }
        profile = self._build(merged)
        profile.check_limits(cfg)
        write_atomic(self.profile_path(profile_id), profile.model_dump(mode="json"))
        return profile

    def _build(self, body: Dict[str, Any]) -> Profile:
        """Validate the questions, then derive the fingerprint from the result.

        Questions are validated on their own first, because the fingerprint has to
        be computed from their canonical form and ``Profile`` refuses a fingerprint
        that disagrees with its questions. Deriving it here -- rather than accepting
        it from the caller -- means it cannot be forged, and cannot be left stale by
        a writer who does not know the field exists.
        """
        raw_questions = body.get("questions")
        if not isinstance(raw_questions, dict) or not raw_questions:
            raise bad_request("'questions' must be a non-empty object of id -> question")
        try:
            questions = QUESTIONS_ADAPTER.validate_python(raw_questions)
        except ValidationError as exc:
            raise bad_request(
                {
                    "msg": "questions are invalid",
                    "detail": decision.format_validation_error(
                        exc, loc_prefix=("questions",)
                    ),
                }
            ) from None

        fingerprint = questions_fingerprint(canonical_questions(questions))
        try:
            return Profile.model_validate(
                {**body, "questions": questions, "questions_fingerprint": fingerprint}
            )
        except ValidationError as exc:
            raise bad_request(
                {
                    "msg": "profile is invalid",
                    "detail": decision.format_validation_error(exc),
                }
            ) from None

    def delete(self, profile_id: str) -> None:
        self.get(profile_id)
        for path in (
            self.profile_path(profile_id),
            self.examples_path(profile_id),
            self.calibration_path(profile_id),
        ):
            try:
                path.unlink()
            except FileNotFoundError:
                pass

    # -- examples ---------------------------------------------------------------

    def read_examples(self, profile_id: str) -> List[Dict[str, Any]]:
        path = self.examples_path(profile_id)
        if not path.is_file():
            return []
        rows: List[Dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            rows.append(json.loads(stripped))
        return rows

    def replace_examples(self, profile_id: str, rows: List[Dict[str, Any]]) -> int:
        payload = "\n".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows
        )
        path = self.examples_path(profile_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        handle, temp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                stream.write(payload + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_name, path)
        except BaseException:
            try:
                os.unlink(temp_name)
            except OSError:  # pragma: no cover
                pass
            raise
        return len(rows)

    # -- calibration ------------------------------------------------------------

    def read_calibration(self, profile_id: str) -> Optional[Dict[str, Any]]:
        path = self.calibration_path(profile_id)
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            # An unreadable artifact is treated as absent rather than fatal: the
            # consequence of "no calibration" is an ungated confidence value, and
            # the consequence of guessing at a corrupt one is a wrong threshold.
            return None

    def write_calibration(self, profile_id: str, payload: Dict[str, Any]) -> None:
        write_atomic(self.calibration_path(profile_id), payload)

    def clear_calibration(self, profile_id: str) -> None:
        try:
            self.calibration_path(profile_id).unlink()
        except FileNotFoundError:
            pass
