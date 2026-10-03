"""Fitting confidence numbers that mean something, per profile.

Laya ships the calibration mathematics; this module is the glue that decides when
it may be trusted and what a caller is allowed to do with the result.

The pipeline, and why each step is where it is::

    rows (JSONL)  ->  expand_target  ->  records_from_labeled
                  ->  fit_temperature_map(compute_ece=True)
                  ->  fit_abstention_thresholds(target_error=...)
                  ->  artifact on disk

**``expand_target`` exists because the SDK wants vectors and humans write labels.**
``_validated_pair`` requires ``len(target) == len(logits)``, so a target must be a
one-hot vector the same width as the question's options. A JSONL file says
``"expected": "high"``. Turning that into ``[0, 1]`` is the entire job here, and it
is a pure function precisely so it can be tested without a checkpoint.

**The ECE holdout is not optional.** ``compute_ece=True`` fits temperatures on a
stratified subset and scores ECE on records the fit never saw. Without it, the
number the console displays is the number the fit optimised, which is not evidence
of anything.

**Below ``MIN_BUCKET_N`` this refuses rather than reporting.** The SDK's per-bucket
floor is 100 examples; under it the bucket is omitted from the map and a
type-level scalar silently covers it. That is a reasonable fitting strategy and a
terrible thing to present as "your calibrated threshold", so the console is told to
collect more examples instead.

**An artifact is bound to the question set that produced it.** The stored
``questions_fingerprint`` is compared on every read; a profile whose questions were
edited since the fit gets ``stale`` and its threshold is not served. A calibration
fitted for one question set and applied to another produces confidently wrong
numbers, and nothing downstream would reveal the mistake.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from fastapi import HTTPException
from pydantic import ValidationError

from app import profiles as profile_mod
from app.schemas import ChoiceQuestion, NoulQuestion, ScoreQuestion

_log = logging.getLogger("laya_service.calibration")

DEFAULT_TARGET_ERROR = 0.10


class CalibrationError(HTTPException):
    """Calibration could not proceed; the detail says what to do about it."""

_SDK: Dict[str, float] = {}


def sdk_floors() -> Dict[str, float]:
    """The SDK's own sample-size floors, read once on first use.

    Imported lazily because pulling in ``laya.calibrate`` drags torch with it, and
    this module is reached from ``app.main`` at import time.

    Two floors matter and they are not the same number, which is easy to get wrong:
    ``MIN_BUCKET_N`` (2000) is what ``fit_temperature_map`` requires before it emits
    a per-bucket temperature, and ``MIN_TYPE_N`` (10) is what it requires before it
    emits a type-level one at all. Between them the fit still succeeds -- it just
    produces a single scalar per question type rather than per option-count bucket,
    and that is a materially different claim from the one a console would display.
    """
    if not _SDK:
        from laya import calibrate as lc

        _SDK["bucket"] = int(lc.MIN_BUCKET_N)
        _SDK["type"] = int(lc.MIN_TYPE_N)
        _SDK["holdout"] = float(lc.ECE_HOLDOUT_FRAC)
    return _SDK


def min_bucket_n() -> int:
    return int(sdk_floors()["bucket"])


def min_type_n() -> int:
    return int(sdk_floors()["type"])


# Scope of a completed fit, reported alongside the numbers.
SCOPE_NONE = "unusable"
SCOPE_TYPE_LEVEL = "type-level"
SCOPE_BUCKETED = "bucketed"


def scope_for(record_count: int) -> str:
    """Whether there is enough data to fit anything at all.

    Only used as a pre-flight refusal. It deliberately does *not* decide the reported
    scope, because per-bucket temperatures depend on each bucket's own count rather
    than the total: 2800 records spread over two buckets of 1400 clears the 2000
    floor overall while qualifying for neither. The reported scope is read back off
    the fitted artifact instead, which is the only statement that cannot disagree
    with the numbers next to it.
    """
    return SCOPE_NONE if record_count < min_type_n() else SCOPE_TYPE_LEVEL


def scope_of(temperature_by_options: Optional[Dict[str, Any]]) -> str:
    """The scope the fitter actually achieved."""
    return SCOPE_BUCKETED if temperature_by_options else SCOPE_TYPE_LEVEL


def insufficient_examples(count: int) -> HTTPException:
    floor = min_type_n()
    return CalibrationError(
        status_code=422,
        detail=(
            f"a calibration needs at least {floor} labelled records, got {count}. "
            f"Below that there is not enough data to fit even a single type-level "
            f"temperature. Collect more labelled examples rather than activating a "
            f"number this service cannot stand behind."
        ),
    )


def scope_caveat(scope: str, record_count: int, buckets: int = 0) -> Optional[str]:
    """The sentence the console shows above the numbers, or None if there is none."""
    floor = min_bucket_n()
    if scope == SCOPE_BUCKETED:
        return None
    return (
        f"type-level only: a temperature per question type, not per option-count "
        f"bucket. The fitter needs {floor} records in a single bucket before it fits "
        f"per-bucket temperatures, and these {record_count} records cover {buckets or 'several'} "
        f"bucket(s), none of which reached it. It is a real fit and the thresholds "
        f"below are real, but they are shared across option counts and are a narrower "
        f"claim than a per-bucket calibration. Collect more examples for a finer one."
    )


# --------------------------------------------------------------------------------------
# Target expansion
# --------------------------------------------------------------------------------------


def expand_target(qid: str, question: Any, expected: Any) -> List[float]:
    """Turn a human-written label into the one-hot vector the fitter needs.

    Accepts what a person would plausibly write: the option label, or its index.
    Index form is bounds-checked, because an out-of-range index that silently
    pointed at the last option would produce a calibration fitted against wrong
    ground truth -- an artifact that loads cleanly and is quietly wrong.

    ``noul`` takes a bool and is ordered ``[true, false]``, matching the key order
    of ``NoulQuestion.criteria``. A length mismatch against the real logits is
    raised by the SDK as a ``ValueError`` during the fit, which surfaces as a
    named error rather than a silently misaligned target.
    """
    if isinstance(question, ChoiceQuestion):
        labels = list(question.criteria)
        width = len(labels)
        if isinstance(expected, bool):
            raise _bad_target(qid, "choice", "expected must be an option label or index")
        if isinstance(expected, str):
            if expected not in labels:
                raise _bad_target(
                    qid,
                    "choice",
                    f"expected {expected!r} is not one of the options {labels}",
                )
            index = labels.index(expected)
        elif isinstance(expected, int):
            index = expected
        else:
            raise _bad_target(qid, "choice", "expected must be an option label or index")
    elif isinstance(question, ScoreQuestion):
        levels = list(question.criteria)
        width = len(levels)
        if isinstance(expected, bool):
            raise _bad_target(qid, "score", "expected must be a level index or level text")
        if isinstance(expected, str):
            if expected not in levels:
                raise _bad_target(
                    qid, "score", f"expected {expected!r} is not one of the levels {levels}"
                )
            index = levels.index(expected)
        elif isinstance(expected, int):
            index = expected
        else:
            raise _bad_target(qid, "score", "expected must be a level index or level text")
    elif isinstance(question, NoulQuestion):
        width = 2
        if not isinstance(expected, bool):
            raise _bad_target(qid, "noul", "expected must be true or false")
        index = 0 if expected else 1
    else:  # pragma: no cover - the union is closed
        raise _bad_target(qid, "unknown", f"unsupported question type {type(question).__name__}")

    if not 0 <= index < width:
        raise _bad_target(
            qid,
            question.type,
            f"expected index {index} is out of range for {width} option(s)",
        )
    return [1.0 if position == index else 0.0 for position in range(width)]


def _bad_target(qid: str, kind: str, detail: str) -> HTTPException:
    return CalibrationError(
        status_code=422, detail=f"labelled example for question {qid!r} ({kind}): {detail}"
    )


def build_pairs(
    profile: "profile_mod.Profile", rows: List[Dict[str, Any]]
) -> List[Tuple[Any, Dict[str, Any], Dict[str, List[float]]]]:
    """Turn stored labelled rows into the ``(state, questions, targets)`` triples.

    Two shapes are accepted per row, because both are reasonable to paste:

    * the profile's own question set, with ``expected`` keyed by question id
    * a per-row ``questions`` override, for a profile being calibrated against
      several variants at once

    A row naming a question the profile does not define is refused rather than
    skipped. Silently dropping it would shrink the fit's ``n`` without saying so,
    which is exactly the number the honesty gate is protecting.
    """
    pairs: List[Tuple[Any, Dict[str, Any], Dict[str, List[float]]]] = []
    for position, row in enumerate(rows):
        if not isinstance(row, dict):
            raise CalibrationError(
                status_code=422, detail=f"labelled example {position} is not an object"
            )
        if "state" not in row:
            raise CalibrationError(
                status_code=422, detail=f"labelled example {position} has no 'state'"
            )
        expected = row.get("expected")
        if not isinstance(expected, dict) or not expected:
            raise CalibrationError(
                status_code=422,
                detail=(
                    f"labelled example {position} needs a non-empty 'expected' object "
                    f"keyed by question id"
                ),
            )

        questions = row.get("questions")
        if questions is None:
            resolved = profile.questions
        else:
            try:
                resolved = profile_mod.QUESTIONS_ADAPTER.validate_python(questions)
            except ValidationError as exc:
                raise CalibrationError(
                    status_code=422,
                    detail=(
                        f"labelled example {position} has invalid questions: "
                        f"{decision_errors(exc)}"
                    ),
                ) from None

        targets = {
            qid: expand_target(qid, resolved[qid], value)
            for qid, value in expected.items()
            if qid in resolved
        }
        missing = sorted(set(expected) - set(targets))
        if missing:
            raise CalibrationError(
                status_code=422,
                detail=(
                    f"labelled example {position} names question(s) {missing} that the "
                    f"profile does not define; defined questions are "
                    f"{sorted(resolved)}"
                ),
            )

        pairs.append((row["state"], profile_mod.canonical_questions(resolved), targets))

    if not pairs:
        raise CalibrationError(
            status_code=422, detail="no labelled examples to fit a calibration from"
        )
    return pairs


def decision_errors(exc: ValidationError) -> str:
    from app.decision import format_validation_error

    return "; ".join(item["msg"] for item in format_validation_error(exc))


# --------------------------------------------------------------------------------------
# Staleness
# --------------------------------------------------------------------------------------


def staleness(
    profile: "profile_mod.Profile", artifact: Optional[Dict[str, Any]]
) -> Optional[str]:
    """Why this artifact must not be served for this profile, or ``None`` if it may.

    Pure and profile-scoped: the question set an artifact was fitted against is a
    property of the profile document alone, so this needs nothing from the Router and
    can be checked at write time as well as read time.

    It is checked on every read rather than once at write time, because the thing that
    changes underneath an artifact is the *profile*: someone edits the questions to add
    one field and every number the artifact claims now describes a different question.

    Checkpoint-revision drift is a separate concern and deliberately not here -- see
    ``CalibrationService.revision_drift``, which needs the Router to know what is
    resident.
    """
    if not artifact:
        return "no calibration has been fitted for this profile"
    fitted = artifact.get("questions_fingerprint")
    if fitted != profile.questions_fingerprint:
        return (
            f"the calibration was fitted for questions {str(fitted)[:12]} but this "
            f"profile's questions now fingerprint as "
            f"{profile.questions_fingerprint[:12]}; re-fit before using it"
        )
    return None


def ece_block(report: Dict[str, Any], scope: Optional[str]) -> Dict[str, Any]:
    """Held-out calibration error, or why there isn't any.

    ECE is only available when a bucket is large enough to be *held out*: the SDK
    excludes any bucket that would fall below ``MIN_BUCKET_N`` after the split and
    names it in ``buckets_excluded_from_eval``. Below that the report carries
    ``NaN`` and ``n_eval: 0``, which rendered as a number would read as "the fit was
    scored and scored badly" rather than "nothing was scored". So the reason is
    reported alongside, and ``NaN`` is never passed through.
    """
    excluded = report.get("buckets_excluded_from_eval") or []
    n_eval = report.get("n_eval")
    has_eval = bool(n_eval) and not excluded
    block: Dict[str, Any] = {
        "available": has_eval,
        "n_eval": n_eval,
        "before": report.get("ece_before") if has_eval else None,
        "after": report.get("ece_after") if has_eval else None,
        "buckets_excluded_from_eval": excluded,
    }
    if has_eval:
        return block
    if excluded:
        block["reason"] = (
            f"no held-out ECE for bucket(s) {excluded}: each would fall below "
            f"{min_bucket_n()} records once 20% is held out, so the fitter scored "
            f"them on their own training records and excluded them from the "
            f"evaluation instead. The temperatures are fitted; they are just not "
            f"independently scored."
        )
    elif not n_eval:
        block["reason"] = (
            "no records reached the held-out split; with a small labelled set every "
            "bucket was fit on all of its records."
        )
    else:
        block["reason"] = "the fit returned no calibration-error report."
    return block


def summarize(
    profile: "profile_mod.Profile", artifact: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """What ``GET .../calibration`` reports: the numbers, and whether to trust them."""
    reason = staleness(profile, artifact)
    floors = sdk_floors()
    payload: Dict[str, Any] = {
        "fitted": artifact is not None,
        "stale": reason is not None,
        "reason": reason,
        "scope": (artifact or {}).get("scope"),
        "caveat": (artifact or {}).get("caveat"),
        "ece": ece_block(
            dict((artifact or {}).get("report") or {}),
            (artifact or {}).get("scope"),
        ),
        "floors": {
            "type_level": int(floors["type"]),
            "per_bucket": int(floors["bucket"]),
            "ece_holdout_frac": floors["holdout"],
            "abstention_per_bucket": 100,
        },
        "questions_fingerprint": profile.questions_fingerprint,
    }
    if artifact:
        payload.update(
            {
                "fitted_at": artifact.get("fitted_at"),
                "n": artifact.get("n"),
                "n_eval": artifact.get("n_eval"),
                "report": artifact.get("report"),
                "temperature": artifact.get("temperature"),
                "temperature_by_options": artifact.get("temperature_by_options"),
                "thresholds": artifact.get("thresholds"),
                "target_error": artifact.get("target_error"),
                "active": bool(artifact.get("active")),
                "revision": artifact.get("revision"),
            }
        )
    return payload


# --------------------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------------------


def fit(
    profile: "profile_mod.Profile",
    rows: List[Dict[str, Any]],
    agent: Any,
    *,
    target_error: float = DEFAULT_TARGET_ERROR,
    seed: int = 0,
    records_from: Optional[Any] = None,
) -> Dict[str, Any]:
    """Run the fit and return the artifact to store.

    ``agent`` is the checkpoint the profile pins. It is passed in rather than
    reached for, so this stays a pure function of its arguments and the caller
    owns the agent's lifetime -- which matters, because a long-lived reference to a
    Router-owned agent defeats the LRU cap and brings back the 20-23s rebuild.

    ``records_from`` overrides the SDK's forward pass with synthetic records. That
    seam exists so the fitting, thresholding and artifact assembly around it can be
    tested without a checkpoint; the SDK's own maths is not reimplemented here, so a
    test using it is still exercising the real ``fit_temperature_map``.
    """
    from laya.calibrate import (
        calibration_payload,
        fit_abstention_thresholds,
        fit_temperature_map,
        records_from_labeled,
    )

    pairs = build_pairs(profile, rows)
    if records_from is not None:
        records = records_from(agent, pairs)
    else:
        records = records_from_labeled(agent, pairs)

    scope = scope_for(len(records))
    if scope == SCOPE_NONE:
        raise insufficient_examples(len(records))

    fitted = fit_temperature_map(records, compute_ece=True, seed=seed)
    # `min_bucket_n` is left at the SDK's own default: it governs which buckets get a
    # threshold of their own, which is the SDK's judgement to make, not this service's.
    by_options = fitted.get("temperature_by_options") or {}
    scope = scope_of(by_options)
    thresholds = fit_abstention_thresholds(
        records,
        fitted["temperature"],
        by_options,
        target_error=target_error,
    )

    payload = calibration_payload(
        fitted["temperature"],
        fitted.get("temperature_by_options") or {},
        # Identity from the agent, not from the profile's friendly name: the
        # payload's identity check compares against these, and "english" would
        # never match the repo id the agent actually loaded.
        model_id_or_path=getattr(agent, "model_id_or_path", None),
        subfolder=getattr(agent, "subfolder", None),
    )

    report = dict(fitted.get("report") or {})
    artifact = {
        **payload,
        "questions_fingerprint": profile.questions_fingerprint,
        "profile_id": profile.id,
        "revision": _revision_of(agent),
        "fitted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "target_error": target_error,
        "seed": seed,
        "thresholds": thresholds,
        "n": len(records),
        "n_eval": report.get("n_eval"),
        "n_buckets": len(thresholds),
        "n_buckets_per_bucket_temperature": len(by_options),
        "report": report,
        "scope": scope,
        "caveat": scope_caveat(scope, len(records), len(thresholds)),
        "active": False,
    }
    return artifact


def _revision_of(agent: Any) -> Optional[str]:
    revision = getattr(agent, "revision", None)
    return str(revision) if revision else None


# --------------------------------------------------------------------------------------
# Which calibration is live for which checkpoint
# --------------------------------------------------------------------------------------


class CalibrationService:
    """The one object the routes talk to about calibration.

    Bundles the store (what is on disk), the registry (which checkpoint currently
    has which profile's temperatures installed) and the staleness rule, because
    those three only make sense together: a threshold may be read only if the
    artifact behind it is both active and not stale, and answering that question
    correctly requires all three.
    """

    def __init__(self, store: Any, router: Any = None) -> None:
        self.store = store
        self.active = ActiveCalibrations()
        self._router: Any = None
        if router is not None:
            self.install(router)

    def install(self, router: Any) -> None:
        # Kept so revision drift can be checked against what is actually resident.
        # Held as a plain reference to the Router -- which is long-lived and owned by
        # the engine -- never to an agent, so this does not interfere with LRU
        # eviction of checkpoints.
        self._router = router
        self.active.install(router)

    def revision_drift(
        self, profile: "profile_mod.Profile", artifact: Optional[Dict[str, Any]]
    ) -> Optional[str]:
        """Why the checkpoint this was fitted against is no longer the one resident.

        A temperature is fitted for particular weights. If the checkpoint is updated
        underneath it -- a new Hub revision, a re-pulled image -- the stored temperature
        is no longer the right one for those weights, and serving it produces
        confidence numbers that look calibrated and are not.

        Only reported when the checkpoint is *resident* and its revision is known. A
        checkpoint that is merely not loaded is not drift: the temperature is still
        correct for those weights, and refusing on eviction would make the threshold
        flap on and off with LRU traffic, which is worse than the problem being solved.
        """
        fitted = (artifact or {}).get("revision")
        if not fitted or self._router is None:
            return None
        model = profile.routing.model or self.active.model_for(profile.id)
        if not model:
            return None
        resident = (getattr(self._router, "loaded_revisions", {}) or {}).get(model)
        if not resident:
            # Not loaded, so nothing is known and nothing has changed.
            return None
        if str(resident) == str(fitted):
            return None
        return (
            f"the calibration was fitted against revision {str(fitted)[:12]} of "
            f"{model!r}, but revision {str(resident)[:12]} is loaded now. The stored "
            f"temperature was computed for different weights, so its confidence numbers "
            f"would not be calibrated. Re-fit against the resident revision."
        )

    def unservable_reason(
        self, profile: "profile_mod.Profile", artifact: Optional[Dict[str, Any]]
    ) -> Optional[str]:
        """Either kind of staleness: profile drift first, then checkpoint drift."""
        return staleness(profile, artifact) or self.revision_drift(profile, artifact)

    def restore(self) -> List[str]:
        """Reload active calibrations from disk into the in-memory registry.

        Called once the Router exists. Without it a restart silently drops every
        activation: the artifact on disk still says ``active: true``, so the report
        claims a threshold is being served while nothing installs it and no
        ``on_load`` hook fires to put the temperature back. The result would be a
        profile whose stated confidence gate quietly stopped applying at the next
        deploy -- the worst possible failure for a threshold, because it looks like
        it is working.

        Conflicts are resolved by first-writer-wins rather than raising: a set of
        artifacts that cannot all be honoured is a state an operator has to see and
        fix, not one that should stop the service booting.
        """
        restored: List[str] = []
        models: List[str] = []
        for profile in self.store.list():
            artifact = self.store.read_calibration(profile.id)
            if not artifact or not artifact.get("active"):
                continue
            if self.unservable_reason(profile, artifact) is not None:
                _log.warning(
                    "not restoring calibration for %s: %s",
                    profile.id,
                    self.unservable_reason(profile, artifact),
                )
                continue
            model = profile.routing.model or ""
            owner = self.active.owner_of(model)
            if owner is not None and owner != profile.id:
                _log.error(
                    "checkpoints conflict on restore: %s and %s are both active for %r; "
                    "keeping %s. Deactivate one before trusting its threshold.",
                    owner,
                    profile.id,
                    model,
                    owner,
                )
                continue
            self.active.activate(model, profile.id, self.store.calibration_path(profile.id))
            restored.append(profile.id)
            models.append(model)
        if restored:
            _log.info("restored %d active calibration(s): %s", len(restored), ", ".join(restored))
            # `preload` may already have the checkpoints resident, in which case no load
            # event fires and the hook never runs. Applying here is what makes a restored
            # calibration effective on the very first request rather than the next reload.
            for model in models:
                self._install_now(self._router, model)
        return restored

    def read_calibration(self, profile_id: str) -> Optional[Dict[str, Any]]:
        return self.store.read_calibration(profile_id)

    def report(self, profile: "profile_mod.Profile") -> Dict[str, Any]:
        artifact = self.store.read_calibration(profile.id)
        summary = summarize(profile, artifact)
        summary["reason"] = self.unservable_reason(profile, artifact)
        summary["stale"] = summary["reason"] is not None
        summary["served"] = self.threshold_for(profile) is not None
        return summary

    def threshold_for(self, profile: "profile_mod.Profile") -> Optional[float]:
        """The threshold to gate this profile's answers on, or ``None``.

        Requires all three of: an artifact exists, it is active, and it is not
        stale. The staleness check is the reason this is a method here and not a
        dictionary lookup -- a threshold read from an artifact whose questions have
        since changed would gate answers on numbers fitted to a different question
        set, and nothing downstream would show it. Checkpoint drift counts too: a
        temperature fitted for weights that are no longer resident is equally not a
        threshold for this checkpoint.
        """
        artifact = self.store.read_calibration(profile.id)
        if not artifact or not artifact.get("active"):
            return None
        if self.unservable_reason(profile, artifact) is not None:
            return None
        thresholds = artifact.get("thresholds") or {}
        values = [float(value) for value in thresholds.values() if value is not None]
        if not values:
            return None
        # A homogeneous question set yields one threshold. A mixed one yields
        # several, and gating on the strictest bucket is the defensible reading:
        # averaging them would gate some question types below a level their own
        # bucket never justified.
        return max(values)

    def fit(
        self,
        profile: "profile_mod.Profile",
        rows: List[Dict[str, Any]],
        agent: Any,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return fit(profile, rows, agent, **kwargs)

    def activate(
        self,
        profile: "profile_mod.Profile",
        artifact: Dict[str, Any],
        router: Any,
        model: str,
    ) -> None:
        """Make this profile's calibration the live one for ``model``.

        Refuses when another profile already owns the checkpoint. Temperature lives
        on the loaded agent, so two profiles cannot both have their own; silently
        letting the second win would change the confidence numbers the first is
        serving, which is a worse failure than an explicit 409.
        """
        owner = self.active.owner_of(model)
        if owner is not None and owner != profile.id:
            raise CalibrationError(
                status_code=409,
                detail=(
                    f"checkpoint {model!r} already has an active calibration for profile "
                    f"{owner!r}. One checkpoint carries one temperature, so activating "
                    f"{profile.id!r} would change {owner!r}'s confidence numbers. "
                    f"Deactivate {owner!r} first, or point this profile at another "
                    f"checkpoint."
                ),
            )

        stored = {**artifact, "active": True}
        self.store.write_calibration(profile.id, stored)
        self.active.activate(model, profile.id, self.store.calibration_path(profile.id))
        self._install_now(router, model)

    def deactivate(
        self, profile: "profile_mod.Profile", model: str, router: Any
    ) -> None:
        artifact = self.store.read_calibration(profile.id)
        if artifact:
            self.store.write_calibration(profile.id, {**artifact, "active": False})
        self.active.deactivate(model, profile.id)
        self._install_now(router, model)

    def _install_now(self, router: Any, model: str) -> None:
        """Apply the temperature on an already-resident checkpoint.

        Without this, activation would only take effect after the next eviction and
        reload, which is not something an operator can observe or verify. Deactivation
        needs no counterpart: the installed temperatures are gone with the artifact's
        ``active`` flag, and the next load installs nothing, so the checkpoint returns
        to shipped temperatures.
        """
        entry = self.active.active_for(model)
        if not entry:
            return
        try:
            from laya.mcp.device import router_agent

            agent = router_agent(router, model) if router is not None else None
        except Exception:  # noqa: BLE001 - a stub router in tests has no agents
            return
        if agent is None:
            # Not resident. The on_load hook applies it when it is.
            return
        try:
            agent.load_calibration(entry["path"])
        except Exception:  # noqa: BLE001
            _log.exception("could not install calibration for %s", model)


class ActiveCalibrations:
    """At most one calibration per checkpoint, reinstalled whenever it is loaded.

    Temperature is a property of a *loaded agent*, not of a request. Two profiles
    pinned to the same checkpoint therefore cannot both have their own temperature:
    the second load would silently change the confidence numbers the first profile
    was calibrated and served with. So activation is exclusive per checkpoint, and
    the conflict is reported rather than resolved by last-write-wins.

    The temperature does not survive an LRU eviction -- the reloaded agent is a
    fresh object with no temperatures -- which is why this installs an ``on_load``
    hook instead of applying once at activation. Without it, a profile's numbers
    would change the first time a mixed-language request evicted its checkpoint,
    which is the exact failure mode the eviction counter exists to reveal.
    """

    def __init__(self) -> None:
        self._by_checkpoint: Dict[str, Dict[str, Any]] = {}

    def activate(self, model: str, profile_id: str, path: Any) -> None:
        self._by_checkpoint[model] = {"profile_id": profile_id, "path": str(path)}

    def deactivate(self, model: str, profile_id: str) -> None:
        current = self._by_checkpoint.get(model)
        if current and current["profile_id"] == profile_id:
            self._by_checkpoint.pop(model, None)

    def deactivate_for_profile(self, profile_id: str) -> None:
        """Release every checkpoint this profile holds, whatever it is pinned to.

        Called when a profile's questions change or it is deleted. A registry entry
        left behind would keep installing temperatures for a profile that no longer
        exists, and -- worse -- keep the checkpoint marked as owned, so a later
        activation of a different profile would be refused with a conflict against a
        profile that is gone.
        """
        for model in [
            model
            for model, entry in self._by_checkpoint.items()
            if entry["profile_id"] == profile_id
        ]:
            self._by_checkpoint.pop(model, None)

    def active_for(self, model: str) -> Optional[Dict[str, Any]]:
        return self._by_checkpoint.get(model)

    def model_for(self, profile_id: str) -> Optional[str]:
        """Which checkpoint this profile's calibration is installed on.

        Recorded rather than re-derived because a profile need not pin a checkpoint:
        an unpinned profile is activated against whatever was resident at the time, so
        the model is only knowable from the activation itself.
        """
        for model, entry in self._by_checkpoint.items():
            if entry["profile_id"] == profile_id:
                return model
        return None

    def owner_of(self, model: str) -> Optional[str]:
        entry = self._by_checkpoint.get(model)
        return entry["profile_id"] if entry else None

    def install(self, router: Any) -> None:
        """Register the observer that reapplies calibrations on every checkpoint load."""
        if getattr(router, "_layadev_calibration_hook", None) is not None:
            return

        registry = self

        class _CalibrationObserver:
            def on_load(self, ctx: Any) -> None:  # noqa: ANN401
                model = getattr(ctx, "model", None)
                if not model:
                    return
                entry = registry._by_checkpoint.get(model)
                if not entry:
                    return
                try:
                    from laya.mcp.device import router_agent

                    agent = router_agent(router, model)
                    if agent is not None:
                        agent.load_calibration(entry["path"])
                except Exception:  # noqa: BLE001
                    # Observability and calibration must not take the service down.
                    # A checkpoint answering with shipped (over-confident)
                    # temperatures is wrong but harmless; a failed import is fatal.
                    _log.exception("could not reinstall calibration for %s", model)

        observer = _CalibrationObserver()
        router._layadev_calibration_hook = observer
        try:
            router.add_hook(observer)
        except Exception:  # noqa: BLE001 - a stub Router without hooks is fine
            router._layadev_calibration_hook = None
