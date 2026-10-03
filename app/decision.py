"""The single-decision core, shared by every endpoint that answers one state.

This module exists so that guard parity is structural rather than aspirational.
``POST /v1/decisions/predict`` and each minted ``/v1/profiles/{id}/predict`` differ
only in how a :class:`~app.schemas.PredictRequest` is built from the body; every
check between "body parsed" and "answer returned" lives here exactly once. A
profile endpoint that reimplemented the guards would be a second copy to keep in
sync, and the copy that drifts is the one with no tests.

The fixed guard order is preserved verbatim, and the reason for it is the ordering:

    schema -> body controls -> lone surrogates -> token budgets
    -> question count/option weight -> state size -> inference

Schema first because a malformed request has no meaningful body to walk. Body
controls next because they are refused rather than interpreted -- honouring a
caller's ``hooks_timeout`` would let a request change how server-side hooks run.
Surrogates after the size checks so the walk only ever sees bodies already bounded
by the character and question limits. Question shape before state, so a request
that is going to be refused for its questions does not also pay to serialise a
state it was never going to be scored against.

A ``RequestBuilder`` is injected rather than branching on request origin here. That
keeps this module ignorant of profiles entirely, so the one place that knows what a
profile is cannot accidentally become the place that decides what a guard means.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from app import limits, metrics
from app.schemas import PredictRequest

_log = logging.getLogger("laya_service.decision")

# Turns a parsed body into a validated request model, or raises HTTPException(422).
RequestBuilder = Callable[[Dict[str, Any]], PredictRequest]


def timing_headers(inference_ms: float) -> Dict[str, str]:
    """Per-request inference cost, for traces and browser devtools.

    The histogram is the right instrument for percentiles; this is what an
    individual request reports about itself.
    """
    return {
        "Server-Timing": f"inference;dur={inference_ms:.2f}",
        "X-Inference-Time-Ms": f"{inference_ms:.2f}",
    }


def format_validation_error(
    exc: ValidationError, loc_prefix: tuple = ()
) -> List[Dict[str, Any]]:
    """Locate each problem and say what to change about it.

    The point is that a caller can act on a 422. "value is not a valid dict" does
    not tell anyone which of nine questions was malformed, so a failure inside
    `questions` is reported against the question id that caused it.

    ``loc_prefix`` lets a caller that validated a nested fragment on its own still
    get question-aware messages. Validating ``{"severity": {...}}`` against
    ``Dict[str, Question]`` yields a loc starting at the question id; prefixing it
    with ``questions`` puts it back in the shape the branch below recognises, so
    there is one formatter rather than two that disagree about wording.
    """
    formatted = []
    for error in exc.errors():
        location = [*loc_prefix, *(str(part) for part in error.get("loc", ()))]
        # loc[0] is the body's top-level field; the question id is loc[1] when the
        # failure is inside `questions`. Surface it as a message so the caller does
        # not have to know the request shape to find the offending question.
        if len(location) >= 3 and location[0] == "questions":
            qid = location[1]
            leaf = location[-1]
            message = error.get("msg", "invalid")
            formatted.append(
                {
                    "loc": location,
                    "msg": f"question {qid!r}: {message}",
                    "question": qid,
                    "field": leaf,
                }
            )
        else:
            formatted.append(
                {
                    "loc": location,
                    "msg": error.get("msg", "invalid"),
                    "field": location[-1] if location else None,
                }
            )
    return formatted


def reject_schema(exc: ValidationError, route: str) -> HTTPException:
    """Record a schema refusal and build the 422."""
    metrics.REJECTED.labels(route=route, reason=metrics.REASON_SCHEMA).inc()
    return HTTPException(status_code=422, detail=format_validation_error(exc))


def validate_predict_request(payload: Dict[str, Any], route: str) -> PredictRequest:
    """The standard builder: validate the body as-is."""
    try:
        return PredictRequest.model_validate(payload)
    except ValidationError as exc:
        raise reject_schema(exc, route) from None


def _guard(payload: Dict[str, Any], request_model: Any, cfg: Any, route: str) -> None:
    """Body controls, surrogates and token budgets, in that order."""
    try:
        limits.refuse_body_controls(payload)
    except HTTPException:
        metrics.REJECTED.labels(route=route, reason=metrics.REASON_BODY_CONTROL).inc()
        raise

    if limits.has_lone_surrogate(payload):
        metrics.REJECTED.labels(route=route, reason=metrics.REASON_SURROGATE).inc()
        raise HTTPException(
            status_code=400,
            detail=(
                "request body contains an unpaired surrogate escape; those cannot be "
                "encoded as UTF-8"
            ),
        )

    for name in ("max_len", "head_max_len"):
        try:
            limits.clamp_budget(name, getattr(request_model, name), cfg.limits.token_budget)
        except HTTPException:
            metrics.REJECTED.labels(route=route, reason=metrics.REASON_BUDGET).inc()
            raise


def check_question_limits(request_model: Any, cfg: Any, route: str) -> None:
    """Question count and total option weight."""
    try:
        limits.check_question_count(request_model.questions, cfg.limits.questions)
        limits.check_option_weights(
            request_model.questions,
            max_choice_options=cfg.limits.choice_options,
            max_score_levels=cfg.limits.score_levels,
            max_total_options=cfg.limits.total_options,
        )
    except HTTPException:
        # The specific reason is not recoverable from the exception type, so the
        # coarse reason is recorded here and the detail carries the specifics.
        metrics.REJECTED.labels(route=route, reason=metrics.REASON_TOTAL_OPTIONS).inc()
        raise


async def decide(
    request: Request,
    payload: Dict[str, Any],
    route: str,
    build: RequestBuilder,
) -> JSONResponse:
    """Answer one state, enforcing every guard on the way.

    Admission and body reading have already happened; this owns everything after
    ``parse_json_object`` so that no caller can reach inference without it. The
    handler is responsible only for reading the body under its own route label,
    because the body cap has to fire before anything is parsed.
    """
    engine = request.app.state.engine
    cfg = request.app.state.config
    started = time.perf_counter()

    request_model = build(payload)
    _guard(payload, request_model, cfg, route)
    check_question_limits(request_model, cfg, route)

    try:
        limits.check_state(request_model.state, cfg.limits.state_chars)
    except HTTPException:
        metrics.REJECTED.labels(route=route, reason=metrics.REASON_STATE).inc()
        raise

    model_name = limits.resolve_model(request_model.model)
    kwargs = request_model.predict_kwargs()
    n_questions = len(request_model.questions)

    try:
        result, inference_ms, queue_s = await engine.run(
            engine.router.predict,
            state=request_model.state,
            questions=request_model.as_laya_questions(),
            model=model_name,
            **kwargs,
        )
    except HTTPException:
        raise
    except ValueError as exc:
        # Question validation from core: names the question and what to fix,
        # so it is safe to return.
        metrics.REJECTED.labels(route=route, reason=metrics.REASON_SCHEMA).inc()
        raise HTTPException(status_code=422, detail=str(exc)) from None
    except Exception as exc:  # noqa: BLE001
        # The caller learns nothing beyond "it failed"; the operator gets the
        # traceback. Without this the logs show only a 500 and a deterministic
        # failure has to be reproduced in-process to be diagnosed at all.
        metrics.REJECTED.labels(route=route, reason=metrics.REASON_INFERENCE).inc()
        _log.exception(
            "inference failed (model=%s, questions=%d, route=%s)",
            limits.sanitize_log_value(model_name or "auto"),
            n_questions,
            route,
        )
        raise HTTPException(status_code=500, detail="inference failed") from None

    elapsed = time.perf_counter() - started
    checkpoint = (result.get("routing") or {}).get("model", "unknown")

    # Inference time is measured, not derived: it is the whole point of having a
    # separate histogram from the request total.
    metrics.INFERENCE_DURATION.labels(
        checkpoint=checkpoint, questions=str(n_questions)
    ).observe(inference_ms / 1000.0)
    metrics.QUEUE_WAIT.labels(route=route).observe(queue_s)
    metrics.REQUEST_DURATION.labels(route=route, outcome="ok").observe(elapsed)

    return JSONResponse(content=result, headers=timing_headers(inference_ms))
