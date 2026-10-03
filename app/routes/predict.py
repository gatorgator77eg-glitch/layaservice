"""The decision endpoints.

``POST /v1/decisions/predict``       one state, one question set, one forward pass
``POST /v1/decisions/predict/batch`` one question set, many states

The handlers are thin on purpose. Guard order is the only thing here with real
subtlety, and it is fixed:

    admission -> body size -> JSON -> schema -> question/state size
    -> body controls -> lone surrogates -> token budgets -> inference

Size before schema so an oversized body is refused without being parsed into
objects first; body controls after the size checks so an oversized request is
refused rather than having its contents interpreted; surrogates last among the
pre-inference checks so the walk only ever sees bodies already bounded by the
character and question limits.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from app import limits, metrics
from app.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    PredictRequest,
    PredictResponse,
)

_log = logging.getLogger("laya_service.predict")

router = APIRouter(tags=["decisions"])

ROUTE_PREDICT = "predict"
ROUTE_BATCH = "batch"


def _engine(request: Request) -> Any:
    return request.app.state.engine


def _timing_headers(inference_ms: float) -> Dict[str, str]:
    """Per-request inference cost, for traces and browser devtools.

    The histogram is the right instrument for percentiles; this is what an
    individual request reports about itself.
    """
    return {
        "Server-Timing": f"inference;dur={inference_ms:.2f}",
        "X-Inference-Time-Ms": f"{inference_ms:.2f}",
    }


def _load_request(app: Any, payload: Any, route: str) -> PredictRequest:
    """Validate the parsed body, converting Pydantic's error shape to a 422."""
    try:
        return PredictRequest.model_validate(payload)
    except ValidationError as exc:
        metrics.REJECTED.labels(route=route, reason=metrics.REASON_SCHEMA).inc()
        raise HTTPException(status_code=422, detail=_format_validation_error(exc)) from None


def _format_validation_error(exc: ValidationError) -> List[Dict[str, Any]]:
    """Locate each problem and say what to change about it.

    The point is that a caller can act on a 422. "value is not a valid dict" does
    not tell anyone which of nine questions was malformed.
    """
    formatted = []
    for error in exc.errors():
        location = [str(part) for part in error.get("loc", ())]
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


def _guard(payload: Dict[str, Any], request_model: Any, cfg: Any, route: str) -> None:
    """Everything between parsing and inference, in the fixed order."""
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

    questions = request_model.questions
    for name in ("max_len", "head_max_len"):
        try:
            limits.clamp_budget(name, getattr(request_model, name), cfg.limits.token_budget)
        except HTTPException:
            metrics.REJECTED.labels(route=route, reason=metrics.REASON_BUDGET).inc()
            raise


def _check_questions(request_model: Any, cfg: Any, route: str) -> None:
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


@router.post(
    "/v1/decisions/predict",
    responses={
        200: {
            "model": PredictResponse,
            "description": (
                "The decision set. Emitted verbatim from Laya rather than re-serialised "
                "through this model, so `usage.options` and the abstention fields stay "
                "absent when they do not apply."
            ),
        },
        400: {"description": "Malformed body, null state, or an unpaired surrogate escape."},
        413: {"description": "A request guardrail was hit; `detail` names which."},
        422: {"description": "The question set is invalid, or a request control has a bad value."},
        500: {"description": "Inference failed. The cause is in the server log, not here."},
        503: {"description": "Admission limit reached; retry after the interval in `Retry-After`."},
    },
    summary="Decide typed questions over one state",
)
async def predict(request: Request) -> JSONResponse:
    engine = _engine(request)
    cfg = request.app.state.config
    started = time.perf_counter()

    async with engine.admit():
        try:
            raw = await limits.read_body_capped(request, cfg.limits.body_bytes)
        except HTTPException:
            metrics.REJECTED.labels(route=ROUTE_PREDICT, reason=metrics.REASON_BODY).inc()
            raise

        try:
            payload = limits.parse_json_object(raw)
        except HTTPException:
            metrics.REJECTED.labels(route=ROUTE_PREDICT, reason=metrics.REASON_JSON).inc()
            raise

        request_model = _load_request(request.app, payload, ROUTE_PREDICT)
        _guard(payload, request_model, cfg, ROUTE_PREDICT)
        _check_questions(request_model, cfg, ROUTE_PREDICT)

        try:
            limits.check_state(request_model.state, cfg.limits.state_chars)
        except HTTPException:
            metrics.REJECTED.labels(route=ROUTE_PREDICT, reason=metrics.REASON_STATE).inc()
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
            metrics.REJECTED.labels(route=ROUTE_PREDICT, reason=metrics.REASON_SCHEMA).inc()
            raise HTTPException(status_code=422, detail=str(exc)) from None
        except Exception as exc:  # noqa: BLE001
            # The caller learns nothing beyond "it failed"; the operator gets the
            # traceback. Without this the logs show only a 500 and a deterministic
            # failure has to be reproduced in-process to be diagnosed at all.
            metrics.REJECTED.labels(route=ROUTE_PREDICT, reason=metrics.REASON_INFERENCE).inc()
            _log.exception(
                "inference failed (model=%s, questions=%d)",
                limits.sanitize_log_value(model_name or "auto"),
                n_questions,
            )
            raise HTTPException(status_code=500, detail="inference failed") from None

    elapsed = time.perf_counter() - started
    checkpoint = (result.get("routing") or {}).get("model", "unknown")

    # Inference time is measured, not derived: it is the whole point of having a
    # separate histogram from the request total.
    metrics.INFERENCE_DURATION.labels(
        checkpoint=checkpoint, questions=str(n_questions)
    ).observe(inference_ms / 1000.0)
    metrics.QUEUE_WAIT.labels(route=ROUTE_PREDICT).observe(queue_s)
    metrics.REQUEST_DURATION.labels(route=ROUTE_PREDICT, outcome="ok").observe(elapsed)

    return JSONResponse(content=result, headers=_timing_headers(inference_ms))


@router.post(
    "/v1/decisions/predict/batch",
    responses={
        200: {
            "model": BatchPredictResponse,
            "description": (
                "One decision set per state, in the order the states were sent. A whole-batch "
                "failure raises instead of returning a partial array."
            ),
        },
        400: {"description": "Malformed body or an unpaired surrogate escape."},
        413: {"description": "A request guardrail was hit; `detail` names which."},
        422: {"description": "The question set is invalid, or a request control has a bad value."},
        500: {"description": "Inference failed. The cause is in the server log, not here."},
        503: {"description": "Admission limit reached; retry after the interval in `Retry-After`."},
    },
    summary="Decide one question set over many states",
)
async def predict_batch(request: Request) -> JSONResponse:
    engine = _engine(request)
    cfg = request.app.state.config
    started = time.perf_counter()

    async with engine.admit():
        try:
            raw = await limits.read_body_capped(request, cfg.limits.body_bytes)
        except HTTPException:
            metrics.REJECTED.labels(route=ROUTE_BATCH, reason=metrics.REASON_BODY).inc()
            raise

        try:
            payload = limits.parse_json_object(raw)
        except HTTPException:
            metrics.REJECTED.labels(route=ROUTE_BATCH, reason=metrics.REASON_JSON).inc()
            raise

        try:
            batch = BatchPredictRequest.model_validate(payload)
        except ValidationError as exc:
            metrics.REJECTED.labels(route=ROUTE_BATCH, reason=metrics.REASON_SCHEMA).inc()
            raise HTTPException(status_code=422, detail=_format_validation_error(exc)) from None

        try:
            limits.refuse_body_controls(payload)
        except HTTPException:
            metrics.REJECTED.labels(route=ROUTE_BATCH, reason=metrics.REASON_BODY_CONTROL).inc()
            raise

        if limits.has_lone_surrogate(payload):
            metrics.REJECTED.labels(route=ROUTE_BATCH, reason=metrics.REASON_SURROGATE).inc()
            raise HTTPException(
                status_code=400,
                detail=(
                    "request body contains an unpaired surrogate escape; those cannot be "
                    "encoded as UTF-8"
                ),
            )

        # Budgets can arrive three ways on a batch -- batch-wide, per-state override, or
        # neither -- and all of them are capped by the server. Checked here rather than
        # only in `_guard`, which covers the single-predict path.
        try:
            for name in ("max_len", "head_max_len"):
                limits.clamp_budget(name, getattr(batch, name), cfg.limits.token_budget)
                for index, override in (batch.overrides or {}).items():
                    limits.clamp_budget(
                        f"overrides[{index}].{name}",
                        getattr(override, name),
                        cfg.limits.token_budget,
                    )
        except HTTPException:
            metrics.REJECTED.labels(route=ROUTE_BATCH, reason=metrics.REASON_BUDGET).inc()
            raise

        try:
            limits.check_question_count(batch.questions, cfg.limits.questions)
            limits.check_option_weights(
                batch.questions,
                max_choice_options=cfg.limits.choice_options,
                max_score_levels=cfg.limits.score_levels,
                max_total_options=cfg.limits.total_options,
            )
        except HTTPException:
            metrics.REJECTED.labels(route=ROUTE_BATCH, reason=metrics.REASON_TOTAL_OPTIONS).inc()
            raise

        if len(batch.states) > cfg.limits.batch_states:
            metrics.REJECTED.labels(route=ROUTE_BATCH, reason=metrics.REASON_BATCH_STATES).inc()
            raise HTTPException(
                status_code=413,
                detail=(
                    f"too many states in batch ({len(batch.states)} > "
                    f"{cfg.limits.batch_states})"
                ),
            )

        for index, state in enumerate(batch.states):
            try:
                limits.check_state(state, cfg.limits.state_chars)
            except HTTPException as exc:
                metrics.REJECTED.labels(route=ROUTE_BATCH, reason=metrics.REASON_STATE).inc()
                raise HTTPException(
                    status_code=exc.status_code,
                    detail=f"states[{index}]: {exc.detail}",
                ) from None

        try:
            metrics.IN_FLIGHT.inc()
            # `predict_batch` takes per-request dicts and returns one result per
            # request in input order. It raises on failure rather than returning a
            # partial result, so there is no per-state error channel to fill in.
            results, inference_ms, queue_s = await engine.run(
                engine.router.predict_batch,
                requests=batch.as_requests(),
                **batch.call_kwargs(),
            )
        except HTTPException:
            raise
        except ValueError as exc:
            metrics.REJECTED.labels(route=ROUTE_BATCH, reason=metrics.REASON_SCHEMA).inc()
            raise HTTPException(status_code=422, detail=str(exc)) from None
        except Exception:  # noqa: BLE001
            metrics.REJECTED.labels(route=ROUTE_BATCH, reason=metrics.REASON_INFERENCE).inc()
            _log.exception("batch inference failed (states=%d)", len(batch.states))
            raise HTTPException(status_code=500, detail="inference failed") from None
        finally:
            metrics.IN_FLIGHT.dec()

    elapsed = time.perf_counter() - started
    checkpoint = "batch"
    for item in results:
        routed = (item.get("routing") or {}).get("model")
        if routed:
            checkpoint = routed
            break

    metrics.INFERENCE_DURATION.labels(
        checkpoint=checkpoint, questions=str(len(batch.questions))
    ).observe(inference_ms / 1000.0)
    metrics.QUEUE_WAIT.labels(route=ROUTE_BATCH).observe(queue_s)
    metrics.REQUEST_DURATION.labels(route=ROUTE_BATCH, outcome="ok").observe(elapsed)

    return JSONResponse(content=results, headers=_timing_headers(inference_ms))