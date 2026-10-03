"""The decision endpoints.

``POST /v1/decisions/predict``       one state, one question set, one forward pass
``POST /v1/decisions/predict/batch`` one question set, many states

The handlers are thin on purpose. Everything from schema validation to the answer
lives in ``app.decision``, shared with the profile endpoints minted at runtime, so
guard parity is structural: there is one implementation, and a second one would be
the one without tests.

What stays here is what cannot move into the shared core, because it is genuinely
specific to these two routes:

* reading the body under the byte cap, before anything is parsed
* the batch shape, which has no profile equivalent -- a minted profile endpoint
  answers one state, and per-state overrides over a shared question set are the
  fixed endpoint's job

Guard order for the single-decision path is fixed and documented in
``app.decision``; for batch it is:

    admission -> body size -> JSON -> schema -> body controls -> lone surrogates
    -> token budgets -> question size -> state size -> inference
"""

from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from app import decision, limits, metrics
from app.openapi import declare_request_body
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

# Both handlers read the body off a raw Request so the byte cap can fire before
# parsing, which leaves FastAPI with no signature to infer a request schema from.
# Declaring them here is what keeps the request side of the API in the document.
declare_request_body("/v1/decisions/predict", "post", PredictRequest)
declare_request_body("/v1/decisions/predict/batch", "post", BatchPredictRequest)


def _engine(request: Request) -> Any:
    return request.app.state.engine


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
    async with _engine(request).admit():
        try:
            raw = await limits.read_body_capped(request, request.app.state.config.limits.body_bytes)
        except HTTPException:
            metrics.REJECTED.labels(route=ROUTE_PREDICT, reason=metrics.REASON_BODY).inc()
            raise

        try:
            payload = limits.parse_json_object(raw)
        except HTTPException:
            metrics.REJECTED.labels(route=ROUTE_PREDICT, reason=metrics.REASON_JSON).inc()
            raise

        return await decision.decide(
            request,
            payload,
            ROUTE_PREDICT,
            lambda body: decision.validate_predict_request(body, ROUTE_PREDICT),
        )


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
            # `reject_schema` records the refusal and builds the 422.
            raise decision.reject_schema(exc, ROUTE_BATCH) from None

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

    return JSONResponse(content=results, headers=decision.timing_headers(inference_ms))