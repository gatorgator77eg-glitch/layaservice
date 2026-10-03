"""Profile administration: the surface the console drives.

This is a privileged surface and it is unauthenticated, on purpose and with a
caveat. The service as a whole has no bearer check because a gateway in front of it
provides one; ``/console`` and ``/v1/profiles*`` are the paths that gateway must
restrict to an admin route. Reachable directly, this router lets a caller mint
endpoints and change the confidence gate, which no inference client should be able
to do. That is a deployment requirement, not a code path, so it is stated here and
in the runbook rather than enforced here.

Reading a request body off the raw ``Request`` -- as the decision endpoints do, so
the byte cap can fire before parsing -- means FastAPI infers no request schema from
these signatures either. The request models are therefore attached to the operations
by hand in ``app.openapi``, which keeps the console's contract documented and
generatable.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from app import calibration as calib
from app import limits
from app import profiles as profile_mod
from app.openapi import declare_request_body
from app.routes import dynamic
from app.routes.admin_schemas import (
    ActivateRequest,
    CalibrationFitRequest,
    ExamplesRequest,
    ProfileCreateRequest,
    ProfileUpdateRequest,
)

_log = logging.getLogger("laya_service.admin")

router = APIRouter(tags=["profiles"])

ROUTE_ADMIN = "admin"

# Documented request bodies; see app.openapi for why they are attached by hand.
declare_request_body("/v1/profiles", "post", ProfileCreateRequest)
declare_request_body("/v1/profiles/{profile_id}", "patch", ProfileUpdateRequest)
declare_request_body("/v1/profiles/{profile_id}/examples", "put", ExamplesRequest)
declare_request_body(
    "/v1/profiles/{profile_id}/calibration/fit", "post", CalibrationFitRequest
)
declare_request_body(
    "/v1/profiles/{profile_id}/calibration/activate", "post", ActivateRequest
)


def _store(request: Request) -> profile_mod.ProfileStore:
    return request.app.state.profiles


def _service(request: Request) -> calib.CalibrationService:
    return request.app.state.calibration


def _cfg(request: Request) -> Any:
    return request.app.state.config


async def _json_body(request: Request) -> Dict[str, Any]:
    """Read and parse a body under the same cap as the decision endpoints.

    An admin body is trusted no more than an inference body. The console sends
    question sets, which are text the operator chose but which arrive over the same
    socket, so the cap applies here too rather than only where a caller is
    untrusted.
    """
    raw = await limits.read_body_capped(request, _cfg(request).limits.body_bytes)
    return limits.parse_json_object(raw)


def _enforce_profile_cap(request: Request) -> None:
    cfg = _cfg(request)
    existing = _store(request).list()
    if len(existing) >= cfg.max_profiles:
        raise HTTPException(
            status_code=409,
            detail=(
                f"this deployment already holds {len(existing)} profiles, at the "
                f"LAYA_MAX_PROFILES limit of {cfg.max_profiles}. Each profile is a route "
                f"label and may pin its own checkpoint, so the cap is what keeps both "
                f"bounded. Delete one, or raise the limit."
            ),
        )


@router.get("/v1/profiles", summary="List profiles")
async def list_profiles(request: Request) -> Dict[str, Any]:
    store = _store(request)
    profiles = store.list()
    cfg = _cfg(request)
    return {
        "profiles": [profile.summary() for profile in profiles],
        "count": len(profiles),
        "max_profiles": cfg.max_profiles,
        "data_dir": cfg.data_dir,
    }


@router.post("/v1/profiles", status_code=201, summary="Create a profile and mint its endpoint")
async def create_profile(request: Request) -> JSONResponse:
    _enforce_profile_cap(request)
    body = await _json_body(request)
    profile_id = body.pop("id", None)
    if not isinstance(profile_id, str) or not profile_id.strip():
        raise HTTPException(status_code=422, detail="'id' is required and must be a string")

    store = _store(request)
    cfg = _cfg(request)
    profile = store.create(profile_id.strip(), body, cfg)
    dynamic.register(request.app, profile, cfg, _service(request))
    _log.info("created profile %s with %d question(s)", profile.id, len(profile.questions))
    return JSONResponse(status_code=201, content={"profile": profile.summary()})


@router.get("/v1/profiles/{profile_id}", summary="Read one profile")
async def read_profile(profile_id: str, request: Request) -> Dict[str, Any]:
    profile = _store(request).get(profile_id)
    return {
        "profile": profile.model_dump(mode="json"),
        "calibration": _service(request).report(profile),
    }


@router.patch("/v1/profiles/{profile_id}", summary="Update a profile")
async def update_profile(profile_id: str, request: Request) -> Dict[str, Any]:
    store = _store(request)
    cfg = _cfg(request)
    before = store.get(profile_id)
    body = await _json_body(request)
    body.pop("id", None)

    profile = store.update(profile_id, body, cfg)

    if profile.questions_fingerprint != before.questions_fingerprint:
        # The endpoint itself is unchanged in shape -- same path, same body model --
        # but its answers now come from different questions, so the route is
        # re-registered to pick up the new set and the calibration is retired.
        _log.info(
            "profile %s questions changed (%s -> %s); retiring its calibration",
            profile_id,
            before.questions_fingerprint[:12],
            profile.questions_fingerprint[:12],
        )
        _service(request).active.deactivate_for_profile(profile_id)
        store.clear_calibration(profile_id)

    dynamic.register(request.app, profile, cfg, _service(request))
    return {"profile": profile.summary()}


@router.delete("/v1/profiles/{profile_id}", summary="Delete a profile and its endpoint")
async def delete_profile(profile_id: str, request: Request) -> Dict[str, Any]:
    store = _store(request)
    store.get(profile_id)
    _service(request).active.deactivate_for_profile(profile_id)
    store.delete(profile_id)
    dynamic.unregister(request.app, profile_id)
    _log.info("deleted profile %s", profile_id)
    return {"deleted": profile_id}


# --------------------------------------------------------------------------------------
# Labelled examples
# --------------------------------------------------------------------------------------


@router.get("/v1/profiles/{profile_id}/examples", summary="List labelled examples")
async def list_examples(profile_id: str, request: Request) -> Dict[str, Any]:
    store = _store(request)
    store.get(profile_id)
    rows = store.read_examples(profile_id)
    return {"profile_id": profile_id, "count": len(rows), "examples": rows}


@router.put("/v1/profiles/{profile_id}/examples", summary="Replace labelled examples")
async def replace_examples(profile_id: str, request: Request) -> Dict[str, Any]:
    store = _store(request)
    profile = store.get(profile_id)
    body = await _json_body(request)
    rows = body.get("examples")
    if not isinstance(rows, list):
        raise HTTPException(
            status_code=422, detail="'examples' must be a list of labelled rows"
        )

    # Parsed before storing so a bad label is refused now rather than at fit time,
    # where the operator has lost the row that caused it.
    calib.build_pairs(profile, rows)
    count = store.replace_examples(profile_id, rows)
    return {"profile_id": profile_id, "count": count}


# --------------------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------------------


@router.get("/v1/profiles/{profile_id}/calibration", summary="Read the calibration report")
async def read_calibration(profile_id: str, request: Request) -> Dict[str, Any]:
    profile = _store(request).get(profile_id)
    return {"profile_id": profile_id, "calibration": _service(request).report(profile)}


@router.post("/v1/profiles/{profile_id}/calibration/fit", summary="Fit temperatures and thresholds")
async def fit_calibration(profile_id: str, request: Request) -> Dict[str, Any]:
    """Run the fit on the single inference worker.

    ``records_from_labeled`` runs a CPU forward per labelled example, so this occupies
    the worker for as long as it takes and live inference queues behind it. That is a
    deliberate choice over a second process: it keeps one copy of the weights resident,
    and the alternative -- a second Router -- doubles resident weights to buy
    concurrency this service does not otherwise have.

    It goes through ``engine.run`` rather than calling the fitter inline, which is the
    difference between a busy service and a wedged one: ``run`` hands the work to the
    executor and releases the event loop, so ``/health`` keeps answering and the
    admission gate keeps counting while a fit is in progress. Called inline it would
    hold the loop for the whole fit and every request in the process would look hung,
    including the one an operator would use to check whether the fit was alive.

    Deliberately not wrapped in ``engine.admit()``: a fit is an administrative job, not
    a caller waiting on inference, and it should not consume one of the concurrency
    slots that bound live traffic. The request is synchronous -- it returns the report
    when the fit is done -- and reports how long it queued and ran, so the wait is
    legible rather than mysterious.
    """
    store = _store(request)
    profile = store.get(profile_id)
    service = _service(request)
    rows = store.read_examples(profile_id)
    if not rows:
        raise HTTPException(
            status_code=422,
            detail=(
                "this profile has no labelled examples. Paste a JSONL of "
                "{state, expected} rows first."
            ),
        )

    body = await _json_body(request)
    target_error = float(body.get("target_error", calib.DEFAULT_TARGET_ERROR))
    if not 0.0 < target_error < 1.0:
        raise HTTPException(
            status_code=422, detail="'target_error' must be between 0 and 1 exclusive"
        )

    engine = request.app.state.engine
    model = profile.routing.model or _cfg(request).default_model or "english"
    agent = _agent_for(engine.router, model)
    if agent is None:
        raise HTTPException(
            status_code=409,
            detail=(
                f"checkpoint {model!r} is not resident, so there is nothing to fit "
                f"against. Send one request to it first, or pin the profile to a "
                f"checkpoint this deployment has loaded."
            ),
        )

    artifact, fit_ms, queue_s = await engine.run(
        calib.fit,
        profile=profile,
        rows=rows,
        agent=agent,
        target_error=target_error,
    )
    store.write_calibration(profile_id, artifact)
    _log.info(
        "fitted calibration for %s: n=%s n_eval=%s scope=%s queued=%.0fms ran=%.0fms",
        profile_id,
        artifact.get("n"),
        artifact.get("n_eval"),
        artifact.get("scope"),
        queue_s * 1000.0,
        fit_ms,
    )
    return {
        "profile_id": profile_id,
        "queued_ms": round(queue_s * 1000.0, 1),
        "fit_ms": round(fit_ms, 1),
        "calibration": service.report(profile),
    }


@router.post("/v1/profiles/{profile_id}/calibration/activate", summary="Adopt a fitted threshold")
async def activate_calibration(profile_id: str, request: Request) -> Dict[str, Any]:
    store = _store(request)
    profile = store.get(profile_id)
    service = _service(request)
    artifact = store.read_calibration(profile_id)
    if not artifact:
        raise HTTPException(
            status_code=409, detail="no calibration has been fitted for this profile yet"
        )
    reason = calib.staleness(profile, artifact)
    if reason is not None:
        raise HTTPException(status_code=409, detail=reason)

    model = profile.routing.model or _cfg(request).default_model or "english"
    service.activate(profile, artifact, request.app.state.engine.router, model)
    _log.info("activated calibration for %s on %s", profile_id, model)
    return {
        "profile_id": profile_id,
        "model": model,
        "calibration": service.report(profile),
    }


@router.post("/v1/profiles/{profile_id}/calibration/deactivate", summary="Drop the threshold")
async def deactivate_calibration(profile_id: str, request: Request) -> Dict[str, Any]:
    store = _store(request)
    profile = store.get(profile_id)
    model = profile.routing.model or _cfg(request).default_model or "english"
    _service(request).deactivate(profile, model, request.app.state.engine.router)
    return {"profile_id": profile_id, "calibration": _service(request).report(profile)}


def _agent_for(router: Any, model: str) -> Any:
    """The resident Agent for ``model``, or None.

    Read through ``laya.mcp.device`` rather than by reaching into the Router's
    internals. The reference is used and dropped inside the caller's frame: holding
    one would keep a checkpoint resident past the LRU's decision to evict it, which
    is the 20-23s rebuild this service already has a metric for.
    """
    try:
        from laya.mcp.device import router_agent

        return router_agent(router, model)
    except Exception:  # noqa: BLE001 - a stub Router has no agents
        return None


__all__ = ["router"]
