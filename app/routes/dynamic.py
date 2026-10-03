"""Runtime route registration for profiles.

Adding a ``APIRoute`` to a live ``app.router.routes`` is the whole mechanism. It
works, and ``get_openapi`` reads ``app.routes``, so a minted endpoint appears in the
service's own ``/openapi.json`` rather than in a side document that can drift.

Two things make it safe to do this per request rather than once at boot.

**The generated handler owns no logic.** It reads the body under the byte cap and
then calls ``app.decision.decide``, the same core the fixed endpoint uses, with a
builder that merges the profile in. Every guard therefore applies to a minted
endpoint by construction rather than by having been remembered.

**Each route carries its own request model.** ``app.openapi`` reads it off the route
object, so the documented body is the body that is validated. A generated endpoint
whose schema was hand-written would drift the first time a field was added.

The path parameter is deliberately *not* used to look the profile up. FastAPI would
resolve ``{profile_id}`` and hand the handler a string, which means the endpoint
would exist for any id including ones with no profile -- a 500 where a 404 belongs.
Each route is registered with its profile closed over, so an unknown id is simply
not a route.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from pydantic import ValidationError

from app import decision, limits, metrics
from app.openapi import declare_request_body, forget_request_body
from app.profiles import Profile
from app.schemas import PredictRequest, ProfilePredictRequest

_log = logging.getLogger("laya_service.dynamic")

ROUTE_PREFIX = "profile"


def route_label(profile_id: str) -> str:
    """Metric label for a profile's requests.

    Includes the id because per-profile latency is the number that tells an operator
    which question set is slow. It is bounded because the profile count is capped;
    without that cap this label would be an unbounded cardinality source.
    """
    return f"{ROUTE_PREFIX}:{profile_id}"


def endpoint_path(profile_id: str) -> str:
    return f"/v1/profiles/{profile_id}/predict"


def make_builder(profile: Profile, cfg: Any, service: Any) -> decision.RequestBuilder:
    """Merge a caller body with the profile, then hand it to the shared validator.

    The profile is authoritative and the caller is advisory. Routing pins win; a
    caller who repeats a pinned value gets the same answer, and there is no way to
    send a different one because the field is not on the request model at all.
    """

    def build(payload: Dict[str, Any]) -> PredictRequest:
        route = route_label(profile.id)
        try:
            request = ProfilePredictRequest.model_validate(payload)
        except ValidationError as exc:
            metrics.REJECTED.labels(route=route, reason=metrics.REASON_SCHEMA).inc()
            raise HTTPException(
                status_code=422, detail=decision.format_validation_error(exc)
            ) from None

        merged: Dict[str, Any] = {"questions": profile.questions, "state": request.state}

        # Layering, cheapest-first: the caller's controls, then the profile's pins over
        # them, then the profile's budgets only where the caller sent nothing. Order is
        # the whole point -- pins land last so a caller cannot choose the checkpoint a
        # profile was calibrated against. `ProfilePredictRequest.controls()` is the
        # single list of what a caller may send, so a control added there is layered
        # here without a second edit; hand-enumerating the names is how `lang` came to
        # be accepted by the schema and then dropped before it reached the SDK.
        merged.update(request.controls())
        merged.update(profile.routing.pinned())

        for name in ("max_len", "head_max_len"):
            if merged.get(name) is None:
                merged[name] = getattr(profile.budgets, name)

        # A calibrated threshold is the profile's default; a caller may still override
        # it for one request, and may still send nothing at all.
        if request.min_confidence is None:
            threshold = service.threshold_for(profile) if service is not None else None
            if threshold is not None:
                merged["min_confidence"] = threshold

        # Reached only if the profile itself is malformed, which create/update
        # refuses. Kept so a corrupted profile cannot reach the SDK unvalidated.
        return decision.validate_predict_request(merged, route)

    return build


def make_endpoint(profile: Profile, cfg: Any, service: Any) -> Any:
    """Build the ASGI handler for one profile."""
    route = route_label(profile.id)

    async def endpoint(request: Request) -> JSONResponse:
        engine = request.app.state.engine
        async with engine.admit():
            try:
                raw = await limits.read_body_capped(
                    request, request.app.state.config.limits.body_bytes
                )
            except HTTPException:
                metrics.REJECTED.labels(route=route, reason=metrics.REASON_BODY).inc()
                raise
            try:
                payload = limits.parse_json_object(raw)
            except HTTPException:
                metrics.REJECTED.labels(route=route, reason=metrics.REASON_JSON).inc()
                raise
            return await decision.decide(
                request, payload, route, make_builder(profile, cfg, service)
            )

    endpoint.__name__ = f"profile_{profile.id}_predict"
    return endpoint


def register(app: FastAPI, profile: Profile, cfg: Any, service: Any) -> APIRoute:
    """Add (or replace) the profile's endpoint and invalidate the cached document."""
    path = endpoint_path(profile.id)
    unregister(app, profile.id)

    route_obj = APIRoute(
        path,
        make_endpoint(profile, cfg, service),
        methods=["POST"],
        response_class=JSONResponse,
        responses={
            200: {
                "model": _response_model(),
                "description": (
                    f"Decision set for profile {profile.id!r}. Emitted verbatim from Laya, "
                    f"so conditional fields stay absent when they do not apply."
                ),
            },
            400: {"description": "Malformed body, null state, or an unpaired surrogate escape."},
            413: {"description": "A request guardrail was hit; `detail` names which."},
            422: {"description": "The body is invalid, or a request control has a bad value."},
            500: {"description": "Inference failed. The cause is in the server log."},
            503: {"description": "Admission limit reached; retry after `Retry-After`."},
        },
        summary=f"Decide {profile.name!r} over one state",
        description=(
            f"Questions are fixed by the profile and cannot be sent per request. "
            f"Question ids: {', '.join(sorted(profile.questions))}."
        ),
        tags=["profiles"],
    )
    # Read back by app.openapi so the documented body is the validated one.
    route_obj.request_body_model = ProfilePredictRequest
    app.router.routes.append(route_obj)
    declare_request_body(path, "post", ProfilePredictRequest)
    app.openapi_schema = None
    _log.info("registered profile endpoint %s", path)
    return route_obj


def _response_model() -> Any:
    from app.schemas import PredictResponse

    return PredictResponse


def unregister(app: FastAPI, profile_id: str) -> None:
    """Remove a profile's endpoint, if present."""
    path = endpoint_path(profile_id)
    kept = [route for route in app.router.routes if getattr(route, "path", None) != path]
    if len(kept) != len(app.router.routes):
        app.router.routes[:] = kept
        forget_request_body(path, "post")
        app.openapi_schema = None


def register_all(app: FastAPI, store: Any, cfg: Any, service: Any) -> List[str]:
    """Register every persisted profile, reporting which ids were restored."""
    restored: List[str] = []
    for profile in store.list():
        try:
            register(app, profile, cfg, service)
            restored.append(profile.id)
        except Exception:  # noqa: BLE001 - one bad profile must not block startup
            _log.exception("could not register profile %s", profile.id)
    return restored
