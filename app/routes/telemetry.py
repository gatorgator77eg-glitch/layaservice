"""Operational endpoints: liveness and the Prometheus scrape target.

Named ``telemetry`` rather than ``metrics`` so it does not shadow ``app.metrics``,
which holds the metric definitions it renders.

``/health`` is documented -- a gateway or load balancer has to know it exists, and
its shape is stable enough to depend on. ``/metrics`` is not: it is a machine-facing
scrape target, and documenting it would imply a shape is stable when the Prometheus
text format is a moving target.
"""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Request, Response

from app import metrics as metric_defs
from app.engine import checkpoint_devices, cpu_fallbacks

router = APIRouter(tags=["operations"])

LIVENESS_ONLY: Dict[str, Any] = {"status": "ok"}


@router.get("/health", summary="Liveness, plus what is resident")
async def health(request: Request) -> Dict[str, Any]:
    """Report liveness and the resident checkpoint set.

    Deliberately off the inference gate. The forward pass is synchronous torch that
    can hold the single worker for seconds; if liveness queued behind it, a load
    balancer would restart a healthy process precisely when it is busiest.

    ``status`` answers only "is this process responding". Whether the checkpoints
    loaded is a separate question, reported in ``loaded`` -- a service answering
    200 with an empty ``loaded`` is alive and unable to serve its first request,
    and the distinction is the reason these are separate fields.

    This deployment sits behind an authenticating gateway, so there is no bearer
    check here. The detail below liveness quotes host hardware: checkpoint names,
    revision SHAs, device state and CPU-fallback reasons. That is acceptable only
    because the gateway is what makes it reachable. If this service is ever
    exposed directly, the full payload becomes an information disclosure and this
    handler needs the bearer check upstream ships, which withholds everything but
    ``status`` from an unauthenticated caller.
    """
    app = request.app
    engine = app.state.engine
    router_obj = engine.router

    devices = checkpoint_devices(router_obj)
    actual = next(iter(devices.values()), None)

    return {
        "status": "ok",
        "loaded": list(router_obj.loaded or []),
        "revisions": dict(getattr(router_obj, "loaded_revisions", {}) or {}),
        "device": actual or app.state.config.device,
        # True exactly while nothing is resident: the difference between a server
        # reporting its configuration and a server reporting where its work is.
        # One that quietly lost its GPU says `device: cpu` rather than echoing the
        # `cuda` it asked for.
        "device_is_preference": actual is None,
        "checkpoint_devices": devices,
        "cpu_fallbacks": cpu_fallbacks(router_obj),
        "torch_threads": app.state.thread_settings,
        "limits": {
            "body_bytes": app.state.config.limits.body_bytes,
            "state_chars": app.state.config.limits.state_chars,
            "questions": app.state.config.limits.questions,
            "choice_options": app.state.config.limits.choice_options,
            "score_levels": app.state.config.limits.score_levels,
            "total_options": app.state.config.limits.total_options,
            "batch_states": app.state.config.limits.batch_states,
            "token_budget": app.state.config.limits.token_budget,
            "max_concurrent": app.state.config.max_concurrent,
        },
    }


@router.get(
    "/metrics",
    include_in_schema=False,
    summary="Prometheus exposition",
)
async def scrape() -> Response:
    """Render the registry in OpenMetrics text format."""
    payload, content_type = metric_defs.render()
    return Response(content=payload, media_type=content_type)