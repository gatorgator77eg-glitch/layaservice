"""Prometheus instrumentation.

Latency is split into three signals rather than one, because under a single
inference worker they answer different questions and only one of them is about
the model:

* ``queue wait`` -- admission to gate acquired. This is what grows with
  concurrency. Reporting it inside the request latency would make a busy service
  indistinguishable from a slow one.
* ``inference`` -- the forward pass alone. This is the only number a checkpoint or
  a hardware change should move.
* ``request`` -- admission to response, the client-visible total.

The default Prometheus buckets (5ms to 10s) are useless here in both directions.
Measured on a 4-core laptop CPU: 198ms for one question on multilingual, 661ms on
english, 1,436ms for three english questions, and 4.7s for a cold checkpoint
build. Everything worth distinguishing falls between 100ms and 2s, which the
default buckets split into three unhelpful rows, and the cold build falls off the
end entirely. The ``laya[fast]`` path's 4.6ms small-request floor sits below the
default's lowest bucket for the same reason.

P50 and P95 come from ``histogram_quantile`` over these buckets. That is an
interpolation within a bucket, not an exact order statistic: a P95 reading is
accurate to the bucket width, so the buckets above are deliberately tight in the
200ms-3s range where this service actually operates.
"""

from __future__ import annotations

from typing import Any

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest
from prometheus_client.openmetrics.exposition import CONTENT_TYPE_LATEST as OPENMETRICS_TYPE

# Tight through the range this service operates in, with headroom for cold builds
# and for a GPU deployment measured an order of magnitude faster.
LATENCY_BUCKETS = (
    0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75,
    1.0, 1.5, 2.0, 3.0, 5.0, 7.5, 10.0, 30.0, 60.0,
)

REGISTRY = CollectorRegistry()

REQUEST_DURATION = Histogram(
    "laya_request_duration_seconds",
    "Admission to response, including queue wait and validation.",
    labelnames=("route", "outcome"),
    buckets=LATENCY_BUCKETS,
    registry=REGISTRY,
)

INFERENCE_DURATION = Histogram(
    "laya_inference_duration_seconds",
    "The forward pass alone, excluding queue wait.",
    labelnames=("checkpoint", "questions"),
    buckets=LATENCY_BUCKETS,
    registry=REGISTRY,
)

QUEUE_WAIT = Histogram(
    "laya_queue_wait_seconds",
    "Admission to inference-gate acquisition. Grows with concurrency; not the model.",
    labelnames=("route",),
    buckets=LATENCY_BUCKETS,
    registry=REGISTRY,
)

REQUESTS = Counter(
    "laya_requests_total",
    "Requests by route and outcome.",
    labelnames=("route", "status"),
    registry=REGISTRY,
)

REJECTED = Counter(
    "laya_rejected_total",
    "Requests refused before inference, by which guard fired.",
    labelnames=("route", "reason"),
    registry=REGISTRY,
)

IN_FLIGHT = Gauge(
    "laya_inference_in_flight",
    "Inference passes currently executing. Always 0 or 1. Non-zero for longer than one "
    "forward pass means the worker is wedged.",
    registry=REGISTRY,
)

CHECKPOINT_EVICTIONS = Counter(
    "laya_checkpoint_evictions_total",
    "Resident checkpoints evicted by the LRU cap.",
    labelnames=("model",),
    registry=REGISTRY,
)


def render() -> tuple[bytes, str]:
    """Expose the registry in OpenMetrics text format."""
    return generate_latest(REGISTRY), OPENMETRICS_TYPE


# Guard reasons. Kept as constants because they become label values, and a typo
# silently creates a second time series.
REASON_BODY = "body_bytes"
REASON_JSON = "malformed_json"
REASON_STATE = "state_chars"
REASON_QUESTIONS = "question_count"
REASON_CHOICE_OPTIONS = "choice_options"
REASON_SCORE_LEVELS = "score_levels"
REASON_TOTAL_OPTIONS = "total_options"
REASON_BATCH_STATES = "batch_states"
REASON_SURROGATE = "lone_surrogate"
REASON_BODY_CONTROL = "body_control"
REASON_BUDGET = "token_budget"
REASON_SCHEMA = "schema"
REASON_ADMISSION = "admission"
REASON_INFERENCE = "inference"


def observe_prometheus() -> None:  # pragma: no cover - helper for manual checks
    """Touch every metric once so ``/metrics`` is non-empty before first traffic.

    Without this, a freshly started service exposes nothing until a request lands,
    which reads as a broken scrape target rather than an idle one.
    """
    for route in ("predict", "batch", "health", "metrics"):
        REQUEST_DURATION.labels(route=route, outcome="idle").observe(0)
        QUEUE_WAIT.labels(route=route).observe(0)
        REQUESTS.labels(route=route, status="none").inc(0)
    for checkpoint in ("english", "multilingual", "typed-decisions"):
        INFERENCE_DURATION.labels(checkpoint=checkpoint, questions="0").observe(0)
        CHECKPOINT_EVICTIONS.labels(model=checkpoint).inc(0)
    for reason in (
        REASON_BODY, REASON_JSON, REASON_STATE, REASON_QUESTIONS,
        REASON_CHOICE_OPTIONS, REASON_SCORE_LEVELS, REASON_TOTAL_OPTIONS,
        REASON_BATCH_STATES, REASON_SURROGATE, REASON_BODY_CONTROL,
        REASON_BUDGET, REASON_SCHEMA, REASON_ADMISSION, REASON_INFERENCE,
    ):
        REJECTED.labels(route="predict", reason=reason).inc(0)
        REJECTED.labels(route="batch", reason=reason).inc(0)


def install_lifecycle_observers(router: Any) -> None:
    """Count checkpoint evictions, which are a capacity signal, not an error.

    A LRU eviction under a mixed-language workload with ``max_loaded`` below what
    routing can choose is the expensive case: rebuilding costs seconds on CPU
    against milliseconds resident, and it looks like a latency regression with no
    error anywhere.

    ``Router.add_hook`` takes the ``laya.hooks.Hook`` protocol -- an object with
    optional lifecycle methods -- not a callback per event, so this needs a class
    rather than a closure. The evicted checkpoint's name is on ``ctx.model``.
    """
    if getattr(router, "_layadev_eviction_hook", None) is not None:
        return

    class _EvictionObserver:
        def on_evict(self, ctx: Any) -> None:  # noqa: ANN401
            CHECKPOINT_EVICTIONS.labels(model=getattr(ctx, "model", None) or "unknown").inc()

    observer = _EvictionObserver()
    router._layadev_eviction_hook = observer
    try:
        router.add_hook(observer)
    except Exception:  # noqa: BLE001 - observability must never break startup
        # Older or stubbed Router without the hook API. Eviction counting is a
        # diagnostic nicety, not worth failing a deploy over.
        router._layadev_eviction_hook = None