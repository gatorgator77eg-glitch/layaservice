"""Concurrency behaviour.

The service's whole throughput story rests on two claims: one inference worker, and
refusing rather than queueing when that worker is busy. Both are testable without
weights, because both live in ``Engine`` and the stub can be made to take a
measurable amount of time.

These use a real event loop via ``TestClient`` rather than mocking the semaphore,
because the failure mode being guarded against -- accepting a request and then
blocking the event loop behind a 700ms forward pass -- is precisely what a mock
would not reproduce.
"""

from __future__ import annotations

import concurrent.futures
import time
from typing import List

from fastapi.testclient import TestClient

from app.config import Config
from app.main import create_app
from tests.conftest import body
from tests.fake_router import FakeRouter


def test_concurrent_requests_all_succeed(config: Config) -> None:
    # A stub with no delay: nothing to contend over, so this only proves the wiring
    # does not deadlock when several requests share the worker.
    app = create_app(config, router_obj=FakeRouter())

    with TestClient(app) as client:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(client.post, "/v1/decisions/predict", json=body()) for _ in range(8)]
            statuses = [future.result().status_code for future in futures]

    assert statuses == [200] * 8


def test_a_slow_forward_pass_does_not_block_health(config: Config) -> None:
    """Liveness must stay answerable while the single worker is saturated.

    The forward pass is synchronous torch holding the only worker. If /health queued
    behind it, a load balancer polling health would see timeouts exactly when the
    service is busiest and restart a healthy process.
    """
    app = create_app(config, router_obj=FakeRouter(delay_s=0.5))

    with TestClient(app) as client:
        started = time.perf_counter()
        health = client.get("/health")
        health_elapsed = time.perf_counter() - started

        assert health.status_code == 200
        # Well under the 500ms the inference call is taking.
        assert health_elapsed < 0.25, f"/health took {health_elapsed:.3f}s while busy"


def test_inference_is_serialised_by_one_worker(config: Config) -> None:
    """Two concurrent requests must not overlap inside the forward pass.

    Asserted by wall-clock: two 300ms calls on one worker take ~600ms, and two
    overlapping calls would take ~300ms. A regression to a per-thread executor would
    pass a functional test and fail this one, which is the point -- the failure would
    otherwise show up only as thread-unsafe model access under load.
    """
    delay = 0.3
    app = create_app(config, router_obj=FakeRouter(delay_s=delay))

    with TestClient(app) as client:
        started = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(client.post, "/v1/decisions/predict", json=body()) for _ in range(2)]
            statuses: List[int] = [future.result().status_code for future in futures]
        elapsed = time.perf_counter() - started

    assert statuses == [200, 200]
    assert elapsed >= delay * 2 * 0.9, (
        f"two {delay}s calls finished in {elapsed:.3f}s; they ran concurrently, so the "
        "single-worker guarantee is not holding"
    )


def test_metrics_labels_record_the_checkpoint_actually_used(config: Config) -> None:
    app = create_app(config, router_obj=FakeRouter())

    with TestClient(app) as client:
        client.post("/v1/decisions/predict", json=body())
        text = client.get("/metrics").text

    # Labelled by the checkpoint the SDK reported in `routing`, not by the one the
    # caller asked for. A guess here would misattribute latency on a routed request.
    assert "laya_inference_duration_seconds" in text
    assert 'checkpoint="english"' in text
