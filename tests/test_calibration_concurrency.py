"""A calibration fit must not be able to wedge the service.

``records_from_labeled`` runs one CPU forward per labelled example. If the fit is
awaited inline in the request handler it holds the event loop for its whole duration,
and every other request in the process -- including ``/health``, which is the one an
operator reaches for to find out whether the service is alive -- looks hung for
minutes. These tests assert the loop is actually released.

Requests go through a real ``httpx.AsyncClient`` over ASGI rather than ``TestClient``,
because the property under test is what happens when a fit and a probe are *in flight
at the same time*. ``TestClient`` drives one request at a time from the calling thread
and could not express that even if the fit did block the loop.

The fitter is replaced with a sleep rather than mocked away: the point is which thread
it runs on, not what it computes.

The primary assertion is ``not fit_task.done()`` when a probe is served. That is
deliberately unit-free. Wall-clock arithmetic here is easy to get subtly wrong -- an
earlier version of this file compared a seconds delta against a milliseconds bound,
which made the assertion permanently true and the test worthless. "Was the fit still
running?" cannot be got wrong that way.
"""

from __future__ import annotations

import asyncio
import threading
import time

import httpx

from app.config import Config
from app.main import create_app
from tests.fake_router import FakeRouter

QUESTION = {"type": "choice", "instructions": "?", "criteria": {"a": "A", "b": "B"}}
FIT_SECONDS = 0.6
# A fit this size must still leave the loop responsive; generous enough not to be
# flaky on a loaded CI box, tight enough that a 600ms block cannot slip through.
PROBE_BUDGET_MS = 300


def build_slow_app(tmp_path, monkeypatch):
    """An app whose fit sleeps, with the checkpoint lookup stubbed out.

    ``_agent_for`` reaches into the real Router for a loaded agent. A ``FakeRouter`` has
    none, and standing up real weights is not what this test is about, so the lookup is
    replaced with a sentinel. Everything downstream of it is the real code path.
    """
    from app import calibration as calib
    from app.routes import admin

    def slow_fit(profile, rows, agent, **kwargs):
        time.sleep(FIT_SECONDS)
        return {
            "profile_id": profile.id,
            "questions_fingerprint": profile.questions_fingerprint,
            "thresholds": {"choice:2": 0.8},
            "temperature": 1.0,
            "temperature_by_options": {},
            "n": len(rows),
            "n_eval": 0,
            "report": {
                "n": float(len(rows)),
                "n_eval": 0.0,
                "ece_before": float("nan"),
                "ece_after": float("nan"),
            },
            "scope": calib.SCOPE_TYPE_LEVEL,
            "caveat": "type-level only, synthetic",
            "target_error": 0.1,
            "fitted_at": "2026-01-01T00:00:00+00:00",
            "revision": None,
            "active": False,
        }

    monkeypatch.setattr(calib, "fit", slow_fit)
    monkeypatch.setattr(admin, "_agent_for", lambda router, model: object())

    cfg = Config(device="cpu", preload=False, threads=1, data_dir=str(tmp_path / "data"))
    return create_app(cfg, router_obj=FakeRouter())


async def _seeded_client(app):
    """A client with one profile and its examples already in place."""
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")
    created = await client.post(
        "/v1/profiles", json={"id": "t", "name": "t", "questions": {"q1": QUESTION}}
    )
    assert created.status_code == 201, created.text
    saved = await client.put(
        "/v1/profiles/t/examples", json={"examples": [{"state": "x", "expected": {"q1": "a"}}]}
    )
    assert saved.status_code == 200, saved.text
    return client


def test_health_is_served_while_a_fit_is_still_running(tmp_path, monkeypatch) -> None:
    """A probe issued during a fit is answered before the fit finishes."""
    app = build_slow_app(tmp_path, monkeypatch)

    async def scenario():
        client = await _seeded_client(app)
        origin = time.perf_counter()

        fit_task = asyncio.create_task(
            client.post("/v1/profiles/t/calibration/fit", json={})
        )
        # Let the fit get under way. With a released loop this costs ~150ms. With a
        # blocked one it cannot return until the fit is already over.
        await asyncio.sleep(0.15)

        fit_running_at_probe = not fit_task.done()
        began = time.perf_counter()
        response = await client.get("/health")
        probe_after_ms = (began - origin) * 1000.0
        probe_took_ms = (time.perf_counter() - began) * 1000.0

        fit = await fit_task
        await client.aclose()
        return fit, fit_running_at_probe, probe_after_ms, probe_took_ms

    fit, fit_running, probe_after_ms, probe_took_ms = asyncio.run(scenario())

    assert fit.status_code == 200, fit.text
    assert fit_running, (
        "the fit had already finished before any probe could be served, which means "
        "the event loop was held for the whole fit; /health would have looked hung "
        "for the duration"
    )
    assert probe_after_ms < PROBE_BUDGET_MS, (
        f"the first /health probe was issued {probe_after_ms:.0f}ms into a "
        f"{FIT_SECONDS * 1000:.0f}ms fit"
    )
    assert probe_took_ms < PROBE_BUDGET_MS, (
        f"/health took {probe_took_ms:.0f}ms while a fit was in flight"
    )


def test_the_fitter_runs_off_the_event_loop_thread(tmp_path, monkeypatch) -> None:
    """Belt and braces: name the thread, so a future refactor cannot quietly regress.

    The behavioural test above is the one that matters, but it infers blocking from
    timing. This one is unambiguous and cheap.
    """
    from app import calibration as calib

    app = build_slow_app(tmp_path, monkeypatch)
    # Wrapped after the app is built: `build_slow_app` installs the sleeping fitter, so
    # wrapping first would just be overwritten.
    installed = calib.fit
    seen: list[str] = []

    def recording_fit(profile, rows, agent, **kwargs):
        seen.append(threading.current_thread().name)
        return installed(profile, rows, agent, **kwargs)

    monkeypatch.setattr(calib, "fit", recording_fit)

    async def scenario():
        client = await _seeded_client(app)
        response = await client.post("/v1/profiles/t/calibration/fit", json={})
        await client.aclose()
        return response

    response = asyncio.run(scenario())

    assert response.status_code == 200, response.text
    assert seen, "the fitter was never called"
    assert threading.current_thread().name not in seen[0], (
        f"the fitter ran on {seen[0]}, which is the event loop thread; it must be "
        f"handed to the worker via engine.run"
    )


def test_live_inference_still_completes_while_the_worker_is_busy(tmp_path, monkeypatch) -> None:
    """The documented trade: inference serialises behind the fit, but never hangs."""
    app = build_slow_app(tmp_path, monkeypatch)

    async def scenario():
        client = await _seeded_client(app)
        statuses = []
        origin = time.perf_counter()

        async def call_predict():
            for _ in range(6):
                response = await client.post(
                    "/v1/decisions/predict",
                    json={"state": "x", "questions": {"q1": QUESTION}},
                )
                statuses.append(response.status_code)
                await asyncio.sleep(0.1)

        fit_task = asyncio.create_task(
            client.post("/v1/profiles/t/calibration/fit", json={})
        )
        await call_predict()
        await fit_task
        total_ms = (time.perf_counter() - origin) * 1000.0
        await client.aclose()
        return statuses, total_ms

    statuses, total_ms = asyncio.run(scenario())

    assert statuses == [200] * 6
    # Bounded, so a deadlock or a lost wakeup shows up as a failure rather than a hang.
    assert total_ms < FIT_SECONDS * 1000 * 5, f"took {total_ms:.0f}ms with a fit running"


def test_fit_reports_how_long_it_queued_and_ran(tmp_path, monkeypatch) -> None:
    """The wait is measurable, so 'still working' is distinguishable from 'hung'."""
    app = build_slow_app(tmp_path, monkeypatch)

    async def scenario():
        client = await _seeded_client(app)
        response = await client.post("/v1/profiles/t/calibration/fit", json={})
        await client.aclose()
        return response

    response = asyncio.run(scenario())

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["fit_ms"] >= FIT_SECONDS * 1000 * 0.9, (
        "the synthetic fit sleeps 600ms; it cannot have reported running faster"
    )
    assert payload["queued_ms"] >= 0.0
