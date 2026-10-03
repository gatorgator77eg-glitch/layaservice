"""Model lifecycle and the concurrency model.

The shape here is deliberate and every part of it is load-bearing:

* **Threads are pinned at startup, before any tensor work.** Intra-op to the
  physical core count, inter-op to exactly 1. A single forward pass has no
  inter-op parallelism to overlap, so torch's defaults were pure overhead: a
  three-question call on a busy host measured p50 9,396ms against 783ms pinned.
  Oversubscribing intra-op to logical cores is also a regression, because SMT
  siblings contend.

* **One inference worker.** ``predict`` is a synchronous torch call taking
  hundreds of milliseconds to seconds, so it must never run on the event loop --
  one request would stall every other client including ``/health``. One worker
  because one forward pass at a time is what a single checkpoint on one device
  wants. Concurrency here means queueing, not parallel compute, which is why
  queue wait is timed separately from inference.

* **Admission is checked before any body byte is read** and held through
  inference, so the bodies buffered at once stay bounded no matter how many
  clients connect. Excess load is refused with 503 rather than queued, so a
  healthcheck is never starved and a retry can take the slot a refused client left.

* **The inference gate is joined only after the body is complete**, so a slow
  client holds an admission slot but never an inference slot.
"""

from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, Optional

from fastapi import HTTPException

from app.config import Config

_log = logging.getLogger("laya_service.engine")


def pin_torch_threads(config: Config) -> Dict[str, Any]:
    """Pin torch's thread pools. Must run before any tensor is created.

    ``set_num_interop_threads`` raises once the pool is initialised, so this is
    called at startup rather than per request and a RuntimeError here means
    something already warmed the pool -- reported, not fatal.
    """
    import torch

    applied: Dict[str, Any] = {}
    torch.set_num_threads(config.threads)
    applied["intra_op"] = config.threads
    try:
        torch.set_num_interop_threads(1)
        applied["inter_op"] = 1
    except RuntimeError as exc:
        applied["inter_op"] = f"left at torch default ({exc})"
        _log.warning("could not pin inter-op threads: %s", exc)
    return applied


def build_router(config: Config) -> Any:
    """Construct the Router from configuration and optionally preload.

    Preloading happens here, before the app starts listening, so a healthy
    container already has its checkpoints resident. Otherwise the first request
    pays the cold build inside its own latency measurement -- measured 4.7s for
    the English checkpoint and 4.1s for multilingual on a 4-core host, against a
    661ms warm forward pass.
    """
    from laya import Router

    router = Router(**config.router_kwargs())
    if config.preload:
        started = time.perf_counter()
        names = config.preload_models
        router.preload(names)
        elapsed = time.perf_counter() - started
        _log.info(
            "preloaded %s in %.1fs", names or "all configured checkpoints", elapsed
        )
    return router


class Engine:
    """Owns the Router, the single inference worker, and the admission gate."""

    def __init__(self, router: Any, config: Config) -> None:
        self.router = router
        self.config = config
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="laya-infer")
        # Both created lazily on first use: an asyncio primitive binds to the loop
        # running when it is first awaited, and the app may be constructed before
        # that loop exists (module scope, TestClient startup, a preload script).
        self._gate: Optional[asyncio.Lock] = None
        self._admission: Optional[asyncio.Semaphore] = None

    # -- admission ---------------------------------------------------------------

    @asynccontextmanager
    async def admit(self) -> AsyncIterator[None]:
        """Bound requests past this point; excess gets 503 rather than a queue.

        Non-blocking on purpose. Queuing here would mean buffering more request
        bodies than the bound is there to allow.
        """
        if self._admission is None:
            self._admission = asyncio.Semaphore(self.config.max_concurrent)
        if self._admission.locked():
            # Admission turns over at inference speed, so one second is the honest
            # Retry-After hint.
            raise HTTPException(
                status_code=503,
                detail="server busy, try again later",
                headers={"Retry-After": "1"},
            )
        await self._admission.acquire()
        try:
            yield
        finally:
            self._admission.release()

    # -- inference ---------------------------------------------------------------

    async def run(self, fn: Any, /, **kwargs: Any) -> tuple:
        """Run one blocking call on the inference worker.

        Returns ``(result, inference_ms)``. The gate serialises forward passes; the
        queue wait is measured separately so a P95 report can distinguish "the model
        was slow" from "the model was busy".
        """
        if self._gate is None:
            self._gate = asyncio.Lock()

        loop = asyncio.get_running_loop()
        queued_at = time.perf_counter()
        async with self._gate:
            started = time.perf_counter()
            result = await loop.run_in_executor(self.pool, lambda: fn(**kwargs))
            inference_ms = (time.perf_counter() - started) * 1000.0
        return result, inference_ms, (started - queued_at)

    async def shutdown(self) -> None:
        self.pool.shutdown(wait=True, cancel_futures=True)


def checkpoint_devices(router: Any) -> Dict[str, str]:
    """Where each resident checkpoint actually computes.

    ``LAYA_DEVICE`` is a preference, not a guarantee: a checkpoint that wants a GPU
    it cannot get falls back to CPU silently and keeps answering correctly. A
    server reporting where its work happens is more useful than one echoing its
    configuration back.
    """
    from laya.mcp.device import agent_device, router_agent

    devices: Dict[str, str] = {}
    for name in router.loaded or []:
        device = agent_device(router_agent(router, name))
        if device:
            devices[name] = device
    return devices


def cpu_fallbacks(router: Any) -> Dict[str, Dict[str, Any]]:
    """Per-checkpoint count of requests that exhausted GPU memory and retried on CPU.

    Read through side-effect-free accessors; ``Router.load()`` would reorder the
    LRU and rebuild evicted checkpoints. getattr-guarded so a stub router without
    the counters stays health-compatible.
    """
    from laya.mcp.device import router_agent

    fallbacks: Dict[str, Dict[str, Any]] = {}
    for name in router.loaded or []:
        agent = router_agent(router, name)
        fallbacks[name] = {
            "count": getattr(agent, "cpu_fallback_count", 0),
            "last_reason": getattr(agent, "last_fallback_reason", None),
        }
    return fallbacks


@asynccontextmanager
async def lifespan(app: Any) -> AsyncIterator[None]:
    """Build the engine, keep it warm, and drain the worker pool on shutdown.

    An engine already on ``app.state`` is adopted rather than replaced: that is how
    a test injects a stub Router and still gets the shutdown path exercised.

    Draining matters for a process supervisor, a TestClient and an embedded ASGI
    server alike -- an abandoned worker holding a checkpoint delays exit by the
    length of an in-flight forward pass.
    """
    config: Config = app.state.config
    existing = getattr(app.state, "engine", None)
    if existing is None:
        app.state.thread_settings = pin_torch_threads(config)
        app.state.engine = Engine(build_router(config), config)
    try:
        yield
    finally:
        engine: Engine = app.state.engine
        await engine.shutdown()