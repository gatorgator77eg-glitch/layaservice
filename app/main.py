"""Application factory.

``create_app`` accepts an injected Router so tests can run the whole HTTP surface
against a stub, with no checkpoint download and no forward pass. Everything
except the SDK call itself is exercised that way.

The service deliberately does not use ``laya.serve``. Upstream's server registers
its routes on literal paths (``/v1/systemone``) with no override, and validates
by hand while declaring no response models, so its ``/openapi.json`` carries no
schema a consumer can generate a client from. The guards are reimplemented here
instead, on top of the same ``laya.Router``.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Optional

# `transformers` probes for TensorFlow at import time, and when TensorFlow is
# installed its abseil runtime can deadlock model construction -- the process hangs
# at load with no error. This must be set before laya is imported anywhere.
os.environ.setdefault("USE_TF", "0")

from fastapi import FastAPI  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from app.calibration import CalibrationService  # noqa: E402
from app.config import Config, ConfigError  # noqa: E402
from app.engine import Engine, lifespan as engine_lifespan, pin_torch_threads  # noqa: E402
from app.openapi import install_openapi  # noqa: E402
from app.profiles import ProfileStore  # noqa: E402
from app.routes import admin, dynamic, predict, telemetry  # noqa: E402

VERSION = "1.0.0"

CONSOLE_DIR = Path(__file__).resolve().parent / "console"


@asynccontextmanager
async def lifespan(app: Any) -> AsyncIterator[None]:
    """The engine's own startup, plus the calibration wiring that needs a Router.

    ``app.engine.lifespan`` stays ignorant of profiles on purpose: it owns checkpoints
    and the worker pool, and reaching into it for an optional application feature
    would couple the two. This wrapper is where the feature-specific startup lives.

    The order matters. The engine lifespan builds the Router first, and only then can
    active calibrations be reloaded and the ``on_load`` hook be attached to the object
    that will actually fire it. Doing it any earlier would be a no-op in production,
    where the Router does not exist until this runs.
    """
    async with engine_lifespan(app):
        service = getattr(app.state, "calibration", None)
        if service is not None:
            # Hook first, then restore: attaching the Router is what lets `restore`
            # compare a fitted revision against what is actually resident, and lets it
            # apply the temperature to an already-preloaded checkpoint instead of
            # waiting for a load event that may never come.
            service.install(app.state.engine.router)
            service.restore()
        yield

DESCRIPTION = """
A reusable inference endpoint over the Laya decision model. Give it a **state**
(a text, email, ticket or JSON document) and a set of **typed questions**, and it
returns structured decisions with calibrated probabilities in a single forward
pass. It never generates text, so there is nothing to parse and nothing to
hallucinate.

## Decision types

| Type | Question | Answer |
| --- | --- | --- |
| `choice` | Which of these options applies? | The selected label, plus a probability per option |
| `score` | Where does this sit on this ordered scale? | The expected level index, plus the distribution |
| `noul` | Is this proposition true? | `P(true)`, a float in [0,1] |

Every question is answered independently against the same state in one forward
pass, so adding a question never changes another question's answer.

## Two confidence numbers

`answer_confidence` is the probability mass on the answer being reported -- the
number that gates, and the one any temperature fit is computed on.

`confidence` means something different per type: normalised entropy
`1 - H(p)/log(k)` on `choice` and `score`, and `max(p_yes, p_no)` on `noul`. It
depends on how many options the question had, so a two-option distribution comes
back around 0.90 on a `noul` and 0.53 on an equivalent `choice`. **Never compare
it against a threshold.** Gate on `answer_confidence`.

## Before you set `min_confidence`

The checkpoints ship over-confident. Measured expected calibration error falls
from 0.466 to 0.081 for the English checkpoint and 0.314 to 0.106 for
multilingual once temperatures are fitted, and the direction is task-dependent:
one documented routing task was *under*-confident instead. Fit and validate
temperatures on held-out examples from your own workload before treating any
confidence value as a decision boundary. Until then, treat confidence as a ranking
signal for human-review triage, not as an automated gate.

## Routing

Requests are routed by script and language detected on the state. The English
checkpoint is not merely weaker outside English, it collapses while staying
confident -- 0.000 accuracy at 0.952 confidence on Khmer -- which is why routing
happens *before* the forward pass rather than as a confidence check afterwards.
Set `model` to pin a checkpoint, `lang` to supply a language hint, or `task` to
force a workflow. The `routing` block in every response records what was chosen
and why.

## Limits and errors

| Status | When |
| --- | --- |
| 400 | Body is not a JSON object, or holds an unpaired surrogate escape |
| 413 | A state or question set is over a size limit; `detail` names which and by how much |
| 422 | A required field is missing or null, the question set is invalid, or a request control has a bad value |
| 500 | Inference failed. The cause is in the server log; the client learns nothing |
| 503 | Admission limit reached. Retry after the interval in `Retry-After` |

The 413/422 split is deliberate. A state that is too *long*, or a question set with
too many *options*, is a 413: the request is bigger than this deployment accepts. A
`max_len` above the server's ceiling is a 422: the request is well-formed and the
value is simply one this service will not honour.

## Token budgets

Only the first 512 tokens are read by the English checkpoint and 1,024 by
multilingual, per question, so put the decisive text first. `usage` is the only
place a cut is visible: `truncated`, `state_tokens_dropped` and
`truncated_questions` report what did not reach the model. `max_len` can widen a
window up to the server's `token_budget` (8,192 by default); the multilingual
encoder accepts it but measurably degrades past roughly 4,000 tokens of preceding
text.
"""


def create_app(config: Optional[Config] = None, router_obj: Optional[Any] = None) -> FastAPI:
    """Build the ASGI application.

    ``router_obj`` is for tests. Passing one skips both the checkpoint download and
    the preload, and the supplied object is used in place of the Router the
    configuration would otherwise build.
    """
    logging.basicConfig(
        level=(config or Config.from_env()).log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    cfg = config or Config.from_env()

    app = FastAPI(
        title="Laya Decision Service",
        version=VERSION,
        description=DESCRIPTION,
        root_path=cfg.root_path,
        lifespan=lifespan,
    )
    app.state.config = cfg

    if router_obj is not None:
        # Test path: adopt the stub so the lifespan adopts it too (rather than
        # building a Router that would download checkpoints), and pin threads so
        # thread assertions still have something real to read.
        app.state.thread_settings = pin_torch_threads(cfg)
        app.state.engine = Engine(router_obj, cfg)

    # Profiles outlive the process, so they are read here rather than created at
    # startup: a minted endpoint that vanished on restart would 404 for every
    # caller with no trace of why.
    app.state.profiles = ProfileStore(Path(cfg.data_dir))
    app.state.calibration = CalibrationService(app.state.profiles)
    # No calibration wiring here: in production the Router does not exist until the
    # lifespan builds it, so `restore`/`install` belong there. With an injected Router
    # the lifespan still runs, so both paths are covered by one implementation.

    install_openapi(app, version=VERSION, title="Laya Decision Service", description=DESCRIPTION)

    app.include_router(predict.router)
    app.include_router(admin.router)
    app.include_router(telemetry.router)
    app.mount("/console", StaticFiles(directory=CONSOLE_DIR, html=True), name="console")

    restored = dynamic.register_all(app, app.state.profiles, cfg, app.state.calibration)
    if restored:
        logging.getLogger("laya_service.main").info(
            "restored %d profile endpoint(s): %s", len(restored), ", ".join(restored)
        )

    return app


def build_default_app() -> FastAPI:
    """Entry point for ``uvicorn app.main:build_default_app --factory``."""
    return create_app()


__all__ = ["create_app", "build_default_app", "ConfigError", "Config", "VERSION"]