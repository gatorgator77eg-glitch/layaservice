"""Shared fixtures.

``client`` is built over the stub Router, so the default suite needs no weights and
runs in well under a second. Live checkpoint tests opt in with ``@pytest.mark.live``.
"""

from __future__ import annotations

import os
from typing import Iterator

import pytest
from fastapi.testclient import TestClient

# Must precede the laya import inside app.main: transformers probes for TensorFlow at
# import time and its abseil runtime can hang model construction.
os.environ.setdefault("USE_TF", "0")

from app.config import Config  # noqa: E402
from app.main import create_app  # noqa: E402
from tests.fake_router import FakeRouter  # noqa: E402


@pytest.fixture
def config() -> Config:
    return Config(device="cpu", preload=False, max_concurrent=4, threads=1)


@pytest.fixture
def fake_router() -> FakeRouter:
    return FakeRouter()


@pytest.fixture
def client(config: Config, fake_router: FakeRouter) -> Iterator[TestClient]:
    app = create_app(config, router_obj=fake_router)
    with TestClient(app) as test_client:
        yield test_client


def question(qid: str = "q1", kind: str = "choice", **extra: object) -> dict:
    """A minimal valid question of the requested type.

    Shaped after ``laya.agent``'s own contract, which is what the service accepts:
    ``instructions`` is required and non-empty for every type, and both choice
    options and score levels are spelled ``criteria``. There is no ``id`` field --
    the question id is the key it sits under in the ``questions`` mapping.
    """
    base: dict = {
        "type": kind,
        "instructions": f"Is this state about {qid}?",
    }
    if kind == "choice":
        base["criteria"] = {"yes": "Yes", "no": "No"}
    elif kind == "score":
        base["criteria"] = ["low", "high"]
    base.update(extra)
    return base


def body(**extra: object) -> dict:
    payload = {
        "state": "The order shipped early and the customer is happy.",
        "questions": {"q1": question()},
    }
    payload.update(extra)
    return payload
