"""Live tests against a real checkpoint.

Excluded from the default run by ``-m 'not live'`` in pyproject, because they need
~800MB of weights in the local Hugging Face cache and take seconds per forward pass.
Run them with::

    python -m pytest -m live

If a checkpoint is not cached these skip rather than fail, so a fresh checkout does
not turn into a 1.4GB download the moment someone runs the suite. Populate the cache
first with ``python -m scripts.warmup --models all``.

What is worth a live test is only what the fake cannot check: that the response shape
the schemas document is the shape the checkpoints actually produce, and that routing
really does pick multilingual for non-Latin script.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.config import Config
from app.main import create_app


def _cached(repo_id: str) -> bool:
    from pathlib import Path

    cache = Path.home() / ".cache" / "huggingface" / "hub"
    folder = cache / ("models--" + repo_id.replace("/", "--"))
    return folder.is_dir() and any(folder.rglob("*.safetensors"))


pytestmark = pytest.mark.live


@pytest.fixture(scope="module")
def live_client():
    if not _cached("convaiinnovations/laya"):
        pytest.skip("english checkpoint not cached; run `python -m scripts.warmup`")
    config = Config(device="cpu", preload=False, threads=1)
    with TestClient(create_app(config)) as client:
        yield client


def test_real_predict_matches_the_documented_response_shape(live_client: TestClient) -> None:
    response = live_client.post(
        "/v1/decisions/predict",
        json={
            "state": "My order arrived four days early and the packaging was damaged.",
            "questions": {
                "sentiment": {
                    "type": "choice",
                    "instructions": "How does the customer feel about the delivery?",
                    "criteria": {"positive": "Satisfied", "negative": "Unhappy"},
                }
            },
        },
    )

    assert response.status_code == 200
    payload = response.json()
    # Every key the response model promises, and no invented envelope around them.
    assert set(payload) >= {"model", "answers", "usage", "routing"}
    answer = payload["answers"]["sentiment"]
    assert answer["type"] == "choice"
    assert answer["choice"] in {"positive", "negative"}
    assert set(answer["probabilities"]) == {"positive", "negative"}
    assert sum(answer["probabilities"].values()) == pytest.approx(1.0, abs=1e-3)
    assert payload["routing"]["model"] == "english"


def test_all_three_question_types_answer_in_one_pass(live_client: TestClient) -> None:
    response = live_client.post(
        "/v1/decisions/predict",
        json={
            "state": "The refund was issued to my card on Tuesday and is not visible yet.",
            "questions": {
                "c": {
                    "type": "choice",
                    "instructions": "What is the customer asking about?",
                    "criteria": {"refund": "A refund", "shipping": "A delivery"},
                },
                "s": {
                    "type": "score",
                    "instructions": "How urgent is this?",
                    "criteria": ["low", "medium", "high"],
                },
                "n": {
                    "type": "noul",
                    "instructions": "Has the customer already been refunded?",
                },
            },
        },
    )

    assert response.status_code == 200
    answers = response.json()["answers"]
    assert answers["c"]["type"] == "choice"
    assert 0 <= answers["s"]["score"] < 3
    assert 0.0 <= answers["n"]["noul"] <= 1.0


def test_devanagari_routes_to_the_multilingual_checkpoint(live_client: TestClient) -> None:
    # The reason routing happens before inference rather than as a confidence check:
    # the English checkpoint is not merely weaker on Khmer, it collapses while
    # staying confident. A correct-looking answer from the wrong checkpoint is the
    # failure mode this asserts against.
    response = live_client.post(
        "/v1/decisions/predict",
        json={
            "state": "मेरा ऑर्डर देर से आया है और मैं बहुत नाराज हूं।",
            "questions": {
                "n": {"type": "noul", "instructions": "क्या ग्राहक नाराज है?"}
            },
        },
    )

    assert response.status_code == 200
    assert response.json()["routing"]["model"] == "multilingual"


def test_real_batch_returns_one_result_per_state_in_order(live_client: TestClient) -> None:
    states = [
        "The package never arrived and nobody has replied to my emails.",
        "Great news, my replacement arrived today and it works perfectly.",
    ]

    response = live_client.post(
        "/v1/decisions/predict/batch",
        json={
            "states": states,
            "questions": {
                "n": {"type": "noul", "instructions": "Is the customer unhappy?"}
            },
        },
    )

    assert response.status_code == 200
    results = response.json()
    assert len(results) == len(states)
    # Order is the contract, so it is worth asserting rather than assuming: the SDK
    # groups by checkpoint internally to share forward passes and restores input
    # order afterwards.
    assert 0.0 <= results[0]["answers"]["n"]["noul"] <= 1.0
    assert 0.0 <= results[1]["answers"]["n"]["noul"] <= 1.0
    assert results[0]["answers"]["n"]["noul"] > results[1]["answers"]["n"]["noul"]
