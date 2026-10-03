"""The HTTP surface, exercised end to end over the stub Router.

These are the tests that would have caught the batch-envelope bug: if the service
invents a response shape the SDK does not produce, or forwards arguments the SDK does
not accept, a stub written against the real signature fails here.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app.config import Config
from app.main import create_app
from tests.conftest import body, question
from tests.fake_router import FakeRouter


# --- happy path -------------------------------------------------------------------


def test_predict_returns_laya_payload_verbatim(client: TestClient, fake_router: FakeRouter) -> None:
    response = client.post("/v1/decisions/predict", json=body())

    assert response.status_code == 200
    payload = response.json()
    # Byte-for-byte what the SDK produced: no re-serialisation through a response
    # model, so conditional keys stay absent rather than becoming null.
    assert payload["model"] == "laya-rl-agent"
    assert payload["answers"]["q1"]["type"] == "choice"
    assert payload["usage"]["truncated"] is False
    # Absent, not null.
    assert "truncated_questions" not in payload["usage"]


def test_predict_reports_inference_time(client: TestClient) -> None:
    response = client.post("/v1/decisions/predict", json=body())

    assert response.status_code == 200
    assert "Server-Timing" in response.headers
    assert float(response.headers["X-Inference-Time-Ms"]) >= 0.0


def test_all_three_question_types_round_trip(client: TestClient) -> None:
    payload = {
        "state": "Order 41 shipped early.",
        "questions": {
            "c": question("c", "choice"),
            "s": question("s", "score"),
            "n": {"type": "noul", "instructions": "Is it true?"},
        },
    }

    response = client.post("/v1/decisions/predict", json=payload)

    assert response.status_code == 200
    answers = response.json()["answers"]
    assert answers["c"]["type"] == "choice"
    assert answers["s"]["type"] == "score"
    assert answers["n"]["type"] == "noul"


def test_state_may_be_a_json_document(client: TestClient, fake_router: FakeRouter) -> None:
    state = {"subject": "Ada", "event": "refund", "amount": 12}

    response = client.post("/v1/decisions/predict", json=body(state=state))

    assert response.status_code == 200
    # Reaches the SDK unchanged, rather than being stringified on the way through.
    assert fake_router.calls[0]["state"] == state


def test_state_may_be_a_list(client: TestClient, fake_router: FakeRouter) -> None:
    state = ["line one", "line two"]

    response = client.post("/v1/decisions/predict", json=body(state=state))

    assert response.status_code == 200
    assert fake_router.calls[0]["state"] == state


# --- batch ---------------------------------------------------------------------


def test_batch_returns_one_result_per_state_in_order(client: TestClient) -> None:
    states = ["first state", "second state", "third state"]

    response = client.post(
        "/v1/decisions/predict/batch",
        json={"states": states, "questions": {"q1": question()}},
    )

    assert response.status_code == 200
    results = response.json()
    # A flat array, not an envelope. Matches what the SDK returns.
    assert isinstance(results, list)
    assert len(results) == 3
    assert all(item["model"] == "laya-rl-agent" for item in results)


def test_batch_builds_one_request_per_state_with_shared_questions(
    client: TestClient, fake_router: FakeRouter
) -> None:
    states = ["a", "b"]

    client.post(
        "/v1/decisions/predict/batch",
        json={"states": states, "questions": {"q1": question()}},
    )

    call = fake_router.calls[0]
    assert call["method"] == "predict_batch"
    assert [item["state"] for item in call["requests"]] == states
    for item in call["requests"]:
        assert item["questions"] == {"q1": question("q1")}


def test_batch_forwards_call_level_controls(client: TestClient, fake_router: FakeRouter) -> None:
    client.post(
        "/v1/decisions/predict/batch",
        json={
            "states": ["a", "b"],
            "questions": {"q1": question()},
            "batch_size": 2,
            "sort_by_length": True,
            "min_confidence": 0.7,
        },
    )

    call = fake_router.calls[0]
    assert call["batch_size"] == 2
    assert call["sort_by_length"] is True
    assert call["min_confidence"] == 0.7


def test_batch_applies_per_state_overrides(client: TestClient, fake_router: FakeRouter) -> None:
    client.post(
        "/v1/decisions/predict/batch",
        json={
            "states": ["english text", "नमस्ते दुनिया"],
            "questions": {"q1": question()},
            "overrides": {"1": {"model": "multilingual", "max_len": 2048}},
        },
    )

    requests = fake_router.calls[0]["requests"]
    assert "model" not in requests[0]
    assert requests[1]["model"] == "multilingual"
    assert requests[1]["max_len"] == 2048


def test_batch_rejects_override_index_out_of_range(client: TestClient) -> None:
    response = client.post(
        "/v1/decisions/predict/batch",
        json={
            "states": ["only one"],
            "questions": {"q1": question()},
            "overrides": {"5": {"model": "multilingual"}},
        },
    )

    assert response.status_code == 422
    # Says which index is wrong, not just "invalid".
    assert "5" in str(response.json()["detail"])


# --- guards ---------------------------------------------------------------------


def test_missing_state_is_422(client: TestClient) -> None:
    # 422, not 400: `state` is a required field like any other, and special-casing
    # it would force a caller to handle 400 for one field and 422 for all others.
    response = client.post("/v1/decisions/predict", json={"questions": {"q1": question()}})

    assert response.status_code == 422
    assert "state" in str(response.json()["detail"])


def test_null_state_is_422_with_one_clear_message(client: TestClient) -> None:
    response = client.post("/v1/decisions/predict", json=body(state=None))

    assert response.status_code == 422
    detail = response.json()["detail"]
    # One problem, one message. Left to the union, null fails all three branches and
    # the caller gets three errors naming three internal type paths.
    messages = detail if isinstance(detail, list) else [detail]
    assert len(messages) == 1
    assert "state" in str(messages[0])


def test_malformed_json_is_400(client: TestClient) -> None:
    response = client.post(
        "/v1/decisions/predict",
        content=b'{"state": "x", ',
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 400


def test_oversized_body_is_413(client: TestClient) -> None:
    response = client.post("/v1/decisions/predict", json=body(state="x" * 5_000_000))

    assert response.status_code == 413
    assert "body" in str(response.json()["detail"]).lower()


def test_oversized_state_is_413(client: TestClient, config: Config) -> None:
    limit = config.limits.state_chars
    response = client.post("/v1/decisions/predict", json=body(state="x" * (limit + 1)))

    assert response.status_code == 413
    assert str(limit) in str(response.json()["detail"])


def test_choice_option_amplification_is_413(client: TestClient, config: Config) -> None:
    limit = config.limits.choice_options
    options = {f"opt{i}": f"Option {i}" for i in range(limit + 1)}

    response = client.post(
        "/v1/decisions/predict",
        json={
                "state": "x",
                "questions": {"q1": {"type": "choice", "instructions": "Which?", "criteria": options}},
            },
    )

    assert response.status_code == 413
    assert str(limit) in str(response.json()["detail"])


def test_invalid_question_is_422_and_names_the_question(client: TestClient) -> None:
    payload = {
        "state": "x",
        "questions": {
            "good": question("good"),
            "bad": {"type": "choice", "instructions": "Which?", "criteria": {}},
        },
    }

    response = client.post("/v1/decisions/predict", json=payload)

    assert response.status_code == 422
    detail = str(response.json()["detail"])
    assert "bad" in detail


def test_unknown_body_field_is_422(client: TestClient) -> None:
    response = client.post("/v1/decisions/predict", json=body(nonsense=1))

    assert response.status_code == 422


def test_min_confidence_out_of_range_is_422(client: TestClient) -> None:
    response = client.post("/v1/decisions/predict", json=body(min_confidence=1.5))

    assert response.status_code == 422


def test_max_len_above_server_budget_is_422(client: TestClient, config: Config) -> None:
    # A well-formed request carrying a value this service will not honour, so 422
    # rather than the 413 used for oversized content.
    response = client.post("/v1/decisions/predict", json=body(max_len=config.limits.token_budget + 1))

    assert response.status_code == 422
    assert str(config.limits.token_budget) in str(response.json()["detail"])


def test_inference_failure_is_500_with_no_internal_detail(config: Config) -> None:
    router_obj = FakeRouter(raise_with=RuntimeError("secret internal detail /etc/passwd"))
    app = create_app(config, router_obj=router_obj)

    with TestClient(app, raise_server_exceptions=False) as test_client:
        response = test_client.post("/v1/decisions/predict", json=body())

    assert response.status_code == 500
    # The operator gets the traceback; the caller does not.
    assert "passwd" not in response.text
    assert "secret internal detail" not in response.text


def test_validation_error_from_core_is_422_with_its_message(config: Config) -> None:
    router_obj = FakeRouter(raise_with=ValueError("question 'q1': options must not be empty"))
    app = create_app(config, router_obj=router_obj)

    with TestClient(app) as test_client:
        response = test_client.post("/v1/decisions/predict", json=body())

    assert response.status_code == 422
    assert "options must not be empty" in str(response.json()["detail"])


# --- operations ------------------------------------------------------------------


def test_health_reports_loaded_and_limits(client: TestClient) -> None:
    payload = client.get("/health").json()

    assert payload["status"] == "ok"
    assert "loaded" in payload
    assert payload["limits"]["state_chars"] > 0


def test_metrics_exposes_the_latency_histograms(client: TestClient) -> None:
    client.post("/v1/decisions/predict", json=body())
    text = client.get("/metrics").text

    assert "laya_request_duration_seconds" in text
    assert "laya_inference_duration_seconds" in text
    assert "laya_queue_wait_seconds" in text


def test_openapi_is_3_0_3_and_portable(client: TestClient) -> None:
    spec = client.get("/openapi.json").json()

    assert spec["openapi"] == "3.0.3"
    # Pydantic v2 emits `const` for single-valued Literals, which is 3.1-only. A 3.0
    # consumer reading it sees an unconstrained string and generates a client that
    # accepts anything, so it must have been rewritten to a one-value enum.
    assert "const" not in json.dumps(spec)
    choice_type = spec["components"]["schemas"]["ChoiceAnswer"]["properties"]["type"]
    assert choice_type.get("enum") == ["choice"] or choice_type.get("$ref")


def test_openapi_documents_predict_batch_and_health(client: TestClient) -> None:
    paths = client.get("/openapi.json").json()["paths"]

    assert "/v1/decisions/predict" in paths
    assert "/v1/decisions/predict/batch" in paths
    assert "/health" in paths
    # Prometheus is a scrape target, not an API surface.
    assert "/metrics" not in paths


def test_batch_200_is_documented_as_an_array(client: TestClient) -> None:
    spec = client.get("/openapi.json").json()

    schema = spec["paths"]["/v1/decisions/predict/batch"]["post"]["responses"]["200"][
        "content"
    ]["application/json"]["schema"]
    assert schema["type"] == "array"
