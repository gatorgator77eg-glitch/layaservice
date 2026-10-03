"""OpenAPI 3.0 normalisation.

The generated document is the only contract a consumer sees before writing code, so
these check the specific 3.1-to-3.0 gaps that would otherwise ship silently.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.openapi import OPENAPI_VERSION, _assert_portable, _rewrite
from tests.conftest import question


def test_const_becomes_a_single_value_enum() -> None:
    rewritten = _rewrite({"const": "choice"})

    # 3.0 has no `const`. A 3.0 consumer reading it sees an unconstrained string and
    # generates a client that accepts any value for the field.
    assert rewritten == {"enum": ["choice"]}


def test_existing_matching_enum_is_not_duplicated() -> None:
    rewritten = _rewrite({"const": "choice", "enum": ["choice"]})

    assert rewritten["enum"] == ["choice"]


def test_schema_examples_become_singular_example() -> None:
    rewritten = _rewrite({"examples": ["first", "second"]})

    # 3.0's Schema Object has `example`; `examples` is a 3.1/JSON-Schema keyword.
    assert rewritten["example"] == "first"
    assert "examples" not in rewritten


def test_union_type_with_null_becomes_nullable() -> None:
    rewritten = _rewrite({"type": ["string", "null"]})

    assert rewritten["type"] == "string"
    assert rewritten["nullable"] is True


def test_union_type_without_null_becomes_any_of() -> None:
    # A union of two non-null types is not expressible as `type` in 3.0.
    rewritten = _rewrite({"type": ["string", "integer"]})

    assert "type" not in rewritten
    assert rewritten["anyOf"] == [{"type": "string"}, {"type": "integer"}]


def test_nested_schemas_are_rewritten_too() -> None:
    # The rewrite has to recurse; a shallow pass leaves `const` behind in every
    # component schema, which is exactly where the discriminator lives.
    rewritten = _rewrite(
        {
            "properties": {
                "answers": {
                    "additionalProperties": {
                        "properties": {"type": {"const": "noul"}}
                    }
                }
            }
        }
    )

    node = rewritten["properties"]["answers"]["additionalProperties"]["properties"]["type"]
    assert node == {"enum": ["noul"]}


def test_assert_portable_rejects_a_surviving_const() -> None:
    with pytest.raises(RuntimeError, match="const"):
        _assert_portable({"components": {"schemas": {"X": {"properties": {"t": {"const": "a"}}}}}})


def test_assert_portable_accepts_a_clean_document() -> None:
    _assert_portable({"openapi": OPENAPI_VERSION, "paths": {}})  # must not raise


# --- against the real application -------------------------------------------------


def test_generated_document_declares_3_0_3(client: TestClient) -> None:
    assert client.get("/openapi.json").json()["openapi"] == OPENAPI_VERSION


def test_generated_document_contains_no_const(client: TestClient) -> None:
    import json

    assert "const" not in json.dumps(client.get("/openapi.json").json())


def test_question_type_is_constrained_in_the_document(client: TestClient) -> None:
    spec = client.get("/openapi.json").json()

    # The question `type` is what selects the discriminated branch, so it has to be
    # constrained or a generated client can send `type: "typo"` and get a 422 the
    # schema never warned about.
    schemas = spec["components"]["schemas"]
    noul = schemas["NoulQuestion"]["properties"]["type"]
    assert noul.get("enum") == ["noul"]


def test_document_passes_its_own_portability_check(client: TestClient) -> None:
    _assert_portable(client.get("/openapi.json").json())


def test_both_endpoints_document_their_request_body(client: TestClient) -> None:
    spec = client.get("/openapi.json").json()

    for path in ("/v1/decisions/predict", "/v1/decisions/predict/batch"):
        body = spec["paths"][path]["post"].get("requestBody")
        assert body is not None, f"{path} has no documented requestBody"
        assert body["required"] is True
        ref = body["content"]["application/json"]["schema"]["$ref"]
        # The $ref has to resolve, or a generator produces a client with no body.
        name = ref.rsplit("/", 1)[-1]
        assert name in spec["components"]["schemas"]


def test_question_schemas_are_in_the_document(client: TestClient) -> None:
    # The nested question models only appear because the request body is registered.
    # Their absence is how an undocumented request side hides.
    schemas = client.get("/openapi.json").json()["components"]["schemas"]

    for name in (
        "PredictRequest",
        "BatchPredictRequest",
        "ChoiceQuestion",
        "ScoreQuestion",
        "NoulQuestion",
        "StateOverride",
    ):
        assert name in schemas


def test_state_override_is_keyed_by_index(client: TestClient) -> None:
    # Keyed by integer index into `states`, so the document has to say integer --
    # a client generating a string key would send overrides that never apply.
    schemas = client.get("/openapi.json").json()["components"]["schemas"]
    overrides = schemas["BatchPredictRequest"]["properties"]["overrides"]

    branches = overrides.get("anyOf", [overrides])
    mapping = next(b for b in branches if b.get("type") == "object")
    assert mapping["additionalProperties"]["$ref"].endswith("/StateOverride")


def test_optional_fields_are_nullable_not_null_typed(client: TestClient) -> None:
    import json

    spec = client.get("/openapi.json").json()
    # Pydantic v2 writes `X | None` as `anyOf: [X, {"type": "null"}]`. OpenAPI 3.0
    # has no null type, so a strict 3.0 validator rejects it and a generator reads the
    # field as required. It must arrive as `nullable: true`.
    assert '"type": "null"' not in json.dumps(spec)
    overrides = spec["components"]["schemas"]["BatchPredictRequest"]["properties"]["overrides"]
    assert overrides.get("nullable") is True
    spec = client.get("/openapi.json").json()

    properties = spec["components"]["schemas"]["ChoiceQuestion"]["properties"]
    # `criteria`, not `options`, and `instructions` is required -- both verified
    # against `laya.agent`'s own validator.
    assert "criteria" in properties
    assert "options" not in properties
    assert "instructions" in spec["components"]["schemas"]["ChoiceQuestion"]["required"]
