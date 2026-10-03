"""OpenAPI 3.0 normalisation.

FastAPI's generated document is OpenAPI 3.1, which is a superset of 3.0. A 3.1
document served from a 3.0 URL is a real interoperability problem for the tooling
the spec exists for: generators, validators and client SDK generators branch on the
version, and several treat 3.0 and 3.1 as different schemas rather than nested.

The gap that actually bites is ``const``. Pydantic v2 emits ``{"const": "choice"}``
for a single-valued ``Literal``, and OpenAPI 3.0 has no ``const`` keyword -- it is
3.1-only. A generator reading 3.0 sees an unconstrained string and produces a client
that accepts any value for ``type``. Rewriting it to ``enum: ["choice"]`` is exactly
equivalent and portable.

Anything else 3.1-only that shows up (``examples`` inside a schema, ``$defs`` naming,
``type: [x, null]`` unions) is handled here too, so the document a consumer
generates from is the document this service actually enforces.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

_log = logging.getLogger("laya_service.openapi")

OPENAPI_VERSION = "3.0.3"

JSON_SCHEMA_2020_12 = "https://json-schema.org/draft/2020-12/schema"


def _rewrite(node: Any) -> Any:
    """Recursively convert 3.1-only constructs to their 3.0 equivalents."""
    if isinstance(node, dict):
        out: Dict[str, Any] = {}

        # `const` -> single-value `enum`.
        if "const" in node:
            const_value = node.pop("const")
            existing = node.get("enum")
            if isinstance(existing, list) and existing == [const_value]:
                pass  # already expressed as a one-value enum
            else:
                node["enum"] = [const_value]

        # `examples` (array form) -> `example` (singular). 3.0's Schema Object has
        # `example`; `examples` is a 3.1/JSON-Schema keyword.
        examples = node.get("examples")
        if isinstance(examples, list) and examples and "example" not in node:
            node["example"] = examples[0]
            node.pop("examples")

        # `type: ["string", "null"]` -> nullable. JSON Schema union types are not
        # expressible in 3.0.
        node_type = node.get("type")
        if isinstance(node_type, list):
            non_null = [t for t in node_type if t != "null"]
            if len(non_null) != len(node_type):
                node["nullable"] = True
            if len(non_null) == 1:
                node["type"] = non_null[0]
            elif non_null:
                # AnyOf is the 3.0 way to say "one of these types".
                node.pop("type")
                node["anyOf"] = [{"type": t} for t in non_null]
            else:
                node.pop("type")

        if node.get("default", False) is None and "default" in node:
            # `default: null` cannot be typed in 3.0, and the field's optionality is
            # already carried by its absence from `required`.
            node.pop("default")

        # `anyOf: [X, {"type": "null"}]` is how Pydantic v2 expresses `X | None`.
        # OpenAPI 3.0 has no null type, so collapse the null branch into `nullable`.
        # Left alone, this is not merely ugly: a strict 3.0 validator rejects the
        # document, and a generator reads the field as required.
        any_of = node.get("anyOf")
        if isinstance(any_of, list):
            non_null = [
                entry
                for entry in any_of
                if not (isinstance(entry, dict) and entry.get("type") == "null")
            ]
            if len(non_null) != len(any_of):
                node["nullable"] = True
                if non_null:
                    node["anyOf"] = non_null
                else:
                    node.pop("anyOf")

        for key, value in node.items():
            out[key] = _rewrite(value)
        return out

    if isinstance(node, list):
        return [_rewrite(item) for item in node]

    return node


def _demote_schema_dialect(schema: Dict[str, Any]) -> None:
    """Point any remaining dialect declaration at the 2020-12 vocabulary.

    Harmless on its own -- the constructs 3.0 uses are valid in 2020-12 -- but a
    validator that honours the declared dialect strictly may then apply 3.1-only
    defaults to a document that claims to be 3.0.
    """
    if schema.get("$schema") == JSON_SCHEMA_2020_12:
        schema["$schema"] = "http://json-schema.org/draft-04/schema#"


def _install_model(spec: Dict[str, Any], model: Any) -> str:
    """Register a Pydantic model in ``components/schemas`` and return a $ref to it.

    Needed because both endpoints read the body off a raw ``Request`` -- that is how
    the body-size cap can reject an oversized payload *before* it is parsed into
    objects -- and FastAPI infers request schemas from handler signatures. So the
    request side of the API was silently undocumented: responses typed, requests
    invisible. A generator would produce a client with no idea what to send.
    """
    from pydantic import BaseModel

    assert issubclass(model, BaseModel)
    components = spec.setdefault("components", {})
    schemas = components.setdefault("schemas", {})

    schema = model.model_json_schema(ref_template="#/components/schemas/{model}")

    # Pydantic hoists nested models into `$defs`; OpenAPI 3.0 wants them beside the
    # top-level schemas so the `$ref`s above resolve.
    defs = schema.pop("$defs", {})
    for name, definition in defs.items():
        nested = dict(definition)
        nested.pop("$defs", None)
        nested.pop("title", None)
        schemas.setdefault(name, nested)

    schema.pop("title", None)
    schemas[model.__name__] = schema
    return f"#/components/schemas/{model.__name__}"


# (path, method) -> request model, declared by whichever module owns the route.
#
# The two decision endpoints and the admin endpoints all read the body off a raw
# `Request`, so FastAPI infers nothing from their signatures and the request side of
# the API is invisible unless it is attached by hand. A registry rather than a
# hardcoded dict because the profile endpoints are minted at runtime: each one
# declares its own body as it registers, and the document is regenerated when the
# route set changes.
_REQUEST_BODIES: Dict[Any, Any] = {}


def declare_request_body(path: str, method: str, model: Any) -> None:
    """Record the body model for one operation, for the document to reference."""
    _REQUEST_BODIES[(path, method.lower())] = model


def forget_request_body(path: str, method: str = "post") -> None:
    """Drop a declaration when its route is removed.

    Without this a deleted profile would keep a ``$ref`` in the document pointing at
    a model for an endpoint that no longer answers, which is a client generator's
    worst case: it emits a call that 404s.
    """
    _REQUEST_BODIES.pop((path, method.lower()), None)


def _attach_request_bodies(spec: Dict[str, Any]) -> None:
    for (path, method), model in list(_REQUEST_BODIES.items()):
        operation = spec.get("paths", {}).get(path, {}).get(method)
        if operation is None:
            continue
        from pydantic import BaseModel

        if not (isinstance(model, type) and issubclass(model, BaseModel)):
            continue
        ref = _install_model(spec, model)
        operation["requestBody"] = {
            "required": True,
            "description": (operation.get("description") or "Request body."),
            "content": {"application/json": {"schema": {"$ref": ref}}},
        }


def build_openapi(app: FastAPI, *, version: str, title: str, description: str) -> Dict[str, Any]:
    """Generate a 3.0.3 document regardless of FastAPI's default 3.1."""
    schema = get_openapi(
        title=title,
        version=version,
        description=description,
        routes=app.routes,
    )
    _attach_request_bodies(schema)
    schema = _rewrite(schema)
    schema["openapi"] = OPENAPI_VERSION
    _demote_schema_dialect(schema)

    # FastAPI emits component schemas under `$defs` in 3.1. 3.0 reads the same
    # references from `#/components/schemas`, which is where FastAPI already put
    # them; only the leftover key is dropped so no tool sees two vocabularies.
    components = schema.get("components")
    if isinstance(components, dict) and "$defs" in components:
        components.pop("$defs")

    _assert_portable(schema)
    return schema


def _assert_portable(schema: Dict[str, Any]) -> None:
    """Fail loudly at startup if a 3.1-only construct survived.

    Better a refused startup than a document that quietly misdescribes the API.
    """
    problems: List[str] = []

    def walk(node: Any, path: str) -> None:
        if isinstance(node, dict):
            if "const" in node:
                problems.append(f"{path}: `const` is OpenAPI 3.1-only")
            if isinstance(node.get("type"), list):
                problems.append(f"{path}: union `type` is not expressible in 3.0")
            if node.get("type") == "null":
                # 3.0 spells an absent value `nullable: true`; it has no null type.
                problems.append(f"{path}: `type: null` is not valid in 3.0")
            for key, value in node.items():
                walk(value, f"{path}/{key}")
        elif isinstance(node, list):
            for index, item in enumerate(node):
                walk(item, f"{path}[{index}]")

    walk(schema, "#")
    if problems:
        seen: List[str] = []
        for problem in problems:
            if problem not in seen:
                seen.append(problem)
        raise RuntimeError(
            "generated OpenAPI document is not portable to 3.0: " + "; ".join(seen[:10])
        )


def install_openapi(app: FastAPI, *, version: str, title: str, description: str) -> None:
    """Override the app's document generator in place."""

    def openapi() -> Dict[str, Any]:
        if app.openapi_schema:
            return app.openapi_schema
        app.openapi_schema = build_openapi(
            app, version=version, title=title, description=description
        )
        return app.openapi_schema

    app.openapi = openapi  # type: ignore[method-assign]