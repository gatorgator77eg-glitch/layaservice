"""Profile management, the endpoints profiles mint, and calibration gating.

The expensive part of calibration -- the forward pass and the fitter -- is the SDK's,
so the tests here do not re-test its maths. They pin the things this service is
responsible for and could get wrong quietly:

  * a profile endpoint answers, and answers with the profile's questions;
  * every guard the fixed endpoint has, the minted one also has, in the same order;
  * a profile cannot be used to smuggle past the guards (extra keys, caller-supplied
    questions, caller-supplied budget, caller-supplied routing);
  * the profile document is the single source of truth for routing, and changing the
    questions invalidates the calibration fitted against the old ones;
  * the fit refuses to report a threshold it cannot support, and says which of the
    SDK's three sample floors was hit rather than quietly returning a type-level
    scalar dressed up as a per-bucket calibration.

Calibration is exercised through the SDK's real ``fit_temperature_map`` and
``fit_abstention_thresholds`` with synthetic records supplied via the ``records_from``
seam, so the artifact shape and the gating logic are tested without a checkpoint.
"""

from __future__ import annotations

import re

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pathlib import Path

from app.config import Config
from app.main import create_app
from tests.conftest import body, question
from tests.fake_router import FakeRouter


# --- helpers ---------------------------------------------------------------------

SIMPLE = {
    "name": "Triage",
    "questions": {"q1": question("q1", "choice")},
}

TWO_TYPES = {
    "name": "Triage",
    "questions": {
        "sev": {"type": "choice", "instructions": "How bad?", "criteria": {"low": "Low", "high": "High"}},
        "ok": {"type": "noul", "instructions": "Is this acceptable?"},
    },
}


def create(client: TestClient, pid: str = "triage", document: dict | None = None) -> dict:
    """Create a profile the way the console does: ``POST /v1/profiles`` with an id."""
    response = client.post("/v1/profiles", json={"id": pid, **(document or SIMPLE)})
    assert response.status_code == 201, response.text
    return response.json()["profile"]


def synthetic_records(n_rows: int, *, correct_rate: float = 0.7, choice_only: bool = False):
    """A ``records_from`` stand-in producing records shaped like the SDK's.

    Real records are ``(qtype, logits, target, k)`` with one-dimensional arrays, which
    is the contract ``fit_temperature_map`` and ``fit_abstention_thresholds`` consume.
    The logits are over-confident and the correctness rate varies with the gap, so the
    fitter has a real relationship to find rather than pure noise -- an unlearnable
    dataset would let a broken gate pass.

    ``choice_only`` puts every record in one bucket, which is how a test reaches the
    per-bucket floor. Spreading the same total over two buckets does not, because the
    floor is per bucket.
    """

    def produce(agent, pairs):
        records = []
        for index in range(n_rows):
            correct = (index % 10) < int(correct_rate * 10)
            gap = 1.8 if correct else -0.3
            target = np.array([1.0, 0.0]) if correct else np.array([0.0, 1.0])
            is_choice = choice_only or index % 2 == 0
            qtype = 0 if is_choice else 2
            records.append((qtype, np.array([gap, 0.0]), target, 2))
        return records

    return produce


def labelled(n_rows: int) -> list[dict]:
    return [
        {"state": f"case {i}", "expected": {"sev": "high" if i % 2 else "low", "ok": bool(i % 2)}}
        for i in range(n_rows)
    ]


@pytest.fixture
def profiles_client(tmp_path, fake_router: FakeRouter):
    """A client with a tighter profile cap, over the shared stub Router.

    Shares ``fake_router`` with the rest of the suite so a test can assert on what
    actually reached the SDK.
    """
    cfg = Config(
        device="cpu",
        preload=False,
        max_concurrent=4,
        threads=1,
        data_dir=str(tmp_path / "data"),
        max_profiles=3,
    )
    with TestClient(create_app(cfg, router_obj=fake_router)) as client:
        yield client


# --- the minted endpoint ---------------------------------------------------------


def test_profile_endpoint_answers_with_the_profile_questions(profiles_client: TestClient) -> None:
    create(profiles_client, document=TWO_TYPES)

    response = profiles_client.post(
        "/v1/profiles/triage/predict", json={"state": "Order 41 shipped early."}
    )

    assert response.status_code == 200, response.text
    answers = response.json()["answers"]
    # Both profile questions were asked, and only those: the caller never names them.
    assert set(answers) == {"sev", "ok"}


def test_profile_endpoint_is_a_real_openapi_path(profiles_client: TestClient) -> None:
    create(profiles_client)

    schema = profiles_client.get("/openapi.json").json()

    assert "/v1/profiles/triage/predict" in schema["paths"]
    body = schema["paths"]["/v1/profiles/triage/predict"]["post"]["requestBody"]
    ref = body["content"]["application/json"]["schema"]["$ref"]
    # It documents the caller-facing body, not the profile document: no questions key.
    name = ref.rsplit("/", 1)[-1]
    assert name != "Profile"
    assert name in schema["components"]["schemas"]


def test_profile_endpoint_rejects_an_unknown_field(profiles_client: TestClient) -> None:
    """A typo must not be silently ignored on a profile endpoint."""
    create(profiles_client)

    response = profiles_client.post(
        "/v1/profiles/triage/predict", json={"state": "x", "min_confidnce": 0.9}
    )

    assert response.status_code == 422


def test_profile_endpoint_rejects_caller_supplied_questions(profiles_client: TestClient) -> None:
    """The profile owns its questions; a caller may not widen them per request."""
    create(profiles_client)

    response = profiles_client.post(
        "/v1/profiles/triage/predict",
        json={"state": "x", "questions": {"other": question("other")}},
    )

    assert response.status_code == 422


def test_profile_endpoint_rejects_caller_supplied_budget(profiles_client: TestClient) -> None:
    """Budgets are the profile's, or the limits'; never the caller's to raise."""
    create(profiles_client)

    response = profiles_client.post(
        "/v1/profiles/triage/predict",
        json={"state": "x", "budgets": {"token_budget": 10_000_000}},
    )

    assert response.status_code == 422


# --- guards are not bypassable through a profile -----------------------------------


def test_profile_endpoint_enforces_the_state_limit(profiles_client: TestClient, config: Config) -> None:
    create(profiles_client)

    response = profiles_client.post(
        "/v1/profiles/triage/predict", json={"state": "x" * (config.limits.state_chars + 10)}
    )

    assert response.status_code in (413, 422)


def test_profile_endpoint_enforces_the_body_byte_cap(profiles_client: TestClient, config: Config) -> None:
    create(profiles_client)

    response = profiles_client.post(
        "/v1/profiles/triage/predict",
        content=b"{" + b'"state":"' + b"x" * (config.limits.body_bytes + 1024) + b'"}',
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 413


def _raw(client: TestClient, url: str, payload: bytes) -> int:
    return client.post(
        url, content=payload, headers={"content-type": "application/json"}
    ).status_code


def test_profile_endpoint_rejects_a_lone_surrogate(profiles_client: TestClient) -> None:
    """A correct-for-this-endpoint body still gets caught before the SDK.

    Sent as raw bytes because an unpaired surrogate is not encodable as JSON text.
    The escaped form is the point: ``\\ud800`` survives ``json.loads`` as a lone
    surrogate, so it reaches the body as a ``str`` the tokenizer cannot handle. Left
    alone it dies deep inside torch, which is a 500 with a stack trace instead of a
    400 that names the cause.
    """
    create(profiles_client)
    payload = b'{"state": "\\ud800"}'

    assert _raw(profiles_client, "/v1/profiles/triage/predict", payload) == 400


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param(b"{not json", id="malformed"),
        pytest.param(b'{"state": "ok"} trailing', id="trailing-bytes"),
    ],
)
def test_profile_endpoint_rejects_malformed_json_like_the_fixed_one(
    profiles_client: TestClient, payload: bytes
) -> None:
    """Byte-level body handling is shared, so this one genuinely must match."""
    create(profiles_client)

    fixed = _raw(profiles_client, "/v1/decisions/predict", payload)

    assert _raw(profiles_client, "/v1/profiles/triage/predict", payload) == fixed


def test_deleted_profile_endpoint_stops_answering(profiles_client: TestClient) -> None:
    create(profiles_client)
    assert profiles_client.post("/v1/profiles/triage/predict", json={"state": "x"}).status_code == 200

    assert profiles_client.delete("/v1/profiles/triage").status_code == 200

    assert profiles_client.post("/v1/profiles/triage/predict", json={"state": "x"}).status_code == 404


def test_deleted_profile_leaves_the_document(profiles_client: TestClient) -> None:
    """/openapi.json must not keep advertising a path that no longer routes."""
    create(profiles_client)
    profiles_client.delete("/v1/profiles/triage")

    schema = profiles_client.get("/openapi.json").json()

    assert "/v1/profiles/triage/predict" not in schema["paths"]


def test_caller_language_hint_reaches_the_sdk(profiles_client: TestClient, fake_router: FakeRouter) -> None:
    """A control on the request model must not be silently dropped before the SDK.

    ``lang`` was accepted by the profile schema and then lost in the merge, so the
    checkpoint routed as if the caller had said nothing. The layering now goes through
    ``ProfilePredictRequest.controls()`` so that cannot happen again.
    """
    create(profiles_client)

    response = profiles_client.post(
        "/v1/profiles/triage/predict", json={"state": "x", "lang": "de"}
    )

    assert response.status_code == 200, response.text
    assert fake_router.calls[0]["lang"] == "de"


def test_profile_pins_beat_the_caller(profiles_client: TestClient, fake_router: FakeRouter) -> None:
    """A profile pinned to a checkpoint cannot be re-pointed per request."""
    create(
        profiles_client,
        document={**SIMPLE, "routing": {"model": "english", "lang": "en"}},
    )

    profiles_client.post("/v1/profiles/triage/predict", json={"state": "x", "lang": "de"})

    call = fake_router.calls[0]
    assert call["model"] == "english"
    assert call["lang"] == "en"


def test_caller_budget_is_honoured_and_profile_budget_is_the_default(
    profiles_client: TestClient, fake_router: FakeRouter
) -> None:
    create(profiles_client, document={**SIMPLE, "budgets": {"max_len": 512}})

    profiles_client.post("/v1/profiles/triage/predict", json={"state": "x"})
    profiles_client.post("/v1/profiles/triage/predict", json={"state": "x", "max_len": 256})

    assert fake_router.calls[0]["max_len"] == 512
    assert fake_router.calls[1]["max_len"] == 256


# --- lifecycle --------------------------------------------------------------------


def test_profile_survives_a_restart(tmp_path) -> None:
    cfg = Config(device="cpu", preload=False, threads=1, data_dir=str(tmp_path / "data"))
    with TestClient(create_app(cfg, router_obj=FakeRouter())) as first:
        create(first)

    with TestClient(create_app(cfg, router_obj=FakeRouter())) as second:
        listed = second.get("/v1/profiles").json()["profiles"]
        assert [p["id"] for p in listed] == ["triage"]
        assert second.post("/v1/profiles/triage/predict", json={"state": "x"}).status_code == 200


def test_active_calibration_is_reloaded_into_the_registry_on_restart(tmp_path) -> None:
    """The in-memory registry, not just the artifact, has to survive a restart.

    ``threshold_for`` reads the artifact off disk, so it keeps answering correctly even
    with an empty registry -- which is exactly why this cannot be tested through the
    report. What an empty registry breaks is everything downstream of it: the
    ``on_load`` hook finds no entry, so a checkpoint evicted and reloaded comes back on
    shipped temperatures, and a second profile is allowed to claim the same checkpoint
    because the conflict check sees no owner. The registry is the thing being restored.
    """
    cfg = Config(device="cpu", preload=False, threads=1, data_dir=str(tmp_path / "data"))
    with TestClient(create_app(cfg, router_obj=FakeRouter())) as first:
        create(first)
        store = first.app.state.profiles
        store.write_calibration(
            "triage",
            {
                "questions_fingerprint": store.get("triage").questions_fingerprint,
                "active": True,
                "thresholds": {"choice:2": 0.77},
                "temperature": 1.2,
                "temperature_by_options": {},
                "revision": None,
            },
        )

    with TestClient(create_app(cfg, router_obj=FakeRouter())) as second:
        service = second.app.state.calibration
        assert service.active.owner_of("") == "triage", (
            "the active calibration was not reloaded, so nothing reinstalls the "
            "temperature after an eviction and the per-checkpoint conflict check is blind"
        )
        # The hook is attached to the Router, not merely present on the service.
        assert getattr(second.app.state.engine.router, "_layadev_calibration_hook", None) is not None


def test_a_second_profile_cannot_take_an_owned_checkpoint_after_restart(tmp_path) -> None:
    """The restored registry is what makes activation conflicts visible again."""
    cfg = Config(device="cpu", preload=False, threads=1, data_dir=str(tmp_path / "data"))
    with TestClient(create_app(cfg, router_obj=FakeRouter())) as first:
        create(first, "owner")
        store = first.app.state.profiles
        store.write_calibration(
            "owner",
            {
                "questions_fingerprint": store.get("owner").questions_fingerprint,
                "active": True,
                "thresholds": {"choice:2": 0.77},
            },
        )
        create(first, "other", {**SIMPLE, "name": "other"})

    with TestClient(create_app(cfg, router_obj=FakeRouter())) as second:
        second.app.state.profiles.write_calibration(
            "other",
            {
                "questions_fingerprint": second.app.state.profiles.get("other").questions_fingerprint,
                "active": True,
                "thresholds": {"choice:2": 0.5},
            },
        )
        service = second.app.state.calibration
        # Both artifacts say active; only one can own the (unpinned) checkpoint.
        assert service.active.owner_of("") == "owner"


def test_a_stale_artifact_is_not_restored(tmp_path) -> None:
    """Questions changed while the service was down: the old threshold stays off."""
    cfg = Config(device="cpu", preload=False, threads=1, data_dir=str(tmp_path / "data"))
    with TestClient(create_app(cfg, router_obj=FakeRouter())) as first:
        create(first)
        store = first.app.state.profiles
        store.write_calibration(
            "triage",
            {
                "questions_fingerprint": "not-this-profile",
                "active": True,
                "thresholds": {"choice:2": 0.77},
            },
        )

    with TestClient(create_app(cfg, router_obj=FakeRouter())) as second:
        profile = second.app.state.profiles.get("triage")
        assert second.app.state.calibration.threshold_for(profile) is None


def test_threshold_is_withheld_when_the_checkpoint_revision_moved(profiles_client: TestClient) -> None:
    """A temperature fitted for other weights must not be served after a checkpoint update.

    The number would still look calibrated and would no longer be calibrated, which is
    the one failure this whole feature exists to prevent.
    """
    create(profiles_client, document={**SIMPLE, "routing": {"model": "english"}})
    store = profiles_client.app.state.profiles
    router = profiles_client.app.state.engine.router
    service = profiles_client.app.state.calibration
    profile = store.get("triage")
    service.active.activate("english", "triage", store.calibration_path("triage"))
    store.write_calibration(
        "triage",
        {
            "questions_fingerprint": profile.questions_fingerprint,
            "active": True,
            "revision": "aaaaaaaaaaaa1111",
            "thresholds": {"choice:2": 0.8},
        },
    )

    router.loaded_revisions = {"english": "aaaaaaaaaaaa1111"}
    assert service.threshold_for(profile) == pytest.approx(0.8), "matching revision must serve"

    router.loaded_revisions = {"bbbbbbbbbbbb2222": None, "english": "bbbbbbbbbbbb2222"}
    assert service.threshold_for(profile) is None, (
        "a checkpoint on different weights is not drift-free; the stored temperature "
        "was fitted for revision aaaa..., not bbbb..."
    )
    reason = service.report(profile)["reason"]
    assert "bbbbbbbbbbbb"[:12] in reason and "aaaaaaaaaaaa"[:12] in reason, reason
    assert service.report(profile)["served"] is False


def test_a_checkpoint_that_is_simply_not_loaded_is_not_drift(profiles_client: TestClient) -> None:
    """Absence of evidence is not evidence of change.

    Refusing here would make the threshold flap on and off with LRU eviction, which is
    a worse failure than the drift it is guarding against.
    """
    create(profiles_client, document={**SIMPLE, "routing": {"model": "english"}})
    store = profiles_client.app.state.profiles
    router = profiles_client.app.state.engine.router
    service = profiles_client.app.state.calibration
    profile = store.get("triage")
    service.active.activate("english", "triage", store.calibration_path("triage"))
    store.write_calibration(
        "triage",
        {
            "questions_fingerprint": profile.questions_fingerprint,
            "active": True,
            "revision": "aaaaaaaaaaaa1111",
            "thresholds": {"choice:2": 0.8},
        },
    )

    router.loaded_revisions = {}

    assert service.threshold_for(profile) == pytest.approx(0.8)
    assert service.report(profile)["served"] is True


def test_the_console_reads_calibration_from_its_own_endpoint(profiles_client: TestClient) -> None:
    """Regression: opening a profile showed "not fitted yet" even after a fit.

    The console read ``data.calibration`` from ``GET /v1/profiles/{id}``, which returns
    only ``{"profile": ...}``. That is undefined, and ``renderCal`` renders undefined as
    "No calibration fitted yet" -- so every profile looked uncalibrated on open and on
    every manual refresh, whichever endpoint had actually been used to fit it. A static
    page cannot be executed here, so each ``renderCal`` argument is checked to name a
    calibration instead, and the endpoint shapes are pinned below.
    """
    html = (Path(__file__).resolve().parents[1] / "app" / "console" / "index.html").read_text(
        encoding="utf-8"
    )
    # Every renderCal(data.calibration) must be fed by a call to a /calibration
    # endpoint. Two of them were fed by GET /v1/profiles/{id}, which returns only
    # {"profile": ...}: the value was undefined and the panel claimed no calibration
    # existed for any profile, on open and on every manual refresh.
    args = [
        match.group(1)
        for match in re.finditer(r"(?<!function )renderCal\(([^()]*)\)", html)
        if not html[max(0, match.start() - 9) : match.start()].endswith("function")
    ]
    assert args, "console no longer renders a calibration report"
    for arg in args:
        assert "calibration" in arg, (
            f"renderCal is fed {arg!r}, which carries no calibration"
        )
    assert "data.profile.calibration" not in html

    create(profiles_client, document={**SIMPLE, "routing": {"model": "english"}})
    body = profiles_client.get("/v1/profiles/triage").json()
    assert "calibration" not in body["profile"]
    report = profiles_client.get("/v1/profiles/triage/calibration").json()
    assert "fitted" in report["calibration"] and "reason" in report["calibration"]


def test_a_fitted_profile_reports_the_fields_the_console_renders(profiles_client: TestClient) -> None:
    """Every key ``renderCal`` reads must exist on the report, even before a fit.

    Otherwise the console silently shows "?" for a value it should be displaying.
    """
    create(profiles_client, document={**SIMPLE, "routing": {"model": "english"}})
    cal = profiles_client.get("/v1/profiles/triage/calibration").json()["calibration"]
    for key in ("fitted", "stale", "reason", "scope", "floors", "n", "active", "thresholds"):
        assert key in cal, f"report is missing {key!r}, which the console renders"


def test_profile_cap_is_enforced(profiles_client: TestClient) -> None:
    for pid in ("one", "two", "three"):
        assert profiles_client.post("/v1/profiles", json={"id": pid, **SIMPLE}).status_code == 201

    response = profiles_client.post("/v1/profiles", json={"id": "four", **SIMPLE})

    assert response.status_code == 409


def test_health_reports_profile_count_and_at_capacity(profiles_client: TestClient) -> None:
    create(profiles_client, "one")

    health = profiles_client.get("/health").json()

    assert health["profiles"]["count"] == 1
    assert health["profiles"]["max"] == 3
    assert health["profiles"]["at_capacity"] is False
    assert health["calibration_floors"]["per_bucket"] >= 2000


def test_unsafe_profile_id_is_refused(profiles_client: TestClient) -> None:
    for bad in ("../escape", "a/b", "UPPER", "with space", "dot.json"):
        response = profiles_client.post("/v1/profiles", json={"id": bad, **SIMPLE})
        assert response.status_code in (404, 422), bad


def test_duplicate_profile_id_conflicts(profiles_client: TestClient) -> None:
    create(profiles_client)

    again = profiles_client.post("/v1/profiles", json={"id": "triage", **SIMPLE})

    assert again.status_code == 409


# --- questions are the contract a calibration is fitted against --------------------


def test_editing_questions_clears_the_calibration(profiles_client: TestClient) -> None:
    """A threshold fitted against different questions is not this profile's threshold."""
    create(profiles_client, document=TWO_TYPES)
    store = profiles_client.app.state.profiles
    fingerprint = store.get("triage").questions_fingerprint
    store.write_calibration(
        "triage",
        {"questions_fingerprint": fingerprint, "active": True, "thresholds": {"choice:2": 0.9}},
    )
    assert store.read_calibration("triage") is not None

    updated = profiles_client.patch(
        "/v1/profiles/triage",
        json={"questions": {"sev": question("sev", "choice"), "extra": question("extra")}},
    )

    assert updated.status_code == 200
    calibration = profiles_client.get("/v1/profiles/triage").json()["calibration"]
    assert calibration["fitted"] is False
    # And the artifact is gone from disk, not merely unreported: a stale threshold that
    # came back to life on the next activation would be worse than no threshold.
    assert store.read_calibration("triage") is None


def test_calibration_is_marked_stale_when_questions_change(profiles_client: TestClient) -> None:
    create(profiles_client)
    store = profiles_client.app.state.profiles
    store.write_calibration(
        "triage",
        {
            "questions_fingerprint": store.get("triage").questions_fingerprint,
            "active": True,
            "thresholds": {},
        },
    )

    changed = profiles_client.patch(
        "/v1/profiles/triage",
        json={"questions": {"q1": question("q1", "choice", instructions="Something else?")}},
    )

    assert changed.status_code == 200
    report = profiles_client.get("/v1/profiles/triage/calibration").json()["calibration"]
    assert report["fitted"] is False, "the artifact should be dropped, not left stale"


# --- calibration gating -----------------------------------------------------------

def test_fit_requires_examples(profiles_client: TestClient) -> None:
    create(profiles_client)

    response = profiles_client.post("/v1/profiles/triage/calibration/fit", json={})

    assert response.status_code == 422


def test_calibration_report_states_the_sdk_floors(profiles_client: TestClient) -> None:
    """The operator is told the real numbers, quoted from the SDK, not from a constant."""
    create(profiles_client)

    report = profiles_client.get("/v1/profiles/triage/calibration").json()["calibration"]

    assert report["fitted"] is False
    assert report["floors"]["per_bucket"] >= 2000
    assert report["floors"]["type_level"] < report["floors"]["per_bucket"]


def test_small_dataset_is_labelled_type_level_not_per_bucket(profiles_client: TestClient) -> None:
    """Below the SDK's 2000 the fit is real but narrower, and must say so."""
    from app import calibration as cb

    create(profiles_client, document=TWO_TYPES)
    store = profiles_client.app.state.profiles
    profile = store.get("triage")
    rows = labelled(120)
    store.replace_examples("triage", rows)

    artifact = cb.fit(profile, rows, None, records_from=synthetic_records(240))

    assert artifact["n"] == 240
    assert artifact["scope"] == cb.SCOPE_TYPE_LEVEL
    assert artifact["caveat"], "a type-level fit must carry the caveat that says so"
    assert cb.SCOPE_BUCKETED not in artifact["scope"]
    # ECE is not scored at this size, and NaN must not be passed through as a number.
    report = cb.summarize(profile, artifact)
    assert report["ece"]["available"] is False
    assert report["ece"]["before"] is None
    assert report["ece"]["reason"]


def test_large_dataset_is_reported_as_bucketed(profiles_client: TestClient) -> None:
    """One bucket past 2000 is enough to earn the per-bucket claim."""
    from app import calibration as cb

    create(profiles_client, document=TWO_TYPES)
    store = profiles_client.app.state.profiles
    profile = store.get("triage")
    rows = labelled(1200)
    store.replace_examples("triage", rows)

    artifact = cb.fit(profile, rows, None, records_from=synthetic_records(2600, choice_only=True))

    assert artifact["n"] == 2600
    assert artifact["scope"] == cb.SCOPE_BUCKETED
    assert artifact["caveat"] is None
    assert artifact["temperature_by_options"], "per-bucket temperatures were expected"
    assert artifact["n_buckets_per_bucket_temperature"] >= 1
    report = cb.summarize(profile, artifact)
    assert report["ece"]["available"] is True
    assert report["ece"]["before"] is not None


def test_scope_is_read_from_the_fit_not_guessed_from_the_total(profiles_client: TestClient) -> None:
    """A total over the floor spread thin across buckets must not claim a per-bucket fit.

    2600 records is past the 2000 floor, so a total-count check would call this
    bucketed. But split over two buckets of 1300 the fitter qualifies for neither, and
    reporting it as per-bucket is exactly the overclaim this reporting exists to stop.
    """
    from app import calibration as cb

    create(profiles_client, document=TWO_TYPES)
    store = profiles_client.app.state.profiles
    profile = store.get("triage")
    rows = labelled(1300)
    store.replace_examples("triage", rows)

    artifact = cb.fit(profile, rows, None, records_from=synthetic_records(2600))

    assert artifact["n"] == 2600
    assert artifact["temperature_by_options"] == {}
    assert artifact["scope"] == cb.SCOPE_TYPE_LEVEL
    assert artifact["caveat"], "a fit that got no per-bucket temperatures must say so"


def test_too_few_records_is_refused_with_the_floor(profiles_client: TestClient) -> None:
    from app import calibration as cb

    create(profiles_client, document=TWO_TYPES)
    store = profiles_client.app.state.profiles
    profile = store.get("triage")
    rows = labelled(4)
    store.replace_examples("triage", rows)

    with pytest.raises(cb.CalibrationError) as caught:
        cb.fit(profile, rows, None, records_from=synthetic_records(8))

    assert "at least" in str(caught.value.detail)


def test_target_expansion_maps_labels_to_one_hot() -> None:
    from app.calibration import expand_target
    from app.schemas import ChoiceQuestion, NoulQuestion

    choice = ChoiceQuestion.model_validate(
        {"type": "choice", "instructions": "How bad?", "criteria": {"low": "Low", "high": "High"}}
    )
    # Both the label and its index are accepted, because both are things a person
    # writes in a labelling tool.
    assert expand_target("sev", choice, "high") == [0.0, 1.0]
    assert expand_target("sev", choice, 1) == [0.0, 1.0]
    assert expand_target("sev", choice, "low") == [1.0, 0.0]

    noul = NoulQuestion.model_validate({"type": "noul", "instructions": "Is this acceptable?"})
    assert expand_target("ok", noul, True) == [1.0, 0.0]
    assert expand_target("ok", noul, False) == [0.0, 1.0]


def test_target_expansion_rejects_a_label_that_is_not_offered() -> None:
    from app.calibration import CalibrationError, expand_target
    from app.schemas import ChoiceQuestion

    choice = ChoiceQuestion.model_validate(
        {"type": "choice", "instructions": "How bad?", "criteria": {"low": "Low", "high": "High"}}
    )

    with pytest.raises(CalibrationError) as caught:
        expand_target("sev", choice, "catastrophic")

    assert "catastrophic" in str(caught.value.detail)


def test_target_expansion_rejects_an_out_of_range_index() -> None:
    """An index past the end must fail loudly rather than silently mean the last option."""
    from app.calibration import CalibrationError, expand_target
    from app.schemas import ChoiceQuestion

    choice = ChoiceQuestion.model_validate(
        {"type": "choice", "instructions": "How bad?", "criteria": {"low": "Low", "high": "High"}}
    )

    with pytest.raises(CalibrationError):
        expand_target("sev", choice, 7)


# --- examples ---------------------------------------------------------------------


def test_examples_round_trip(profiles_client: TestClient) -> None:
    create(profiles_client, document=TWO_TYPES)
    rows = labelled(3)

    saved = profiles_client.put("/v1/profiles/triage/examples", json={"examples": rows})

    assert saved.status_code == 200
    assert saved.json()["count"] == 3
    listed = profiles_client.get("/v1/profiles/triage/examples").json()
    assert len(listed["examples"]) == 3
    assert listed["examples"][0]["expected"]["sev"] in ("low", "high")


def test_examples_replace_rather_than_append(profiles_client: TestClient) -> None:
    create(profiles_client, document=TWO_TYPES)
    profiles_client.put("/v1/profiles/triage/examples", json={"examples": labelled(5)})

    profiles_client.put("/v1/profiles/triage/examples", json={"examples": labelled(2)})

    listed = profiles_client.get("/v1/profiles/triage/examples").json()
    assert listed["count"] == 2


def test_examples_are_validated_on_the_way_in(profiles_client: TestClient) -> None:
    """A label the profile does not offer is rejected at upload, not at fit time."""
    create(profiles_client, document=TWO_TYPES)

    response = profiles_client.put(
        "/v1/profiles/triage/examples",
        json={"examples": [{"state": "x", "expected": {"sev": "catastrophic", "ok": True}}]},
    )

    assert response.status_code == 422
    assert "catastrophic" in response.text


# --- console ----------------------------------------------------------------------


def test_console_is_served(profiles_client: TestClient) -> None:
    response = profiles_client.get("/console/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "/v1/profiles" in response.text


def test_console_uses_no_external_resources(profiles_client: TestClient) -> None:
    """The deployment is air-gapped: no CDN, no remote font, no import from a URL."""
    body_text = profiles_client.get("/console/").text

    for marker in ("http://", "https://", "//cdn", "integrity="):
        assert marker not in body_text, marker
