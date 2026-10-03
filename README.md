# Laya Decision Service

A reusable inference endpoint over the [Laya](https://huggingface.co/convaiinnovations/laya)
decision model. Give it a **state** (a text, email, ticket or JSON document) and a set of
**typed questions**, and it returns structured decisions with probabilities in a single
forward pass. It never generates text, so there is nothing to parse and nothing to
hallucinate.

Built directly on `laya.Router` rather than on upstream's `laya-serve`, because
`laya-serve` registers its routes on literal paths with no override and declares no
response models — its `/openapi.json` carries no schema a consumer can generate a client
from.

---

## Quick start

```powershell
python -m scripts.warmup                      # cache the two routing checkpoints
python -m app                                 # serve on $LAYA_HOST:$LAYA_PORT
```

Then:

```powershell
curl.exe -X POST http://127.0.0.1:8000/v1/decisions/predict `
  -H "content-type: application/json" `
  -d '{
        "state": "The refund was issued on Tuesday and is still not visible.",
        "questions": {
          "topic": {
            "type": "choice",
            "instructions": "What is the customer asking about?",
            "criteria": {"refund": "A refund", "shipping": "A delivery"}
          },
          "urgent": {"type": "noul", "instructions": "Is the customer angry?"}
        }
      }'
```

Interactive docs at `/docs`, the schema at `/openapi.json`, liveness at `/health`, and
Prometheus metrics at `/metrics`.

---

## Decision types

| Type | Question shape | Answer |
| --- | --- | --- |
| `choice` | `criteria`: label → description, or a bare list of labels | The selected label plus a probability per option |
| `score` | `criteria`: ordered level list, least to most severe | The expected level index plus the distribution |
| `noul` | `criteria`/`labels` optional, keyed `true`/`false` | `P(true)`, in [0,1] |

`instructions` is required and must be non-empty for every type — that is the text the
model answers. Every question is answered against the same state in one forward pass, so
adding a question never changes another question's answer.

Note the field name: both a choice's options and a score's levels are spelled `criteria`,
matching `laya`'s own contract. There is no `id` field on a question; the id is the key it
sits under in the `questions` mapping.

### Two confidence numbers, and only one you should gate on

`answer_confidence` is the probability mass on the answer being reported. It is the number
`min_confidence` compares against, and the one any temperature fit is computed on.

`confidence` means something different per type: normalised entropy `1 - H(p)/log(k)` on
`choice` and `score`, and `max(p_yes, p_no)` on `noul`. It depends on how many options the
question had, so a two-option distribution comes back around 0.90 on a `noul` and 0.53 on
an equivalent `choice`. **Do not compare it against a threshold.**

### Calibration: read this before setting `min_confidence`

The shipped checkpoints are over-confident as shipped. Measured expected calibration error
falls from 0.466 to 0.081 for English and 0.314 to 0.106 for multilingual once temperatures
are fitted — and the direction is task-dependent: one documented routing task was
*under*-confident instead. The English checkpoint also ships an invalid temperature for
`choice:11+` (0.10, outside the usable [0.5, 5] range), which `laya` clamps to 0.5 and
flags at load; confidence from those answers is uncalibrated.

There is no default `min_confidence` in this service, deliberately. Fit and validate
temperatures on held-out examples from your own workload first. Until then, treat
confidence as a **ranking signal for human-review triage**, not as an automated gate.

---

## Routing

Requests are routed by script and language detected on the state; the choice is reported
in every response's `routing` block. This matters more than it sounds: the English
checkpoint is not merely weaker outside English, it collapses while staying confident —
measured at **0.000 accuracy on Khmer while reporting 0.952 confidence**. Routing therefore
happens *before* the forward pass, not as a confidence check afterwards.

Override with `model` (an alias or a public Hugging Face id), `lang` (a hint such as `de`
or `pt-BR`, which skips detection), or `task` (forces a checkpoint by workflow).

Keep `LAYA_MAX_LOADED` at 2 or more. Setting it to 1 measured 20–23 second rebuilds under
a mixed-language workload, because each request evicts the checkpoint the next one needs.

---

## Profiles

A profile is a named, stored set of questions plus the routing pins and budgets to use
with them. Creating one mints its own endpoint, so a caller cannot drift from the
question set it was meant to ask:

```
POST /v1/profiles                      -> 201, and POST /v1/profiles/triage/predict now exists
GET  /v1/profiles/triage               -> the stored document
PATCH /v1/profiles/triage              -> replace questions/routing/budgets
DELETE /v1/profiles/triage             -> the endpoint stops existing
PUT  /v1/profiles/triage/examples      -> replace the labelled set (JSONL rows)
POST /v1/profiles/triage/calibration/fit        -> fit temperatures and thresholds
POST /v1/profiles/triage/calibration/activate   -> adopt the fitted threshold
POST /v1/profiles/triage/calibration/deactivate -> drop it
```

`GET /console` serves a static page that drives all of the above. It has no build step
and no external resources, so it works air-gapped.

**The minted endpoint has the same guards as `/v1/decisions/predict`, by
construction.** Its handler owns no logic: it reads the body under the same byte cap,
parses it the same way, and calls the same `app.decision.decide` core with a builder
that merges the profile in. Body cap, malformed JSON, unpaired surrogates, token
budgets, question limits, state size and the admission gate all apply.

**The profile owns the questions.** They cannot be sent per request — an attempt is
rejected by the schema — and routing pins and budgets come from the profile. The one
deliberate exception is `min_confidence`: a calibrated threshold is the profile's
default, and a caller may still override it for a single request.

**Editing the questions discards the calibration.** A threshold fitted against one
question set is not evidence about another, so the artifact is deleted rather than left
in place to be reactivated. The fingerprint that ties a calibration to its questions is
in the profile document.

**A checkpoint update invalidates the calibration too.** A temperature is fitted for
particular weights, so if the checkpoint is replaced underneath it — a new Hub revision,
a re-pulled image — the stored temperature is not the right one and the threshold is
withheld until you re-fit. A checkpoint that is merely *not loaded* is not treated as
drift: absence of evidence is not evidence of change, and refusing there would make
every threshold flap on and off with LRU eviction.

### Calibrating a profile, and what the numbers mean

Upload labelled rows — one JSON object per line, `{"state": ..., "expected": {...}}`.
`expected` takes whatever a person would write: an option label, an index, a level name,
or a bool for `noul`. Labels are validated on upload, so a bad one is reported with its
row rather than surfacing later at fit time.

`fit` runs the SDK's `fit_temperature_map` and `fit_abstention_thresholds`. **Read the
floors before trusting a fit.** They are three different numbers and are reported under
`calibration.floors` in `GET /v1/profiles/{id}/calibration`, quoted from the SDK rather
than restated here:

| Floor | Records | Below it |
| --- | --- | --- |
| `type_level` | 10 | Nothing is fitted; the fit is refused |
| `abstention_per_bucket` | 100 | No abstention threshold for that bucket |
| `per_bucket` | 2000 | No per-bucket temperature, and no held-out ECE |

The `per_bucket` floor is the one that matters for interpretation, and it is **per
bucket**: a bucket being "question type × option count", e.g. `choice:2`. 2600 records
split over two buckets of 1300 clear the floor in total and qualify for neither. When
that happens the fit is still real — one temperature per question type, shared across
option counts — and the report says so via `scope: "type-level"` and a `caveat` string
rather than presenting it as a per-bucket calibration.

For the same reason `ece.available` is `false` on small fits rather than `NaN`: the SDK
excludes any bucket that would fall below the floor once 20% is held out, and names it
in `buckets_excluded_from_eval`. When `available` is true, `before → after` is the
measured improvement from the fitted temperature on records the fitter did not train on.

**Temperatures live on a loaded checkpoint, so activation is exclusive per checkpoint.**
Activating a profile for a checkpoint that another active profile already owns is a
409. This is not a policy choice — two profiles cannot hold different temperatures on
one set of weights. Temperatures are reinstalled automatically if the Router evicts and
reloads the checkpoint.

`LAYA_MAX_LOADED` should stay at 2 or more for the same reason: a calibration never
holds a reference to a Router-owned agent, so eviction works normally, but pinning one
would defeat the LRU and bring back the 20–23s rebuilds.

---

## Configuration

All via environment variables; see `app/config.py` for the full set.

| Variable | Default | Notes |
| --- | --- | --- |
| `LAYA_DEVICE` | `cpu` | `cuda` where available |
| `LAYA_MAX_LOADED` | `2` | Resident checkpoints. Below 2, mixed-language traffic thrashes |
| `LAYA_THREADS` | physical cores | Torch intra-op cap |
| `LAYA_MAX_CONCURRENT` | `16` | Admission limit; excess gets a 503 |
| `LAYA_PRELOAD` | `0` | Build checkpoints at startup instead of on first request |
| `LAYA_MAX_BODY_BYTES` | `1048576` | Request body cap |
| `LAYA_MAX_STATE_CHARS` | `50000` | Characters of state |
| `LAYA_MAX_QUESTIONS` | `32` | Questions per request |
| `LAYA_MAX_CHOICE_OPTIONS` | `32` | Per choice question |
| `LAYA_MAX_SCORE_LEVELS` | `32` | Per score question |
| `LAYA_MAX_TOTAL_OPTIONS` | `512` | Across one request |
| `LAYA_MAX_BATCH_STATES` | `16` | States per batch call |
| `LAYA_TOKEN_BUDGET` | `8192` | Ceiling on a caller-supplied `max_len` |
| `LAYA_DATA_DIR` | `data` | Where profiles, examples and calibration artifacts are stored |
| `LAYA_MAX_PROFILES` | `32` | Profiles per deployment |

The profile cap is a hard operational limit, not a suggestion: each profile is a minted
endpoint with its own metric label, so without a cap `route` would be an unbounded
cardinality source in Prometheus. `GET /health` reports `profiles.count` and
`at_capacity`.

`USE_TF=0` is set by the app before importing `laya`, and is required. `transformers`
probes for TensorFlow at import time, and when TensorFlow is installed its abseil runtime
can deadlock model construction — the process hangs at load with no error.

---

## Errors

| Status | When |
| --- | --- |
| 400 | Body is not a JSON object, or holds an unpaired surrogate escape |
| 413 | A state or question set is over a size limit; `detail` names which and by how much |
| 422 | A required field is missing or null, the question set is invalid, or a request control has a bad value |
| 500 | Inference failed. The cause is in the server log; the client learns nothing |
| 503 | Admission limit reached. Retry after the interval in `Retry-After` |

The 413/422 split is deliberate: a state that is too *long* is too big (413), while a
`max_len` above the server ceiling is a well-formed request carrying a value the service
will not honour (422).

A 500 returns a generic message and logs the traceback. A caller cannot act on an internal
error, and echoing it leaks paths and configuration to whoever found the failure.

---

## Limits and why they exist

The option caps are not arbitrary. All options for one request are collated into a single
tensor and the state is tokenized once per question, so an unbounded option array is an
amplification primitive: a small request produces a large forward pass. A count cap is the
cheap proxy available before tokenization; the token budget itself is enforced later by
`laya`, which refuses a question whose option texts do not fit.

`/health` is deliberately **not** behind the inference gate. The forward pass is synchronous
torch holding the single worker; if liveness queued behind it, a load balancer would restart
a healthy process precisely when it is busiest.

---

## Concurrency model

One inference worker (`ThreadPoolExecutor(max_workers=1)`), because torch model objects are
not documented as thread-safe and a second concurrent forward pass against the same weights
is undefined behaviour. Throughput comes from replicas, not from threads.

Beyond that, admission is non-blocking: a request that arrives when `LAYA_MAX_CONCURRENT`
are already in flight gets a 503 immediately rather than queueing behind a 700 ms forward
pass. Latency is reported as three separate signals, because under one worker they answer
different questions and only one is about the model:

| Metric | Measures |
| --- | --- |
| `laya_queue_wait_seconds` | Admission to gate. Grows with concurrency |
| `laya_inference_duration_seconds` | The forward pass alone |
| `laya_request_duration_seconds` | Admission to response, the client-visible total |

The default Prometheus buckets (5 ms–10 s) are useless here: everything worth
distinguishing falls between 100 ms and 2 s, and a cold checkpoint build falls off the end.
Custom buckets start at 5 ms and run to 60 s.

---

## Token budgets

Only the first 512 tokens are read by the English checkpoint and 1,024 by multilingual, per
question. Put the decisive text first. `usage` is the only place a cut is visible:
`truncated`, `state_tokens_dropped` and `truncated_questions` report what did not reach the
model.

`max_len` can widen a window up to `LAYA_TOKEN_BUDGET`. The multilingual encoder accepts it,
but measurably degrades past roughly 4,000 tokens of preceding text.

---

## Metrics

| Metric | Type | Labels |
| --- | --- | --- |
| `laya_request_duration_seconds` | histogram | `route`, `outcome` |
| `laya_inference_duration_seconds` | histogram | `checkpoint`, `questions` |
| `laya_queue_wait_seconds` | histogram | `route` |
| `laya_requests_total` | counter | `route`, `status` |
| `laya_rejected_total` | counter | `route`, `reason` |
| `laya_inference_in_flight` | gauge | — |
| `laya_checkpoint_evictions_total` | counter | `model` |

P50 and P95 come from `histogram_quantile` over these buckets, which interpolates within a
bucket rather than computing an exact order statistic.

---

## Batch

`POST /v1/decisions/predict/batch` applies one question set to many states, sharing forward
passes across states of the same checkpoint. It returns **a flat array of decision sets, in
the order the states were sent** — not an envelope.

That shape is not a stylistic choice. `Router.predict_batch` raises on failure rather than
returning a partial result, so a per-item `errors` list could only ever be empty, and an
envelope would advertise a partial-success mode that cannot occur. A client coded against
one would be handling a case the server never produces.

Per-state `overrides` (keyed by index into `states`) set `model`, `task`, `lang`,
`max_len` or `head_max_len` for individual states. `batch_size`, `sort_by_length` and
`min_confidence` describe the forward pass rather than any one state, so they are
call-level.

---

## OpenAPI

The document is served as **OpenAPI 3.0.3**, not FastAPI's default 3.1. A 3.1 document
served from a 3.0 URL breaks the generators and validators the spec exists for, which treat
the versions as different schemas rather than nested ones.

The gap that actually bites is `const`: Pydantic v2 emits it for single-valued `Literal`s,
and it is 3.1-only. A 3.0 consumer sees an unconstrained string and generates a client that
accepts anything, so it is rewritten to an equivalent one-value `enum`. The document is
asserted portable at startup — better a refused start than a document that quietly
misdescribes the API.

`/metrics` is excluded from the schema: it is a machine-facing scrape target, not an API
surface. `/health` is documented, because a gateway has to know it exists.

---

## Testing

```powershell
python -m pytest            # 62 tests, no weights needed, ~4s
python -m pytest -m live    # 4 tests against real checkpoints, ~35s
```

The default suite runs against a stub Router written to the SDK's real signature and return
shape, so the entire HTTP surface — guards, error precedence, OpenAPI, concurrency — is
verified without downloading weights. This is deliberate: an earlier version of the batch
schema assumed an `{"results", "errors"}` envelope that the SDK does not produce, and a
stub written to the real contract fails that mistake immediately instead of passing it.

Live tests skip rather than fail when a checkpoint is not cached.

---

## Operations runbook

### Air-gapped / offline deployment

`laya` bundles all three checkpoints as subfolders of `convaiinnovations/laya`, and
resolves `typed-decisions` to a **subfolder**, not to the standalone
`convaiinnovations/laya-typed-decisions` repo. Warming the standalone repos produces a
cache that looks complete and is not: startup under `HF_HUB_OFFLINE=1` then fails with
`IncompleteSnapshotError` for a file the warm-up never looked at. `scripts/warmup.py`
reads the table from `laya.router.DEFAULT_MODELS` for exactly this reason, so it cannot
drift from what the Router loads.

```powershell
# On a networked machine, once per release:
python -m scripts.warmup --models all --verify-offline

# Ship the whole HF_HOME directory to the target.

# On the air-gapped host:
$env:HF_HUB_OFFLINE = "1"
$env:USE_TF = "0"
$env:LAYA_PRELOAD = "1"
python -m app
```

`--verify-offline` reloads every checkpoint through a fresh Router with `HF_HUB_OFFLINE=1`
set. It reloads rather than reuses the already-built agents on purpose: built objects
perform no file lookup, so reusing them would pass against an incomplete cache.

Confirm on the target: `GET /health` shows a populated `loaded` array and a `revisions`
object.

Measured on a 4-core laptop CPU with all three checkpoints cached: 16 s to a
preload-ready server, and a warm cache-to-build of 4.5 s for English, 3.8 s for
multilingual. The first `typed-decisions` build took 77 s, most of it the 421 MB download,
which is why `--models all` is opt-in rather than the default.

The three checkpoints come to **2.2 GB** in the cache. Ship the whole `HF_HOME` directory,
not the model directories alone — the `blobs`, `refs` and `snapshots` layout is part of the
cache.

Without `HF_HUB_OFFLINE=1`, `huggingface_hub` issues metadata requests to
huggingface.co at startup even with a fully warm cache. On a slow or firewalled network
that adds tens of seconds before `/health` answers, and it fails outright with no egress.

### Startup and shutdown

`LAYA_PRELOAD=1` builds checkpoints before the server accepts traffic, trading slower
startup for no cold first request. Expect a few seconds per checkpoint on CPU. Leave it off
if startup time matters more than first-request latency.

Keep `LAYA_MAX_LOADED` at 2 or more. Evictions are the expensive failure mode here: a
rebuild costs seconds on CPU against milliseconds resident, and `laya` fires a
`gc.collect()` and `empty_cache()` on each one. Watch `laya_checkpoint_evictions_total`; a
sustained non-zero rate under steady traffic means the cap is too low for the language mix.

### State cleanliness

There is no redaction hook in this service, by design. Whatever you send as `state` is
tokenized and processed. Strip credentials, tokens, PII and bulky binary or base64 blobs
before calling. Note also that `laya` normalises literal boolean criterion keys to strings,
and that a `noul` question whose criteria are not keyed `true`/`false` is refused here with
a 422 rather than silently answered against the default pair.

### No Dockerfile

Deployment is expected to be handled by your own image build or platform; this repo ships a
runbook and an executable warm-up instead. Nothing here depends on a container.

### Authentication

There is no bearer check in this service. It is intended to sit behind an authenticating
gateway, which is why `/health` may report checkpoint names, revision SHAs, device state and
token-thread configuration. **If you expose this service directly, `/health` becomes an
information disclosure** — put the gateway in front of it first.

### Gateway routes: the console is admin, not inference

This matters more now that the service can define its own endpoints. The gateway must
route on **exact path**, not prefix:

| Path | Who |
| --- | --- |
| `/v1/decisions/predict`, `/v1/decisions/batch`, `/v1/decisions/options` | inference callers |
| `/v1/profiles`, `/v1/profiles/*` | administrators only |
| `/console`, `/console/*` | administrators only |
| `/health`, `/metrics` | your existing policy |

`/v1/profiles/*` includes the minted `…/predict` endpoints. A caller-scoped rule that
forwards `/v1/` to the inference audience would also expose profile creation, calibration
fits and activation to that audience — and activation is what changes the confidence
gate for every subsequent request.

Two concrete reasons this is not a theoretical concern:

- `POST /v1/profiles` writes to disk and mints a route. Left open, any caller can mint up
  to `LAYA_MAX_PROFILES` endpoints and fill the metric-label cardinality.
- `POST /v1/profiles/{id}/calibration/activate` installs temperatures onto a resident
  checkpoint. Left open, any caller can change the confidence gate for other tenants'
  requests.

Both are refused only by the gateway. The service applies the profile cap and the
per-checkpoint activation conflict (409), but it does not and will not authenticate.

**Do not put profile examples in a multi-tenant gateway cache keyed only by path.**
Examples and calibration artifacts are per-profile administrative state, and `/health`
reports the profile count.
