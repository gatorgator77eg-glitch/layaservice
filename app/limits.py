"""Request guardrails, enforced before any tokenization.

Every check here is a bound on work the server would otherwise do on a caller's
behalf, so an oversized request costs the bytes it took to read and nothing more.
That ordering is deliberate: refusing a 2 MiB body is cheaper than serialising it,
and serialising it is cheaper than tokenizing it.

Two behaviours are inherited from upstream and are easy to get wrong when
rewriting them:

* the state length is measured on ``json.dumps(state, ensure_ascii=False)``,
  which is what actually gets tokenized -- not ``str(state)``. ``str`` measures a
  different length in both directions, so a gate built on it once admitted a state
  that serialised to twice the documented limit.
* the lone-surrogate walk runs *after* the size checks, so an oversized body is
  refused first and the character and question limits bound what it can reach.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from fastapi import HTTPException


# --------------------------------------------------------------------------------------
# Body reading
# --------------------------------------------------------------------------------------


async def read_body_capped(request: Any, max_bytes: int) -> bytes:
    """Read the request body, refusing to buffer more than ``max_bytes``.

    ``Content-Length`` cannot be the only gate. It is a value the client chooses,
    it is absent entirely under ``Transfer-Encoding: chunked``, and HTTP/2 and
    HTTP/3 have no such header at all. The body is streamed and abandoned as soon
    as it passes the cap, so the bound holds for every framing.
    """
    declared = request.headers.get("content-length")
    if declared:
        try:
            if int(declared) > max_bytes:
                raise HTTPException(
                    status_code=413,
                    detail=f"request body too large ({int(declared)} > {max_bytes} bytes)",
                )
        except ValueError:
            pass  # a malformed header is not this layer's problem to diagnose

    total = 0
    chunks: List[bytes] = []
    async for chunk in request.stream():
        if not chunk:
            continue
        total += len(chunk)
        if total > max_bytes:
            # Stop reading rather than draining the rest: the peer is already over
            # the limit and nothing further can make the request acceptable.
            raise HTTPException(
                status_code=413,
                detail=f"request body too large (exceeds {max_bytes} bytes)",
            )
        chunks.append(chunk)
    return b"".join(chunks)


def parse_json_object(raw: bytes) -> Dict[str, Any]:
    """Decode the body and require a JSON object."""
    try:
        parsed = json.loads(raw)
    except (ValueError, RecursionError):
        raise HTTPException(status_code=400, detail="request body must be valid JSON") from None
    if not isinstance(parsed, dict):
        raise HTTPException(status_code=400, detail="request body must be a JSON object")
    return parsed


# --------------------------------------------------------------------------------------
# State measurement
# --------------------------------------------------------------------------------------

# How many values the cheap lower-bound walk inspects before giving up and letting
# the exact measurement decide. This bounds values *inspected*, not total work:
# pushing a container's elements onto the stack is not charged against it, so a
# 2 MiB body of many small values walks in well under a millisecond.
_PROBE_VALUES = 64


def _state_length_lower_bound_over(state: Any, cap: int) -> int:
    """A measured lower bound on the serialised length, or 0 if not proven.

    Non-zero only when exceeding ``cap`` is certain, and 0 for "not proven" --
    never the reverse, so a 0 defers to the exact measurement and no verdict
    changes.

    The raw length of a string value is a lower bound on its JSON length: escaping
    maps each character to one or more, and keys, separators, brackets and quotes
    only add. That makes this an under-estimate, which is the safe direction --
    under-estimating can only fail to refuse, and then the exact encoder decides.

    Worth having because the exact ``json.dumps`` runs before the gate refuses, so
    its cost is bounded by the body cap rather than by the character cap. Measured
    on a near-cap body: 5.69ms to serialise a request that is then rejected,
    against 0.0003ms to notice with a single ``len()``.
    """
    total = 0
    budget = _PROBE_VALUES
    stack = [state]
    while stack and budget > 0:
        budget -= 1
        item = stack.pop()
        if isinstance(item, str):
            total += len(item)
            if total > cap:
                return total
        elif type(item) is dict:
            # `.values()` and `extend` stay at C level; keys are ignored, which is
            # what keeps this a lower bound.
            stack.extend(item.values())
        elif type(item) is list or type(item) is tuple:
            stack.extend(item)
        else:
            # EXACT types only. A dict subclass may override `values()` while
            # `json.dumps` reads the real items, which would let this "lower
            # bound" exceed the true length -- measured: a 13-character state
            # refused as "60000 > 50000". Anything else, including a subclass, a
            # set or a cycle, goes to the encoder, which decides or raises.
            return 0
    return 0


def state_length(state: Any, cap: int) -> int:
    """Length of the text that will be tokenized.

    ``cap`` bounds the cheap lower-bound walk so a large state can be refused
    without paying to serialise it first.
    """
    if isinstance(state, str):
        # `serialize_state` returns a string state unchanged.
        return len(state)
    proven = _state_length_lower_bound_over(state, cap)
    if proven:
        return proven
    try:
        return len(json.dumps(state, ensure_ascii=False))
    except (TypeError, ValueError, RecursionError):
        # Not a size problem. An earlier revision reported these as 413 with a
        # count nothing had measured; it is a malformed request, so it is a 400.
        raise HTTPException(status_code=400, detail="'state' must be JSON-serializable") from None


def check_state(state: Any, max_chars: int) -> None:
    if state is None:
        # `serialize_state(None)` is `json.dumps(None)` == the four characters
        # "null", so a missing state was answered as a decision about the literal
        # text "null" -- HTTP 200, at ~0.94 confidence, byte-identical to sending
        # the string "null". Nothing downstream can tell those apart.
        raise HTTPException(status_code=400, detail="'state' is required and must not be null")
    length = state_length(state, max_chars)
    if length > max_chars:
        raise HTTPException(
            status_code=413,
            detail=f"state too large ({length} > {max_chars} chars)",
        )


# --------------------------------------------------------------------------------------
# Question guardrails
# --------------------------------------------------------------------------------------


def check_option_weights(
    questions: Dict[str, Any],
    *,
    max_choice_options: int,
    max_score_levels: int,
    max_total_options: int,
) -> None:
    """Bound the option weight a question set allocates.

    A ``choice`` question's options share one fixed token budget
    (``head_max_len``, 192 by default). That budget -- not the option count -- is
    what actually limits quality: the 77-label banking77 run scored 0.425 on
    *both* checkpoints, which is the signature of labels squeezed to roughly four
    tokens each rather than of a capability gap. A count cap is the cheap proxy
    available before tokenization, so it is enforced here; the token budget itself
    is enforced later by the SDK, which refuses a question whose option texts do
    not fit.

    Refusing before tokenization matters: without the cap, an unbounded option
    array is an amplification primitive, because the state is tokenized once per
    question and collated into a single tensor.
    """
    total = 0
    for qid, question in questions.items():
        kind = question.type
        if kind == "choice":
            count = question.option_count
            total += count
            if count > max_choice_options:
                raise HTTPException(
                    status_code=413,
                    detail=(
                        f"question {qid!r} has {count} options, over the limit of "
                        f"{max_choice_options}. Options share a fixed token budget, so a "
                        f"large label space stops being distinguishable; split it into a "
                        f"coarse question and a fine one."
                    ),
                )
        elif kind == "score":
            count = question.option_count
            total += count
            if count > max_score_levels:
                raise HTTPException(
                    status_code=413,
                    detail=(
                        f"score question {qid!r} has {count} levels, over the limit of "
                        f"{max_score_levels}"
                    ),
                )
    if total > max_total_options:
        raise HTTPException(
            status_code=413,
            detail=f"too many answer options across questions ({total} > {max_total_options})",
        )


def check_question_count(questions: Dict[str, Any], max_questions: int) -> None:
    if not questions:
        raise HTTPException(status_code=400, detail="'questions' must contain at least one question")
    if len(questions) > max_questions:
        raise HTTPException(
            status_code=413,
            detail=f"too many questions ({len(questions)} > {max_questions})",
        )


# --------------------------------------------------------------------------------------
# Surrogate walk
# --------------------------------------------------------------------------------------

_LONE_SURROGATE = re.compile("[\ud800-\udfff]")


def has_lone_surrogate(value: Any) -> bool:
    """True if any string in the parsed body holds an unpaired surrogate.

    ``json.loads`` accepts a ``\\udXXX`` escape and builds a ``str`` holding that
    code point, which cannot be encoded to UTF-8, so the tokenizer raises
    ``TypeError`` from inside the forward pass -- the caller's own mistake arriving
    as a 500 plus a traceback per request. A *pair* of escapes is combined by the
    decoder into one ordinary astral character, so an emoji is unaffected.

    Walks with an explicit stack: ``json.loads`` accepts nesting far deeper than
    Python's recursion limit, and a recursive walk turned a body the parser handles
    into a ``RecursionError``.
    """
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, str):
            if _LONE_SURROGATE.search(item):
                return True
        elif isinstance(item, dict):
            stack.extend(item.keys())
            stack.extend(item.values())
        elif isinstance(item, (list, tuple)):
            stack.extend(item)
    return False


# --------------------------------------------------------------------------------------
# Request controls
# --------------------------------------------------------------------------------------

# `Router.predict` takes five arguments beyond state and questions that a JSON body
# must not be able to set: a hook is a callable that runs inside this process, and
# `hooks_raise` / `hooks_timeout` say how the hooks *this deployment* installed
# execute, so honouring a caller's value would let a request change server-side
# behaviour. Refused with 422 rather than dropped, so a client that sends one is
# told rather than silently ignored.
BODY_REFUSALS = (
    "hooks",
    "on_predict_start",
    "on_predict_end",
    "hooks_raise",
    "hooks_timeout",
)


def refuse_body_controls(body: Dict[str, Any]) -> None:
    given = sorted(key for key in BODY_REFUSALS if body.get(key) is not None)
    if given:
        raise HTTPException(
            status_code=422,
            detail=(
                f"{', '.join(given)} run inside the server process and cannot be sent to "
                f"this endpoint; install them where the service runs, or drop them"
            ),
        )


def clamp_budget(name: str, value: Any, cap: int) -> Optional[int]:
    """Validate a caller-supplied token budget against the server's ceiling."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise HTTPException(status_code=422, detail=f"{name} must be an integer")
    if value < 1:
        raise HTTPException(status_code=422, detail=f"{name} must be a positive integer")
    if value > cap:
        raise HTTPException(
            status_code=422,
            detail=f"{name} exceeds the server limit ({value} > {cap}); the deployment "
            f"caps token budgets at {cap}",
        )
    return value


def sanitize_log_value(value: Any) -> str:
    """Make caller-supplied text safe to write on one log line.

    ``model`` is caller text that reaches a log call. Newlines in it would forge
    additional log entries.
    """
    return str(value).replace("\n", "\\n").replace("\r", "\\r")


def resolve_model(value: Optional[str]) -> Optional[str]:
    """Map a caller's ``model`` onto a checkpoint name, or None to auto-route.

    Accepted: the checkpoint names, their aliases, and the public Hugging Face ids.
    Anything else -- including a client id such as ``jev-1`` -- means "let the
    router choose", and the response's ``routing`` block records what it picked.
    """
    if not value:
        return None
    from laya.router import normalise_name

    text = str(value).strip().lower()
    try:
        return normalise_name(text)
    except Exception:  # noqa: BLE001 - normalise_name's rejection is the contract
        return None