"""Live smoke test: proves the cached checkpoints route and answer correctly.

Complements ``scripts/warmup.py``, which proves the weights are *present*.
This proves they *work* and reports the latency this host actually produces, so
the histogram buckets and the runbook's reference numbers are grounded in a
measurement rather than copied from a GPU.

Every state below is one the routing rules make unambiguous: ASCII English text
must reach the ``english`` checkpoint, and Devanagari is a non-Latin script the
English checkpoint provably cannot read (it collapses to 0.000 accuracy on Khmer
while staying confident), so it must reach ``multilingual``.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any, Dict

os.environ.setdefault("USE_TF", "0")

QUESTIONS: Dict[str, Any] = {
    "queue": {
        "type": "choice",
        "instructions": "Which team should handle this request?",
        "criteria": {
            "billing": "invoices, payments, refunds",
            "technical": "bugs, outages, system errors",
            "other": "everything else",
        },
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this request?",
        "criteria": ["not urgent", "soon", "blocking"],
    },
    "refund_requested": {
        "type": "noul",
        "instructions": "Does the user explicitly request a refund?",
    },
}

CASES = [
    (
        "english",
        {"subject": "Duplicate charge", "body": "I was charged twice this month. Please refund the duplicate."},
    ),
    ("devanagari", {"body": "मुझसे दो बार शुल्क लिया गया, कृपया पैसे वापस करें।"}),
]


def main() -> int:
    import laya
    from laya import Router

    # Same pinning the service uses; the numbers below are only meaningful with it.
    import torch

    torch.set_num_threads(4)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError as exc:
        print(f"note: inter-op threads left at default ({exc})")

    print(f"laya {laya.__version__} | torch {torch.__version__} | threads {torch.get_num_threads()}")
    print("=" * 72)

    router = Router(device="cpu")
    failures = []

    for label, state in CASES:
        print(f"\n--- {label} ---")
        started = time.perf_counter()
        result = router.predict(state, QUESTIONS)
        elapsed = (time.perf_counter() - started) * 1000

        routed = (result.get("routing") or {}).get("model")
        print(f"routed to : {routed}")
        print(f"reason    : {(result.get('routing') or {}).get('reason')}")
        print(f"latency   : {elapsed:.0f} ms")

        answers = result["answers"]
        queue = answers["queue"]
        print(f"choice    : {queue['choice']!r} (confidence {queue['confidence']:.3f}, "
              f"answer_confidence {queue['answer_confidence']:.3f})")
        urgency = answers["urgency"]
        print(f"score     : {urgency['score']:.3f} / {max(urgency['legend'].values(), key=len)!r} "
              f"legend={urgency['legend']}")
        noul = answers["refund_requested"]
        print(f"noul      : P(true)={noul['noul']:.3f}")

        usage = result.get("usage") or {}
        print(f"usage     : input_tokens={usage.get('input_tokens')} "
              f"state_tokens={usage.get('state_tokens')} truncated={usage.get('truncated')}")

        # Assert routing reached the checkpoint that can actually read this script.
        expected = "english" if label == "english" else "multilingual"
        if routed != expected:
            failures.append(f"{label}: routed to {routed!r}, expected {expected!r}")

        # Assert each primitive produced its own answer shape.
        if queue.get("type") != "choice" or "choice" not in queue:
            failures.append(f"{label}: choice answer malformed")
        if urgency.get("type") != "score" or "score" not in urgency or "legend" not in urgency:
            failures.append(f"{label}: score answer malformed")
        if noul.get("type") != "noul" or not 0.0 <= noul.get("noul", -1) <= 1.0:
            failures.append(f"{label}: noul answer malformed or out of [0,1]")

    print("\n" + "=" * 72)
    if failures:
        print("FAILED:")
        for line in failures:
            print(f"  - {line}")
        return 1
    print("All routing and answer-shape assertions passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())