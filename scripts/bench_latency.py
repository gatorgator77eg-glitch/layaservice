"""Measure steady-state inference latency on this host, with checkpoints warm.

The first call in ``scripts/smoke.py`` conflates three costs that the service
has to keep separate: downloading the checkpoint, building it from cache, and
running the forward pass. Only the third is the request latency an SLA is
written against -- and it is the only one a warm deployment ever pays.

Reported per checkpoint and question count, because both matter and they do not
scale the same way: the encoder runs once per state, while each extra question
costs an option-prompt forward pass over a shared budget.

Usage::

    python -m scripts.bench_latency --models english,multilingual
    python -m scripts.bench_latency --questions 1,5 --repeat 5
"""

from __future__ import annotations

import argparse
import os
import statistics
import time
from typing import Any, Dict, List

os.environ.setdefault("USE_TF", "0")


def build_questions(n: int) -> Dict[str, Any]:
    """``n`` questions mixing all three primitives, cycling if n > 3."""
    templates = [
        {
            "type": "choice",
            "instructions": "Which team should handle this request?",
            "criteria": {
                "billing": "invoices, payments, refunds",
                "technical": "bugs, outages, system errors",
                "other": "everything else",
            },
        },
        {
            "type": "score",
            "instructions": "How urgent is this request?",
            "criteria": ["not urgent", "soon", "blocking"],
        },
        {"type": "noul", "instructions": "Does the user explicitly request a refund?"},
    ]
    return {f"q{i}": dict(templates[i % len(templates)]) for i in range(n)}


STATE = {
    "subject": "Duplicate charge on invoice 4411",
    "body": (
        "I was charged twice this month for the same subscription. The second "
        "charge posted on the 14th for 49.99 and the first on the 13th. Please "
        "refund the duplicate. This is the third time I have had to raise this."
    ),
}


def percentile(values: List[float], pct: float) -> float:
    """Nearest-rank percentile; honest about small samples."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(pct / 100 * len(ordered) + 0.5)) - 1))
    return ordered[index]


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="english,multilingual")
    parser.add_argument("--questions", default="1,3")
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)

    import torch

    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError as exc:
        print(f"note: inter-op threads left at default ({exc})")

    import laya
    from laya.router import Router

    from scripts.warmup import CHECKPOINTS

    print(f"laya {laya.__version__} | torch {torch.__version__} | "
          f"intra_op={args.threads} inter_op=1 | repeat={args.repeat} warmup={args.warmup}")
    print("=" * 78)
    print(f"{'model':<15}{'qs':>4}{'p50 ms':>12}{'p95 ms':>12}{'min ms':>12}{'max ms':>12}")
    print("-" * 78)

    for model in [m.strip() for m in args.models.split(",") if m.strip()]:
        cold = time.perf_counter()
        laya.load(CHECKPOINTS[model], device="cpu")
        cold_ms = (time.perf_counter() - cold) * 1000
        print(f"[{model}] build from cache: {cold_ms / 1000:.1f}s")

        for n_q in [int(x) for x in args.questions.split(",") if x.strip()]:
            questions = build_questions(n_q)
            router = Router(device="cpu")
            # Pin this model so routing cannot send the timing to another checkpoint.
            for _ in range(args.warmup):
                router.predict(STATE, questions, model=model)

            samples = []
            for _ in range(args.repeat):
                started = time.perf_counter()
                router.predict(STATE, questions, model=model)
                samples.append((time.perf_counter() - started) * 1000)

            print(
                f"{model:<15}{n_q:>4}{statistics.median(samples):>12.0f}"
                f"{percentile(samples, 95):>12.0f}{min(samples):>12.0f}{max(samples):>12.0f}"
            )
        print("-" * 78)

    print("\nThese are CPU numbers on this host. GPU deployments are ~1 order of")
    print("magnitude faster (a T4 measures 39.5ms p50 for one question on the")
    print("english checkpoint), so re-measure before reusing a threshold.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())