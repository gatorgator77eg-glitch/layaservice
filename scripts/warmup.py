"""Download and cache Laya checkpoints into the local Hugging Face cache.

This is the warm-up half of the offline story: run it once on a networked host
so the weights exist on disk, then deploy with ``HF_HUB_OFFLINE=1`` and the
service loads without reaching the Hub.

It builds each checkpoint with Laya's own loader rather than calling
``huggingface_hub.snapshot_download`` directly, so the cached file set is
exactly the one the runtime asks for -- downloading the whole repository would
pull all three checkpoints (~2.4 GB) when a deployment only needs two.

Usage::

    python -m scripts.warmup                      # english + multilingual
    python -m scripts.warmup --models all         # all three
    python -m scripts.warmup --models english --verify-offline

``--verify-offline`` re-loads every checkpoint with ``HF_HUB_OFFLINE=1`` set,
which is the only honest way to prove an air-gapped deploy will work: a cache
that is merely populated can still miss a file and only fail at first request.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List, Tuple

# Deliberately NOT a hardcoded table of hub repo ids.
#
# `laya.router.DEFAULT_MODELS` resolves all three checkpoints as subfolders of one
# bundled repo (`convaiinnovations/laya`), which is what `Router` actually loads. A
# hardcoded list of standalone repo ids warms *different weights* than the service
# will ask for: it appears to succeed, and then the first request fails under
# `HF_HUB_OFFLINE=1` with `IncompleteSnapshotError` for a file the warm-up never
# looked at. That is exactly the failure this script exists to prevent.
#
# The one thing that must not drift is `ROUTING_DEFAULT` below, which encodes a
# policy choice (what to warm by default) rather than a fact about where the
# weights live.
DEFAULT_SPECS: Dict[str, Tuple[str, str | None]] = {}


def load_specs() -> Dict[str, Tuple[str, str | None]]:
    """Read the checkpoint table from the installed SDK."""
    global DEFAULT_SPECS
    if not DEFAULT_SPECS:
        from laya.router import DEFAULT_MODELS

        DEFAULT_SPECS = dict(DEFAULT_MODELS)
    return DEFAULT_SPECS


def spec_str(name: str) -> str:
    """Human-readable ``repo`` or ``repo/subfolder`` for messages."""
    repo, subfolder = load_specs()[name]
    return f"{repo}/{subfolder}" if subfolder else repo

# `typed-decisions` comes last so a default run builds the two checkpoints
# automatic routing actually chooses between. Building `typed-decisions` is only
# reachable through an explicit `model=` request or `LAYA_AUTO_TASK`, so it costs
# startup time nobody uses by default.
ROUTING_DEFAULT = ["english", "multilingual"]


def configure_threads(intra: int | None) -> Dict[str, int | None]:
    """Pin torch thread counts before any parallel work starts.

    Both settings are load-bearing and neither is cosmetic:

    * intra-op should be the *physical* core count. One thread per logical core
      is measurably slower -- SMT siblings contend, and the reported p50 for a
      one-question call regressed from 329 ms to 388 ms going from 8 threads to
      10 on a 4-core/8-thread host.
    * inter-op must be 1. A single forward pass has no inter-op parallelism to
      overlap, and leaving the default produced a p50 of 9,396 ms against 783 ms
      pinned -- a 12x regression on a three-question call.

    `set_num_interop_threads` raises if the pool is already initialised, so this
    has to run at import-adjacent time, before any tensor work.
    """
    import torch

    applied: Dict[str, int | None] = {}
    if intra is None:
        applied["intra_op"] = None
    else:
        torch.set_num_threads(intra)
        applied["intra_op"] = intra
    try:
        torch.set_num_interop_threads(1)
        applied["inter_op"] = 1
    except RuntimeError as exc:
        # Harmless if something already warmed the pool; report rather than fail,
        # because the download this script exists for does not depend on it.
        applied["inter_op"] = f"unchanged ({exc})"
    return applied


def parse_models(raw: str) -> List[str]:
    known = load_specs()
    if raw.strip().lower() == "all":
        return list(known)
    names = [part.strip() for part in raw.replace(",", " ").split() if part.strip()]
    unknown = [n for n in names if n not in known]
    if unknown:
        raise SystemExit(
            f"unknown checkpoint(s): {', '.join(unknown)}; "
            f"choose from {', '.join(known)} or 'all'"
        )
    if not names:
        raise SystemExit("no checkpoints selected")
    return names


def cache_report() -> str:
    """Describe the Hugging Face cache this process will load from."""
    home = os.environ.get("HF_HOME")
    lines = [f"HF_HOME={home or '(default)'}"]
    if home:
        lines.append(f"cache dir={os.path.join(home, 'hub')}")
    lines.append(
        f"HF_HUB_OFFLINE={os.environ.get('HF_HUB_OFFLINE', '0')} "
        "(0 = may reach the network)"
    )
    return "\n".join(lines)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        default=",".join(ROUTING_DEFAULT),
        help="comma-separated checkpoint names, or 'all' (default: english,multilingual)",
    )
    parser.add_argument(
        "--device",
        default=os.environ.get("LAYA_DEVICE", "cpu"),
        help="torch device to build on (default: cpu)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="torch intra-op thread cap; defaults to the physical core count",
    )
    # A real flag, not a scan of sys.argv. Scanning argv looks like it works and then
    # argparse rejects the unknown argument, so the documented command fails.
    parser.add_argument(
        "--verify-offline",
        action="store_true",
        help="reload every checkpoint with HF_HUB_OFFLINE=1 to prove the cache is complete",
    )
    args = parser.parse_args(argv)

    # `transformers` probes for TensorFlow at import time, and when TensorFlow
    # is installed its abseil runtime can deadlock model construction. Set this
    # before laya is imported anywhere in the process.
    os.environ.setdefault("USE_TF", "0")

    names = parse_models(args.models)

    if args.threads is None:
        # psutil is not a dependency; the portable reading of "physical cores"
        # without it is to ask torch, which is already a hard dependency.
        import torch

        args.threads = torch.get_num_threads() or 4

    print("Laya checkpoint warm-up")
    print("=" * 60)
    print(cache_report())
    print(f"checkpoints: {', '.join(names)}")
    print(f"device: {args.device}")

    import laya
    from laya.router import Router

    print(f"laya {laya.__version__}")
    applied = configure_threads(args.threads)
    print(f"torch threads: {applied}")

    router = Router(device=args.device, max_loaded=len(names) or 1)

    failures = {}
    revisions = {}
    for name in names:
        print("-" * 60)
        print(f"[{name}] building from {spec_str(name)} ...")
        started = time.perf_counter()
        try:
            # Through the Router, not `laya.load(repo)`. The Router applies the
            # subfolder and any agent kwargs the service will use, so this warms the
            # exact files the service requests. Loading a bare repo id instead can
            # populate a cache that is complete for the wrong path.
            router.load(name)
        except Exception as exc:  # noqa: BLE001 - report and continue to the next checkpoint
            elapsed = time.perf_counter() - started
            failures[name] = exc
            print(f"[{name}] FAILED after {elapsed:.1f}s: {type(exc).__name__}: {exc}")
            continue
        elapsed = time.perf_counter() - started
        revision = (getattr(router, "loaded_revisions", {}) or {}).get(name) or "unknown"
        revisions[name] = revision
        print(f"[{name}] ready in {elapsed:.1f}s (revision {revision})")

    print("-" * 60)
    print(f"router.loaded after warm-up: {router.loaded or '(none - loaded per-agent above)'}")

    if failures:
        print()
        print("FAILED checkpoints:")
        for name, exc in failures.items():
            print(f"  {name}: {type(exc).__name__}: {exc}")
        print()
        print("A partial cache is worse than none: it looks warm and still pays a")
        print("network fetch on first use. Resolve the failures above before deploying")
        print("with HF_HUB_OFFLINE=1.")
        return 1

    print()
    print("All requested checkpoints are cached. For an air-gapped deploy:")
    print("  1. ship this HF_HOME directory in the image")
    print("  2. set HF_HUB_OFFLINE=1 and LAYA_PRELOAD=1")
    print("  3. confirm with: GET /health  -> loaded[] populated, revisions{} present")

    if args.verify_offline:
        print()
        print("Re-loading with HF_HUB_OFFLINE=1 to prove the cache is complete ...")
        os.environ["HF_HUB_OFFLINE"] = "1"
        # A fresh Router, and the same `load` path the service uses. Reloading the
        # objects already in memory would prove nothing: they are built, so no file
        # lookup happens, and an incomplete cache would still pass.
        offline_router = Router(device=args.device, max_loaded=len(names) or 1)
        for name in names:
            print(f"[{name}] offline reload ...", flush=True)
            try:
                offline_router.load(name)
            except Exception as exc:  # noqa: BLE001
                print(f"[{name}] OFFLINE RELOAD FAILED: {type(exc).__name__}: {exc}")
                print()
                print("This is the failure an air-gapped deploy would hit at startup.")
                print("Re-run WITHOUT --verify-offline so the network can complete the")
                print("snapshot, then verify again. A cache that is merely populated is")
                print("not a cache that is complete.")
                return 1
            print(f"[{name}] loads offline")
        print("Offline verification passed.")
        print("The service will now start under HF_HUB_OFFLINE=1 with no network.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())