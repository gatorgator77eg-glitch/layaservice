"""``python -m app`` -- run the service without uvicorn's CLI.

Configuration is read from the environment by ``app.config.Config.from_env``, so
the same command works for a dev run and for a supervisor that only execs one
thing. ``--reload`` and ``--workers`` are passed through because they are the two
operator flags worth having here; note that ``--workers`` must stay 1, since each
worker would build its own copy of every checkpoint and the single-worker
inference model already assumes one process owns one set of weights.
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app", description=__doc__)
    parser.add_argument("--host", default=None, help="override LAYA_HOST")
    parser.add_argument("--port", type=int, default=None, help="override LAYA_PORT")
    parser.add_argument("--reload", action="store_true", help="reload on source changes")
    parser.add_argument(
        "--workers", type=int, default=1, help="must be 1; see module docstring"
    )
    parser.add_argument("--log-level", default=None, help="override LAYA_LOG_LEVEL")
    args = parser.parse_args(argv)

    if args.workers != 1:
        parser.error(
            "--workers must be 1. Each worker builds its own copy of every "
            "checkpoint, and this service is single-worker by design: scale with "
            "replicas behind the gateway, not with processes behind one port."
        )

    import uvicorn

    from app.config import Config

    try:
        cfg = Config.from_env()
    except Exception as exc:  # noqa: BLE001 - surface the bad variable name, not a traceback
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2

    host = args.host or cfg.host
    port = args.port or cfg.port
    log_level = args.log_level or cfg.log_level

    uvicorn.run(
        "app.main:build_default_app",
        factory=True,
        host=host,
        port=port,
        log_level=log_level,
        reload=args.reload,
        # One in-process worker: see the --workers note above.
        workers=1,
        root_path=cfg.root_path or "",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())