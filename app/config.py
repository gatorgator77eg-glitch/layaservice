"""Environment-driven configuration.

Every setting is an environment variable so one image serves a laptop dev run
and a production deployment, and so a config file is never a second source of
truth. Names deliberately reuse upstream ``LAYA_*`` spellings where the meaning
is identical: this service does not run ``laya-serve``, so there is no collision,
and an operator who knows one does not have to learn a second vocabulary.

Parsing is fail-fast. A limit that silently falls back to a default is a
deployment serving bounds nobody asked for, so a malformed value stops startup
with the variable named. The one exception is thread count, where a fallback is
safe and a hard failure would be worse than a slower service.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

CHECKPOINT_NAMES = ("english", "multilingual", "typed-decisions")


class ConfigError(Exception):
    """A setting is present but unusable. Startup must not continue past this."""


def _raw(name: str) -> Optional[str]:
    value = os.environ.get(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _int(name: str, default: int, *, minimum: int = 1, maximum: Optional[int] = None) -> int:
    raw = _raw(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ConfigError(f"{name}={raw!r} is not an integer") from None
    if value < minimum:
        raise ConfigError(f"{name}={value} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise ConfigError(f"{name}={value} must be <= {maximum}")
    return value


def _bool(name: str, default: bool) -> bool:
    raw = _raw(name)
    if raw is None:
        return default
    lowered = raw.lower()
    if lowered in ("1", "true", "yes", "on"):
        return True
    if lowered in ("0", "false", "no", "off"):
        return False
    raise ConfigError(f"{name}={raw!r} is not a boolean (use 1/0, true/false, yes/no, on/off)")


def _physical_cores() -> int:
    """Physical cores, not logical ones.

    Torch defaults to one intra-op thread per logical core, and SMT siblings
    contend: a one-question call regressed from 329ms p50 to 388ms going from 8
    threads to 10 on a 4-core/8-thread host. On Windows the processor groups /
    core-count APIs are awkward, so this reads the environment override first
    and otherwise asks torch, which has already resolved the topology by the time
    it is imported.
    """
    override = _raw("LAYA_THREADS")
    if override:
        try:
            n = int(override)
        except ValueError:
            raise ConfigError(f"LAYA_THREADS={override!r} is not an integer") from None
        if n < 1:
            raise ConfigError(f"LAYA_THREADS={n} must be >= 1")
        return n
    try:
        import torch

        return max(1, torch.get_num_threads())
    except Exception:  # noqa: BLE001 - torch absent at config time is fine
        return max(1, (os.cpu_count() or 4) // 2)


@dataclass(frozen=True)
class Limits:
    """Request guardrails, checked before tokenization.

    An oversized request then costs the bytes it took to read and nothing more,
    because nothing is tokenized on the way to being refused.
    """

    body_bytes: int = 2 * 1024 * 1024
    state_chars: int = 50_000
    questions: int = 32
    choice_options: int = 32
    score_levels: int = 32
    total_options: int = 512
    batch_states: int = 16
    token_budget: int = 8192


@dataclass(frozen=True)
class Config:
    device: str = "cpu"
    preload: bool = True
    preload_models: Optional[List[str]] = None
    threads: int = 4
    max_loaded: Optional[int] = None
    auto_task_detection: bool = False
    default_model: Optional[str] = None
    max_concurrent: int = 16
    host: str = "0.0.0.0"  # noqa: S104 - container bind; a gateway fronts this
    port: int = 8000
    root_path: str = ""
    log_level: str = "info"
    data_dir: str = "data"
    max_profiles: int = 32
    limits: Limits = field(default_factory=Limits)

    @classmethod
    def from_env(cls) -> "Config":
        models_raw = _raw("LAYA_MODELS")
        preload_models = None
        if models_raw is not None:
            names = [n.strip() for n in models_raw.replace(",", " ").split() if n.strip()]
            unknown = [n for n in names if n not in CHECKPOINT_NAMES]
            if unknown:
                raise ConfigError(
                    f"LAYA_MODELS names unknown checkpoint(s) {unknown}; "
                    f"expected any of {list(CHECKPOINT_NAMES)}"
                )
            preload_models = names or None

        max_loaded_raw = _raw("LAYA_MAX_LOADED")
        max_loaded = None
        if max_loaded_raw is not None:
            # A cap below what routing can choose unloads the least-recently-used
            # checkpoint on every language switch. Measured 20-23s per request to
            # rebuild one on CPU against 49-136ms with it resident, so this is
            # validated rather than merely bounded.
            try:
                parsed = int(max_loaded_raw)
            except ValueError:
                raise ConfigError(f"LAYA_MAX_LOADED={max_loaded_raw!r} is not an integer") from None
            if parsed < 1:
                raise ConfigError(f"LAYA_MAX_LOADED={parsed} must be >= 1")
            max_loaded = parsed

        default_model = _raw("LAYA_DEFAULT_MODEL")
        if default_model is not None and default_model not in CHECKPOINT_NAMES:
            raise ConfigError(
                f"LAYA_DEFAULT_MODEL={default_model!r} is not one of {list(CHECKPOINT_NAMES)}"
            )

        limits = Limits(
            body_bytes=_int("LAYA_MAX_BODY_BYTES", 2 * 1024 * 1024),
            state_chars=_int("LAYA_MAX_STATE_CHARS", 50_000),
            questions=_int("LAYA_MAX_QUESTIONS", 32),
            # 32 rather than upstream's 100. The real ceiling is the shared
            # head_max_len budget, not the option count, and the 77-label
            # banking77 run scored 0.425 on *both* checkpoints -- the signature
            # of labels squeezed to ~4 tokens each, not of a capability gap.
            choice_options=_int("LAYA_MAX_CHOICE_OPTIONS", 32),
            score_levels=_int("LAYA_MAX_SCORE_LEVELS", 32),
            total_options=_int("LAYA_MAX_TOTAL_OPTIONS", 512),
            batch_states=_int("LAYA_MAX_BATCH_STATES", 16),
            token_budget=_int("LAYA_MAX_TOKEN_BUDGET", 8192),
        )

        return cls(
            device=_raw("LAYA_DEVICE") or "cpu",
            preload=_bool("LAYA_PRELOAD", True),
            preload_models=preload_models,
            threads=_physical_cores(),
            max_loaded=max_loaded,
            auto_task_detection=_bool("LAYA_AUTO_TASK", False),
            default_model=default_model,
            max_concurrent=_int("LAYA_MAX_CONCURRENT", 16),
            host=_raw("LAYA_HOST") or "0.0.0.0",  # noqa: S104
            port=_int("LAYA_PORT", 8000, maximum=65535),
            root_path=_raw("LAYA_ROOT_PATH") or "",
            log_level=_raw("LAYA_LOG_LEVEL") or "info",
            data_dir=_raw("LAYA_DATA_DIR") or "data",
            # Capped because a profile becomes a Prometheus route label, and also
            # because each profile pinned to a different checkpoint multiplies
            # resident weights against a max_loaded that is deliberately small.
            max_profiles=_int("LAYA_MAX_PROFILES", 32),
            limits=limits,
        )

    def router_kwargs(self) -> Dict[str, object]:
        """Constructor arguments for ``laya.Router``.

        Only non-default values are passed. ``Router`` reads an absent argument as
        "use my own default", so sending ``None`` for e.g. ``max_loaded`` would
        override the library default rather than defer to it.
        """
        kwargs: Dict[str, object] = {"device": self.device}
        if self.auto_task_detection:
            kwargs["auto_task_detection"] = True
        if self.max_loaded is not None:
            kwargs["max_loaded"] = self.max_loaded
        if self.default_model is not None:
            # Router's parameter is `default`; LAYA_DEFAULT_MODEL is the
            # operator-facing spelling so it does not collide with the
            # `default_model` field used for a different thing elsewhere.
            kwargs["default"] = self.default_model
        return kwargs