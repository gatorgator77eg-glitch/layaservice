"""A stub ``laya.Router`` with the real signature and return shape.

The point of this fake is that it is written against the SDK's actual contract --
``predict(state, questions, model=..., ...)`` returning one dict, and
``predict_batch(requests, batch_size=..., ...)`` returning one result per request
in input order -- not against what the service *assumes* the contract is. An earlier
version of the batch schema assumed an ``{"results", "errors"}`` envelope; this fake
returns a flat list because that is what the SDK does, so the same mistake would fail
the tests instead of passing them.

Nothing here loads weights, so the whole HTTP surface is verifiable in milliseconds.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence


def _answer_for(question: Dict[str, Any]) -> Dict[str, Any]:
    """Deterministic, schema-correct answer for one question.

    Reads ``criteria``, because that is what ``laya`` calls both a choice's options
    and a score's levels.
    """
    kind = question["type"]
    criteria = question.get("criteria")
    if kind == "choice":
        labels = list(criteria) if isinstance(criteria, dict) else list(criteria or [])
        chosen = labels[0]
        # Uniform, then the chosen label takes the remainder, so the fake's own
        # output would survive a confidence check.
        rest = (1.0 - 0.7) / max(len(labels) - 1, 1)
        probabilities = {
            label: (0.7 if label == chosen else rest) for label in labels
        }
        return {
            "type": "choice",
            "choice": chosen,
            "probabilities": probabilities,
            "confidence": 0.5,
        }
    if kind == "score":
        levels = list(criteria or [])
        return {
            "type": "score",
            "score": 0,
            "distribution": [1.0 / len(levels)] * len(levels) if levels else [],
            "confidence": 0.5,
        }
    return {"type": "noul", "noul": True, "confidence": 0.5}


def make_result(
    questions: Dict[str, Any],
    routing_model: str = "english",
    *,
    truncated: bool = False,
    state_tokens_dropped: int = 0,
) -> Dict[str, Any]:
    """One realistic decision set, with the conditional keys present as documented."""
    return {
        "model": "laya-rl-agent",
        "answers": {qid: _answer_for(q) for qid, q in questions.items()},
        "usage": {
            "input_tokens": 128,
            "truncated": truncated,
            "state_tokens_dropped": state_tokens_dropped,
            # Conditional: only present when something was actually dropped.
            **({"truncated_questions": ["q"]} if truncated else {}),
        },
        "routing": {
            "model": routing_model,
            "reason": "lang_detected" if routing_model == "multilingual" else "default",
            "detection": {"language": "eng", "confidence": 0.99},
        },
    }


class FakeRouter:
    """Stand-in for ``laya.Router``. Records calls so tests can assert forwarding."""

    def __init__(
        self,
        *,
        delay_s: float = 0.0,
        raise_with: Optional[BaseException] = None,
        loaded: Optional[List[str]] = None,
    ) -> None:
        self.delay_s = delay_s
        self.raise_with = raise_with
        self.loaded = loaded if loaded is not None else ["english"]
        self.loaded_revisions: Dict[str, str] = {}
        self.agent_kwargs: Dict[str, Any] = {}
        self.calls: List[Dict[str, Any]] = []
        self.hooks: List[Any] = []

    def add_hook(self, hook: Any) -> Any:
        self.hooks.append(hook)
        return self

    def predict(
        self,
        state: Any,
        questions: Dict[str, Any],
        model: Optional[str] = None,
        task: Optional[str] = None,
        lang: Optional[str] = None,
        lang_guess: Optional[str] = None,
        max_len: Optional[int] = None,
        head_max_len: Optional[int] = None,
        min_confidence: Optional[float] = None,
    ) -> Dict[str, Any]:
        self.calls.append(
            {
                "method": "predict",
                "state": state,
                "questions": questions,
                "model": model,
                "task": task,
                "lang": lang,
                "max_len": max_len,
                "head_max_len": head_max_len,
                "min_confidence": min_confidence,
            }
        )
        if self.delay_s:
            time.sleep(self.delay_s)
        if self.raise_with is not None:
            raise self.raise_with
        return make_result(questions)

    def predict_batch(
        self,
        requests: Sequence[Dict[str, Any]],
        batch_size: Optional[int] = None,
        hooks_timeout: Optional[float] = None,
        min_confidence: Optional[float] = None,
        sort_by_length: bool = False,
    ) -> List[Dict[str, Any]]:
        self.calls.append(
            {
                "method": "predict_batch",
                "requests": list(requests),
                "batch_size": batch_size,
                "min_confidence": min_confidence,
                "sort_by_length": sort_by_length,
            }
        )
        if self.delay_s:
            time.sleep(self.delay_s)
        if self.raise_with is not None:
            raise self.raise_with
        # One result per request, input order. Same as the SDK.
        return [make_result(item["questions"]) for item in requests]
