"""Request and response models.

These exist for two reasons that are easy to conflate:

1. Validation. A request whose question set cannot be answered is refused with a
   named 422 before it takes an inference slot, rather than failing inside a
   forward pass.
2. The OpenAPI document. Upstream ``laya-serve`` validates by hand and declares
   no response models, so its ``/openapi.json`` carries no schema for a consumer
   to generate a client from. Everything here is documentation that is also
   enforced.

The request models are the contract; the response models describe the payload
Laya returns. The response models are declared to FastAPI for the document only
-- the handler emits Laya's own dict, because two of its fields are conditional
in a way a serialiser would flatten (see ``PredictResponse``).
"""

from __future__ import annotations

from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# --------------------------------------------------------------------------------------
# Requests
# --------------------------------------------------------------------------------------


class ChoiceQuestion(BaseModel):
    """Pick one label from a caller-defined option set.

    ``criteria`` is either a mapping of label -> description or a bare list of
    labels. The mapping is strongly preferred: a description gives the model
    something to match against, and the published guidance is that bare label
    names measurably weaken the answer. Both are accepted because upstream
    accepts both and refusing the list form would break existing callers.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["choice"]
    instructions: str = Field(min_length=1, description="The question, in plain language.")
    criteria: Union[Dict[str, str], List[str]] = Field(
        min_length=1,
        description="Option labels mapped to descriptions, or a bare list of labels.",
    )

    @property
    def option_count(self) -> int:
        return len(self.criteria)

    def as_laya(self) -> Dict[str, Any]:
        # Dict form is the documented shape; a bare list is normalised into it so
        # the SDK only ever sees one representation.
        if isinstance(self.criteria, dict):
            return {"type": self.type, "instructions": self.instructions, "criteria": self.criteria}
        return {
            "type": self.type,
            "instructions": self.instructions,
            "criteria": {label: label for label in self.criteria},
        }


class ScoreQuestion(BaseModel):
    """Place the state on an ordered, sequential rubric.

    The array order *is* the scale, so it is ascending from the least to the most
    severe. Levels must be non-empty strings: a null would reach the response
    ``legend`` as ``{"<i>": null}``, which strict clients cannot parse.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["score"]
    instructions: str = Field(min_length=1, description="The question, in plain language.")
    criteria: List[str] = Field(
        min_length=2,
        description="Ordinal levels, ordered least to most severe.",
    )

    @field_validator("criteria")
    @classmethod
    def no_blank_levels(cls, value: List[str]) -> List[str]:
        for index, level in enumerate(value):
            if not isinstance(level, str) or not level.strip():
                raise ValueError(
                    f"level {index} is empty; every level needs text, because the "
                    f"answer's legend is keyed by it"
                )
        return value

    @property
    def option_count(self) -> int:
        return len(self.criteria)

    def as_laya(self) -> Dict[str, Any]:
        return {"type": self.type, "instructions": self.instructions, "criteria": self.criteria}


class NoulCriteria(BaseModel):
    """Optional yes/no framing for a ``noul`` question.

    Keys must be exactly ``true`` and ``false``. ``extra="forbid"`` is the whole
    point: upstream once accepted ``{"yes": ..., "no": ...}``, silently dropped it
    and answered against the default ``false:``/``true:`` pair anyway, which cost
    two of three clearly positive reviews on the English checkpoint. Refusing the
    shape at the edge turns a silent wrong answer into a 422 the caller can see.
    """

    model_config = ConfigDict(extra="forbid")

    true: str = Field(min_length=1, description="What counts as the proposition being true.")
    false: str = Field(min_length=1, description="What counts as it being false.")

    def as_laya(self) -> Dict[str, str]:
        return {"true": self.true, "false": self.false}


class NoulLabels(BaseModel):
    """Model-facing label text for the two slots, without changing the meaning.

    ``noul`` always returns P(true) regardless of how the slots are worded. This
    only replaces the words shown to the model, for the case where the default
    ``false:``/``true:`` pair dominates the answer. Distinctness is enforced
    because two identical labels make the question unanswerable.
    """

    model_config = ConfigDict(extra="forbid")

    true: str = Field(min_length=1)
    false: str = Field(min_length=1)

    @model_validator(mode="after")
    def labels_must_differ(self) -> "NoulLabels":
        if self.true == self.false:
            raise ValueError("noul labels must differ; identical text makes the question unanswerable")
        return self

    def as_laya(self) -> Dict[str, str]:
        return {"true": self.true, "false": self.false}


class NoulQuestion(BaseModel):
    """Estimate P(true) for a yes/no proposition.

    The returned value is the probability of the ``true`` slot and always lies in
    [0.0, 1.0].
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["noul"]
    instructions: str = Field(min_length=1, description="The proposition to estimate.")
    criteria: Optional[NoulCriteria] = Field(
        default=None,
        description="Optional yes/no framing. Keys must be exactly 'true' and 'false'.",
    )
    labels: Optional[NoulLabels] = Field(
        default=None,
        description="Optional replacement model-facing slot text; does not change the meaning.",
    )

    @property
    def option_count(self) -> int:
        return 2

    def as_laya(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"type": self.type, "instructions": self.instructions}
        if self.criteria is not None:
            payload["criteria"] = self.criteria.as_laya()
        if self.labels is not None:
            payload["labels"] = self.labels.as_laya()
        return payload


Question = Annotated[
    Union[ChoiceQuestion, ScoreQuestion, NoulQuestion],
    Field(discriminator="type"),
]


class PredictRequest(BaseModel):
    """One state, one typed question set, one forward pass.

    Every question is answered against the same state independently, so adding a
    question never changes the answer to another.
    """

    model_config = ConfigDict(extra="forbid")

    state: Union[str, Dict[str, Any], List[Any]] = Field(
        description="Text, email, ticket or JSON document to decide on.",
    )
    questions: Dict[str, Question] = Field(
        min_length=1,
        description="Question id -> question definition.",
    )

    @field_validator("state", mode="before")
    @classmethod
    def state_must_be_present(cls, value: Any) -> Any:
        """Reject a null state with one clear message.

        Left to the union, ``null`` fails all three branches and the caller gets
        three near-identical errors naming three internal type paths, none of
        which says what to do. A missing `state` already fails as "Field required",
        so this makes the explicit null read the same way.

        Deliberately 422, not 400: `state` is a required field like any other, and
        special-casing it to 400 would mean a caller handles 400 for one field and
        422 for every other.
        """
        if value is None:
            raise ValueError(
                "state must not be null; it is the text or document to decide on"
            )
        return value
    model: Optional[str] = Field(
        default=None,
        description=(
            "Pin a checkpoint: 'english', 'multilingual', 'typed-decisions', a public "
            "Hugging Face id, or an alias. Omit or send an unrecognised value to let "
            "the router choose; the choice is reported in `routing`."
        ),
    )
    task: Optional[str] = Field(
        default=None, description="Force a checkpoint by workflow name instead of by routing."
    )
    lang: Optional[str] = Field(
        default=None,
        description="Language hint such as 'de' or 'pt-BR'. Skips detection, which can "
        "mistake short Latin-script text for English.",
    )
    lang_guess: Optional[str] = Field(
        default=None,
        description="Language code from the caller's own identifier. Consulted after `lang`.",
    )
    max_len: Optional[int] = Field(
        default=None, ge=1, description="Total token window for this request."
    )
    head_max_len: Optional[int] = Field(
        default=None, ge=1, description="Token window the option prompt shares."
    )
    min_confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Abstention threshold. Answers below it are kept but marked "
            "`low_confidence`. No shipped default: the checkpoints are not calibrated "
            "for your domain, so a threshold here is a guess until you fit one."
        ),
    )

    def as_laya_questions(self) -> Dict[str, Dict[str, Any]]:
        return {qid: question.as_laya() for qid, question in self.questions.items()}

    def option_total(self) -> int:
        return sum(question.option_count for question in self.questions.values())

    def predict_kwargs(self) -> Dict[str, Any]:
        """Controls forwarded to ``Router.predict``, present only when sent.

        An absent control must stay absent: core reads a missing argument as
        "inherit the Router's setting", so passing None would override a
        deployment's own ``lang_guess`` or abstention threshold with the server's
        default.
        """
        kwargs: Dict[str, Any] = {}
        for name in ("task", "lang", "lang_guess", "max_len", "head_max_len", "min_confidence"):
            value = getattr(self, name)
            if value is not None:
                kwargs[name] = value
        return kwargs


class ProfilePredictRequest(BaseModel):
    """Body of a minted profile endpoint: a state and per-request controls.

    Identical for every profile -- what differs between profiles is the question
    set, which the profile supplies and the caller cannot send. That is enforced by
    ``extra="forbid"`` rather than by stripping keys: a caller who typos
    ``max_lan`` gets a 422 naming the field instead of silently losing the control
    they thought they had set.

    ``model`` and ``task`` are absent by design. A profile's checkpoint is part of
    what makes its calibration meaningful, so a request cannot re-route it.
    """

    model_config = ConfigDict(extra="forbid")

    state: Union[str, Dict[str, Any], List[Any]] = Field(
        description="Text, email, ticket or JSON document to decide on.",
    )
    lang: Optional[str] = Field(
        default=None,
        description="Language hint such as 'de' or 'pt-BR'. Skips detection.",
    )
    lang_guess: Optional[str] = Field(
        default=None,
        description="Language code from the caller's own identifier. Consulted after `lang`.",
    )
    max_len: Optional[int] = Field(default=None, ge=1)
    head_max_len: Optional[int] = Field(default=None, ge=1)
    min_confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Overrides this profile's calibrated threshold for one request. Set by a "
            "calibration the operator fitted; absent means use the profile's."
        ),
    )

    @field_validator("state", mode="before")
    @classmethod
    def state_must_be_present(cls, value: Any) -> Any:
        """Reject a null state, matching ``PredictRequest`` exactly.

        The message and status are deliberately identical to the fixed endpoint's:
        a caller moving between the two should not have to learn a second rule for
        the same mistake.
        """
        if value is None:
            raise ValueError(
                "state must not be null; it is the text or document to decide on"
            )
        return value

    def controls(self) -> Dict[str, Any]:
        """Controls the caller actually sent, for layering under the profile's."""
        return {
            name: value
            for name, value in (
                ("lang", self.lang),
                ("lang_guess", self.lang_guess),
                ("max_len", self.max_len),
                ("head_max_len", self.head_max_len),
                ("min_confidence", self.min_confidence),
            )
            if value is not None
        }


class StateOverride(BaseModel):
    """Per-state controls, for the states in a batch that need something different.

    ``Router.predict_batch`` routes each request independently, so a mixed-language
    batch costs one checkpoint build per language either way. Being able to say so
    per state is what makes that explicit rather than accidental.
    """

    model_config = ConfigDict(extra="forbid")

    model: Optional[str] = None
    task: Optional[str] = None
    lang: Optional[str] = None
    lang_guess: Optional[str] = None
    max_len: Optional[int] = Field(default=None, ge=1)
    head_max_len: Optional[int] = Field(default=None, ge=1)

    def as_kwargs(self) -> Dict[str, Any]:
        return {
            name: value
            for name, value in (
                ("model", self.model),
                ("task", self.task),
                ("lang", self.lang),
                ("lang_guess", self.lang_guess),
                ("max_len", self.max_len),
                ("head_max_len", self.head_max_len),
            )
            if value is not None
        }


class BatchPredictRequest(BaseModel):
    """One question set applied to many states, sharing forward passes.

    Batching is where the throughput is: 7.2ms per question batched against 33ms
    unbatched on a T4. On CPU the gain is smaller but still real.

    The question set is shared because that is the shape batching is for. Per-state
    routing and token-budget overrides go in ``overrides``, keyed by index, so a
    partial override cannot silently drift out of alignment with ``states`` the way
    a parallel list would.
    """

    model_config = ConfigDict(extra="forbid")

    states: List[Union[str, Dict[str, Any], List[Any]]] = Field(
        min_length=1, description="States to decide on, in request order."
    )
    questions: Dict[str, Question] = Field(min_length=1)
    overrides: Optional[Dict[int, StateOverride]] = Field(
        default=None,
        description=(
            "Per-state controls, keyed by index into `states`. Keys outside the range "
            "are rejected rather than ignored."
        ),
    )
    model: Optional[str] = None
    task: Optional[str] = None
    lang: Optional[str] = None
    lang_guess: Optional[str] = None
    max_len: Optional[int] = Field(default=None, ge=1)
    head_max_len: Optional[int] = Field(default=None, ge=1)
    batch_size: Optional[int] = Field(
        default=None, ge=1, description="Maximum states per forward-pass batch."
    )
    min_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    sort_by_length: Optional[bool] = Field(
        default=None,
        description="Sort states by length before batching to reduce padding.",
    )

    @model_validator(mode="after")
    def _check_override_indices(self) -> "BatchPredictRequest":
        """Reject an override naming a state that does not exist.

        Keyed-by-index only works if the keys are real. An out-of-range index that
        was silently dropped would answer a state with different controls than the
        caller asked for, which is worse than a 422.
        """
        if not self.overrides:
            return self
        invalid = sorted(
            index for index in self.overrides if not 0 <= index < len(self.states)
        )
        if invalid:
            raise ValueError(
                f"overrides references state indices {invalid}, but states has "
                f"{len(self.states)} item(s) (valid: 0-{len(self.states) - 1})"
            )
        return self

    def as_laya_questions(self) -> Dict[str, Dict[str, Any]]:
        return {qid: question.as_laya() for qid, question in self.questions.items()}

    def option_total(self) -> int:
        return sum(question.option_count for question in self.questions.values())

    def call_kwargs(self) -> Dict[str, Any]:
        """Call-level controls applied to the whole batch.

        Note that ``min_confidence`` and ``batch_size`` are arguments to
        ``predict_batch`` itself rather than per-request fields: they describe the
        forward pass, not any one state.
        """
        kwargs: Dict[str, Any] = {}
        for name in ("batch_size", "min_confidence", "sort_by_length"):
            value = getattr(self, name)
            if value is not None:
                kwargs[name] = value
        return kwargs

    def defaults_for_state(self) -> Dict[str, Any]:
        """Batch-wide routing and token-budget controls."""
        return {
            name: value
            for name, value in (
                ("model", self.model),
                ("task", self.task),
                ("lang", self.lang),
                ("lang_guess", self.lang_guess),
                ("max_len", self.max_len),
                ("head_max_len", self.head_max_len),
            )
            if value is not None
        }

    def as_requests(self) -> List[Dict[str, Any]]:
        """Build the per-request dicts ``Router.predict_batch`` consumes.

        Returns one dict per state, in input order, each carrying the batch-wide
        defaults with that state's own overrides layered on top.
        """
        defaults = self.defaults_for_state()
        requests: List[Dict[str, Any]] = []
        for index, state in enumerate(self.states):
            item = dict(defaults)
            if self.overrides and index in self.overrides:
                item.update(self.overrides[index].as_kwargs())
            item["state"] = state
            item["questions"] = self.as_laya_questions()
            requests.append(item)
        return requests


# --------------------------------------------------------------------------------------
# Responses
# --------------------------------------------------------------------------------------


class Action(BaseModel):
    """The act head's output."""

    act_probability: float = Field(description="Probability that the decision should be acted on.")


class ChoiceAnswer(BaseModel):
    type: Literal["choice"]
    choice: str = Field(description="The selected label.")
    probabilities: Dict[str, float] = Field(description="Probability per option.")
    confidence: float = Field(
        ge=0.0, le=1.0,
        description=(
            "1 - normalised entropy. Depends on how many options the question had, so it "
            "does not compare against a threshold. Gate on `answer_confidence` instead."
        ),
    )
    answer_confidence: float = Field(
        ge=0.0, le=1.0, description="Probability mass on the reported answer. The gateable number."
    )
    action: Action
    abstention: Optional[Literal["passed", "abstained", "unevaluated"]] = Field(
        default=None,
        description="Present only when the request set `min_confidence`. Absence means ungated.",
    )
    abstention_threshold: Optional[float] = None
    low_confidence: Optional[bool] = Field(
        default=None, description="True only when the answer fell below the gate."
    )


class ScoreAnswer(BaseModel):
    type: Literal["score"]
    score: float = Field(description="Expected level index; may fall between two levels.")
    legend: Dict[str, str] = Field(description="Level index -> level text.")
    probabilities: Dict[str, float] = Field(description="Distribution keyed by level index.")
    confidence: float = Field(ge=0.0, le=1.0)
    answer_confidence: float = Field(ge=0.0, le=1.0)
    action: Action
    abstention: Optional[Literal["passed", "abstained", "unevaluated"]] = None
    abstention_threshold: Optional[float] = None
    low_confidence: Optional[bool] = None


class NoulAnswer(BaseModel):
    type: Literal["noul"]
    noul: float = Field(ge=0.0, le=1.0, description="P(true).")
    confidence: float = Field(ge=0.0, le=1.0)
    answer_confidence: float = Field(ge=0.0, le=1.0)
    action: Action
    abstention: Optional[Literal["passed", "abstained", "unevaluated"]] = None
    abstention_threshold: Optional[float] = None
    low_confidence: Optional[bool] = None


Answer = Annotated[
    Union[ChoiceAnswer, ScoreAnswer, NoulAnswer],
    Field(discriminator="type"),
]


class OptionsCollapse(BaseModel):
    """Report that a question's options no longer each fit a token span."""

    total: int = Field(description="Options the question defines.")
    distinct: int = Field(description="Spans that reached the sequence.")
    tokens_per_option: float


class Usage(BaseModel):
    input_tokens: int = Field(description="Non-pad tokens across the state's rows, one row per question.")
    output_tokens: int = Field(description="Always 0; the head answers in one pass and generates nothing.")
    state_tokens: int = Field(description="Tokens the whole serialised state needs.")
    state_tokens_dropped: int = Field(
        description="Tokens of the state at least one question did not get (worst case)."
    )
    truncated: bool = Field(description="True when that worst case dropped anything.")
    truncated_questions: List[str] = Field(description="Ids of questions whose own window was cut.")
    options: Optional[Dict[str, OptionsCollapse]] = Field(
        default=None,
        description="Present only when some question's options were squeezed.",
    )


class Detection(BaseModel):
    """``laya.lang.analyse()`` on the state."""

    script: Optional[str] = None
    script_profile: Dict[str, float] = Field(default_factory=dict)
    language: Optional[str] = None
    is_english: bool = False
    language_undecided: bool = False
    diacritic_rate: float = 0.0
    non_latin_fraction: float = 0.0
    mixed_segment: Optional[str] = None


class Routing(BaseModel):
    model: Literal["english", "multilingual", "typed-decisions"] = Field(
        description="Which checkpoint answered."
    )
    repo: str = Field(description="Its public Hugging Face id.")
    reason: str = Field(description="Why this checkpoint, naming the evidence it acted on.")
    detection: Optional[Detection] = Field(
        default=None, description="Null when routing decided without reading the state."
    )
    workflow: Optional[str] = Field(
        default=None, description="Typed-decisions workflow the question ids matched, if any."
    )


class PredictResponse(BaseModel):
    """A decision set.

    Documented here, emitted verbatim by the handler. Two fields are conditional in
    a way that matters: ``usage.options`` appears only when some question's options
    were squeezed, and ``abstention`` / ``abstention_threshold`` / ``low_confidence``
    appear on an answer only when the request set ``min_confidence``. Routing the
    payload through this model's serialiser would emit all of them as ``null``,
    which reads as "evaluated and absent" rather than "never evaluated" -- so the
    handler returns Laya's own dict and this class documents the contract.
    """

    model: Literal["laya-rl-agent"] = Field(
        default="laya-rl-agent", description="Constant name of the decision head."
    )
    answers: Dict[str, Answer] = Field(description="Question id -> answer.")
    usage: Usage
    routing: Routing


# One decision set per state, in the order the states were sent. A plain array, not an
# envelope: `Router.predict_batch` returns one result per request in input order and
# raises rather than returning a partial result, so a per-item `errors` list would only
# ever be empty or the whole call would have failed. An envelope here would advertise a
# partial-success mode that cannot happen, and a client coded against it would be
# handling a case the server never produces.
BatchPredictResponse = List[PredictResponse]