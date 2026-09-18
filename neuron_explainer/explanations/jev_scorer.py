"""Explanation scorers backed by TypeSafe's Jev (https://docs.typesafe.ai).

Jev answers typed questions about a JSON `state` with calibrated probabilities instead of
generated text, and evaluates every question in one request. One explanation costs one request
(plus one small request for the optional logit fit).

Three score types share this module:

- detection: one yes/no question per example on the plain text. Score = balanced accuracy.
- fuzz: the same, but the tokens the feature fires on are wrapped in << >>. Score = balanced
  accuracy. Catches explanations that are true of any text ("the word 'the'").
- holistic: one 5-level rating of the explanation against the top examples. Score = Jev's
  probability-weighted level / 4, so it is continuous in [0, 1].

All three can also ask whether the feature's top output logits fit the explanation. That answer
is stored as `logit_fit` and never enters `score`: for many features the top logits are byte-pair
fragments, and folding that in penalizes a correct explanation.

These scorers do not fit `NeuronSimulator`, which predicts one activation per token, so they are
a separate entry point rather than a simulator subclass. Usage::

    scorer = JevScorer()  # reads TYPESAFE_API_KEY
    result = await scorer.score(
        JevScoreType.FUZZ,
        explanation="references to dogs",
        activating_records=neuron_record.most_positive_activation_records[:20],
        non_activating_records=neuron_record.random_sample[:5],
        top_logits=[" dog", " dogs", " puppy"],
    )
    result.score  # 0..1
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Sequence

import httpx

from neuron_explainer.activations.activations import ActivationRecord
from neuron_explainer.api_client import exponential_backoff
from neuron_explainer.fast_dataclasses import FastDataclass, register_dataclass

JEV_API_URL = "https://api.typesafe.ai/v1/systemone"
JEV_MODEL = "jev-latest"
API_KEY_ENV_VAR = "TYPESAFE_API_KEY"
REQUEST_TIMEOUT_SECONDS = 30

# Activations are normalized so the max token is 10. Tokens at or above this are the ones the
# fuzz prompt marks; marking every non-zero token drowns the signal in function words.
FUZZ_MARK_MIN = 3
NOUL_THRESHOLD = 0.5
HOLISTIC_MAX_EXAMPLES = 12
MAX_TOP_LOGITS = 10

JEV_SCORE_LEVELS = [
    "Does not describe the marked tokens at all.",
    "Describes a few of the marked tokens. Most do not fit.",
    "Describes about half of the marked tokens.",
    "Describes most of the marked tokens, with minor gaps or being too broad.",
    "Describes the marked tokens perfectly and specifically.",
]

DETECTION_INSTRUCTIONS = (
    "Does `examples[{i}]` contain any word or phrase that matches the neuron explanation in "
    "`explanation`?"
)
FUZZ_INSTRUCTIONS = "Do the words wrapped in << >> in `examples[{i}]` match the description in `explanation`?"
HOLISTIC_INSTRUCTIONS = (
    "In each string in `examples`, the tokens wrapped in << >> are where one neural network "
    "feature fires. How well does `explanation` describe the marked tokens across `examples`?"
)
LOGIT_FIT_INSTRUCTIONS = (
    "`top_output_tokens` are the next-token predictions a feature promotes. Do most of the tokens "
    "in `top_output_tokens` fit the topic described in `explanation`?"
)

# Fixed negatives, added to every detection/fuzz run so a feature with no stored non-activating
# texts still has negatives. The marked tokens are where an unrelated feature fired. Same texts
# as Neuronpedia's recall scorer, so both give comparable scores.
DECOY_RECORDS: list[ActivationRecord] = [
    ActivationRecord(
        tokens=[
            "Sources",
            ":",
            " Cowboys",
            "'",
            " Dak",
            " Prescott",
            " agrees",
            " to",
            " four",
            "-year",
            ",",
            " $",
            "136",
            "M",
            " deal",
        ],
        activations=[0, 0, 10, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0],
    ),
    ActivationRecord(
        tokens=[
            " The",
            " second",
            " element",
            " is",
            " the",
            " ",
            "1",
            "0",
            "-",
            "residue",
            " MT",
            "ase",
            "-",
            "Rd",
            "RP",
            " linker",
            " that",
            " overall",
            " exhibits",
            " low",
        ],
        activations=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0],
    ),
    ActivationRecord(
        tokens=[
            "The",
            " effect",
            " of",
            " a",
            " good",
            " metabolic",
            " control",
            " in",
            " the",
            " natural",
            " history",
            " of",
            " diabetic",
            " retin",
            "opathy",
            " is",
            " discussed",
        ],
        activations=[0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ),
    ActivationRecord(
        tokens=[
            "Category",
            ":",
            "Military",
            " history",
            " of",
            " the",
            " Soviet",
            " Union",
            " during",
            " World",
            " War",
            " II",
        ],
        activations=[0, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0],
    ),
    ActivationRecord(
        tokens=[
            " Wells",
            " Fargo",
            " issued",
            " three",
            " Forms",
            " ",
            "1",
            "0",
            "9",
            "9",
            "-",
            "C",
            ",",
            " Cancellation",
            " of",
            " Debt",
            ", ",
            " to",
        ],
        activations=[0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ),
]


class JevScoreType(str, Enum):
    DETECTION = "detection"
    FUZZ = "fuzz"
    HOLISTIC = "holistic"


# ---------------------------------------------------------------------------------------------
# Token text
# ---------------------------------------------------------------------------------------------


def _build_byte_decoder() -> dict[str, int]:
    """Inverse of GPT-2's bytes_to_unicode: byte-level characters such as Ġ and Ċ back to bytes."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1))
    bs += list(range(ord("®"), ord("ÿ") + 1))
    cs = list(bs)
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


_BYTE_DECODER = _build_byte_decoder()


def decode_token(token: str) -> str:
    """Return a token as readable text.

    Handles GPT-2 byte-level tokens, also when half-decoded (real spaces but `Ċ` for newlines),
    and SentencePiece's `▁` word marker. Returns the input unchanged when the bytes are not valid
    UTF-8, so a stray character never turns a whole token into replacement characters.
    """
    token = token.replace("\u2581", " ")
    out = bytearray()
    for ch in token:
        b = _BYTE_DECODER.get(ch)
        if b is not None:
            out.append(b)
        else:
            out += ch.encode("utf-8")
    try:
        return out.decode("utf-8")
    except UnicodeDecodeError:
        return token


def plain_text(tokens: Sequence[str]) -> str:
    """Join tokens into one line of readable text."""
    return "".join(decode_token(t) for t in tokens).replace("\n", " ")


def _mark_one_word(text: str) -> str:
    """Pick one word by a hash of the text, so the choice is stable across runs."""
    words = text.split(" ")
    h = 0
    for ch in text:
        h = (h * 31 + ord(ch)) % 1_000_003
    k = h % len(words)
    words[k] = f"<<{words[k]}>>"
    return " ".join(words)


def marked_text(
    tokens: Sequence[str], activations: Sequence[float], mark_min: float = FUZZ_MARK_MIN
) -> str:
    """Wrap the tokens the feature fires on in << >>.

    A text with no activation still gets one marked word, so the marker itself is not the signal
    that separates positives from negatives.
    """
    max_activation = max([0.0, *activations])
    if max_activation <= 0:
        return _mark_one_word(plain_text(tokens))
    parts = []
    for token, value in zip(tokens, activations):
        decoded = decode_token(token)
        parts.append(
            f"<<{decoded}>>" if value * 10 / max_activation >= mark_min else decoded
        )
    return "".join(parts).replace("\n", " ")


def balanced_accuracy(predictions: Sequence[bool], truths: Sequence[bool]) -> float:
    """Mean of the true-positive rate and the true-negative rate, in [0, 1]. Chance is 0.5."""
    pos = sum(1 for t in truths if t)
    neg = len(truths) - pos
    if pos == 0 or neg == 0:
        return 0.0
    tp = sum(1 for p, t in zip(predictions, truths) if t and p)
    tn = sum(1 for p, t in zip(predictions, truths) if not t and not p)
    return 0.5 * (tp / pos + tn / neg)


# ---------------------------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------------------------


class JevApiError(Exception):
    """A non-2xx response from the TypeSafe API."""

    def __init__(self, status_code: int, body: str):
        super().__init__(f"TypeSafe API returned {status_code}: {body[:500]}")
        self.status_code = status_code
        self.body = body


def _is_retryable(err: Exception) -> bool:
    """Retry on rate limits, overload and transport errors; a 4xx will not change on retry."""
    if isinstance(err, JevApiError):
        return err.status_code in (429, 529) or err.status_code >= 500
    return isinstance(err, (httpx.TransportError,))


class JevClient:
    """Minimal async client for TypeSafe's /v1/systemone endpoint.

    Separate from `ApiClient`, which only speaks OpenAI chat/completions.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = JEV_MODEL,
        base_api_url: str = JEV_API_URL,
        timeout_seconds: float = REQUEST_TIMEOUT_SECONDS,
    ):
        self.api_key = api_key or os.environ.get(API_KEY_ENV_VAR, "")
        if not self.api_key:
            raise ValueError(f"Pass api_key or set {API_KEY_ENV_VAR}.")
        self.model = model
        self.base_api_url = base_api_url
        self.timeout_seconds = timeout_seconds

    @exponential_backoff(retry_on=_is_retryable)
    async def ask(
        self, state: dict[str, Any], questions: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
        """POST one state and its questions; return the parsed response (`answers`, `usage`)."""
        body = {"state": state, "model": self.model, "questions": questions}
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.post(self.base_api_url, headers=headers, json=body)
        if response.status_code >= 400:
            raise JevApiError(response.status_code, response.text)
        return response.json()


# ---------------------------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------------------------


@register_dataclass
@dataclass
class JevExampleResult(FastDataclass):
    """One example's answer in a detection or fuzz run."""

    kind: str
    """'activating', 'non_activating' or 'decoy'."""
    text: str
    marked_text: str
    tokens: list[str]
    activations: list[float]
    ground_truth: bool
    noul: float
    """Jev's probability that the answer is yes."""
    prediction: bool
    correct: bool


@register_dataclass
@dataclass
class JevLogitFit(FastDataclass):
    noul: float
    top_logits: list[str]


@register_dataclass
@dataclass
class JevScoredExplanation(FastDataclass):
    """Result of scoring one explanation with one Jev score type."""

    explanation: str
    score_type: str
    """A `JevScoreType` value."""
    score: float
    """In [0, 1]. Balanced accuracy for detection/fuzz, expected_level / 4 for holistic."""
    model: str
    input_tokens: int
    example_results: list[JevExampleResult] = field(default_factory=list)
    """Detection and fuzz only."""
    expected_level: Optional[float] = None
    """Holistic only: Jev's probability-weighted level, in [0, len(JEV_SCORE_LEVELS) - 1]."""
    top_level: Optional[int] = None
    """Holistic only: the most probable index into JEV_SCORE_LEVELS."""
    level_probabilities: Optional[list[float]] = None
    """Holistic only: one probability per level, in order."""
    confidence: Optional[float] = None
    logit_fit: Optional[JevLogitFit] = None

    def get_preferred_score(self) -> float:
        return self.score


# ---------------------------------------------------------------------------------------------
# Request builders (pure, so they can be tested without the network)
# ---------------------------------------------------------------------------------------------


def _record_text(record: ActivationRecord, mark_min: float) -> tuple[str, str]:
    return plain_text(record.tokens), marked_text(
        record.tokens, record.activations, mark_min
    )


def build_classification_request(
    score_type: JevScoreType,
    explanation: str,
    activating_records: Sequence[ActivationRecord],
    non_activating_records: Sequence[ActivationRecord],
    decoy_records: Sequence[ActivationRecord],
    mark_min: float = FUZZ_MARK_MIN,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Return (state, questions, examples) for a detection or fuzz run."""
    is_fuzz = score_type == JevScoreType.FUZZ
    examples: list[dict[str, Any]] = []
    for kind, records, truth in (
        ("activating", activating_records, True),
        ("non_activating", non_activating_records, False),
        ("decoy", decoy_records, False),
    ):
        for record in records:
            text, marked = _record_text(record, mark_min)
            examples.append(
                {
                    "kind": kind,
                    "record": record,
                    "text": text,
                    "marked_text": marked,
                    "truth": truth,
                }
            )
    instructions = FUZZ_INSTRUCTIONS if is_fuzz else DETECTION_INSTRUCTIONS
    questions = {
        f"ex_{i}": {"type": "noul", "instructions": instructions.format(i=i)}
        for i in range(len(examples))
    }
    state = {
        "explanation": explanation,
        "examples": [e["marked_text"] if is_fuzz else e["text"] for e in examples],
    }
    return state, questions, examples


def build_holistic_request(
    explanation: str,
    activating_records: Sequence[ActivationRecord],
    max_examples: int = HOLISTIC_MAX_EXAMPLES,
    mark_min: float = FUZZ_MARK_MIN,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Return (state, questions) for a holistic run over the strongest examples."""
    top = sorted(activating_records, key=lambda r: -max([0.0, *r.activations]))[
        :max_examples
    ]
    state = {
        "explanation": explanation,
        "examples": [marked_text(r.tokens, r.activations, mark_min) for r in top],
    }
    questions = {
        "rating": {
            "type": "score",
            "instructions": HOLISTIC_INSTRUCTIONS,
            "criteria": list(JEV_SCORE_LEVELS),
        }
    }
    return state, questions


def build_logit_fit_request(
    explanation: str, top_logits: Sequence[str]
) -> Optional[tuple[dict[str, Any], dict[str, dict[str, Any]], list[str]]]:
    """Return (state, questions, logits), or None when no readable logits remain."""
    logits = [decode_token(t) for t in list(top_logits)[:MAX_TOP_LOGITS]]
    logits = [t for t in logits if t.strip()]
    if not logits:
        return None
    state = {"explanation": explanation, "top_output_tokens": logits}
    questions = {"fit": {"type": "noul", "instructions": LOGIT_FIT_INSTRUCTIONS}}
    return state, questions, logits


# ---------------------------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------------------------


class JevScorer:
    """Scores explanations against activation records with Jev.

    Neuronpedia passes the top 20 activating records and up to 5 non-activating ones; the five
    fixed decoys are always added as negatives.
    """

    def __init__(
        self,
        client: Optional[JevClient] = None,
        decoy_records: Sequence[ActivationRecord] = tuple(DECOY_RECORDS),
        mark_min: float = FUZZ_MARK_MIN,
        noul_threshold: float = NOUL_THRESHOLD,
        holistic_max_examples: int = HOLISTIC_MAX_EXAMPLES,
    ):
        self.client = client or JevClient()
        self.decoy_records = list(decoy_records)
        self.mark_min = mark_min
        self.noul_threshold = noul_threshold
        self.holistic_max_examples = holistic_max_examples

    async def score(
        self,
        score_type: JevScoreType,
        explanation: str,
        activating_records: Sequence[ActivationRecord],
        non_activating_records: Sequence[ActivationRecord] = (),
        top_logits: Optional[Sequence[str]] = None,
    ) -> JevScoredExplanation:
        """Score one explanation. `top_logits`, when given, adds the `logit_fit` side signal."""
        if not activating_records:
            raise ValueError("Jev scoring needs at least one activating record.")
        if score_type == JevScoreType.HOLISTIC:
            result = await self._score_holistic(explanation, activating_records)
        else:
            result = await self._score_classification(
                score_type, explanation, activating_records, non_activating_records
            )
        if top_logits:
            result.logit_fit = await self.logit_fit(explanation, top_logits)
        return result

    async def _score_classification(
        self,
        score_type: JevScoreType,
        explanation: str,
        activating_records: Sequence[ActivationRecord],
        non_activating_records: Sequence[ActivationRecord],
    ) -> JevScoredExplanation:
        state, questions, examples = build_classification_request(
            score_type,
            explanation,
            activating_records,
            non_activating_records,
            self.decoy_records,
            self.mark_min,
        )
        response = await self.client.ask(state, questions)
        results = []
        for i, example in enumerate(examples):
            noul = float(response["answers"][f"ex_{i}"]["noul"])
            prediction = noul > self.noul_threshold
            record: ActivationRecord = example["record"]
            results.append(
                JevExampleResult(
                    kind=example["kind"],
                    text=example["text"],
                    marked_text=example["marked_text"],
                    tokens=list(record.tokens),
                    activations=[float(a) for a in record.activations],
                    ground_truth=example["truth"],
                    noul=noul,
                    prediction=prediction,
                    correct=prediction == example["truth"],
                )
            )
        return JevScoredExplanation(
            explanation=explanation,
            score_type=score_type.value,
            score=balanced_accuracy(
                [r.prediction for r in results], [r.ground_truth for r in results]
            ),
            model=response.get("model", self.client.model),
            input_tokens=int(response.get("usage", {}).get("input_tokens", 0)),
            example_results=results,
        )

    async def _score_holistic(
        self, explanation: str, activating_records: Sequence[ActivationRecord]
    ) -> JevScoredExplanation:
        state, questions = build_holistic_request(
            explanation, activating_records, self.holistic_max_examples, self.mark_min
        )
        response = await self.client.ask(state, questions)
        answer = response["answers"]["rating"]
        # `score` is the probability-weighted level (e.g. 2.39), not an index.
        expected_level = float(answer["score"])
        # Probabilities come keyed by level index as strings; keep them in level order.
        probabilities = answer.get("probabilities", {})
        level_probabilities = [
            float(probabilities.get(str(i), 0.0)) for i in range(len(JEV_SCORE_LEVELS))
        ]
        return JevScoredExplanation(
            explanation=explanation,
            score_type=JevScoreType.HOLISTIC.value,
            score=expected_level / (len(JEV_SCORE_LEVELS) - 1),
            model=response.get("model", self.client.model),
            input_tokens=int(response.get("usage", {}).get("input_tokens", 0)),
            expected_level=expected_level,
            top_level=max(
                range(len(level_probabilities)), key=lambda i: level_probabilities[i]
            ),
            level_probabilities=level_probabilities,
            confidence=float(answer["confidence"]) if "confidence" in answer else None,
        )

    async def logit_fit(
        self, explanation: str, top_logits: Sequence[str]
    ) -> Optional[JevLogitFit]:
        """Side signal: do the feature's top output logits fit the explanation? Never in `score`."""
        built = build_logit_fit_request(explanation, top_logits)
        if built is None:
            return None
        state, questions, logits = built
        response = await self.client.ask(state, questions)
        return JevLogitFit(
            noul=float(response["answers"]["fit"]["noul"]), top_logits=logits
        )
