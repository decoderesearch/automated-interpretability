import asyncio
from typing import Any

from neuron_explainer.activations.activations import ActivationRecord
from neuron_explainer.explanations.jev_scorer import (
    DECOY_RECORDS,
    JEV_SCORE_LEVELS,
    JevClient,
    JevScoredExplanation,
    JevScorer,
    JevScoreType,
    balanced_accuracy,
    build_classification_request,
    build_holistic_request,
    build_logit_fit_request,
    decode_token,
    marked_text,
    plain_text,
)
from neuron_explainer.fast_dataclasses import dumps, loads

DOG_RECORDS = [
    ActivationRecord(
        tokens=["Ġmy", "Ġdog", "Ġbarks", "Ċ"], activations=[0.0, 8.0, 1.0, 0.0]
    ),
    ActivationRecord(tokens=["▁your", "▁dogs", "▁sleep"], activations=[0.0, 5.0, 0.0]),
]
ZERO_RECORDS = [
    ActivationRecord(
        tokens=["Ġthe", "Ġsky", "Ġis", "Ġblue"], activations=[0.0, 0.0, 0.0, 0.0]
    )
]


def test_decode_token_handles_gpt2_and_sentencepiece() -> None:
    assert decode_token("Ġdog") == " dog"
    assert decode_token("Ċ") == "\n"
    assert decode_token("âĢĻs") == "\u2019s"
    assert decode_token("▁dog") == " dog"
    # Half-decoded input: a real space and a byte-level newline in one token.
    assert decode_token(" dogĊ") == " dog\n"
    # Plain text passes through.
    assert decode_token("hello") == "hello"


def test_plain_and_marked_text() -> None:
    assert plain_text(DOG_RECORDS[0].tokens) == " my dog barks "
    # Only tokens at or above 3/10 of the max are marked: 1.0/8.0 is below.
    assert (
        marked_text(DOG_RECORDS[0].tokens, DOG_RECORDS[0].activations)
        == " my<< dog>> barks "
    )
    assert (
        marked_text(DOG_RECORDS[1].tokens, DOG_RECORDS[1].activations)
        == " your<< dogs>> sleep"
    )


def test_marked_text_marks_one_word_when_nothing_fires() -> None:
    zero = ZERO_RECORDS[0]
    out = marked_text(zero.tokens, zero.activations)
    assert out.count("<<") == 1 and out.count(">>") == 1
    # Stable across calls.
    assert out == marked_text(zero.tokens, zero.activations)


def test_balanced_accuracy() -> None:
    assert (
        balanced_accuracy([True, True, False, False], [True, True, False, False]) == 1.0
    )
    assert (
        balanced_accuracy([True, True, True, True], [True, True, False, False]) == 0.5
    )
    assert (
        balanced_accuracy([True, False, False, False], [True, True, False, False])
        == 0.75
    )
    # One class only: undefined, reported as 0.
    assert balanced_accuracy([True], [True]) == 0.0


def test_build_classification_request_detection() -> None:
    state, questions, examples = build_classification_request(
        JevScoreType.DETECTION, "dogs", DOG_RECORDS, ZERO_RECORDS, DECOY_RECORDS
    )
    n = len(DOG_RECORDS) + len(ZERO_RECORDS) + len(DECOY_RECORDS)
    assert len(examples) == n and len(questions) == n and len(state["examples"]) == n
    assert state["explanation"] == "dogs"
    assert state["examples"][0] == " my dog barks "
    assert questions["ex_0"] == {
        "type": "noul",
        "instructions": "Does `examples[0]` contain any word or phrase that matches the neuron explanation in `explanation`?",
    }
    assert [e["kind"] for e in examples] == ["activating"] * 2 + ["non_activating"] + [
        "decoy"
    ] * 5
    assert [e["truth"] for e in examples] == [True, True] + [False] * 6


def test_build_classification_request_fuzz_uses_marked_text() -> None:
    state, questions, _ = build_classification_request(
        JevScoreType.FUZZ, "dogs", DOG_RECORDS, [], []
    )
    assert state["examples"] == [" my<< dog>> barks ", " your<< dogs>> sleep"]
    assert (
        "<< >>" in questions["ex_1"]["instructions"]
        and "examples[1]" in questions["ex_1"]["instructions"]
    )


def test_build_holistic_request_sorts_and_caps() -> None:
    records = [
        ActivationRecord(tokens=["a"], activations=[1.0]),
        ActivationRecord(tokens=["b"], activations=[9.0]),
        ActivationRecord(tokens=["c"], activations=[5.0]),
    ]
    state, questions = build_holistic_request("x", records, max_examples=2)
    assert state["examples"] == ["<<b>>", "<<c>>"]
    assert questions["rating"]["type"] == "score"
    assert questions["rating"]["criteria"] == JEV_SCORE_LEVELS


def test_build_logit_fit_request() -> None:
    built = build_logit_fit_request("dogs", ["Ġdog", "▁dogs", " ", "x"] + ["y"] * 20)
    assert built is not None
    state, questions, logits = built
    # Blank logits are dropped and the list is capped at 10 before dropping.
    assert logits == [" dog", " dogs", "x"] + ["y"] * 6
    assert state == {"explanation": "dogs", "top_output_tokens": logits}
    assert questions["fit"]["type"] == "noul"
    assert build_logit_fit_request("dogs", [" ", ""]) is None


class FakeClient(JevClient):
    """Answers yes for texts that contain 'dog', so activating records score correctly."""

    def __init__(self) -> None:
        super().__init__(api_key="test")
        self.calls: list[dict[str, Any]] = []

    async def ask(
        self, state: dict[str, Any], questions: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
        self.calls.append({"state": state, "questions": questions})
        answers: dict[str, Any] = {}
        for key, question in questions.items():
            if question["type"] == "score":
                # `score` is the probability-weighted level, as the real API returns it.
                answers[key] = {
                    "type": "score",
                    "score": 2.7,
                    "probabilities": {"0": 0.0, "1": 0.1, "2": 0.2, "3": 0.6, "4": 0.1},
                    "confidence": 0.6,
                }
            elif key == "fit":
                answers[key] = {"type": "noul", "noul": 0.9}
            else:
                i = int(key.split("_")[1])
                answers[key] = {
                    "type": "noul",
                    "noul": 0.95 if "dog" in state["examples"][i] else 0.05,
                }
        return {
            "model": "jev-test",
            "answers": answers,
            "usage": {"input_tokens": 123, "output_tokens": 0},
        }


def test_scorer_detection_end_to_end() -> None:
    client = FakeClient()
    scorer = JevScorer(client=client)
    result = asyncio.run(
        scorer.score(
            JevScoreType.DETECTION,
            "dogs",
            DOG_RECORDS,
            ZERO_RECORDS,
            top_logits=["Ġdog"],
        )
    )
    assert result.score == 1.0
    assert result.score_type == "detection"
    assert result.model == "jev-test" and result.input_tokens == 123
    assert len(result.example_results) == 8
    assert all(r.correct for r in result.example_results)
    assert result.logit_fit is not None and result.logit_fit.noul == 0.9
    assert len(client.calls) == 2  # one classification call, one logit-fit call

    # Round-trips through the fast_dataclasses serializer like ScoredSimulation does.
    restored = loads(dumps(result))
    assert isinstance(restored, JevScoredExplanation)
    assert restored.score == 1.0 and restored.example_results[0].kind == "activating"
    assert restored.logit_fit is not None and restored.logit_fit.top_logits == [" dog"]


def test_scorer_holistic_end_to_end() -> None:
    scorer = JevScorer(client=FakeClient())
    result = asyncio.run(scorer.score(JevScoreType.HOLISTIC, "dogs", DOG_RECORDS))
    assert result.score == 2.7 / 4
    assert result.expected_level == 2.7
    assert result.top_level == 3
    assert result.level_probabilities == [0.0, 0.1, 0.2, 0.6, 0.1]
    assert result.confidence == 0.6
    assert result.example_results == [] and result.logit_fit is None


def test_scorer_rejects_no_activating_records() -> None:
    scorer = JevScorer(client=FakeClient())
    try:
        asyncio.run(scorer.score(JevScoreType.FUZZ, "dogs", []))
    except ValueError:
        return
    raise AssertionError("expected ValueError")
