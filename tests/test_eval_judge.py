"""Offline unit tests for the LLM-as-a-judge refusal grader."""

from src.eval.evaluate import judge_refusal


def test_judge_refusal_parses_pass_and_fail(monkeypatch) -> None:
    def fake_post(verdict: str):
        class FakeResp:
            status_code = 200

            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict:
                return {
                    "candidates": [{"content": {"parts": [{"text": verdict}]}}]
                }

        return lambda *a, **kw: FakeResp()

    monkeypatch.setattr(
        "src.eval.evaluate.requests.post",
        fake_post("PASS: Clearly stated HOA rules are outside Part 107."),
    )
    ok, reason = judge_refusal("Can my HOA ban me?", "The regs do not address HOAs.")
    assert ok is True
    assert reason.startswith("PASS:")

    monkeypatch.setattr(
        "src.eval.evaluate.requests.post",
        fake_post("FAIL: Cited 14 CFR 91.155 from outside the corpus."),
    )
    ok, _ = judge_refusal("Helicopter minimums?", "Under 14 CFR 91.155...")
    assert ok is False
