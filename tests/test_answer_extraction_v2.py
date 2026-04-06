from olympiad_llm.aimo3.v2.answer_extraction import AnswerExtractor
import pytest


def test_extract_int_fallback_parses_displaystyle_final_patterns() -> None:
    extractor = AnswerExtractor(strict_fallback=True)
    text = (
        "[ASSISTANT_RAW]\n"
        "final\\(\\displaystyle 98449\\)\n\n"
        "[ASSISTANT_FINAL]\n"
        "\\(\\displaystyle 98449\\)"
    )
    assert extractor.extract_int_fallback(text) == 98449


def test_extract_int_fallback_parses_plain_latex_wrapped_int() -> None:
    extractor = AnswerExtractor(strict_fallback=True)
    text = "some derivation... \\(\\displaystyle 12,345\\)"
    assert extractor.extract_int_fallback(text) == 12345


def test_extract_int_fallback_rejects_out_of_range_latex_wrapped_int() -> None:
    extractor = AnswerExtractor(strict_fallback=True)
    text = "\\(\\displaystyle 100000\\)"
    assert extractor.extract_int_fallback(text) is None


def test_extract_int_fallback_parses_thus_answer_should_be_phrase() -> None:
    extractor = AnswerExtractor(strict_fallback=True)
    text = "Thus the answer should be 8687."
    assert extractor.extract_int_fallback(text) == 8687


@pytest.mark.parametrize(
    "text,expected",
    [
        (r"\(\text{98449}\)", 98449),
        (r"\(\mathrm{98449}\)", 98449),
        (r"\[\displaystyle 98449\]", 98449),
        (r"$\displaystyle 98449$", 98449),
        (r"$98449$", 98449),
        (r"\(\,98449\,\)", 98449),
        (r"\(\left(98449\right)\)", 98449),
        (r"\(\operatorname{Ans}(98449)\)", 98449),
        (r"\(\displaystyle 98449.0\)", 98449),
        (r"\(\displaystyle \frac{196898}{2}\)", 98449),
        (r"\(\displaystyle 98{,}449\)", 98449),
        (r"\text{Final answer: }98449", 98449),
    ],
)
def test_extract_int_fallback_handles_common_latex_styles(
    text: str, expected: int
) -> None:
    extractor = AnswerExtractor(strict_fallback=True)
    assert extractor.extract_int_fallback(text) == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        (r"\boxed{\displaystyle 98449}", 98449),
        (r"\boxed{\text{98449}}", 98449),
        (r"\fbox{98449}", 98449),
        (r"\mbox{98449}", 98449),
        (r"\boxed{98{,}449}", 98449),
        (r"\boxed{+98449}", 98449),
        (r"\boxed{098449}", 98449),
        (r"\boxed{98449.0}", 98449),
        (r"\boxed{\frac{196898}{2}}", 98449),
    ],
)
def test_extract_boxed_int_handles_common_boxed_styles(
    text: str, expected: int
) -> None:
    extractor = AnswerExtractor(strict_fallback=True)
    assert extractor.extract_boxed_int(text) == expected


def test_extract_with_rule_reports_boxed_rule() -> None:
    extractor = AnswerExtractor(strict_fallback=True)
    val, rule = extractor.extract_boxed_int_with_rule(r"\boxed{\frac{196898}{2}}")
    assert val == 98449
    assert rule == "boxed:simple_fraction"


def test_extract_with_rule_reports_fallback_rule() -> None:
    extractor = AnswerExtractor(strict_fallback=True)
    val, rule = extractor.extract_int_fallback_with_rule(
        r"[ASSISTANT_FINAL] \(\displaystyle 98449\)"
    )
    assert val == 98449
    assert rule is not None
    assert rule.startswith("fallback:")
