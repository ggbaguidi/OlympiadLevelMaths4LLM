from olympiad_llm.aimo3.v2.answer_extraction import AnswerExtractor


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
