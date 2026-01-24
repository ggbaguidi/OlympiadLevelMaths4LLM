from olympiad_llm.aimo3.answer_extraction import AnswerExtractor


def test_extract_boxed_int_last_one_wins():
    ex = AnswerExtractor(aimo_lo=0, aimo_hi=99999)
    txt = "blah \\boxed{12} and later \\boxed{1,234}"
    assert ex.extract_boxed_int(txt) == 1234


def test_extract_boxed_int_out_of_range_none():
    ex = AnswerExtractor(aimo_lo=0, aimo_hi=99999)
    assert ex.extract_boxed_int("\\boxed{100000}") is None


def test_normalize_final_answer_prefers_boxed_content():
    ex = AnswerExtractor()
    assert ex.normalize_final_answer_text("Answer is \\boxed{x^2+1}.") == "x^2+1"
