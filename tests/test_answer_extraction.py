from olympiad_llm.aimo3.answer_extraction import AnswerExtractor


def test_extract_boxed_int_last_one_wins():
    ex = AnswerExtractor(aimo_lo=0, aimo_hi=99999)
    txt = "blah \\boxed{12} and later \\boxed{1,234}"
    assert ex.extract_boxed_int(txt) == 1234


def test_extract_boxed_int_handles_text_wrapper_and_spacing():
    ex = AnswerExtractor(aimo_lo=0, aimo_hi=99999)
    txt = "Answer is \\boxed{\\text{\\,1,234\\,}}."
    assert ex.extract_boxed_int(txt) == 1234


def test_extract_boxed_int_handles_nested_braces():
    ex = AnswerExtractor(aimo_lo=0, aimo_hi=99999)
    txt = "We conclude \\boxed{{1234}}."
    assert ex.extract_boxed_int(txt) == 1234


def test_extract_boxed_content_handles_nested_braces():
    ex = AnswerExtractor()
    assert ex.extract_boxed_content("X=\\boxed{{x^2+1}}") == "{x^2+1}"


def test_extract_boxed_int_out_of_range_none():
    ex = AnswerExtractor(aimo_lo=0, aimo_hi=99999)
    assert ex.extract_boxed_int("\\boxed{100000}") is None


def test_normalize_final_answer_prefers_boxed_content():
    ex = AnswerExtractor()
    assert ex.normalize_final_answer_text("Answer is \\boxed{x^2+1}.") == "x^2+1"


def test_extract_int_fallback_prefers_answer_hint():
    ex = AnswerExtractor(aimo_lo=0, aimo_hi=99999)
    txt = "We found 3 lemmas. Final answer: 1234."
    assert ex.extract_int_fallback(txt) == 1234


def test_extract_int_fallback_handles_dollar_math_wrapper():
    ex = AnswerExtractor(aimo_lo=0, aimo_hi=99999)
    txt = "Final answer is $1,234$. (Earlier we mentioned 99999.)"
    assert ex.extract_int_fallback(txt) == 1234


def test_extract_int_fallback_handles_bold_wrapper():
    ex = AnswerExtractor(aimo_lo=0, aimo_hi=99999)
    txt = "Final answer is **1234**. Ignore 777 later in the writeup. 777"
    assert ex.extract_int_fallback(txt) == 1234


def test_extract_int_fallback_handles_paren_math_wrapper():
    ex = AnswerExtractor(aimo_lo=0, aimo_hi=99999)
    txt = "final answer is \\((1234)\\)."
    assert ex.extract_int_fallback(txt) == 1234


def test_extract_int_fallback_last_int_in_range():
    ex = AnswerExtractor(aimo_lo=0, aimo_hi=99999)
    txt = "Some numbers 12 99 100000 and then 777"
    assert ex.extract_int_fallback(txt) == 777

def test_extract_int_fallback_no_valid_int_none():
    ex = AnswerExtractor(aimo_lo=0, aimo_hi=50)
    txt = "Some numbers 60 99 100000 and then 77"
    assert ex.extract_int_fallback(txt) is None


def test_extract_int_fallback_handles_paren_math_wrapper_():
    ex = AnswerExtractor(aimo_lo=0, aimo_hi=99999)
    txt = "Thus, the answer is 8687."
    assert ex.extract_int_fallback(txt) == 8687


def test_extract_int_fallback_handles_paren_math_wrapper_v():
    ex = AnswerExtractor(aimo_lo=0, aimo_hi=99999)
    txt = "Thus answer is 8687."
    assert ex.extract_int_fallback(txt) == 8687
