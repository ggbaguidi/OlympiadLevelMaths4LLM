from olympiad_llm.aimo3.wickelgren import augment_system_prompt, select_strategy


def test_select_strategy_cycles():
    a = select_strategy(0).name
    b = select_strategy(1).name
    assert a != b


# def test_augment_system_prompt_appends_card():
#     base = "BASE PROMPT"
#     out = augment_system_prompt(base, attempt_index=0)
#     assert out.startswith("BASE PROMPT")
#     assert "strategy card" in out.lower()
