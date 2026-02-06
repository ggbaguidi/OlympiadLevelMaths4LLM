from olympiad_llm.aimo3.wickelgren import (
    augment_system_prompt_with_meta,
    detect_fe_combi,
    select_strategy_pack,
)


def test_detect_fe_combi_cues():
    assert detect_fe_combi("Find all functions f such that f(x+y)=f(x)+f(y)") is True
    assert detect_fe_combi("How many subsets of {1,2,3} have even size?") is True
    assert detect_fe_combi("Compute the integral from 0 to 1 of x^2 dx") is False


def test_select_strategy_pack_round_robin():
    packs = "generic,fe_combi"
    p0 = select_strategy_pack(
        attempt_index=0, problem_text=None, mode="round_robin", enabled_packs=packs
    )
    p1 = select_strategy_pack(
        attempt_index=1, problem_text=None, mode="round_robin", enabled_packs=packs
    )
    assert p0 != p1


def test_select_strategy_pack_auto_uses_fe_combi_when_detected():
    packs = "generic,fe_combi"
    p = select_strategy_pack(
        attempt_index=0,
        problem_text="Find all functions f: Z -> Z such that f(x+y)=f(x)f(y)",
        mode="auto",
        enabled_packs=packs,
    )
    assert p == "fe_combi"


# def test_augment_system_prompt_with_meta_returns_pack_and_card():
#     out, meta = augment_system_prompt_with_meta(
#         "BASE",
#         attempt_index=0,
#         problem_text="Find all functions f such that ...",
#         mode="auto",
#         enabled_packs="generic,fe_combi",
#     )
#     assert out.startswith("BASE")
#     assert "strategy card" in out.lower()
#     assert "pack" in meta and "card" in meta
