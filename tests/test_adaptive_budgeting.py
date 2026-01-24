from olympiad_llm.aimo3.budget import adaptive_verify_budget, reserve_fraction_for_budget


def test_reserve_fraction_for_budget_short_increases_reserve():
    # Tight budget: should be at least the max_fraction default (0.30)
    f = reserve_fraction_for_budget(budget_s=60.0, base_fraction=0.15)
    assert f >= 0.30


def test_reserve_fraction_for_budget_long_decreases_reserve():
    # Large budget: should be at most min_fraction default (0.10)
    f = reserve_fraction_for_budget(budget_s=1000.0, base_fraction=0.15)
    assert f <= 0.10


def test_reserve_fraction_for_budget_mid_keeps_base():
    f = reserve_fraction_for_budget(budget_s=300.0, base_fraction=0.15)
    assert abs(f - 0.15) < 1e-9


def test_adaptive_verify_budget_respects_cap_and_remaining():
    b = adaptive_verify_budget(remaining_s=10.0, base_fraction=0.9, cap_s=100.0, multiplier=2.0)
    assert abs(b - 10.0) < 1e-9


def test_adaptive_verify_budget_applies_multiplier_and_cap():
    # remaining 100, base_fraction 0.2 => 20; multiplier 1.5 => 30; cap 25 => 25
    b = adaptive_verify_budget(remaining_s=100.0, base_fraction=0.2, cap_s=25.0, multiplier=1.5)
    assert abs(b - 25.0) < 1e-9
