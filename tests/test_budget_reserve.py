from olympiad_llm.aimo3.budget import compute_attempt_and_verify_deadlines


def test_reserve_fraction_applied():
    now = 100.0
    overall = 200.0  # 100s remaining
    attempt_deadline, overall_deadline = compute_attempt_and_verify_deadlines(
        now=now,
        overall_deadline=overall,
        reserve_fraction=0.2,
        reserve_cap_s=999.0,
        reserve_min_s=0.0,
    )
    assert overall_deadline == overall
    # reserve 20s => attempt_deadline 180
    assert abs(attempt_deadline - 180.0) < 1e-9


def test_reserve_capped_and_minimum():
    now = 0.0
    overall = 100.0

    # fraction would reserve 50s, but cap to 10s
    attempt_deadline, _ = compute_attempt_and_verify_deadlines(
        now=now,
        overall_deadline=overall,
        reserve_fraction=0.5,
        reserve_cap_s=10.0,
        reserve_min_s=0.0,
    )
    assert abs(attempt_deadline - 90.0) < 1e-9

    # fraction would reserve 1s, but min to 7s
    attempt_deadline, _ = compute_attempt_and_verify_deadlines(
        now=now,
        overall_deadline=overall,
        reserve_fraction=0.01,
        reserve_cap_s=999.0,
        reserve_min_s=7.0,
    )
    assert abs(attempt_deadline - 93.0) < 1e-9
