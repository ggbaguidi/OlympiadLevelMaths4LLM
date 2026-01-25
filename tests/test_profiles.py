def test_apply_profile_does_not_override_by_default():
    from olympiad_llm.aimo3.profiles import apply_profile

    env = {"AIMO3_PROFILE": "full", "AIMO3_SANDBOX_POOL_SIZE": "9"}
    applied = apply_profile("lean", env=env, force=False)

    # Existing keys preserved
    assert env["AIMO3_PROFILE"] == "full"
    assert env["AIMO3_SANDBOX_POOL_SIZE"] == "9"
    # New keys can still be added
    assert "AIMO3_PYTHON_TOOL_TIMEOUT_CAP_S" in env
    assert "AIMO3_PYTHON_TOOL_TIMEOUT_CAP_S" in applied


def test_apply_profile_force_overrides():
    from olympiad_llm.aimo3.profiles import apply_profile

    env = {"AIMO3_PROFILE": "full", "AIMO3_SANDBOX_POOL_SIZE": "9"}
    applied = apply_profile("lean", env=env, force=True)

    assert env["AIMO3_PROFILE"] == "lean"
    assert env["AIMO3_SANDBOX_POOL_SIZE"] == "2"
    assert applied["AIMO3_PROFILE"] == "lean"


def test_profile_env_unknown_raises():
    from olympiad_llm.aimo3.profiles import profile_env

    try:
        profile_env("nope")
        assert False, "expected ValueError"
    except ValueError:
        pass
