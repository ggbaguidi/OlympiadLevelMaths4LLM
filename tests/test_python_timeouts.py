from olympiad_llm.aimo3.python_timeouts import parse_timeout_directive, parse_timeout_error


def test_parse_timeout_directive_basic():
    assert parse_timeout_directive("# timeout: 120\nprint(1)") == 120.0
    assert parse_timeout_directive("#timeout=60\nprint(1)") == 60.0


def test_parse_timeout_directive_only_first_nonempty_line():
    # If the first non-empty line is not a directive, we don't scan further.
    assert parse_timeout_directive("print(1)\n# timeout: 120") is None


def test_parse_timeout_error():
    assert parse_timeout_error("[ERROR] Execution timed out after 60.0 seconds") == 60.0
    assert parse_timeout_error("Execution timed out after 5 seconds") == 5.0
    assert parse_timeout_error("no timeout") is None
