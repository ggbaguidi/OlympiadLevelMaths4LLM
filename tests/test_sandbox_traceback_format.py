from olympiad_llm.aimo3.sandbox import AIMO3Sandbox


def test_format_error_keeps_ipykernel_frames():
    tb = [
        "---------------------------------------------------------------------------\n",
        "NameError                                 Traceback (most recent call last)\n",
        'File "/tmp/ipykernel_10164/4047503851.py", line 1\n',
        "----> 1 print(even_set)\n",
        "NameError: name 'even_set' is not defined\n",
    ]
    out = AIMO3Sandbox._format_error(tb)
    assert "/tmp/ipykernel_" in out
    assert "NameError" in out


def test_format_error_falls_back_if_everything_filtered():
    tb = [
        'File "/some/other/path.py", line 1\n',
        "ValueError: boom\n",
    ]
    out = AIMO3Sandbox._format_error(tb)
    assert "ValueError" in out
