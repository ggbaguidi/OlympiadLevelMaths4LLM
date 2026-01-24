import os


def test_cuda_visible_devices_disable(monkeypatch):
    from olympiad_llm.aimo3.vllm_server import VLLMServer

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert VLLMServer._cuda_visible_devices_allows_gpu() is False

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    assert VLLMServer._cuda_visible_devices_allows_gpu() is False

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    assert VLLMServer._cuda_visible_devices_allows_gpu() is True


def test_cuda_driver_present_fast_checks(monkeypatch):
    from olympiad_llm.aimo3.vllm_server import VLLMServer

    # Force-disable via CUDA_VISIBLE_DEVICES => should return False.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    assert VLLMServer._cuda_driver_present() is False


def test_vllm_start_fails_fast_without_cuda(monkeypatch, tmp_path):
    from olympiad_llm.aimo3.config import AIMO3Config
    from olympiad_llm.aimo3.vllm_server import VLLMServer

    # Ensure our fast check concludes "no CUDA" in most CI/CPU environments.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")

    cfg = AIMO3Config(model_path="/some/model", require_cuda=True)
    s = VLLMServer(cfg=cfg, port=8000, log_path=str(tmp_path / "vllm.log"))

    try:
        s.start()
        assert False, "Expected start() to fail fast when CUDA is not detected"
    except RuntimeError as e:
        msg = str(e)
        assert "CUDA/NVIDIA driver not detected" in msg
        assert "AIMO3_REQUIRE_CUDA" in msg
