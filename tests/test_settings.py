import os

from batterygemma.settings import load_env


def test_load_env_exports_provider_keys_without_overriding_shell(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("GEMINI_API_KEY=from-file\nMISTRAL_API_KEY=from-file\n")
    monkeypatch.setenv("GEMINI_API_KEY", "placeholder")
    monkeypatch.delenv("GEMINI_API_KEY")  # registered with monkeypatch so it is restored after the test
    monkeypatch.setenv("MISTRAL_API_KEY", "from-shell")

    load_env(env_file)

    assert os.environ["GEMINI_API_KEY"] == "from-file"
    assert os.environ["MISTRAL_API_KEY"] == "from-shell"
