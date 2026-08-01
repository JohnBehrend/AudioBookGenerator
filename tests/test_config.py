"""Tests for the centralized configuration module."""

from audiobook_generator import config


class TestSettingsShape:
    def test_llm_settings_has_required_keys(self):
        for key in ("endpoint", "port", "api_key", "default_model"):
            assert key in config.LLM_SETTINGS

    def test_audio_settings_has_required_keys(self):
        for key in ("default_device", "default_tts_engine", "gradio_port"):
            assert key in config.AUDIO_SETTINGS

    def test_defaults_has_audio_generation_keys(self):
        for key in ("short_text_postfix", "inter_line_pause_ms", "enable_postfix", "max_retries"):
            assert key in config.DEFAULTS

    def test_get_llm_port_returns_int(self):
        assert isinstance(config.get_llm_port(), int)


class TestValidate:
    def test_returns_list(self):
        assert isinstance(config.validate(), list)

    def test_warns_unknown_engine(self, monkeypatch):
        monkeypatch.setitem(config.AUDIO_SETTINGS, "default_tts_engine", "not-a-real-engine")
        warnings = config.validate()
        assert any("Unknown TTS engine" in w for w in warnings)

    def test_no_unknown_engine_warning_for_valid_engine(self, monkeypatch):
        monkeypatch.setitem(config.AUDIO_SETTINGS, "default_tts_engine", "omni")
        warnings = config.validate()
        assert not any("Unknown TTS engine" in w for w in warnings)
