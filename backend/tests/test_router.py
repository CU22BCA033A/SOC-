from unittest.mock import patch

import pytest

from app.agent.errors import AgentError
from app.agent.router import generate_general_answer, generate_grounded_answer
from app.config import get_settings


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_routes_to_anthropic_by_default(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    with patch("app.agent.router.claude_client.generate_grounded_answer") as mock_claude:
        mock_claude.return_value = "claude answer"
        result = generate_grounded_answer("q", [], [])
    assert result == "claude answer"
    mock_claude.assert_called_once_with("q", [], [])


def test_routes_to_nvidia_when_configured(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "nvidia")
    with patch("app.agent.router.nvidia_client.generate_grounded_answer") as mock_nvidia:
        mock_nvidia.return_value = "nvidia answer"
        result = generate_grounded_answer("q", [], [])
    assert result == "nvidia answer"
    mock_nvidia.assert_called_once_with("q", [], [])


def test_unknown_provider_raises_agent_error(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "not-a-real-provider")
    with pytest.raises(AgentError):
        generate_grounded_answer("q", [], [])


def test_routes_general_answer_to_configured_provider(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    with patch("app.agent.router.claude_client.generate_general_answer") as mock_general:
        mock_general.return_value = "Paris."
        assert generate_general_answer("capital of France?", []) == "Paris."
    mock_general.assert_called_once_with("capital of France?", [])


def test_routes_general_answer_none_signals_escalation(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "nvidia")
    with patch("app.agent.router.nvidia_client.generate_general_answer") as mock_general:
        mock_general.return_value = None
        assert generate_general_answer("what's our refund policy?", []) is None
