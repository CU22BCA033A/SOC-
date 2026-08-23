"""Picks which LLM backend answers a turn, based on LLM_PROVIDER.

Callers (app/api/chat.py) import from here rather than a specific client
module, so switching providers is a config change, not a code change.
"""

from app.agent import claude_client, nvidia_client
from app.agent.errors import AgentError
from app.config import get_settings

__all__ = ["AgentError", "generate_general_answer", "generate_grounded_answer"]

_PROVIDER_MODULES = {
    "anthropic": claude_client,
    "nvidia": nvidia_client,
}


def _module():
    settings = get_settings()
    provider = settings.llm_provider.lower()
    module = _PROVIDER_MODULES.get(provider)
    if module is None:
        raise AgentError(
            f"Unknown LLM_PROVIDER '{settings.llm_provider}' — expected 'anthropic' or 'nvidia'."
        )
    return module


def generate_grounded_answer(
    question: str,
    history: list[dict[str, str]],
    retrieved_chunks: list[dict],
) -> str:
    # Looked up on the module at call time (not bound at import time) so that
    # patching module.generate_grounded_answer in tests takes effect.
    return _module().generate_grounded_answer(question, history, retrieved_chunks)


def generate_general_answer(question: str, history: list[dict[str, str]]) -> str | None:
    """Returns None when the model determined this needs to escalate instead."""
    return _module().generate_general_answer(question, history)
