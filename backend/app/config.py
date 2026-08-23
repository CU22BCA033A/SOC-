from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BACKEND_DIR / ".env", extra="ignore")

    # Which LLM backend answers chat turns: "anthropic" (default, what this
    # project is tuned for) or "nvidia" (free-tier open models via NIM).
    llm_provider: str = "anthropic"

    anthropic_api_key: str = ""
    agent_model: str = "claude-sonnet-4-6"

    # Only used when llm_provider = "nvidia". Get a free key at build.nvidia.com.
    nvidia_api_key: str = ""
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    # An 8B model, not NVIDIA's largest, because it responds noticeably faster
    # on the free tier -- every chat turn is at most one LLM call (see
    # app/agent/prompts.py), so model latency directly is turn latency. Trade
    # a little quality for speed; bump to a 70B model in .env if you'd rather
    # go the other way.
    nvidia_model: str = "meta/llama-3.1-8b-instruct"

    database_url: str = f"sqlite:///{BACKEND_DIR / 'support_agent.db'}"
    chroma_dir: str = str(BACKEND_DIR / "chroma_data")
    knowledge_base_dir: str = str(BACKEND_DIR / "knowledge_base")

    retrieval_confidence_threshold: float = 0.35
    retrieval_top_k: int = 4

    cors_origins: str = "http://localhost:5173,http://localhost:3000"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
