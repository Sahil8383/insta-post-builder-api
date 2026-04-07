"""Application settings (replaces Django config/settings.py)."""

from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()

_BACKEND_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    debug: bool = True

    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-20250514"

    tavily_api_key: str = ""

    openai_api_key: str = ""
    openai_image_model: str = "dall-e-3"

    pexels_api_key: str = ""

    usage_anthropic_input_per_mtok_usd: float = 3.0
    usage_anthropic_output_per_mtok_usd: float = 15.0
    usage_openai_image_per_call_usd: float = 0.04
    usage_tavily_per_search_usd: float = 0.008
    usage_pexels_per_request_usd: float = 0.0

    @property
    def data_dir(self) -> Path:
        d = _BACKEND_ROOT / "data"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def database_url(self) -> str:
        p = (self.data_dir / "db.sqlite3").resolve()
        return f"sqlite:///{p.as_posix()}"


@lru_cache
def get_settings() -> Settings:
    return Settings()
