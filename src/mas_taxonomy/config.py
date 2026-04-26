from __future__ import annotations #can help with forward references and speed.

from pathlib import Path #pathlib is a module that provides a class for representing file paths in a platform-independent manner.
from pydantic_settings import BaseSettings, SettingsConfigDict #Imports Pydantic Settings base class and config model, used to build settings from environment variables (and optionally .env).
from pydantic import Field

#defines the paths for the project.
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8") #configures the settings to load from .env file and encoding.

    project_root: Path = Path(__file__).resolve().parents[2] #resolves the path to the project root.

    data_dir: Path = project_root / "data"
    input_pdfs_dir: Path = data_dir / "input_pdfs"
    extracted_text_dir: Path = data_dir / "extracted_text"
    runs_dir: Path = data_dir / "runs"

    # API keys (all optional – only the one for the chosen provider is required)
    openai_api_key: str = Field(default="", validation_alias="OPENAI_API_KEY")
    anthropic_api_key: str = Field(default="", validation_alias="ANTHROPIC_API_KEY")
    gemini_api_key: str = Field(default="", validation_alias="GEMINI_API_KEY")

    # Google Vertex AI settings (alternative to gemini API key – uses service account auth)
    vertex_project: str = Field(default="", validation_alias="VERTEX_PROJECT")
    vertex_location: str = Field(default="us-central1", validation_alias="VERTEX_LOCATION")
    # Path to service account JSON key file; if set, used as GOOGLE_APPLICATION_CREDENTIALS
    vertex_credentials_file: str = Field(default="", validation_alias="VERTEX_CREDENTIALS_FILE")

    # Provider priority – comma-separated, e.g. "openai,anthropic,gemini,vertexai"
    llm_provider_priority: str = Field(
        default="openai,anthropic,gemini,vertexai",
        validation_alias="LLM_PROVIDER_PRIORITY",
    )

    # Default models per provider
    default_model_openai: str = "gpt-5.4-2026-03-05"
    default_model_anthropic: str = "claude-haiku-4-5"
    default_model_gemini: str = "gemini-2.5-pro"
    default_model_vertexai: str = "gemini-2.5-flash"

    max_chars_per_doc: int = Field(default=100000, description="Maximum characters per document to send to LLM")
    debug_mode: bool = Field(default=False, description="Enable debug file dumps (prompt, tool schema) per LLM call", validation_alias="DEBUG_MODE")


def get_settings() -> Settings: #function which creates the settings object and ensures that the data directories exist.
    s = Settings()
    s.data_dir.mkdir(parents=True, exist_ok=True)
    s.input_pdfs_dir.mkdir(parents=True, exist_ok=True)
    s.extracted_text_dir.mkdir(parents=True, exist_ok=True)
    s.runs_dir.mkdir(parents=True, exist_ok=True)
    return s
