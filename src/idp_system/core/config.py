"""Configuration placeholders for the IDP system."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Settings:
    """Minimal settings container for Phase 1."""

    project_name: str = "Intelligent Document Processing System"
    data_dir: Path = Path("data")
    model_dir: Path = Path("models")


settings = Settings()
