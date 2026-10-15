"""Configuration and environment management for ConvoGuard."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables from .env if present
load_dotenv()


class ProjectConfig(BaseModel):
    """Project-wide settings and paths."""

    project_root: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent.parent.parent
    )
    data_root: Path = Field(
        default_factory=lambda: Path(
            os.getenv("CONVOGUARD_DATA_ROOT")
            or os.getenv("DATA_ROOT")
            or "data/raw"
        )
    )
    artifacts_dir: Path = Field(
        default_factory=lambda: Path(
            os.getenv("CONVOGUARD_ARTIFACTS_DIR")
            or os.getenv("ARTIFACTS_DIR")
            or "artifacts"
        )
    )
    log_level: str = Field(
        default_factory=lambda: os.getenv("LOG_LEVEL", "INFO")
    )

    def resolve_data_root(self, candidate_path: str | Path | None = None) -> Path:
        """Resolve data root path, checking candidate first, then config, then fallback paths."""
        if candidate_path:
            p = Path(candidate_path)
            if p.is_absolute():
                return p
            return (self.project_root / p).resolve() if not p.exists() else p.resolve()

        if self.data_root.is_absolute() and self.data_root.exists():
            return self.data_root

        # Check relative to project root
        candidate = (self.project_root / self.data_root).resolve()
        if candidate.exists():
            return candidate

        return self.data_root


def get_config() -> ProjectConfig:
    """Instantiate and return the project configuration."""
    return ProjectConfig()
