"""
Centralized Configuration for GuardHier Benchmark.
Enforces deterministic evaluation: temperature=0.0, seed=42.
Supports OpenAI, Azure, Groq, or local vLLM endpoints with fallback parsing
from TalkHier's config_llm.ini.
"""

import configparser
import os
import random
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field

# Research Constants
TEMPERATURE: float = 0.0
SEED: int = 42
DEFAULT_NUM_MEMBERS: int = 4
COMPROMISED_COUNT: int = 1
TRUST_THRESHOLD: float = 0.55
QUARANTINE_STRIKES: int = 2

# Seed randomness globally for strict reproducibility
random.seed(SEED)
try:
    import numpy as np
    np.random.seed(SEED)
except ImportError:
    pass

try:
    import torch
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
except ImportError:
    pass


def _load_talkhier_ini_fallback() -> dict:
    """Parses TalkHier's config_llm.ini if available as a fallback."""
    workspace_root = Path(__file__).resolve().parents[2]
    candidates = [
        workspace_root / "talkhier" / "config" / "config_llm.ini",
        workspace_root / "talkhier" / "template" / "config_llm.ini",
    ]
    config = configparser.ConfigParser()
    for c in candidates:
        if c.exists():
            try:
                config.read(c, encoding="utf-8")
                api_key = config.get("KEYS", "OPENAI_API_KEY", fallback="")
                endpoint = config.get("KEYS", "OPENAI_ENDPOINT", fallback="")
                deployment = config.get("LLM", "deployment_name", fallback="")
                if api_key and "******" not in api_key:
                    return {
                        "api_key": api_key,
                        "base_url": endpoint or "https://api.openai.com/v1",
                        "model_name": deployment or "gpt-4o-mini",
                    }
            except Exception:
                pass
    return {}


class Settings(BaseModel):
    """Central configuration for LLM endpoints, datasets, defenses, and paths."""

    # Fixed Determinism Research Constants
    temperature: float = Field(default=TEMPERATURE, description="Strict temperature 0.0 for deterministic evaluation")
    seed: int = Field(default=SEED, description="Fixed random seed 42")
    default_num_members: int = Field(default=DEFAULT_NUM_MEMBERS, description="Default number of agents in team")
    compromised_count: int = Field(default=COMPROMISED_COUNT, description="Number of compromised member agents")
    trust_threshold: float = Field(default=TRUST_THRESHOLD, description="Centroid alignment threshold for outlier detection")
    quarantine_strikes: int = Field(default=QUARANTINE_STRIKES, description="Consecutive violation strikes triggering quarantine")

    # LLM Endpoint Configuration
    openai_api_key: str = Field(
        default_factory=lambda: os.getenv(
            "OPENAI_API_KEY",
            _load_talkhier_ini_fallback().get("api_key", "mock-api-key")
        ),
        description="API key for OpenAI-compatible endpoint"
    )
    openai_base_url: str = Field(
        default_factory=lambda: os.getenv(
            "OPENAI_BASE_URL",
            _load_talkhier_ini_fallback().get("base_url", "https://api.openai.com/v1")
        ),
        description="Base URL for OpenAI-compatible endpoint (Groq/vLLM/OpenAI/Azure)"
    )
    openai_model_name: str = Field(
        default_factory=lambda: os.getenv(
            "MODEL_NAME",
            os.getenv("OPENAI_MODEL_NAME", _load_talkhier_ini_fallback().get("model_name", "gpt-4o-mini"))
        ),
        description="Model name to evaluate"
    )
    max_tokens: int = Field(default=512, description="Maximum completion tokens")
    timeout_seconds: float = Field(default=60.0, description="API request timeout")

    # Defense Parameters
    embedding_model_name: str = Field(
        default="BAAI/bge-small-en-v1.5",
        description="Sentence embedding model for C2 trust scoring"
    )

    # Repository & Dataset Paths
    workspace_root: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parents[2],
        description="Root workspace directory containing clean baselines"
    )

    @property
    def talkhier_path(self) -> Path:
        return self.workspace_root / "talkhier"

    @property
    def cracks_path(self) -> Path:
        return self.workspace_root / "Cracks-in-Agent-Collaboration"

    @property
    def qa_data_path(self) -> Path:
        return self.cracks_path / "qa-data.json"

    @property
    def medqa_data_path(self) -> Path:
        return self.cracks_path / "medalpaca-medical-meadow-medqa-500.json"

    @property
    def results_dir(self) -> Path:
        res_dir = self.workspace_root / "guardhier_bench" / "results"
        res_dir.mkdir(parents=True, exist_ok=True)
        return res_dir

    @property
    def figures_dir(self) -> Path:
        fig_dir = self.results_dir / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        return fig_dir

    @property
    def traces_path(self) -> Path:
        return self.results_dir / "traces.jsonl"


_GLOBAL_SETTINGS: Optional[Settings] = None


def get_settings() -> Settings:
    """Singleton getter for centralized settings."""
    global _GLOBAL_SETTINGS
    if _GLOBAL_SETTINGS is None:
        _GLOBAL_SETTINGS = Settings()
    return _GLOBAL_SETTINGS
