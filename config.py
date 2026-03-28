"""
config.py – Configuration loader for PlotWeaver V2.0.

Priority order (highest → lowest):
  1. Environment variables
  2. config.yaml values
"""

import os
import yaml
from pathlib import Path

_CONFIG_FILE = Path(__file__).parent / "config.yaml"


def _load_yaml() -> dict:
    if _CONFIG_FILE.exists():
        with open(_CONFIG_FILE, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


_yaml = _load_yaml()


def _get(section: str, key: str, env_var: str, default=None):
    """Return env var if set, else yaml value, else default."""
    env_val = os.environ.get(env_var)
    if env_val is not None:
        return env_val
    return _yaml.get(section, {}).get(key, default)


# ── DeepSeek ──────────────────────────────────────────────────────────────────
DEEPSEEK_API_KEY: str = _get("deepseek", "api_key", "DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL: str = _get("deepseek", "base_url", "DEEPSEEK_BASE_URL",
                               "https://api.deepseek.com/v1")
DEEPSEEK_MODEL: str = _get("deepseek", "model", "DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_MAX_TOKENS: int = int(
    _get("deepseek", "max_tokens", "DEEPSEEK_MAX_TOKENS", 8192)
)
DEEPSEEK_TEMPERATURE: float = float(
    _get("deepseek", "temperature", "DEEPSEEK_TEMPERATURE", 0.7)
)

# ── ChromaDB ──────────────────────────────────────────────────────────────────
CHROMADB_MODE: str = _get("chromadb", "mode", "CHROMADB_MODE", "local")
CHROMADB_PATH: str = _get("chromadb", "path", "CHROMADB_PATH", "./chroma_data")
CHROMADB_HOST: str = _get("chromadb", "host", "CHROMADB_HOST", "localhost")
CHROMADB_PORT: int = int(_get("chromadb", "port", "CHROMADB_PORT", 8000))

# ── Paths ─────────────────────────────────────────────────────────────────────
INPUT_DIR: str = _get("paths", "input_dir", "INPUT_DIR", "./input")
OUTPUT_DIR: str = _get("paths", "output_dir", "OUTPUT_DIR", "./output")
INTERMEDIATE_DIR: str = _get("paths", "intermediate_dir", "INTERMEDIATE_DIR", "./intermediate_data")

# Derived state-file paths for Steps 3-5 (used by save/load helpers).
STEP3_STATE_FILE: str = str(Path(INTERMEDIATE_DIR) / "step3_fused_world.json")
STEP4_STATE_FILE: str = str(Path(INTERMEDIATE_DIR) / "step4_protagonist.json")
STEP5_STATE_FILE: str = str(Path(INTERMEDIATE_DIR) / "step5_reassembled_plot.json")

# ── Pipeline tuning ───────────────────────────────────────────────────────────
_pipe = _yaml.get("pipeline", {})
MAX_RETRY_STEPS: int = int(
    os.environ.get("MAX_RETRY_STEPS", _pipe.get("max_retry_steps", 2))
)
PLAGIARISM_NER_THRESHOLD: float = float(
    os.environ.get(
        "PLAGIARISM_NER_THRESHOLD", _pipe.get("plagiarism_ner_threshold", 0.05)
    )
)
SEMANTIC_CHUNK_MIN_CHAPTERS: int = int(
    os.environ.get(
        "SEMANTIC_CHUNK_MIN_CHAPTERS", _pipe.get("semantic_chunk_min_chapters", 3)
    )
)
SEMANTIC_CHUNK_MAX_CHAPTERS: int = int(
    os.environ.get(
        "SEMANTIC_CHUNK_MAX_CHAPTERS", _pipe.get("semantic_chunk_max_chapters", 5)
    )
)
# Maximum characters of text passed to DeepSeek in a single prompt fragment.
# Keeps token usage predictable; increase if your API plan supports it.
MAX_TEXT_CHUNK_LENGTH: int = int(
    os.environ.get("MAX_TEXT_CHUNK_LENGTH", _pipe.get("max_text_chunk_length", 3000))
)
# Characters used as a source novel synopsis in the adversarial plagiarism check.
PLAGIARISM_SOURCE_SYNOPSIS_LENGTH: int = int(
    os.environ.get(
        "PLAGIARISM_SOURCE_SYNOPSIS_LENGTH",
        _pipe.get("plagiarism_source_synopsis_length", 1000),
    )
)
# Characters of the new outline excerpt fed to the adversarial plagiarism check.
PLAGIARISM_OUTLINE_EXCERPT_LENGTH: int = int(
    os.environ.get(
        "PLAGIARISM_OUTLINE_EXCERPT_LENGTH",
        _pipe.get("plagiarism_outline_excerpt_length", 3000),
    )
)


def validate():
    """Raise ValueError if critical settings are missing."""
    if not DEEPSEEK_API_KEY or DEEPSEEK_API_KEY == "YOUR_DEEPSEEK_API_KEY":
        raise ValueError(
            "DEEPSEEK_API_KEY is not set. "
            "Please set it in config.yaml or via the DEEPSEEK_API_KEY environment variable."
        )
    Path(INPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(INTERMEDIATE_DIR).mkdir(parents=True, exist_ok=True)
