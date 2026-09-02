"""
AGHR System — Common Helpers
=============================
Shared utility functions used across modules.
"""

import os
import json
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from loguru import logger


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """
    Load YAML configuration file.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Dictionary of configuration values.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded config from {config_path}")
    return config


def load_prompts(prompts_path: str = "config/prompts.yaml") -> Dict[str, str]:
    """
    Load prompt templates from YAML.

    Args:
        prompts_path: Path to the prompts YAML file.

    Returns:
        Dictionary of prompt template strings.
    """
    path = Path(prompts_path)
    if not path.exists():
        raise FileNotFoundError(f"Prompts file not found: {prompts_path}")

    with open(path, "r", encoding="utf-8") as f:
        prompts = yaml.safe_load(f)

    logger.info(f"Loaded {len(prompts)} prompt templates from {prompts_path}")
    return prompts


def ensure_dir(dir_path: Union[str, Path]) -> Path:
    """Create directory if it doesn't exist, return Path object."""
    path = Path(dir_path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: Any, filepath: Union[str, Path]) -> None:
    """Save data to JSON file."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logger.debug(f"Saved JSON → {filepath}")


def load_json(filepath: Union[str, Path]) -> Any:
    """Load data from JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_jsonl(records: List[Dict], filepath: Union[str, Path]) -> None:
    """Save records to JSONL (one JSON object per line)."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.debug(f"Saved {len(records)} records → {filepath}")


def load_jsonl(filepath: Union[str, Path]) -> List[Dict]:
    """Load records from JSONL file."""
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def compute_hash(text: str) -> str:
    """Compute MD5 hash of text for deduplication."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """
    Count tokens in text using tiktoken.

    Args:
        text: Input text.
        encoding_name: Tiktoken encoding name.

    Returns:
        Number of tokens.
    """
    import tiktoken

    enc = tiktoken.get_encoding(encoding_name)
    return len(enc.encode(text))


def truncate_text(text: str, max_tokens: int, encoding_name: str = "cl100k_base") -> str:
    """Truncate text to a maximum number of tokens."""
    import tiktoken

    enc = tiktoken.get_encoding(encoding_name)
    tokens = enc.encode(text)

    if len(tokens) <= max_tokens:
        return text

    return enc.decode(tokens[:max_tokens])


def format_size(size_bytes: int) -> str:
    """Format byte count to human-readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"
