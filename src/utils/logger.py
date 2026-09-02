"""
AGHR System — Logging Configuration
====================================
Centralized logging using loguru with file rotation and console output.
"""

import sys
from pathlib import Path
from loguru import logger


def setup_logger(
    log_level: str = "INFO",
    log_file: str = "logs/aghr.log",
    rotation: str = "10 MB",
    retention: str = "7 days",
) -> None:
    """
    Configure the global loguru logger.

    Args:
        log_level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Path to the log file.
        rotation: When to rotate the log file (e.g., "10 MB", "1 day").
        retention: How long to keep rotated log files.
    """
    # Remove default handler
    logger.remove()

    # Console handler — colorized, concise
    logger.add(
        sys.stderr,
        level=log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> — "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    # File handler — detailed, with rotation
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger.add(
        str(log_path),
        level="DEBUG",
        format=(
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
            "{level: <8} | "
            "{name}:{function}:{line} — {message}"
        ),
        rotation=rotation,
        retention=retention,
        encoding="utf-8",
    )

    logger.info(f"Logger initialized — level={log_level}, file={log_file}")


def get_logger(name: str = "aghr"):
    """Return a contextualized logger instance."""
    return logger.bind(name=name)
