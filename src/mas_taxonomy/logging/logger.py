from __future__ import annotations

import logging
from pathlib import Path


def get_logger(name: str, log_dir: Path | None = None) -> logging.Logger: #returns a logger object with the given name and log directory.
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    # Prevent child loggers (e.g. "mas_taxonomy.graph.empirical") from propagating
    # to parent loggers (e.g. "mas_taxonomy") which would cause double-logging
    # when both have handlers attached to the same log file.
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / "app.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
