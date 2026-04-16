from __future__ import annotations

import logging
from pathlib import Path


def build_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("computer_use_agent")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    )
    logger.addHandler(file_handler)
    return logger
