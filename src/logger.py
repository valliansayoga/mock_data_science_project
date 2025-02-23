import logging
from pathlib import Path
from src.config import get_config

cfg = get_config()
FORMAT = "%(asctime)s | %(filename)s (%(levelname)s)\n\t%(message)s"


def get_logger():
    filename = Path("logs.txt")
    root_dir = Path(cfg.paths.pythonpath).resolve()
    log_dir = root_dir / cfg.paths.log
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / filename
    log_level = cfg.general.log_level.upper()
    if log_level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
        print(f"Invalid logging level: {log_level}. Defaulting to INFO.")
        log_level = "INFO"

    logger = logging.getLogger(filename.stem)
    logger.setLevel(log_level)

    if not logger.handlers:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter(FORMAT, datefmt="%Y-%m-%d %H:%M:%S")
        )

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger


if __name__ == "__main__":
    logger = get_logger()
    logger.debug("Debug test message working!")
    logger.info("Info test message working!")
    logger.warning("Warning test message working!")
    logger.error("Error test message working!")
    logger.critical("Critical test message working!")
