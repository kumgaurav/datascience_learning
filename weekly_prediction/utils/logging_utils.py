import logging
import os
from typing import Optional, Union
from logging.handlers import RotatingFileHandler


def _parse_level(level: Optional[Union[str, int]]) -> int:
    if isinstance(level, int):
        return level
    if isinstance(level, str):
        lvl = level.strip().upper()
        return getattr(logging, lvl, logging.INFO)
    env = os.getenv('LOG_LEVEL', 'INFO').strip().upper()
    return getattr(logging, env, logging.INFO)


def configure_logging(level: Optional[Union[str, int]] = None, log_file: Optional[str] = None) -> None:
    """Configure root logging once, respecting env vars LOG_LEVEL, LOG_FILE, LOG_FORMAT.

    If handlers already exist, this function is a no-op except to add a FileHandler
    if a new log_file is provided and not already attached.
    """
    root = logging.getLogger()
    fmt = os.getenv('LOG_FORMAT', '%(asctime)s | %(levelname)s | %(name)s | %(message)s')
    level_val = _parse_level(level)
    file_from_env = os.getenv('LOG_FILE')
    log_file = log_file or (file_from_env if isinstance(file_from_env, str) and file_from_env.strip() else None)

    # If not configured, set up a basic console handler
    if not root.handlers:
        root.setLevel(level_val)
        sh = logging.StreamHandler()
        sh.setLevel(level_val)
        sh.setFormatter(logging.Formatter(fmt))
        root.addHandler(sh)
    else:
        # Ensure root level matches requested level if provided
        root.setLevel(level_val)
        for h in root.handlers:
            try:
                h.setLevel(level_val)
            except Exception:
                pass

    # Add file handler if requested and not already present
    if log_file:
        # Ensure directory exists
        try:
            os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        except Exception:
            pass
        already = any(isinstance(h, logging.FileHandler) and getattr(h, 'baseFilename', None) == os.path.abspath(log_file) for h in root.handlers)
        if not already:
            # Size-based rotation by default
            try:
                max_bytes = int(os.getenv('LOG_MAX_BYTES', str(5 * 1024 * 1024)))  # 5 MB
            except Exception:
                max_bytes = 5 * 1024 * 1024
            try:
                backups = int(os.getenv('LOG_BACKUP_COUNT', '5'))
            except Exception:
                backups = 5
            fh = RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backups)
            fh.setLevel(level_val)
            fh.setFormatter(logging.Formatter(fmt))
            root.addHandler(fh)


