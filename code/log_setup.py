"""
Lightweight terminal output and message capture for this project.

- Captures both stdout and stderr into a timestamped log file
- Records OpenAI-style messages to a JSON array file
- Files live in a per-run session directory: code/log/<SESSION_TS>/
- Minimal coupling: standalone module with setup()/stop() and record_message()
- Easy toggles via env vars:
  - LOG_CAPTURE: 1/true/on to enable (default: on)
  - MSG_CAPTURE: 1/true/on to enable (default: on)

Usage (in your entrypoint, e.g., main.py):

    from log_setup import setup, record_message
    setup()  # honors LOG_CAPTURE/MSG_CAPTURE; default is enabled
    record_message("system", "session started")

You can also disable or override the log directory:

    setup(enable=False)
    setup(log_dir="D:/path/to/logs")
    setup(enable_messages=False)

This implements a Tee to duplicate writes to both the original console and the log file.
"""
from __future__ import annotations

import atexit
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

__all__ = [
    "setup",
    "stop",
    "is_active",
    "record_message",
    "is_message_active",
    "current_session_dir",
    "current_log_path",
    "current_message_path",
]


class _Tee:
    def __init__(self, original, file_obj):
        self._orig = original
        self._file = file_obj

    def write(self, data):
        # Some writers may send None; guard to avoid crashes
        if data is None:
            return 0
        try:
            written = self._orig.write(data)
        except Exception:
            # If console write fails, keep logging to file
            written = 0
        try:
            self._file.write(data)
        except Exception:
            # Avoid raising during logging
            pass
        return written

    def flush(self):
        try:
            self._orig.flush()
        except Exception:
            pass
        try:
            self._file.flush()
        except Exception:
            pass

    # Provide common stream attributes for compatibility
    @property
    def encoding(self):
        return getattr(self._orig, "encoding", "utf-8")

    def isatty(self):
        try:
            return self._orig.isatty()
        except Exception:
            return False

    def fileno(self):
        try:
            return self._orig.fileno()
        except Exception:
            raise OSError("fileno not supported")


# Global state
_active = False
_log_file_handle = None
_orig_stdout = None
_orig_stderr = None
_log_path: Optional[Path] = None
_session_ts: Optional[str] = None
_session_dir: Optional[Path] = None

# Message capture state
_msg_active = False
_msg_file_handle = None
_msg_path: Optional[Path] = None
_msg_first = True  # track comma placement in JSON array


def _env_truthy(value: Optional[str]) -> Optional[bool]:
    if value is None:
        return None
    v = value.strip().lower()
    if v in {"1", "true", "yes", "on"}:
        return True
    if v in {"0", "false", "no", "off"}:
        return False
    return None


def setup(
    enable: Optional[bool] = None,
    log_dir: Optional[str | os.PathLike] = None,
    enable_messages: Optional[bool] = None,
) -> Optional[Path]:
    """
    Start capturing terminal output to a timestamped file.

    Args:
        enable: Force enable/disable. If None, read from env LOG_CAPTURE. Default: enabled.
        log_dir: Target directory for logs. Default: "code/log" beside this file.
    Returns:
        Path to the created log file if active, else None.
    """
    global _active, _log_file_handle, _orig_stdout, _orig_stderr, _log_path
    global _session_ts, _session_dir
    global _msg_active, _msg_file_handle, _msg_path, _msg_first

    if _active:
        return _log_path

    env_enable = _env_truthy(os.getenv("LOG_CAPTURE"))
    if enable is None:
        # Default to enabled if not specified
        enable = True if env_enable is None else env_enable
    else:
        # If explicitly set, it overrides env
        pass

    if not enable:
        return None

    base_dir = Path(__file__).resolve().parent
    root_dir = Path(log_dir) if log_dir else (base_dir / "log")
    root_dir.mkdir(parents=True, exist_ok=True)

    _session_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    _session_dir = root_dir / _session_ts
    _session_dir.mkdir(parents=True, exist_ok=True)

    _log_path = _session_dir / f"{_session_ts}.log"

    # Open the file in text mode with utf-8 to support CJK characters on Windows
    _log_file_handle = _log_path.open("w", encoding="utf-8", buffering=1)

    # Header for debugging context
    _log_file_handle.write(
        f"==== RUN START {datetime.now().isoformat()} ====" "\n"
    )
    _log_file_handle.write(f"CWD: {os.getcwd()}\n")
    _log_file_handle.write(f"PYTHON: {sys.version.splitlines()[0]}\n")
    _log_file_handle.write(f"CMD: {' '.join(sys.argv)}\n")
    _log_file_handle.write("=================================\n\n")
    _log_file_handle.flush()

    # Swap stdout/stderr with tees
    _orig_stdout = sys.stdout
    _orig_stderr = sys.stderr
    sys.stdout = _Tee(_orig_stdout, _log_file_handle)
    sys.stderr = _Tee(_orig_stderr, _log_file_handle)

    _active = True

    # Ensure restoration at exit
    atexit.register(stop)

    # Setup message capture
    env_msg_enable = _env_truthy(os.getenv("MSG_CAPTURE"))
    if enable_messages is None:
        enable_messages = True if env_msg_enable is None else env_msg_enable

    if enable_messages:
        _msg_path = _session_dir / f"{_session_ts}.json"
        _msg_file_handle = _msg_path.open("w", encoding="utf-8", buffering=1)
        # Write JSON array start
        _msg_file_handle.write("[\n")
        _msg_first = True
        _msg_active = True

        # Optional: write a system message for context
        record_message(
            role="system",
            content="session started",
            meta={
                "cwd": os.getcwd(),
                "python": sys.version.splitlines()[0],
                "argv": sys.argv,
                "start": datetime.now().isoformat(),
            },
        )

    return _log_path


def stop():
    """Stop capturing and restore original streams."""
    global _active, _log_file_handle, _orig_stdout, _orig_stderr, _log_path
    global _session_ts, _session_dir
    global _msg_active, _msg_file_handle, _msg_path, _msg_first

    if not _active:
        return

    # Write footer
    try:
        _log_file_handle.write("\n=================================\n")
        _log_file_handle.write(
            f"==== RUN END   {datetime.now().isoformat()} ====\n"
        )
    except Exception:
        pass

    # Restore streams
    if _orig_stdout is not None:
        sys.stdout = _orig_stdout
    if _orig_stderr is not None:
        sys.stderr = _orig_stderr

    # Close file
    try:
        _log_file_handle.flush()
        _log_file_handle.close()
    except Exception:
        pass

    # Reset state
    _active = False
    _log_file_handle = None
    _orig_stdout = None
    _orig_stderr = None
    _log_path = None
    _session_ts = None
    _session_dir = None

    # Close message file (write closing bracket)
    if _msg_active and _msg_file_handle is not None:
        try:
            _msg_file_handle.write("\n]\n")
            _msg_file_handle.flush()
            _msg_file_handle.close()
        except Exception:
            pass
    _msg_active = False
    _msg_file_handle = None
    _msg_path = None
    _msg_first = True


def is_active() -> bool:
    """Return True if capture is active."""
    return _active


def is_message_active() -> bool:
    """Return True if message capture is active."""
    return _msg_active


def record_message(role: str, content: str, meta: Optional[dict] = None):
    """Append an OpenAI-style message to the JSON file if enabled.

    Message format: {"role": "user|assistant|system", "content": str, "timestamp": iso, "meta": {...}}
    """
    if not _msg_active or _msg_file_handle is None:
        return

    msg = {
        "role": role,
        "content": content,
        "timestamp": datetime.now().isoformat(),
    }
    if meta is not None:
        msg["meta"] = meta

    try:
        global _msg_first
        if not _msg_first:
            _msg_file_handle.write(",\n")
        json.dump(msg, _msg_file_handle, ensure_ascii=False)
        _msg_first = False
        _msg_file_handle.flush()
    except Exception:
        # Do not break application on logging errors
        pass


def current_session_dir() -> Optional[Path]:
    return _session_dir


def current_log_path() -> Optional[Path]:
    return _log_path


def current_message_path() -> Optional[Path]:
    return _msg_path
