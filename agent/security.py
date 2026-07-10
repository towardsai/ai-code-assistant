"""Shared workspace and outbound-data security helpers."""
from __future__ import annotations

import pathlib
import re
from typing import Union


SENSITIVE_FILE_NAMES = {
    ".env", ".npmrc", ".pypirc", "credentials", "credentials.json",
    "id_rsa", "id_ed25519", "secrets", "secrets.json",
}
SENSITIVE_DIR_NAMES = {".aws", ".ssh"}
SENSITIVE_SUFFIXES = {".key", ".pem", ".p12", ".pfx"}
SAFE_ENV_SUFFIXES = (".example", ".sample", ".template")

_SECRET_PATTERNS = (
    re.compile(r"\b(?:sk|pk)-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(
        r"(?i)\b(?:api[_-]?key|access[_-]?token|password|secret)\b"
        r"\s*[:=]\s*['\"]?[^\s,'\"]+"
    ),
)


def resolve_in_workspace(root: pathlib.Path,
                         path: Union[str, pathlib.Path]) -> pathlib.Path:
    """Resolve ``path`` and reject absolute, parent, or symlink escapes."""
    resolved_root = pathlib.Path(root).resolve()
    candidate = pathlib.Path(path)
    resolved = (candidate if candidate.is_absolute() else resolved_root / candidate).resolve()
    if not resolved.is_relative_to(resolved_root):
        raise ValueError(f"path '{path}' escapes the working directory")
    return resolved


def is_sensitive_path(path: Union[str, pathlib.Path]) -> bool:
    """Return whether a normalized path commonly contains credentials or keys."""
    candidate = pathlib.Path(path)
    parts = tuple(part.lower() for part in candidate.parts)
    name = candidate.name.lower()
    if any(part in SENSITIVE_DIR_NAMES for part in parts):
        return True
    if name in SENSITIVE_FILE_NAMES or candidate.suffix.lower() in SENSITIVE_SUFFIXES:
        return True
    if name.startswith(".env.") and not name.endswith(SAFE_ENV_SUFFIXES):
        return True
    return ".git" in parts and name == "config"


def redact_outbound_secrets(text: str) -> str:
    """Redact common credential shapes before text crosses the network boundary."""
    redacted = text
    for pattern in _SECRET_PATTERNS:
        redacted = pattern.sub("[REDACTED]", redacted)
    return redacted
