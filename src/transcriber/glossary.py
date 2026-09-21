"""Optional user files: a glossary of names for Whisper and custom Hinglish spellings."""

import json
from pathlib import Path


def load_glossary(path: Path | None) -> list[str]:
    """Read names from a text file, one per line. Blank lines and '#' comments are ignored."""
    if path is None:
        return []
    names = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            names.append(line)
    return names


def hotwords_from(names: list[str]) -> str | None:
    """Whisper takes hotwords as one comma-separated string."""
    return ", ".join(names) or None


def load_custom_words(path: Path | None) -> dict[str, str]:
    """Read a JSON object of {"देवनागरी": "hinglish"} spellings."""
    if path is None:
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in data.items()):
        raise ValueError(f'{path}: expected a JSON object of {{"देवनागरी": "hinglish"}} pairs')
    return {key.strip(): value.strip() for key, value in data.items()}
