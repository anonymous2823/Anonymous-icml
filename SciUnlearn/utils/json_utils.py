import json
from pathlib import Path
from typing import Any


def save_json(data: Any, path: Path) -> None:
    """
    Save data as JSON with UTF-8 encoding and pretty formatting.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)