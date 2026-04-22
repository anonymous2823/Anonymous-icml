import sys
from pathlib import Path
from typing import List


def load_corpus_ids(path: str) -> List[int]:
    """
    Load integer corpus IDs from a text file.
    Ignores empty lines and comment lines starting with '#'.
    """
    ids: List[int] = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            try:
                ids.append(int(s))
            except ValueError:
                print(f"[WARN] Skipping non-integer line: {s}", file=sys.stderr)

    if not ids:
        raise RuntimeError(f"No valid corpus IDs found in {path}")

    return ids


def ensure_dir(path: Path) -> None:
    """
    Ensure a directory exists.
    """
    path.mkdir(parents=True, exist_ok=True)