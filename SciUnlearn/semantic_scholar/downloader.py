from pathlib import Path

import requests


def download_pdf(pdf_url: str, out_path: Path) -> bool:
    """
    Stream the PDF to disk; returns True if saved successfully.
    """
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with requests.get(pdf_url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

        return True

    except requests.RequestException as e:
        print(f"[ERROR] Download failed {pdf_url} -> {out_path.name}: {e}")
        return False