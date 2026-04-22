import time
from typing import Any, Dict, Optional

import requests

from config import AppConfig


def fetch_metadata(corpus_id: int, config: AppConfig) -> Optional[Dict[str, Any]]:
    """
    Call the Semantic Scholar Graph API for a single corpus ID and return JSON metadata.
    Retries on transient errors and 429 (rate limit).
    """
    url = f"{config.semantic_scholar_api_base}/CorpusId:{corpus_id}"
    params = {
        "fields": "title,year,isOpenAccess,openAccessPdf,url,fieldsOfStudy,abstract"
    }

    headers = {}
    if config.semantic_scholar_api_key:
        headers["x-api-key"] = config.semantic_scholar_api_key

    for attempt in range(1, config.retries + 1):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=30)

            if resp.status_code == 404:
                print(f"[MISS] {corpus_id}: not found (404)")
                return None

            if resp.status_code == 429:
                wait = config.backoff * attempt
                print(f"[RATE] 429 on {corpus_id}; sleeping {wait:.1f}s then retrying...")
                time.sleep(wait)
                continue

            resp.raise_for_status()
            return resp.json()

        except requests.RequestException as e:
            if attempt < config.retries:
                wait = config.backoff * attempt
                print(f"[WARN] {corpus_id}: request failed ({e}); retrying in {wait:.1f}s...")
                time.sleep(wait)
            else:
                print(f"[ERROR] {corpus_id}: request failed after {config.retries} attempts ({e})")
                return None

    return None