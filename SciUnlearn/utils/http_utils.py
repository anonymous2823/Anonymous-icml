import time
from typing import Optional

import requests


def request_with_retries(
    method: str,
    url: str,
    *,
    params: Optional[dict] = None,
    json_body: Optional[dict] = None,
    headers: Optional[dict] = None,
    retries: int = 3,
    backoff: float = 5.0,
    timeout: float = 30.0,
    stream: bool = False,
):
    """
    Generic HTTP request helper with retry handling.
    Returns a requests.Response or None on failure / 404.
    """
    headers = headers or {}

    for attempt in range(1, retries + 1):
        try:
            if method.upper() == "GET":
                resp = requests.get(
                    url,
                    params=params,
                    headers=headers,
                    timeout=timeout,
                    stream=stream,
                    allow_redirects=True,
                )
            elif method.upper() == "POST":
                resp = requests.post(
                    url,
                    params=params,
                    json=json_body,
                    headers=headers,
                    timeout=timeout,
                    allow_redirects=True,
                )
            elif method.upper() == "HEAD":
                resp = requests.head(
                    url,
                    params=params,
                    headers=headers,
                    timeout=timeout,
                    allow_redirects=True,
                )
            else:
                raise ValueError(f"Unsupported method: {method}")

            if resp.status_code == 404:
                return None

            if resp.status_code == 429:
                wait = backoff * attempt
                print(f"[RATE] 429 for {url}; sleeping {wait:.1f}s and retrying...")
                time.sleep(wait)
                continue

            resp.raise_for_status()
            return resp

        except requests.RequestException as e:
            if attempt < retries:
                wait = backoff * attempt
                print(f"[WARN] Request failed ({e}); retrying in {wait:.1f}s...")
                time.sleep(wait)
            else:
                print(f"[ERROR] Request failed after {retries} attempts: {e}")
                return None

    return None