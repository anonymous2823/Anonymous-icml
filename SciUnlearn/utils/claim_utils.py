import json
import re
from typing import Any, Dict, List


def parse_claims_json(raw: str) -> List[str]:
    """
    Parse claim JSON of the form:
    {
      "claims": [
        {"text": "..."},
        {"text": "..."}
      ]
    }
    """
    raw = raw.strip()

    if raw.startswith("```"):
        raw = re.sub(
            r"^```(?:json)?\s*|\s*```$",
            "",
            raw,
            flags=re.IGNORECASE | re.DOTALL,
        ).strip()

    data = json.loads(raw)

    claims: List[str] = []
    for item in data.get("claims", []):
        t = (item or {}).get("text", "")
        if isinstance(t, str) and t.strip():
            claims.append(" ".join(t.split()))

    return claims


def parse_json_block(raw: str) -> Dict[str, Any]:
    """
    Strip code fences and parse JSON into a dict.
    """
    raw = raw.strip()

    if raw.startswith("```"):
        raw = re.sub(
            r"^```(?:json)?\s*|\s*```$",
            "",
            raw,
            flags=re.IGNORECASE | re.DOTALL,
        ).strip()

    return json.loads(raw)