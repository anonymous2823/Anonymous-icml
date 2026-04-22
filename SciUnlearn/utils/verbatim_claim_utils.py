import re
from typing import Dict, List


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace so line breaks/tabs don't break matching.
    """
    return re.sub(r"\s+", " ", text).strip()


def verify_verbatim_claim_against_full_text(
    claim_text: str,
    full_text: str,
) -> Dict[str, bool]:
    """
    Verify whether a verbatim claim appears in the full extracted paper text:
    - exact raw substring match
    - whitespace-normalized substring match
    """
    claim_raw = (claim_text or "").strip()
    full_raw = full_text or ""

    exact_in_full_text = claim_raw in full_raw if claim_raw else False

    claim_norm = normalize_whitespace(claim_raw)
    full_norm = normalize_whitespace(full_raw)

    normalized_in_full_text = claim_norm in full_norm if claim_norm else False

    return {
        "exact_in_full_text": exact_in_full_text,
        "normalized_in_full_text": normalized_in_full_text,
    }


def verify_verbatim_claims(
    claims: List[Dict],
    full_text: str,
) -> List[Dict]:
    """
    Add verification flags to each verbatim claim record.
    Expects each item to have:
      - text
      - source_section
    """
    verified = []

    for item in claims:
        claim_text = (item.get("text") or "").strip()
        source_section = (item.get("source_section") or "").strip().lower()

        flags = verify_verbatim_claim_against_full_text(
            claim_text=claim_text,
            full_text=full_text,
        )

        verified.append({
            "text": claim_text,
            "source_section": source_section,
            **flags,
        })

    return verified
