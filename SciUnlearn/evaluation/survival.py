from typing import Any, Dict


def record_survives(record: Dict[str, Any]) -> bool:
    """
    A record survives if, after all filtering stages, it still has:
    - at least one final claim
    - at least one qa_by_claim entry
    """
    return bool(record.get("paper_claims")) and bool(record.get("qa_by_claim"))