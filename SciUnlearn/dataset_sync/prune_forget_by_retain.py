from pathlib import Path
from typing import Any, Dict, List, Set
import shutil

from config import AppConfig
from utils.json_utils import load_json, save_json


def _forget_id_from_pdf_name(pdf_name: str) -> str:
    """
    Example:
      '119529102.pdf' -> '119529102'
    """
    return Path(pdf_name).stem.strip()


def get_successful_anchor_ids_from_retain(
    retain_records: List[Dict[str, Any]]
) -> Set[str]:
    """
    Extract forget anchor IDs that successfully produced retain samples.
    """
    successful_ids: Set[str] = set()

    for rec in retain_records:
        anchor_id = (
            (rec.get("anchor_forget_paper_id") or rec.get("anchor_corpus_id") or "")
            .strip()
        )
        survived = bool(rec.get("retain_survived", False))

        # fallback if retain_survived is missing in some old records
        if not survived:
            survived = bool(rec.get("paper_claims")) and bool(rec.get("qa_by_claim"))

        if anchor_id and survived:
            successful_ids.add(anchor_id)

    return successful_ids


def prune_forget_records(
    forget_records: List[Dict[str, Any]],
    successful_anchor_ids: Set[str],
) -> List[Dict[str, Any]]:
    """
    Keep only forget records whose pdf_name stem is in successful_anchor_ids.
    """
    kept: List[Dict[str, Any]] = []

    for rec in forget_records:
        pdf_name = rec.get("pdf_name", "")
        forget_id = _forget_id_from_pdf_name(pdf_name)

        if forget_id in successful_anchor_ids:
            kept.append(rec)

    return kept


def prune_forget_json_by_retain(config: AppConfig) -> Dict[str, Any]:
    """
    Read:
      - forget JSON from config.claim_output_json
      - retain JSON from config.retain_output_json

    Keep only forget records that successfully produced retain samples.
    """
    if not config.claim_output_json.exists():
        raise FileNotFoundError(f"Forget JSON not found: {config.claim_output_json}")

    if not config.retain_output_json.exists():
        raise FileNotFoundError(f"Retain JSON not found: {config.retain_output_json}")

    forget_records = load_json(config.claim_output_json)
    retain_records = load_json(config.retain_output_json)

    if not isinstance(forget_records, list):
        raise ValueError("Forget JSON must contain a JSON array.")

    if not isinstance(retain_records, list):
        raise ValueError("Retain JSON must contain a JSON array.")

    successful_anchor_ids = get_successful_anchor_ids_from_retain(retain_records)
    pruned_forget_records = prune_forget_records(forget_records, successful_anchor_ids)

    original_count = len(forget_records)
    pruned_count = len(pruned_forget_records)
    removed_count = original_count - pruned_count

    # Safe default: write a new file
    final_output_path = config.pruned_forget_output_json
    save_json(pruned_forget_records, final_output_path)

    

    print(f"[PRUNE] Forget records before pruning: {original_count}")
    print(f"[PRUNE] Forget records after pruning : {pruned_count}")
    print(f"[PRUNE] Forget records removed       : {removed_count}")
    print(f"[PRUNE] Final pruned forget JSON    : {final_output_path}")

    return {
        "original_count": original_count,
        "pruned_count": pruned_count,
        "removed_count": removed_count,
        "successful_anchor_ids": sorted(successful_anchor_ids),
        "final_output_path": str(final_output_path),
    }