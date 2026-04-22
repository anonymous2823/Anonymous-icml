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


def _retain_anchor_id(rec: Dict[str, Any]) -> str:
    """
    Extract anchor forget paper id from a retain external record.
    """
    return (
        rec.get("anchor_forget_paper_id")
        or rec.get("anchor_corpus_id")
        or ""
    ).strip()


def get_successful_anchor_ids_from_internal_retain(
    internal_records: List[Dict[str, Any]]
) -> Set[str]:
    """
    Collect forget paper ids that successfully produced internal retain samples.
    """
    successful_ids: Set[str] = set()

    for rec in internal_records:
        anchor_id = (rec.get("anchor_forget_paper_id") or "").strip()
        survived = bool(rec.get("internal_retain_survived", False))

        if anchor_id and survived:
            successful_ids.add(anchor_id)

    return successful_ids


def prune_forget_records(
    forget_records: List[Dict[str, Any]],
    allowed_anchor_ids: Set[str],
) -> List[Dict[str, Any]]:
    """
    Keep only forget records whose pdf_name stem is in allowed_anchor_ids.
    """
    kept: List[Dict[str, Any]] = []

    for rec in forget_records:
        pdf_name = rec.get("pdf_name", "")
        forget_id = _forget_id_from_pdf_name(pdf_name)

        if forget_id in allowed_anchor_ids:
            kept.append(rec)

    return kept


def prune_retain_external_records(
    retain_records: List[Dict[str, Any]],
    allowed_anchor_ids: Set[str],
) -> List[Dict[str, Any]]:
    """
    Keep only retain external records whose anchor_forget_paper_id is still allowed.
    """
    kept: List[Dict[str, Any]] = []

    for rec in retain_records:
        anchor_id = _retain_anchor_id(rec)
        if anchor_id in allowed_anchor_ids:
            kept.append(rec)

    return kept


def prune_forget_and_retain_external_by_anchor_ids(
    config: AppConfig,
    allowed_anchor_ids: Set[str],
) -> Dict[str, Any]:
    """
    Jointly prune:
    - forget JSON
    - retain external JSON

    using the same allowed_anchor_ids set.
    """
    if not config.claim_output_json.exists():
        raise FileNotFoundError(f"Forget JSON not found: {config.claim_output_json}")

    if not config.retain_output_json.exists():
        raise FileNotFoundError(f"Retain external JSON not found: {config.retain_output_json}")

    forget_records = load_json(config.claim_output_json)
    retain_external_records = load_json(config.retain_output_json)

    if not isinstance(forget_records, list):
        raise ValueError("Forget JSON must contain a JSON array.")

    if not isinstance(retain_external_records, list):
        raise ValueError("Retain external JSON must contain a JSON array.")

    pruned_forget = prune_forget_records(forget_records, allowed_anchor_ids)
    pruned_retain_external = prune_retain_external_records(retain_external_records, allowed_anchor_ids)

    # ---------------- Write forget pruned file ----------------
    save_json(pruned_forget, config.pruned_forget_output_json)

    # ---------------- Write retain external pruned file ----------------
    save_json(pruned_retain_external, config.pruned_retain_output_json)

    # ---------------- Optional overwrite forget ----------------
    # if config.overwrite_forget_json_after_prune:
    #     if config.backup_forget_json_before_overwrite and config.claim_output_json.exists():
    #         backup_path = config.claim_output_json.with_suffix(config.claim_output_json.suffix + ".bak")
    #         shutil.copy2(config.claim_output_json, backup_path)
    #         print(f"[SYNC] Forget backup created: {backup_path}")

    #     save_json(pruned_forget, config.claim_output_json)
    #     print(f"[SYNC] Overwrote forget JSON: {config.claim_output_json}")

    # # ---------------- Optional overwrite retain external ----------------
    # if config.overwrite_retain_json_after_prune:
    #     if config.backup_retain_json_before_overwrite and config.retain_output_json.exists():
    #         backup_path = config.retain_output_json.with_suffix(config.retain_output_json.suffix + ".bak")
    #         shutil.copy2(config.retain_output_json, backup_path)
    #         print(f"[SYNC] Retain external backup created: {backup_path}")

    #     save_json(pruned_retain_external, config.retain_output_json)
    #     print(f"[SYNC] Overwrote retain external JSON: {config.retain_output_json}")

    return {
        "forget_original_count": len(forget_records),
        "forget_pruned_count": len(pruned_forget),
        "forget_removed_count": len(forget_records) - len(pruned_forget),

        "retain_external_original_count": len(retain_external_records),
        "retain_external_pruned_count": len(pruned_retain_external),
        "retain_external_removed_count": len(retain_external_records) - len(pruned_retain_external),

        "allowed_anchor_ids_count": len(allowed_anchor_ids),
        "pruned_forget_output": str(config.pruned_forget_output_json),
        "pruned_retain_output": str(config.pruned_retain_output_json),
    }


def prune_after_internal_retain(config: AppConfig) -> Dict[str, Any]:
    """
    Use internal retain survivors to jointly prune:
    - forget JSON
    - retain external JSON
    """
    if not config.retain_internal_output_json.exists():
        raise FileNotFoundError(f"Internal retain JSON not found: {config.retain_internal_output_json}")

    internal_records = load_json(config.retain_internal_output_json)
    if not isinstance(internal_records, list):
        raise ValueError("Internal retain JSON must contain a JSON array.")

    allowed_anchor_ids = get_successful_anchor_ids_from_internal_retain(internal_records)

    return prune_forget_and_retain_external_by_anchor_ids(
        config=config,
        allowed_anchor_ids=allowed_anchor_ids,
    )