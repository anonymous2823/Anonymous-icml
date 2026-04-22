from pathlib import Path
from typing import Any, Dict, List, Set
import shutil

from config import AppConfig
from utils.json_utils import load_json, save_json


def _forget_id_from_pdf_name(pdf_name: str) -> str:
    return Path(pdf_name).stem.strip()


def _retain_external_anchor_id(rec: Dict[str, Any]) -> str:
    return (
        rec.get("anchor_forget_paper_id")
        or rec.get("anchor_corpus_id")
        or ""
    ).strip()


def _retain_internal_anchor_id(rec: Dict[str, Any]) -> str:
    return (rec.get("anchor_forget_paper_id") or "").strip()


def _derived_anchor_id(rec: Dict[str, Any]) -> str:
    return (rec.get("anchor_forget_paper_id") or "").strip()


def get_successful_anchor_ids_from_derived(
    derived_records: List[Dict[str, Any]]
) -> Set[str]:
    """
    Collect forget anchor ids that successfully produced a derived sample.
    """
    successful_ids: Set[str] = set()

    for rec in derived_records:
        anchor_id = _derived_anchor_id(rec)
        survived = bool(rec.get("derived_survived", False))

        # fallback if old records don't have explicit flag
        if not survived:
            survived = bool(rec.get("derived_qa"))

        if anchor_id and survived:
            successful_ids.add(anchor_id)

    return successful_ids


def prune_forget_records(
    forget_records: List[Dict[str, Any]],
    allowed_anchor_ids: Set[str],
) -> List[Dict[str, Any]]:
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
    kept: List[Dict[str, Any]] = []

    for rec in retain_records:
        anchor_id = _retain_external_anchor_id(rec)
        if anchor_id in allowed_anchor_ids:
            kept.append(rec)

    return kept


def prune_retain_internal_records(
    internal_records: List[Dict[str, Any]],
    allowed_anchor_ids: Set[str],
) -> List[Dict[str, Any]]:
    kept: List[Dict[str, Any]] = []

    for rec in internal_records:
        anchor_id = _retain_internal_anchor_id(rec)
        if anchor_id in allowed_anchor_ids:
            kept.append(rec)

    return kept


def prune_all_by_derived_set(config: AppConfig) -> Dict[str, Any]:
    """
    Use derived-set survivors to jointly prune:
    - forget JSON
    - retain external JSON
    - retain internal JSON
    """
    if not config.derived_output_json.exists():
        raise FileNotFoundError(f"Derived JSON not found: {config.derived_output_json}")

    if not config.claim_output_json.exists():
        raise FileNotFoundError(f"Forget JSON not found: {config.claim_output_json}")

    if not config.retain_output_json.exists():
        raise FileNotFoundError(f"Retain external JSON not found: {config.retain_output_json}")

    if not config.retain_internal_output_json.exists():
        raise FileNotFoundError(f"Retain internal JSON not found: {config.retain_internal_output_json}")

    derived_records = load_json(config.derived_output_json)
    forget_records = load_json(config.claim_output_json)
    retain_external_records = load_json(config.retain_output_json)
    retain_internal_records = load_json(config.retain_internal_output_json)

    if not isinstance(derived_records, list):
        raise ValueError("Derived JSON must contain a JSON array.")
    if not isinstance(forget_records, list):
        raise ValueError("Forget JSON must contain a JSON array.")
    if not isinstance(retain_external_records, list):
        raise ValueError("Retain external JSON must contain a JSON array.")
    if not isinstance(retain_internal_records, list):
        raise ValueError("Retain internal JSON must contain a JSON array.")

    allowed_anchor_ids = get_successful_anchor_ids_from_derived(derived_records)

    pruned_forget = prune_forget_records(forget_records, allowed_anchor_ids)
    pruned_retain_external = prune_retain_external_records(retain_external_records, allowed_anchor_ids)
    pruned_retain_internal = prune_retain_internal_records(retain_internal_records, allowed_anchor_ids)

    # ---------------- Write pruned files ----------------
    save_json(pruned_forget, config.pruned_forget_output_json)
    save_json(pruned_retain_external, config.pruned_retain_output_json)
    save_json(pruned_retain_internal, config.pruned_retain_internal_output_json)

    # # ---------------- Optional overwrite forget ----------------
    # if config.overwrite_forget_json_after_prune:
    #     if config.backup_forget_json_before_overwrite and config.claim_output_json.exists():
    #         backup_path = config.claim_output_json.with_suffix(config.claim_output_json.suffix + ".bak")
    #         shutil.copy2(config.claim_output_json, backup_path)
    #         print(f"[PRUNE][DERIVED] Forget backup created: {backup_path}")

    #     save_json(pruned_forget, config.claim_output_json)
    #     print(f"[PRUNE][DERIVED] Overwrote forget JSON: {config.claim_output_json}")

    # # ---------------- Optional overwrite retain external ----------------
    # if config.overwrite_retain_json_after_prune:
    #     if config.backup_retain_json_before_overwrite and config.retain_output_json.exists():
    #         backup_path = config.retain_output_json.with_suffix(config.retain_output_json.suffix + ".bak")
    #         shutil.copy2(config.retain_output_json, backup_path)
    #         print(f"[PRUNE][DERIVED] Retain external backup created: {backup_path}")

    #     save_json(pruned_retain_external, config.retain_output_json)
    #     print(f"[PRUNE][DERIVED] Overwrote retain external JSON: {config.retain_output_json}")

    # # ---------------- Optional overwrite retain internal ----------------
    # if config.overwrite_retain_internal_json_after_prune:
    #     if config.backup_retain_internal_json_before_overwrite and config.retain_internal_output_json.exists():
    #         backup_path = config.retain_internal_output_json.with_suffix(config.retain_internal_output_json.suffix + ".bak")
    #         shutil.copy2(config.retain_internal_output_json, backup_path)
    #         print(f"[PRUNE][DERIVED] Retain internal backup created: {backup_path}")

    #     save_json(pruned_retain_internal, config.retain_internal_output_json)
    #     print(f"[PRUNE][DERIVED] Overwrote retain internal JSON: {config.retain_internal_output_json}")

    return {
        "allowed_anchor_ids_count": len(allowed_anchor_ids),

        "forget_original_count": len(forget_records),
        "forget_pruned_count": len(pruned_forget),
        "forget_removed_count": len(forget_records) - len(pruned_forget),

        "retain_external_original_count": len(retain_external_records),
        "retain_external_pruned_count": len(pruned_retain_external),
        "retain_external_removed_count": len(retain_external_records) - len(pruned_retain_external),

        "retain_internal_original_count": len(retain_internal_records),
        "retain_internal_pruned_count": len(pruned_retain_internal),
        "retain_internal_removed_count": len(retain_internal_records) - len(pruned_retain_internal),

        "pruned_forget_output": str(config.pruned_forget_output_json),
        "pruned_retain_external_output": str(config.pruned_retain_output_json),
        "pruned_retain_internal_output": str(config.pruned_retain_internal_output_json),
    }