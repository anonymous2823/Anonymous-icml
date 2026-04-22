from pathlib import Path
from typing import Any, Dict, List, Set

from config import AppConfig
from utils.json_utils import load_json, save_json


def ensure_list(data: Any, name: str) -> List[Dict[str, Any]]:
    """
    Ensure loaded JSON is a list of dict records.
    """
    if isinstance(data, dict):
        data = [data]

    if not isinstance(data, list):
        raise ValueError(f"{name} must contain a JSON array (or a single dict).")

    cleaned: List[Dict[str, Any]] = []
    for i, rec in enumerate(data):
        if not isinstance(rec, dict):
            raise ValueError(f"{name}[{i}] is not a JSON object.")
        cleaned.append(rec)

    return cleaned


def forget_id_from_record(rec: Dict[str, Any]) -> str:
    """
    Forget JSON records identify the paper using pdf_name.
    Example:
        '119529102.pdf' -> '119529102'
    """
    pdf_name = (rec.get("pdf_name") or "").strip()
    if not pdf_name:
        return ""
    return Path(pdf_name).stem.strip()


def anchor_id_from_record(rec: Dict[str, Any]) -> str:
    """
    Retain / derived records identify the forget anchor using one of:
      1. anchor_forget_paper_id
      2. anchor_corpus_id
      3. anchor_forget_pdf_name
      4. fallback: pdf_name
    """
    anchor_id = (rec.get("anchor_forget_paper_id") or "").strip()
    if anchor_id:
        return anchor_id

    anchor_id = (rec.get("anchor_corpus_id") or "").strip()
    if anchor_id:
        return anchor_id

    anchor_pdf = (rec.get("anchor_forget_pdf_name") or "").strip()
    if anchor_pdf:
        return Path(anchor_pdf).stem.strip()

    pdf_name = (rec.get("pdf_name") or "").strip()
    if pdf_name:
        return Path(pdf_name).stem.strip()

    return ""


def collect_forget_ids(forget_records: List[Dict[str, Any]]) -> Set[str]:
    """
    Forget file: use ALL records.
    """
    ids: Set[str] = set()

    for rec in forget_records:
        fid = forget_id_from_record(rec)
        if fid:
            ids.add(fid)

    return ids


def collect_retain_external_ids(retain_external_records: List[Dict[str, Any]]) -> Set[str]:
    """
    Retain external:
    - if retain_survived exists, use only retain_survived == True
    - otherwise use all records in the file
    """
    ids: Set[str] = set()

    for rec in retain_external_records:
        if "retain_survived" in rec and not rec.get("retain_survived", False):
            continue

        aid = anchor_id_from_record(rec)
        if aid:
            ids.add(aid)

    return ids


def collect_retain_internal_ids(retain_internal_records: List[Dict[str, Any]]) -> Set[str]:
    """
    Retain internal:
    use only internal_retain_survived == True
    """
    ids: Set[str] = set()

    for rec in retain_internal_records:
        if not rec.get("internal_retain_survived", False):
            continue

        aid = anchor_id_from_record(rec)
        if aid:
            ids.add(aid)

    return ids


def collect_derived_ids(derived_records: List[Dict[str, Any]]) -> Set[str]:
    """
    Derived:
    use only derived_survived == True
    """
    ids: Set[str] = set()

    for rec in derived_records:
        if not rec.get("derived_survived", False):
            continue

        aid = anchor_id_from_record(rec)
        if aid:
            ids.add(aid)

    return ids


def filter_forget_records(
    forget_records: List[Dict[str, Any]],
    common_ids: Set[str],
) -> List[Dict[str, Any]]:
    kept: List[Dict[str, Any]] = []

    for rec in forget_records:
        fid = forget_id_from_record(rec)
        if fid in common_ids:
            kept.append(rec)

    return kept


def filter_retain_external_records(
    retain_external_records: List[Dict[str, Any]],
    common_ids: Set[str],
) -> List[Dict[str, Any]]:
    kept: List[Dict[str, Any]] = []

    for rec in retain_external_records:
        if "retain_survived" in rec and not rec.get("retain_survived", False):
            continue

        aid = anchor_id_from_record(rec)
        if aid in common_ids:
            kept.append(rec)

    return kept


def filter_retain_internal_records(
    retain_internal_records: List[Dict[str, Any]],
    common_ids: Set[str],
) -> List[Dict[str, Any]]:
    kept: List[Dict[str, Any]] = []

    for rec in retain_internal_records:
        if not rec.get("internal_retain_survived", False):
            continue

        aid = anchor_id_from_record(rec)
        if aid in common_ids:
            kept.append(rec)

    return kept


def filter_derived_records(
    derived_records: List[Dict[str, Any]],
    common_ids: Set[str],
) -> List[Dict[str, Any]]:
    kept: List[Dict[str, Any]] = []

    for rec in derived_records:
        if not rec.get("derived_survived", False):
            continue

        aid = anchor_id_from_record(rec)
        if aid in common_ids:
            kept.append(rec)

    return kept


def keep_common_across_all(config: AppConfig) -> Dict[str, Any]:
    """
    Read forget / retain external / retain internal / derived JSONs from config,
    keep only papers common across all four under the correct survival conditions,
    and save 4 new filtered JSON files.
    """
    forget_path = config.claim_output_json
    retain_external_path = config.retain_output_json
    retain_internal_path = config.retain_internal_output_json
    derived_path = config.derived_output_json

    for p in [forget_path, retain_external_path, retain_internal_path, derived_path]:
        if not p.exists():
            raise FileNotFoundError(f"Input file not found: {p}")

    forget_records = ensure_list(load_json(forget_path), "forget JSON")
    retain_external_records = ensure_list(load_json(retain_external_path), "retain external JSON")
    retain_internal_records = ensure_list(load_json(retain_internal_path), "retain internal JSON")
    derived_records = ensure_list(load_json(derived_path), "derived JSON")

    # Collect IDs under the correct logic
    forget_ids = collect_forget_ids(forget_records)
    retain_external_ids = collect_retain_external_ids(retain_external_records)
    retain_internal_ids = collect_retain_internal_ids(retain_internal_records)
    derived_ids = collect_derived_ids(derived_records)

    # Final intersection
    common_ids = forget_ids & retain_external_ids & retain_internal_ids & derived_ids

    # Filter records
    filtered_forget = filter_forget_records(forget_records, common_ids)
    filtered_retain_external = filter_retain_external_records(retain_external_records, common_ids)
    filtered_retain_internal = filter_retain_internal_records(retain_internal_records, common_ids)
    filtered_derived = filter_derived_records(derived_records, common_ids)

    # Output paths (use config if available, otherwise safe defaults)
    common_forget_out = Path(
        getattr(config, "common_forget_output_json", Path("forget_common_3.json"))
    )
    common_retain_out = Path(
        getattr(config, "common_retain_output_json", Path("retain_external_common_3.json"))
    )
    common_retain_internal_out = Path(
        getattr(config, "common_retain_internal_output_json", Path("retain_internal_common_3.json"))
    )
    common_derived_out = Path(
        getattr(config, "common_derived_output_json", Path("derived_common_3.json"))
    )

    save_json(filtered_forget, common_forget_out)
    save_json(filtered_retain_external, common_retain_out)
    save_json(filtered_retain_internal, common_retain_internal_out)
    save_json(filtered_derived, common_derived_out)

    print("\n" + "=" * 100)
    print("COMMON-PAPER FILTER SUMMARY")
    print("=" * 100)
    print(f"Forget input count                 : {len(forget_records)}")
    print(f"Retain external input count        : {len(retain_external_records)}")
    print(f"Retain internal input count        : {len(retain_internal_records)}")
    print(f"Derived input count                : {len(derived_records)}")
    print("-" * 100)
    print(f"Unique forget ids                  : {len(forget_ids)}")
    print(f"Valid retain external ids          : {len(retain_external_ids)}")
    print(f"Valid retain internal ids          : {len(retain_internal_ids)}")
    print(f"Valid derived ids                  : {len(derived_ids)}")
    print("-" * 100)
    print(f"Common paper ids across all 4      : {len(common_ids)}")
    print("-" * 100)
    print(f"Filtered forget count              : {len(filtered_forget)}")
    print(f"Filtered retain external count     : {len(filtered_retain_external)}")
    print(f"Filtered retain internal count     : {len(filtered_retain_internal)}")
    print(f"Filtered derived count             : {len(filtered_derived)}")
    print("-" * 100)
    print(f"Forget output path                 : {common_forget_out}")
    print(f"Retain external output path        : {common_retain_out}")
    print(f"Retain internal output path        : {common_retain_internal_out}")
    print(f"Derived output path                : {common_derived_out}")

    return {
        "common_ids_count": len(common_ids),

        "forget_input_count": len(forget_records),
        "retain_external_input_count": len(retain_external_records),
        "retain_internal_input_count": len(retain_internal_records),
        "derived_input_count": len(derived_records),

        "forget_output_count": len(filtered_forget),
        "retain_external_output_count": len(filtered_retain_external),
        "retain_internal_output_count": len(filtered_retain_internal),
        "derived_output_count": len(filtered_derived),

        "common_forget_output_json": str(common_forget_out),
        "common_retain_output_json": str(common_retain_out),
        "common_retain_internal_output_json": str(common_retain_internal_out),
        "common_derived_output_json": str(common_derived_out),
    }


if __name__ == "__main__":
    config = AppConfig()
    keep_common_across_all(config)