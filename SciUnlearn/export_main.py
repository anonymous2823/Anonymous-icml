
import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pyarrow as pa
import pyarrow.parquet as pq

from config import AppConfig
from utils.json_utils import load_json, save_json

QA_TYPES = ["mcq", "true_false", "fill_blank", "assertion_reason"]


# ============================================================
# Generic helpers
# ============================================================

def ensure_list(data: Any, name: str) -> List[Dict[str, Any]]:
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        raise ValueError(f"{name} must contain a JSON array (or a single dict).")
    out: List[Dict[str, Any]] = []
    for i, rec in enumerate(data):
        if not isinstance(rec, dict):
            raise ValueError(f"{name}[{i}] is not a JSON object.")
        out.append(rec)
    return out


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_jsonl(records: List[Dict[str, Any]], output_path: Path) -> None:
    ensure_parent(output_path)
    with output_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def write_parquet(records: List[Dict[str, Any]], output_path: Path) -> None:
    ensure_parent(output_path)
    table = pa.Table.from_pylist(records)
    pq.write_table(table, output_path)


# ============================================================
# Forget split logic
# ============================================================

def split_forget_record_into_sc1_sc2(record: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Split one forget record into two sibling records:
      - forget_sc_1
      - forget_sc_2

    For each claim and each QA type:
      sc1 = first half
      sc2 = second half

    If count is odd, sc2 gets the extra item.
    """
    rec1 = copy.deepcopy(record)
    rec2 = copy.deepcopy(record)

    qa1: List[Dict[str, Any]] = []
    qa2: List[Dict[str, Any]] = []

    qa_by_claim = record.get("qa_by_claim", []) or []

    for claim_obj in qa_by_claim:
        if not isinstance(claim_obj, dict):
            continue

        claim_text = claim_obj.get("claim", "")
        claim_part_1 = {"claim": claim_text}
        claim_part_2 = {"claim": claim_text}

        for qt in QA_TYPES:
            items = claim_obj.get(qt, []) or []
            half = len(items) // 2
            claim_part_1[qt] = items[:half]
            claim_part_2[qt] = items[half:]

        for extra_key in claim_obj.keys():
            if extra_key not in ["claim", *QA_TYPES]:
                claim_part_1[extra_key] = claim_obj[extra_key]
                claim_part_2[extra_key] = claim_obj[extra_key]

        qa1.append(claim_part_1)
        qa2.append(claim_part_2)

    rec1["qa_by_claim"] = qa1
    rec2["qa_by_claim"] = qa2
    return rec1, rec2


def split_forget_records(records: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    sc1_records: List[Dict[str, Any]] = []
    sc2_records: List[Dict[str, Any]] = []

    for rec in records:
        r1, r2 = split_forget_record_into_sc1_sc2(rec)
        sc1_records.append(r1)
        sc2_records.append(r2)

    return sc1_records, sc2_records


# ============================================================
# Flatten helpers
# ============================================================

def make_row(
    pdf_name: str,
    dataset_key: str,
    qtype: str,
    question_index: int,
    question: str,
    answer: str,
    task: str,
    split: str,
) -> Dict[str, Any]:
    return {
        "id": f"{pdf_name}|{dataset_key}|{qtype}|Q{question_index}",
        "input": question,
        "output": answer,
        "task": task,
        "split": split,
    }


def flatten_claim_based_records(
    records: List[Dict[str, Any]],
    dataset_key: str,
    task: str,
    split: str,
) -> List[Dict[str, Any]]:
    """
    For claim-based datasets:
      - forget_sc_1
      - forget_sc_2
      - retain_external_sc
    """
    rows: List[Dict[str, Any]] = []

    for rec in records:
        pdf_name = rec.get("pdf_name", "")
        qa_by_claim = rec.get("qa_by_claim", []) or []

        local_counter = 1
        for claim_obj in qa_by_claim:
            if not isinstance(claim_obj, dict):
                continue

            for qtype in QA_TYPES:
                items = claim_obj.get(qtype, []) or []
                for item in items:
                    if not isinstance(item, dict):
                        continue

                    rows.append(
                        make_row(
                            pdf_name=pdf_name,
                            dataset_key=dataset_key,
                            qtype=qtype,
                            question_index=local_counter,
                            question=item.get("question", ""),
                            answer=item.get("answer", ""),
                            task=task,
                            split=split,
                        )
                    )
                    local_counter += 1

    return rows


def flatten_internal_records(
    records: List[Dict[str, Any]],
    dataset_key: str,
    task: str,
    split: str,
) -> List[Dict[str, Any]]:
    """
    For retain_internal_sc

    Structure:
      rec["qa_by_base"] = {
        "mcq": [...],
        ...
      }
    """
    rows: List[Dict[str, Any]] = []

    for rec in records:
        pdf_name = rec.get("pdf_name", "")
        qa_by_base = rec.get("qa_by_base", {}) or {}

        local_counter = 1
        for qtype in QA_TYPES:
            items = qa_by_base.get(qtype, []) or []
            for item in items:
                if not isinstance(item, dict):
                    continue

                rows.append(
                    make_row(
                        pdf_name=pdf_name,
                        dataset_key=dataset_key,
                        qtype=qtype,
                        question_index=local_counter,
                        question=item.get("question", ""),
                        answer=item.get("answer", ""),
                        task=task,
                        split=split,
                    )
                )
                local_counter += 1

    return rows


def flatten_derived_records(
    records: List[Dict[str, Any]],
    dataset_key: str,
    task: str,
    split: str,
) -> List[Dict[str, Any]]:
    """
    For derived_sc

    Structure:
      rec["derived_qa"] = {
        "mcq": [...],
        ...
      }
    """
    rows: List[Dict[str, Any]] = []

    for rec in records:
        pdf_name = rec.get("pdf_name", "")
        derived_qa = rec.get("derived_qa", {}) or {}

        local_counter = 1
        for qtype in QA_TYPES:
            items = derived_qa.get(qtype, []) or []
            for item in items:
                if not isinstance(item, dict):
                    continue

                rows.append(
                    make_row(
                        pdf_name=pdf_name,
                        dataset_key=dataset_key,
                        qtype=qtype,
                        question_index=local_counter,
                        question=item.get("question", ""),
                        answer=item.get("answer", ""),
                        task=task,
                        split=split,
                    )
                )
                local_counter += 1

    return rows


# ============================================================
# Export wrapper
# ============================================================

def export_rows(rows: List[Dict[str, Any]], name: str, output_dir: Path) -> Dict[str, Any]:
    jsonl_path = output_dir / f"{name}.jsonl"
    parquet_path = output_dir / f"{name}.parquet"

    write_jsonl(rows, jsonl_path)
    write_parquet(rows, parquet_path)

    return {
        "dataset": name,
        "row_count": len(rows),
        "output_jsonl": str(jsonl_path),
        "output_parquet": str(parquet_path),
    }


# ============================================================
# Main export
# ============================================================

def export_all_common_datasets(config: AppConfig) -> Dict[str, Any]:
    output_dir = Path(getattr(config, "common_export_dir", Path("export_data")))
    output_dir.mkdir(parents=True, exist_ok=True)

    forget_path = Path(getattr(config, "common_forget_output_json", Path("forget_common.json")))
    retain_external_path = Path(getattr(config, "common_retain_output_json", Path("retain_external_common.json")))
    retain_internal_path = Path(getattr(config, "common_retain_internal_output_json", Path("retain_internal_common.json")))
    derived_path = Path(getattr(config, "common_derived_output_json", Path("derived_common.json")))

    for p in [forget_path, retain_external_path, retain_internal_path, derived_path]:
        if not p.exists():
            raise FileNotFoundError(f"Common dataset input file not found: {p}")

    forget_records = ensure_list(load_json(forget_path), "forget common JSON")
    retain_external_records = ensure_list(load_json(retain_external_path), "retain external common JSON")
    retain_internal_records = ensure_list(load_json(retain_internal_path), "retain internal common JSON")
    derived_records = ensure_list(load_json(derived_path), "derived common JSON")

    # split forget
    forget_sc_1_records, forget_sc_2_records = split_forget_records(forget_records)

    # flatten forget splits
    forget_sc_1_rows = flatten_claim_based_records(
        forget_sc_1_records,
        dataset_key="forget_sc_1",
        task="ForgetSet",
        split="forget_sc_1",
    )
    forget_sc_2_rows = flatten_claim_based_records(
        forget_sc_2_records,
        dataset_key="forget_sc_2",
        task="ForgetSet",
        split="forget_sc_2",
    )

    # flatten retain external
    retain_external_rows = flatten_claim_based_records(
        retain_external_records,
        dataset_key="retain_external",
        task="RetainExternalSet",
        split="retain_external",
    )

    # flatten retain internal
    retain_internal_rows = flatten_internal_records(
        retain_internal_records,
        dataset_key="retain_internal",
        task="RetainInternalSet",
        split="retain_internal",
    )

    # merge into total retain
    retain_total_rows = retain_external_rows + retain_internal_rows
    retain_total_rows = [
        {
            **row,
            "id": row["id"].replace("|retain_external|", "|retain|").replace("|retain_internal|", "|retain|"),
            "task": "RetainSet",
            "split": "retain",
        }
        for row in retain_total_rows
    ]

    # flatten derived
    derived_rows = flatten_derived_records(
        derived_records,
        dataset_key="derived",
        task="DerivedSet",
        split="derived",
    )

    summaries: List[Dict[str, Any]] = []
    summaries.append(export_rows(forget_sc_1_rows, "forget_sc_1", output_dir))
    summaries.append(export_rows(forget_sc_2_rows, "forget_sc_2", output_dir))
    summaries.append(export_rows(retain_external_rows, "retain_external_sc", output_dir))
    summaries.append(export_rows(retain_internal_rows, "retain_internal_sc", output_dir))
    summaries.append(export_rows(retain_total_rows, "retain_sc", output_dir))
    summaries.append(export_rows(derived_rows, "derived_sc", output_dir))

    manifest = {
        "export_dir": str(output_dir),
        "inputs": {
            "forget": str(forget_path),
            "retain_external": str(retain_external_path),
            "retain_internal": str(retain_internal_path),
            "derived": str(derived_path),
        },
        "datasets": summaries,
    }

    manifest_path = output_dir / "common_export_manifest.json"
    save_json(manifest, manifest_path)

    print()
    print("=" * 120)
    print("COMMON DATASET EXPORT SUMMARY")
    print("=" * 120)
    for item in summaries:
        print(
            f"{item['dataset']:<22} | rows={item['row_count']:<6} | "
            f"jsonl={item['output_jsonl']} | parquet={item['output_parquet']}"
        )
    print("-" * 120)
    print(f"Manifest: {manifest_path}")

    return manifest


if __name__ == "__main__":
    config = AppConfig()
    export_all_common_datasets(config)
