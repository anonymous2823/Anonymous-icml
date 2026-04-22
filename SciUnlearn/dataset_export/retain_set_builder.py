# -*- coding: utf-8 -*-

from pathlib import Path
from typing import List, Dict, Any

import pyarrow as pa
import pyarrow.parquet as pq

from config import AppConfig
from utils.json_utils import load_json


def make_row(row_id: str, question: str, answer: str, config: AppConfig) -> Dict[str, str]:
    """
    Build one retain-set export row using the same schema style as forget-set export.
    """
    return {
        "id": row_id,
        "input": (question or "").strip(),
        "output": (answer or "").strip(),
        "task": config.retain_export_task_name,
        "split": config.retain_export_split_name,
    }


def to_table(rows: List[Dict[str, str]]) -> pa.Table:
    """
    Convert rows to a pyarrow Table with a stable schema (even if empty).
    """
    schema = pa.schema([
        pa.field("id", pa.string()),
        pa.field("input", pa.string()),
        pa.field("output", pa.string()),
        pa.field("task", pa.string()),
        pa.field("split", pa.string()),
    ])

    if not rows:
        return pa.Table.from_arrays(
            [pa.array([], type=f.type) for f in schema],
            schema=schema,
        )

    tbl = pa.Table.from_pylist(rows)
    return tbl.cast(schema)


def write_jsonl(rows: List[Dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            import json
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_retain_rows(records: List[Dict[str, Any]], config: AppConfig) -> List[Dict[str, str]]:
    """
    Flatten retain_paper_claims.json into one export row list.

    Unlike forget-set export:
    - we do NOT split into q1/q2 files
    - we keep every available QA item in one list
    """
    rows: List[Dict[str, str]] = []

    for rec in records:
        anchor_corpus_id = rec.get("anchor_corpus_id", "")
        qa_by_claim = rec.get("qa_by_claim", []) or []

        for claim_idx, claim_obj in enumerate(qa_by_claim):
            for qtype in config.all_qa_types:
                items = claim_obj.get(qtype, []) or []

                for item_idx, item in enumerate(items, start=1):
                    question = (item.get("question") or "").strip()
                    answer = (item.get("answer") or "").strip()

                    if not question or not answer:
                        continue

                    row_id = f"{anchor_corpus_id}|claim{claim_idx}|{qtype}|Q{item_idx}"
                    rows.append(make_row(row_id, question, answer, config))

    return rows


def build_retain_set_export(config: AppConfig) -> Dict[str, int]:
    """
    Read retain_paper_claims.json and create:
    - one Parquet file
    - one JSONL file
    """
    if not config.pruned_retain_output_json.exists():
        raise FileNotFoundError(f"Retain input JSON not found: {config.pruned_retain_output_json}")

    records = load_json(config.pruned_retain_output_json)
    if not isinstance(records, list):
        raise ValueError("retain_paper_claims.json must contain a JSON array.")

    rows = build_retain_rows(records, config)

    out_dir = config.retain_export_out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- Write Parquet ----------
    out_parquet = out_dir / config.retain_export_parquet_name
    table = to_table(rows)
    pq.write_table(table, out_parquet)

    print(f"[OK] Retain Parquet created: {out_parquet}    rows={table.num_rows}")

    # ---------- Write JSONL ----------
    out_jsonl = out_dir / config.retain_export_jsonl_name
    write_jsonl(rows, out_jsonl)

    print(f"[OK] Retain JSONL created: {out_jsonl}    lines={len(rows)}")

    return {
        "records": len(records),
        "rows": len(rows),
    }