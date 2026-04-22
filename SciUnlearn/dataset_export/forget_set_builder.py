# -*- coding: utf-8 -*-

import json
from pathlib import Path
from typing import List, Dict, Any

import pyarrow as pa
import pyarrow.parquet as pq

from config import AppConfig


def load_json(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input JSON not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Input must be a JSON array (list of records).")

    return data


def make_row(row_id: str, question: str, answer: str, config: AppConfig) -> Dict[str, str]:
    """
    Build one forget-set row with the exact target schema.
    """
    return {
        "id": row_id,
        "input": (question or "").strip(),
        "output": (answer or "").strip(),
        "task": config.forget_set_task_name,
        "split": config.forget_set_split_name,
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
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_forget_set_from_records(
    data: List[Dict[str, Any]],
    config: AppConfig,
) -> Dict[str, int]:
    """
    Build q1/q2 forget-set files from already-filtered final JSON records.
    Returns summary counts.
    """
    out_dir = config.forget_set_out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows_q1: List[Dict[str, str]] = []
    rows_q2: List[Dict[str, str]] = []

    for rec in data:
        pdf_name = rec.get("pdf_name", "")
        qa_by_claim = rec.get("qa_by_claim", []) or []

        for claim_idx, claim_obj in enumerate(qa_by_claim):
            for qtype in config.all_qa_types:
                items = claim_obj.get(qtype, []) or []
                if len(items) < 2:
                    continue

                # Q1
                q1 = items[0]
                q1_question = (q1.get("question") or "").strip()
                q1_answer = (q1.get("answer") or "").strip()

                # Q2
                q2 = items[1]
                q2_question = (q2.get("question") or "").strip()
                q2_answer = (q2.get("answer") or "").strip()

                base_id = f"{pdf_name}|claim{claim_idx}|{qtype}"
                row1_id = f"{base_id}|Q1"
                row2_id = f"{base_id}|Q2"

                rows_q1.append(make_row(row1_id, q1_question, q1_answer, config))
                rows_q2.append(make_row(row2_id, q2_question, q2_answer, config))

    # ---------- Write Parquet ----------
    out_q1_parquet = out_dir / config.forget_set_q1_parquet
    out_q2_parquet = out_dir / config.forget_set_q2_parquet

    table_q1 = to_table(rows_q1)
    table_q2 = to_table(rows_q2)

    pq.write_table(table_q1, out_q1_parquet)
    pq.write_table(table_q2, out_q2_parquet)

    print(f"[OK] Parquet created: {out_q1_parquet}    rows={table_q1.num_rows}")
    print(f"[OK] Parquet created: {out_q2_parquet}    rows={table_q2.num_rows}")

    # ---------- Write JSONL ----------
    out_q1_jsonl = out_dir / config.forget_set_q1_jsonl
    out_q2_jsonl = out_dir / config.forget_set_q2_jsonl

    write_jsonl(rows_q1, out_q1_jsonl)
    write_jsonl(rows_q2, out_q2_jsonl)

    print(f"[OK] JSONL created: {out_q1_jsonl}    lines={len(rows_q1)}")
    print(f"[OK] JSONL created: {out_q2_jsonl}    lines={len(rows_q2)}")

    return {
        "papers": len(data),
        "q1_rows": len(rows_q1),
        "q2_rows": len(rows_q2),
    }


def build_forget_set(config: AppConfig) -> Dict[str, int]:
    """
    Load the final covered JSON and create forget-set outputs.
    """
    data = load_json(config.pruned_forget_output_json)
    return build_forget_set_from_records(data, config)