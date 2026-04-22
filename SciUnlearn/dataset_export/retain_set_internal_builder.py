# -*- coding: utf-8 -*-

from pathlib import Path
from typing import List, Dict, Any

import pyarrow as pa
import pyarrow.parquet as pq

from config import AppConfig
from utils.json_utils import load_json


def make_row(row_id: str, question: str, answer: str, config: AppConfig) -> Dict[str, str]:
    return {
        "id": row_id,
        "input": (question or "").strip(),
        "output": (answer or "").strip(),
        "task": config.retain_internal_task_name,
        "split": config.retain_internal_split_name,
    }


def to_table(rows: List[Dict[str, str]]) -> pa.Table:
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


def build_internal_rows(records: List[Dict[str, Any]], config: AppConfig) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []

    for rec in records:
        pdf_name = rec.get("pdf_name", "")
        qa_by_base = rec.get("qa_by_base", {}) or {}

        for qtype in config.all_qa_types:
            items = qa_by_base.get(qtype, []) or []

            for idx, qa in enumerate(items, start=1):
                question = (qa.get("question") or "").strip()
                answer = (qa.get("answer") or "").strip()

                if not question or not answer:
                    continue

                row_id = f"{pdf_name}|base|{qtype}|Q{idx}"
                rows.append(make_row(row_id, question, answer, config))

    return rows


def build_retain_internal_export(config: AppConfig) -> Dict[str, int]:
    if not config.pruned_retain_internal_output_json.exists():
        raise FileNotFoundError(f"Internal retain input JSON not found: {config.pruned_retain_internal_output_json}")

    records = load_json(config.pruned_retain_internal_output_json)
    if not isinstance(records, list):
        raise ValueError("retain_set_internal.json must contain a JSON array.")

    survived_records = [
        rec for rec in records
        if rec.get("internal_retain_survived", False) and rec.get("qa_by_base")
    ]

    failed_records = [
        rec for rec in records
        if not rec.get("internal_retain_survived", False)
    ]

    rows = build_internal_rows(survived_records, config)

    out_dir = config.retain_internal_export_out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    out_parquet = out_dir / config.retain_internal_export_parquet_name
    table = to_table(rows)
    pq.write_table(table, out_parquet)

    out_jsonl = out_dir / config.retain_internal_export_jsonl_name
    write_jsonl(rows, out_jsonl)

    print(f"[OK] Retain internal Parquet created: {out_parquet}    rows={table.num_rows}")
    print(f"[OK] Retain internal JSONL created: {out_jsonl}    lines={len(rows)}")

    return {
        "records_total": len(records),
        "records_survived": len(survived_records),
        "records_failed": len(failed_records),
        "rows": len(rows),
    }