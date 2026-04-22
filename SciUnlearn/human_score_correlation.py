from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from scipy.stats import spearmanr


DEFAULT_INPUT = Path("claim_metric_verification_5papers_with_human.json")
DEFAULT_CORR_OUTPUT = Path("claim_metric_verification_5papers_spearman.json")
METRIC_KEYS = ("accuracy", "independence", "clarity")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute Spearman correlation between model score and existing human_score "
            "values from a with-human metric JSON file."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input metric JSON file")
    parser.add_argument(
        "--output-correlation",
        type=Path,
        default=DEFAULT_CORR_OUTPUT,
        help="Output file for computed correlation summary",
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def collect_pairs_from_human_scores(
    payload: Dict[str, Any],
) -> Tuple[Dict[str, List[Tuple[int, int]]], int, int]:
    pairs_by_metric: Dict[str, List[Tuple[int, int]]] = {k: [] for k in METRIC_KEYS}
    total_pairs = 0
    skipped_missing_human_score = 0

    evaluations = payload.get("evaluations", [])
    for evaluation in evaluations:
        for side_key in ("forget", "retain"):
            side = evaluation.get(side_key)
            if not isinstance(side, dict):
                continue

            claims = side.get("claims", [])
            for claim in claims:
                metrics = claim.get("metrics")
                if not isinstance(metrics, dict):
                    continue

                for metric_name in METRIC_KEYS:
                    metric_obj = metrics.get(metric_name)
                    if not isinstance(metric_obj, dict):
                        continue

                    score = metric_obj.get("score")
                    if not isinstance(score, int):
                        continue

                    human_score = metric_obj.get("human_score")
                    if not isinstance(human_score, int):
                        skipped_missing_human_score += 1
                        continue

                    pairs_by_metric[metric_name].append((score, human_score))
                    total_pairs += 1

    return pairs_by_metric, total_pairs, skipped_missing_human_score


def compute_spearman(pairs: List[Tuple[int, int]]) -> Dict[str, Any]:
    if len(pairs) < 2:
        return {
            "n": len(pairs),
            "spearman_rho": None,
            "p_value": None,
            "note": "Need at least 2 paired scores to compute correlation.",
        }

    x = [a for a, _ in pairs]
    y = [b for _, b in pairs]
    rho, p_val = spearmanr(x, y)

    if rho != rho:  # NaN check
        return {
            "n": len(pairs),
            "spearman_rho": None,
            "p_value": None,
            "note": "Correlation undefined (likely because one series is constant).",
        }

    return {
        "n": len(pairs),
        "spearman_rho": float(rho),
        "p_value": float(p_val),
    }


def main() -> None:
    args = parse_args()

    payload = load_json(args.input)
    pairs_by_metric, paired_count, skipped_count = collect_pairs_from_human_scores(payload)

    overall_pairs: List[Tuple[int, int]] = []
    for pairs in pairs_by_metric.values():
        overall_pairs.extend(pairs)

    report = {
        "input_file": str(args.input),
        "metric_pairs_used": paired_count,
        "metric_entries_skipped_missing_human_score": skipped_count,
        "correlation_method": "Spearman rank correlation",
        "why_this_method": "Scores are ordinal (0-5), so rank-based correlation is the appropriate default.",
        "results": {
            "accuracy": compute_spearman(pairs_by_metric["accuracy"]),
            "independence": compute_spearman(pairs_by_metric["independence"]),
            "clarity": compute_spearman(pairs_by_metric["clarity"]),
            "overall": compute_spearman(overall_pairs),
        },
    }
    save_json(args.output_correlation, report)

    print(f"Input with human scores: {args.input}")
    print(f"Paired score/human_score entries used: {paired_count}")
    print(f"Skipped entries without human_score: {skipped_count}")
    print(f"Correlation report: {args.output_correlation}")


if __name__ == "__main__":
    main()
