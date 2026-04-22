
# # # # import json
# # # # import re
# # # # import statistics as stats
# # # # from collections import Counter
# # # # from pathlib import Path
# # # # from typing import Any, Dict, List, Optional, Set, Tuple

# # # # try:
# # # #     from scipy.stats import chi2_contingency, ks_2samp, mannwhitneyu
# # # # except Exception:
# # # #     chi2_contingency = None
# # # #     ks_2samp = None
# # # #     mannwhitneyu = None

# # # # from config import AppConfig
# # # # from utils.json_utils import load_json, save_json


# # # # def safe_median(xs: List[float]) -> Optional[float]:
# # # #     return float(stats.median(xs)) if xs else None


# # # # def safe_mean(xs: List[float]) -> Optional[float]:
# # # #     return float(stats.mean(xs)) if xs else None


# # # # def safe_stdev(xs: List[float]) -> Optional[float]:
# # # #     return float(stats.pstdev(xs)) if len(xs) > 1 else (0.0 if xs else None)


# # # # def numeric_summary(xs: List[float]) -> Dict[str, Optional[float]]:
# # # #     if not xs:
# # # #         return {
# # # #             "count": 0,
# # # #             "mean": None,
# # # #             "median": None,
# # # #             "std": None,
# # # #             "min": None,
# # # #             "max": None,
# # # #             "sum": 0,
# # # #         }
# # # #     return {
# # # #         "count": len(xs),
# # # #         "mean": safe_mean(xs),
# # # #         "median": safe_median(xs),
# # # #         "std": safe_stdev(xs),
# # # #         "min": float(min(xs)),
# # # #         "max": float(max(xs)),
# # # #         "sum": float(sum(xs)),
# # # #     }


# # # # def stem_id(name: str) -> str:
# # # #     return Path((name or "").strip()).stem.strip()


# # # # def anchor_id_from_record(rec: Dict[str, Any]) -> str:
# # # #     for key in ("anchor_forget_paper_id", "anchor_corpus_id", "anchor_forget_pdf_name", "pdf_name"):
# # # #         val = (rec.get(key) or "").strip()
# # # #         if val:
# # # #             return stem_id(val)
# # # #     return ""


# # # # def ensure_list(data: Any, name: str) -> List[Dict[str, Any]]:
# # # #     if isinstance(data, dict):
# # # #         data = [data]
# # # #     if not isinstance(data, list):
# # # #         raise ValueError(f"{name} must contain a JSON array (or a single dict).")
# # # #     out: List[Dict[str, Any]] = []
# # # #     for i, rec in enumerate(data):
# # # #         if not isinstance(rec, dict):
# # # #             raise ValueError(f"{name}[{i}] is not a JSON object.")
# # # #         out.append(rec)
# # # #     return out


# # # # def infer_year_index_dir(config: AppConfig) -> Path:
# # # #     for attr in ("year_wise_corpus_ids_dir", "year_corpus_ids_dir", "corpus_ids_dir"):
# # # #         p = getattr(config, attr, None)
# # # #         if p:
# # # #             return Path(p)
# # # #     return Path("year_wise_corpus_ids")


# # # # def build_year_lookup(year_dir: Path) -> Tuple[Dict[str, int], Dict[str, int]]:
# # # #     """
# # # #     Build corpus-id -> year lookup from files like corpus_ids_2018.txt.
# # # #     Returns:
# # # #       - id_to_year
# # # #       - file_counts_by_year
# # # #     """
# # # #     id_to_year: Dict[str, int] = {}
# # # #     file_counts_by_year: Dict[str, int] = {}

# # # #     if not year_dir.exists():
# # # #         return id_to_year, file_counts_by_year

# # # #     for path in sorted(year_dir.glob("*.txt")):
# # # #         # FIXED: only one backslash inside the raw regex
# # # #         m = re.search(r"(19|20)\d{2}", path.stem)
# # # #         if not m:
# # # #             continue

# # # #         year = int(m.group(0))
# # # #         count = 0

# # # #         with path.open("r", encoding="utf-8") as f:
# # # #             for line in f:
# # # #                 cid = line.strip()
# # # #                 if not cid:
# # # #                     continue
# # # #                 id_to_year[cid] = year
# # # #                 count += 1

# # # #         file_counts_by_year[str(year)] = count

# # # #     return id_to_year, file_counts_by_year


# # # # def count_questions_claim_based(rec: Dict[str, Any]) -> Dict[str, int]:
# # # #     counts = {"mcq": 0, "true_false": 0, "fill_blank": 0, "assertion_reason": 0}
# # # #     for claim_obj in rec.get("qa_by_claim", []) or []:
# # # #         for qt in counts:
# # # #             counts[qt] += len(claim_obj.get(qt, []) or [])
# # # #     counts["total"] = sum(counts.values())
# # # #     return counts


# # # # def count_questions_internal(rec: Dict[str, Any]) -> Dict[str, int]:
# # # #     counts = {"mcq": 0, "true_false": 0, "fill_blank": 0, "assertion_reason": 0}
# # # #     qa_by_base = rec.get("qa_by_base", {}) or {}
# # # #     for qt in counts:
# # # #         counts[qt] = len(qa_by_base.get(qt, []) or [])
# # # #     counts["total"] = sum(counts.values())
# # # #     return counts


# # # # def count_questions_derived(rec: Dict[str, Any]) -> Dict[str, int]:
# # # #     counts = {"mcq": 0, "true_false": 0, "fill_blank": 0, "assertion_reason": 0}
# # # #     derived_qa = rec.get("derived_qa", {}) or {}
# # # #     for qt in counts:
# # # #         counts[qt] = len(derived_qa.get(qt, []) or [])
# # # #     counts["total"] = sum(counts.values())
# # # #     return counts


# # # # def forget_pair_split_stats(forget_records: List[Dict[str, Any]]) -> Dict[str, Any]:
# # # #     q1_counter = Counter()
# # # #     q2_counter = Counter()
# # # #     odd_groups = []
# # # #     groups_total = 0

# # # #     for rec in forget_records:
# # # #         pdf_name = rec.get("pdf_name", "")
# # # #         for claim_idx, claim_obj in enumerate(rec.get("qa_by_claim", []) or []):
# # # #             for qt in ["mcq", "true_false", "fill_blank", "assertion_reason"]:
# # # #                 items = claim_obj.get(qt, []) or []
# # # #                 n = len(items)
# # # #                 if n == 0:
# # # #                     continue
# # # #                 groups_total += 1
# # # #                 half = n // 2
# # # #                 q1_n = half
# # # #                 q2_n = n - half
# # # #                 q1_counter[qt] += q1_n
# # # #                 q2_counter[qt] += q2_n
# # # #                 if n % 2 != 0:
# # # #                     odd_groups.append({
# # # #                         "pdf_name": pdf_name,
# # # #                         "claim_index": claim_idx,
# # # #                         "qtype": qt,
# # # #                         "count": n,
# # # #                     })

# # # #     q1_counter["total"] = sum(q1_counter.values())
# # # #     q2_counter["total"] = sum(q2_counter.values())

# # # #     return {
# # # #         "groups_total": groups_total,
# # # #         "q1_counts": dict(q1_counter),
# # # #         "q2_counts": dict(q2_counter),
# # # #         "absolute_imbalance_total": abs(q1_counter["total"] - q2_counter["total"]),
# # # #         "odd_groups_count": len(odd_groups),
# # # #         "odd_groups_examples": odd_groups[:20],
# # # #     }


# # # # def summarize_forget(records: List[Dict[str, Any]], year_lookup: Dict[str, int]) -> Dict[str, Any]:
# # # #     year_counter = Counter()
# # # #     qtype_counter = Counter()
# # # #     claims_per_paper: List[int] = []
# # # #     questions_per_paper: List[int] = []
# # # #     verbatim_counts: List[int] = []
# # # #     missing_year_ids: List[str] = []
# # # #     duplicate_ids = []
# # # #     seen = set()

# # # #     for rec in records:
# # # #         fid = stem_id(rec.get("pdf_name", ""))
# # # #         if fid in seen:
# # # #             duplicate_ids.append(fid)
# # # #         seen.add(fid)

# # # #         year = year_lookup.get(fid)
# # # #         if year is None:
# # # #             missing_year_ids.append(fid)
# # # #         else:
# # # #             year_counter[str(year)] += 1

# # # #         claims = rec.get("paper_claims", []) or []
# # # #         claims_per_paper.append(len(claims))

# # # #         qcounts = count_questions_claim_based(rec)
# # # #         questions_per_paper.append(qcounts["total"])
# # # #         for qt in ["mcq", "true_false", "fill_blank", "assertion_reason"]:
# # # #             qtype_counter[qt] += qcounts[qt]

# # # #         verbatim_counts.append(len(rec.get("verbatim_claims", []) or []))

# # # #     qtype_counter["total"] = sum(qtype_counter.values())
# # # #     return {
# # # #         "dataset": "forget",
# # # #         "record_count": len(records),
# # # #         "unique_paper_ids": len(seen),
# # # #         "duplicate_ids_count": len(set(duplicate_ids)),
# # # #         "duplicate_ids_examples": sorted(set(duplicate_ids))[:20],
# # # #         "year_distribution": dict(year_counter),
# # # #         "missing_year_ids_count": len(set(missing_year_ids)),
# # # #         "missing_year_ids_examples": sorted(set(missing_year_ids))[:20],
# # # #         "claims_per_paper": numeric_summary(claims_per_paper),
# # # #         "verbatim_claims_per_paper": numeric_summary(verbatim_counts),
# # # #         "questions_per_paper": numeric_summary(questions_per_paper),
# # # #         "question_type_counts": dict(qtype_counter),
# # # #         "forget_pair_split": forget_pair_split_stats(records),
# # # #     }


# # # # def summarize_retain_external(records: List[Dict[str, Any]], year_lookup: Dict[str, int]) -> Dict[str, Any]:
# # # #     year_counter = Counter()
# # # #     qtype_counter = Counter()
# # # #     claims_per_paper: List[int] = []
# # # #     questions_per_paper: List[int] = []
# # # #     missing_year_ids: List[str] = []
# # # #     survived_true = 0
# # # #     survived_false = 0
# # # #     selection_year_counter = Counter()

# # # #     for rec in records:
# # # #         aid = anchor_id_from_record(rec)
# # # #         year = year_lookup.get(aid)
# # # #         if year is None:
# # # #             missing_year_ids.append(aid)
# # # #         else:
# # # #             year_counter[str(year)] += 1

# # # #         if rec.get("retain_survived", False):
# # # #             survived_true += 1
# # # #         else:
# # # #             survived_false += 1

# # # #         claims = rec.get("paper_claims", []) or []
# # # #         claims_per_paper.append(len(claims))

# # # #         qcounts = count_questions_claim_based(rec)
# # # #         questions_per_paper.append(qcounts["total"])
# # # #         for qt in ["mcq", "true_false", "fill_blank", "assertion_reason"]:
# # # #             qtype_counter[qt] += qcounts[qt]

# # # #         sel = rec.get("selected_reference", {}) or {}
# # # #         selected_corpus_id = (sel.get("corpusId") or "").strip()
# # # #         selected_year = year_lookup.get(selected_corpus_id)
# # # #         if selected_year is not None:
# # # #             selection_year_counter[str(selected_year)] += 1

# # # #     qtype_counter["total"] = sum(qtype_counter.values())

# # # #     return {
# # # #         "dataset": "retain_external",
# # # #         "record_count": len(records),
# # # #         "retain_survived_true": survived_true,
# # # #         "retain_survived_false": survived_false,
# # # #         "anchor_year_distribution": dict(year_counter),
# # # #         "selected_reference_year_distribution": dict(selection_year_counter),
# # # #         "missing_year_anchor_ids_count": len(set(missing_year_ids)),
# # # #         "missing_year_anchor_ids_examples": sorted(set(missing_year_ids))[:20],
# # # #         "claims_per_paper": numeric_summary(claims_per_paper),
# # # #         "questions_per_paper": numeric_summary(questions_per_paper),
# # # #         "question_type_counts": dict(qtype_counter),
# # # #     }


# # # # def summarize_retain_internal(records: List[Dict[str, Any]], year_lookup: Dict[str, int]) -> Dict[str, Any]:
# # # #     year_counter = Counter()
# # # #     qtype_counter = Counter()
# # # #     questions_per_paper: List[int] = []
# # # #     missing_year_ids: List[str] = []
# # # #     survived_true = 0
# # # #     survived_false = 0
# # # #     paper_type_counter = Counter()
# # # #     topic_nonempty = 0

# # # #     for rec in records:
# # # #         aid = anchor_id_from_record(rec)
# # # #         year = year_lookup.get(aid)
# # # #         if year is None:
# # # #             missing_year_ids.append(aid)
# # # #         else:
# # # #             year_counter[str(year)] += 1

# # # #         survived = bool(rec.get("internal_retain_survived", False))
# # # #         if survived:
# # # #             survived_true += 1
# # # #         else:
# # # #             survived_false += 1

# # # #         qcounts = count_questions_internal(rec)
# # # #         questions_per_paper.append(qcounts["total"])
# # # #         for qt in ["mcq", "true_false", "fill_blank", "assertion_reason"]:
# # # #             qtype_counter[qt] += qcounts[qt]

# # # #         paper_base = rec.get("paper_base", {}) or {}
# # # #         ptype = (paper_base.get("paper_type") or "").strip()
# # # #         if ptype:
# # # #             paper_type_counter[ptype] += 1
# # # #         if (paper_base.get("topic") or "").strip():
# # # #             topic_nonempty += 1

# # # #     qtype_counter["total"] = sum(qtype_counter.values())
# # # #     return {
# # # #         "dataset": "retain_internal",
# # # #         "record_count": len(records),
# # # #         "internal_retain_survived_true": survived_true,
# # # #         "internal_retain_survived_false": survived_false,
# # # #         "year_distribution": dict(year_counter),
# # # #         "missing_year_anchor_ids_count": len(set(missing_year_ids)),
# # # #         "missing_year_anchor_ids_examples": sorted(set(missing_year_ids))[:20],
# # # #         "questions_per_paper": numeric_summary(questions_per_paper),
# # # #         "question_type_counts": dict(qtype_counter),
# # # #         "paper_base_type_distribution": dict(paper_type_counter),
# # # #         "paper_base_topic_nonempty_count": topic_nonempty,
# # # #     }


# # # # def summarize_derived(records: List[Dict[str, Any]], year_lookup: Dict[str, int]) -> Dict[str, Any]:
# # # #     year_counter = Counter()
# # # #     qtype_counter = Counter()
# # # #     questions_per_paper: List[int] = []
# # # #     source_questions_per_paper: List[int] = []
# # # #     missing_year_ids: List[str] = []
# # # #     survived_true = 0
# # # #     survived_false = 0
# # # #     source_type_counter = Counter()

# # # #     for rec in records:
# # # #         aid = anchor_id_from_record(rec)
# # # #         year = year_lookup.get(aid)
# # # #         if year is None:
# # # #             missing_year_ids.append(aid)
# # # #         else:
# # # #             year_counter[str(year)] += 1

# # # #         survived = bool(rec.get("derived_survived", False))
# # # #         if survived:
# # # #             survived_true += 1
# # # #         else:
# # # #             survived_false += 1

# # # #         qcounts = count_questions_derived(rec)
# # # #         questions_per_paper.append(qcounts["total"])
# # # #         for qt in ["mcq", "true_false", "fill_blank", "assertion_reason"]:
# # # #             qtype_counter[qt] += qcounts[qt]

# # # #         sq = rec.get("source_questions", []) or []
# # # #         source_questions_per_paper.append(len(sq))
# # # #         for item in sq:
# # # #             st = (item.get("source_type") or "").strip()
# # # #             if st:
# # # #                 source_type_counter[st] += 1

# # # #     qtype_counter["total"] = sum(qtype_counter.values())
# # # #     return {
# # # #         "dataset": "derived",
# # # #         "record_count": len(records),
# # # #         "derived_survived_true": survived_true,
# # # #         "derived_survived_false": survived_false,
# # # #         "year_distribution": dict(year_counter),
# # # #         "missing_year_anchor_ids_count": len(set(missing_year_ids)),
# # # #         "missing_year_anchor_ids_examples": sorted(set(missing_year_ids))[:20],
# # # #         "questions_per_paper": numeric_summary(questions_per_paper),
# # # #         "question_type_counts": dict(qtype_counter),
# # # #         "source_questions_per_paper": numeric_summary(source_questions_per_paper),
# # # #         "source_question_type_distribution": dict(source_type_counter),
# # # #     }


# # # # def compute_id_sets(forget_records, retain_external_records, retain_internal_records, derived_records) -> Dict[str, Set[str]]:
# # # #     forget_ids = {stem_id(r.get("pdf_name", "")) for r in forget_records if stem_id(r.get("pdf_name", ""))}
# # # #     retain_external_ids = {anchor_id_from_record(r) for r in retain_external_records if anchor_id_from_record(r)}
# # # #     retain_internal_ids = {anchor_id_from_record(r) for r in retain_internal_records if anchor_id_from_record(r)}
# # # #     derived_ids = {anchor_id_from_record(r) for r in derived_records if anchor_id_from_record(r)}
# # # #     return {
# # # #         "forget": forget_ids,
# # # #         "retain_external": retain_external_ids,
# # # #         "retain_internal": retain_internal_ids,
# # # #         "derived": derived_ids,
# # # #     }


# # # # def cross_dataset_consistency(id_sets: Dict[str, Set[str]]) -> Dict[str, Any]:
# # # #     names = list(id_sets.keys())
# # # #     common_all = set.intersection(*(id_sets[n] for n in names)) if names else set()
# # # #     pairwise = {}
# # # #     for i, a in enumerate(names):
# # # #         for b in names[i + 1:]:
# # # #             inter = id_sets[a] & id_sets[b]
# # # #             union = id_sets[a] | id_sets[b]
# # # #             jaccard = len(inter) / len(union) if union else 0.0
# # # #             pairwise[f"{a}__{b}"] = {
# # # #                 "intersection": len(inter),
# # # #                 "union": len(union),
# # # #                 "jaccard": jaccard,
# # # #                 "a_minus_b_examples": sorted((id_sets[a] - id_sets[b]))[:20],
# # # #                 "b_minus_a_examples": sorted((id_sets[b] - id_sets[a]))[:20],
# # # #             }
# # # #     return {
# # # #         "common_across_all_count": len(common_all),
# # # #         "common_across_all_examples": sorted(common_all)[:50],
# # # #         "pairwise": pairwise,
# # # #     }


# # # # def year_distribution_test(year_counter_a, year_counter_b, label_a: str, label_b: str) -> Dict[str, Any]:
# # # #     years = sorted(set(year_counter_a.keys()) | set(year_counter_b.keys()))
# # # #     if not years:
# # # #         return {"dataset_a": label_a, "dataset_b": label_b, "test": None}
# # # #     vec_a = [year_counter_a.get(y, 0) for y in years]
# # # #     vec_b = [year_counter_b.get(y, 0) for y in years]
# # # #     out = {
# # # #         "dataset_a": label_a,
# # # #         "dataset_b": label_b,
# # # #         "years": years,
# # # #         "counts_a": vec_a,
# # # #         "counts_b": vec_b,
# # # #     }
# # # #     if chi2_contingency is not None:
# # # #         chi2, p, dof, expected = chi2_contingency([vec_a, vec_b])
# # # #         out["chi_square"] = {
# # # #             "chi2": float(chi2),
# # # #             "p_value": float(p),
# # # #             "dof": int(dof),
# # # #             "expected": [[float(x) for x in row] for row in expected],
# # # #         }
# # # #     else:
# # # #         out["chi_square"] = None
# # # #     return out


# # # # def question_count_distribution_test(per_paper_a, per_paper_b, label_a: str, label_b: str) -> Dict[str, Any]:
# # # #     out = {
# # # #         "dataset_a": label_a,
# # # #         "dataset_b": label_b,
# # # #         "a_summary": numeric_summary(per_paper_a),
# # # #         "b_summary": numeric_summary(per_paper_b),
# # # #     }
# # # #     if mannwhitneyu is not None and per_paper_a and per_paper_b:
# # # #         try:
# # # #             stat, p = mannwhitneyu(per_paper_a, per_paper_b, alternative="two-sided")
# # # #             out["mann_whitney_u"] = {"statistic": float(stat), "p_value": float(p)}
# # # #         except Exception:
# # # #             out["mann_whitney_u"] = None
# # # #     else:
# # # #         out["mann_whitney_u"] = None
# # # #     if ks_2samp is not None and per_paper_a and per_paper_b:
# # # #         try:
# # # #             stat, p = ks_2samp(per_paper_a, per_paper_b)
# # # #             out["kolmogorov_smirnov"] = {"statistic": float(stat), "p_value": float(p)}
# # # #         except Exception:
# # # #             out["kolmogorov_smirnov"] = None
# # # #     else:
# # # #         out["kolmogorov_smirnov"] = None
# # # #     return out


# # # # def build_markdown_report(report: Dict[str, Any]) -> str:
# # # #     lines: List[str] = []
# # # #     lines.append("# Common-Dataset Evaluation Report")
# # # #     lines.append("")
# # # #     lines.append("## 1. Input Files")
# # # #     for k, v in report["paths"].items():
# # # #         lines.append(f"- **{k}**: `{v}`")
# # # #     lines.append("")
# # # #     lines.append("## 2. Common-ID Consistency")
# # # #     c = report["cross_dataset_consistency"]
# # # #     lines.append(f"- Common papers across all 4 JSONs: **{c['common_across_all_count']}**")
# # # #     lines.append("")
# # # #     lines.append("## 3. Dataset Summaries")
# # # #     for key in ["forget", "retain_external", "retain_internal", "derived"]:
# # # #         ds = report["datasets"][key]
# # # #         lines.append(f"### {key}")
# # # #         lines.append(f"- Record count: **{ds.get('record_count')}**")
# # # #         if "question_type_counts" in ds:
# # # #             q = ds["question_type_counts"]
# # # #             lines.append(
# # # #                 f"- Question counts — MCQ: **{q.get('mcq', 0)}**, TF: **{q.get('true_false', 0)}**, Fill: **{q.get('fill_blank', 0)}**, AR: **{q.get('assertion_reason', 0)}**, Total: **{q.get('total', 0)}**"
# # # #             )
# # # #         if "year_distribution" in ds:
# # # #             lines.append(f"- Year coverage: `{ds['year_distribution']}`")
# # # #         elif "anchor_year_distribution" in ds:
# # # #             lines.append(f"- Anchor year coverage: `{ds['anchor_year_distribution']}`")
# # # #         lines.append("")
# # # #     lines.append("## 4. Forget Q1/Q2 Balance")
# # # #     split = report["datasets"]["forget"].get("forget_pair_split", {})
# # # #     lines.append(f"- Total QA groups inspected: **{split.get('groups_total', 0)}**")
# # # #     lines.append(f"- Q1 counts: `{split.get('q1_counts', {})}`")
# # # #     lines.append(f"- Q2 counts: `{split.get('q2_counts', {})}`")
# # # #     lines.append(f"- Odd QA groups count: **{split.get('odd_groups_count', 0)}**")
# # # #     lines.append("")
# # # #     lines.append("## 5. Statistical Tests")
# # # #     for block in report.get("statistical_tests", []):
# # # #         lines.append(f"### {block.get('name', 'test')}")
# # # #         lines.append("```json")
# # # #         lines.append(json.dumps(block, ensure_ascii=False, indent=2))
# # # #         lines.append("```")
# # # #         lines.append("")
# # # #     return "\n".join(lines)


# # # # def evaluate_common_datasets(config: AppConfig) -> Dict[str, Any]:
# # # #     forget_path = Path(getattr(config, "common_forget_output_json", Path("forget_common.json")))
# # # #     retain_external_path = Path(getattr(config, "common_retain_output_json", Path("retain_external_common.json")))
# # # #     retain_internal_path = Path(getattr(config, "common_retain_internal_output_json", Path("retain_internal_common.json")))
# # # #     derived_path = Path(getattr(config, "common_derived_output_json", Path("derived_common.json")))

# # # #     for p in [forget_path, retain_external_path, retain_internal_path, derived_path]:
# # # #         if not p.exists():
# # # #             raise FileNotFoundError(f"Common dataset input file not found: {p}")

# # # #     forget_records = ensure_list(load_json(forget_path), "forget common JSON")
# # # #     retain_external_records = ensure_list(load_json(retain_external_path), "retain external common JSON")
# # # #     retain_internal_records = ensure_list(load_json(retain_internal_path), "retain internal common JSON")
# # # #     derived_records = ensure_list(load_json(derived_path), "derived common JSON")

# # # #     year_dir = infer_year_index_dir(config)
# # # #     year_lookup, year_file_counts = build_year_lookup(year_dir)

# # # #     report = {
# # # #         "paths": {
# # # #             "forget": str(forget_path),
# # # #             "retain_external": str(retain_external_path),
# # # #             "retain_internal": str(retain_internal_path),
# # # #             "derived": str(derived_path),
# # # #             "year_index_dir": str(year_dir),
# # # #         },
# # # #         "year_index_file_counts": year_file_counts,
# # # #         "datasets": {
# # # #             "forget": summarize_forget(forget_records, year_lookup),
# # # #             "retain_external": summarize_retain_external(retain_external_records, year_lookup),
# # # #             "retain_internal": summarize_retain_internal(retain_internal_records, year_lookup),
# # # #             "derived": summarize_derived(derived_records, year_lookup),
# # # #         },
# # # #     }

# # # #     id_sets = compute_id_sets(forget_records, retain_external_records, retain_internal_records, derived_records)
# # # #     report["cross_dataset_consistency"] = cross_dataset_consistency(id_sets)

# # # #     tests: List[Dict[str, Any]] = []
# # # #     f_year = report["datasets"]["forget"].get("year_distribution", {})
# # # #     re_year = report["datasets"]["retain_external"].get("anchor_year_distribution", {})
# # # #     ri_year = report["datasets"]["retain_internal"].get("year_distribution", {})
# # # #     d_year = report["datasets"]["derived"].get("year_distribution", {})

# # # #     tests.append({"name": "year_distribution_forget_vs_retain_external", **year_distribution_test(f_year, re_year, "forget", "retain_external")})
# # # #     tests.append({"name": "year_distribution_forget_vs_retain_internal", **year_distribution_test(f_year, ri_year, "forget", "retain_internal")})
# # # #     tests.append({"name": "year_distribution_forget_vs_derived", **year_distribution_test(f_year, d_year, "forget", "derived")})

# # # #     fq = [count_questions_claim_based(x)["total"] for x in forget_records]
# # # #     req = [count_questions_claim_based(x)["total"] for x in retain_external_records]
# # # #     riq = [count_questions_internal(x)["total"] for x in retain_internal_records]
# # # #     dq = [count_questions_derived(x)["total"] for x in derived_records]

# # # #     tests.append({"name": "question_count_distribution_forget_vs_retain_external", **question_count_distribution_test(fq, req, "forget", "retain_external")})
# # # #     tests.append({"name": "question_count_distribution_forget_vs_retain_internal", **question_count_distribution_test(fq, riq, "forget", "retain_internal")})
# # # #     tests.append({"name": "question_count_distribution_forget_vs_derived", **question_count_distribution_test(fq, dq, "forget", "derived")})

# # # #     report["statistical_tests"] = tests

# # # #     report_json_path = Path(getattr(config, "common_eval_report_json", Path("common_dataset_evaluation_report.json")))
# # # #     report_md_path = Path(getattr(config, "common_eval_report_md", Path("common_dataset_evaluation_report.md")))
# # # #     save_json(report, report_json_path)
# # # #     report_md_path.write_text(build_markdown_report(report), encoding="utf-8")

# # # #     print() 
# # # #     print("=" * 120)
# # # #     print("COMMON DATASET EVALUATION SUMMARY")
# # # #     print("=" * 120)
# # # #     print(f"Forget records           : {report['datasets']['forget']['record_count']}")
# # # #     print(f"Retain external records  : {report['datasets']['retain_external']['record_count']}")
# # # #     print(f"Retain internal records  : {report['datasets']['retain_internal']['record_count']}")
# # # #     print(f"Derived records          : {report['datasets']['derived']['record_count']}")
# # # #     print("-" * 120)
# # # #     print(f"Common ids across all 4  : {report['cross_dataset_consistency']['common_across_all_count']}")
# # # #     print(f"Report JSON              : {report_json_path}")
# # # #     print(f"Report Markdown          : {report_md_path}")
# # # #     return report


# # # # if __name__ == "__main__":
# # # #     config = AppConfig()
# # # #     evaluate_common_datasets(config)


# # # from __future__ import annotations

# # # import json
# # # import re
# # # import statistics as stats
# # # from collections import Counter
# # # from pathlib import Path
# # # from typing import Any, Dict, List, Optional, Set, Tuple

# # # from config import AppConfig
# # # from utils.json_utils import load_json, save_json


# # # # ============================================================
# # # # Generic helpers
# # # # ============================================================

# # # def ensure_list(data: Any, name: str) -> List[Dict[str, Any]]:
# # #     if isinstance(data, dict):
# # #         data = [data]

# # #     if not isinstance(data, list):
# # #         raise ValueError(f"{name} must contain a JSON array (or a single dict).")

# # #     out: List[Dict[str, Any]] = []
# # #     for i, rec in enumerate(data):
# # #         if not isinstance(rec, dict):
# # #             raise ValueError(f"{name}[{i}] is not a JSON object.")
# # #         out.append(rec)
# # #     return out


# # # def safe_mean(xs: List[float]) -> Optional[float]:
# # #     return float(stats.mean(xs)) if xs else None


# # # def safe_median(xs: List[float]) -> Optional[float]:
# # #     return float(stats.median(xs)) if xs else None


# # # def safe_stdev(xs: List[float]) -> Optional[float]:
# # #     return float(stats.pstdev(xs)) if len(xs) > 1 else (0.0 if xs else None)


# # # def numeric_summary(xs: List[float]) -> Dict[str, Optional[float]]:
# # #     if not xs:
# # #         return {
# # #             "count": 0,
# # #             "mean": None,
# # #             "median": None,
# # #             "std": None,
# # #             "min": None,
# # #             "max": None,
# # #             "sum": 0,
# # #         }

# # #     return {
# # #         "count": len(xs),
# # #         "mean": safe_mean(xs),
# # #         "median": safe_median(xs),
# # #         "std": safe_stdev(xs),
# # #         "min": float(min(xs)),
# # #         "max": float(max(xs)),
# # #         "sum": float(sum(xs)),
# # #     }


# # # def stem_id(name: str) -> str:
# # #     return Path((name or "").strip()).stem.strip()


# # # def forget_id_from_record(rec: Dict[str, Any]) -> str:
# # #     return stem_id(rec.get("pdf_name", ""))


# # # def anchor_id_from_record(rec: Dict[str, Any]) -> str:
# # #     """
# # #     Retain / derived records usually identify the forget anchor using one of:
# # #       1. anchor_forget_paper_id
# # #       2. anchor_corpus_id
# # #       3. anchor_forget_pdf_name
# # #       4. fallback: pdf_name
# # #     """
# # #     for key in ("anchor_forget_paper_id", "anchor_corpus_id", "anchor_forget_pdf_name", "pdf_name"):
# # #         val = (rec.get(key) or "").strip()
# # #         if val:
# # #             return stem_id(val)
# # #     return ""


# # # # ============================================================
# # # # Year lookup
# # # # ============================================================

# # # def infer_year_index_dir(config: AppConfig) -> Path:
# # #     """
# # #     Infer the directory containing year-wise corpus-id files.
# # #     """
# # #     for attr in ("year_wise_corpus_ids_dir", "year_corpus_ids_dir", "corpus_ids_dir"):
# # #         p = getattr(config, attr, None)
# # #         if p:
# # #             return Path(p)

# # #     return Path("year_wise_corpus_ids")


# # # def build_year_lookup(year_dir: Path) -> Tuple[Dict[str, int], Dict[str, int]]:
# # #     """
# # #     Build corpus_id -> year from files like corpus_ids_2020.txt
# # #     """
# # #     id_to_year: Dict[str, int] = {}
# # #     file_counts_by_year: Dict[str, int] = {}

# # #     if not year_dir.exists():
# # #         return id_to_year, file_counts_by_year

# # #     for path in sorted(year_dir.glob("*.txt")):
# # #         m = re.search(r"(19|20)\d{2}", path.stem)
# # #         if not m:
# # #             continue

# # #         year = int(m.group(0))
# # #         count = 0

# # #         with path.open("r", encoding="utf-8") as f:
# # #             for line in f:
# # #                 cid = line.strip()
# # #                 if not cid:
# # #                     continue
# # #                 id_to_year[cid] = year
# # #                 count += 1

# # #         file_counts_by_year[str(year)] = count

# # #     return id_to_year, file_counts_by_year


# # # # ============================================================
# # # # Question / claim counters
# # # # ============================================================

# # # def count_questions_claim_based(rec: Dict[str, Any]) -> Dict[str, int]:
# # #     counts = {"mcq": 0, "true_false": 0, "fill_blank": 0, "assertion_reason": 0}

# # #     for claim_obj in rec.get("qa_by_claim", []) or []:
# # #         for qt in counts:
# # #             counts[qt] += len(claim_obj.get(qt, []) or [])

# # #     counts["total"] = sum(counts.values())
# # #     return counts


# # # def count_questions_internal(rec: Dict[str, Any]) -> Dict[str, int]:
# # #     counts = {"mcq": 0, "true_false": 0, "fill_blank": 0, "assertion_reason": 0}
# # #     qa_by_base = rec.get("qa_by_base", {}) or {}

# # #     for qt in counts:
# # #         counts[qt] = len(qa_by_base.get(qt, []) or [])

# # #     counts["total"] = sum(counts.values())
# # #     return counts


# # # def count_questions_derived(rec: Dict[str, Any]) -> Dict[str, int]:
# # #     counts = {"mcq": 0, "true_false": 0, "fill_blank": 0, "assertion_reason": 0}
# # #     derived_qa = rec.get("derived_qa", {}) or {}

# # #     for qt in counts:
# # #         counts[qt] = len(derived_qa.get(qt, []) or [])

# # #     counts["total"] = sum(counts.values())
# # #     return counts


# # # def total_claims_for_forget_or_retain_external(rec: Dict[str, Any]) -> int:
# # #     return len(rec.get("paper_claims", []) or [])


# # # # ============================================================
# # # # Rejected-QA counters
# # # # ============================================================

# # # def count_rejected_claim_based(rec: Dict[str, Any]) -> Dict[str, int]:
# # #     """
# # #     For:
# # #       - forget rejected file
# # #       - retain external rejected file

# # #     Structure expected:
# # #       rec["rejected_qa_by_claim"] = [
# # #         {
# # #           "claim": "...",
# # #           "mcq": [...],
# # #           "true_false": [...],
# # #           "fill_blank": [...],
# # #           "assertion_reason": [...]
# # #         },
# # #         ...
# # #       ]
# # #     """
# # #     counts = {"mcq": 0, "true_false": 0, "fill_blank": 0, "assertion_reason": 0}

# # #     for claim_obj in rec.get("rejected_qa_by_claim", []) or []:
# # #         for qt in counts:
# # #             counts[qt] += len(claim_obj.get(qt, []) or [])

# # #     counts["total"] = sum(counts.values())
# # #     return counts


# # # def count_rejected_internal(rec: Dict[str, Any]) -> Dict[str, int]:
# # #     """
# # #     For retain internal rejected file:
# # #       rec["rejected_qa_by_base"] = {
# # #         "mcq": [...],
# # #         ...
# # #       }
# # #     """
# # #     counts = {"mcq": 0, "true_false": 0, "fill_blank": 0, "assertion_reason": 0}
# # #     rejected = rec.get("rejected_qa_by_base", {}) or {}

# # #     for qt in counts:
# # #         counts[qt] = len(rejected.get(qt, []) or [])

# # #     counts["total"] = sum(counts.values())
# # #     return counts


# # # def count_rejected_derived(rec: Dict[str, Any]) -> Dict[str, int]:
# # #     """
# # #     For derived rejected file:
# # #       rec["rejected_derived_qa"] = {
# # #         "mcq": [...],
# # #         ...
# # #       }
# # #     """
# # #     counts = {"mcq": 0, "true_false": 0, "fill_blank": 0, "assertion_reason": 0}
# # #     rejected = rec.get("rejected_derived_qa", {}) or {}

# # #     for qt in counts:
# # #         counts[qt] = len(rejected.get(qt, []) or [])

# # #     counts["total"] = sum(counts.values())
# # #     return counts


# # # # ============================================================
# # # # Forget q1 / q2 split diagnostics
# # # # ============================================================

# # # def forget_pair_split_stats(forget_records: List[Dict[str, Any]]) -> Dict[str, Any]:
# # #     """
# # #     The forget set is later split into q1 / q2 by dividing each QA-type group
# # #     equally. This computes the implied balance directly from the common forget JSON.
# # #     """
# # #     q1_counter = Counter()
# # #     q2_counter = Counter()
# # #     odd_groups = []
# # #     groups_total = 0

# # #     for rec in forget_records:
# # #         pdf_name = rec.get("pdf_name", "")

# # #         for claim_idx, claim_obj in enumerate(rec.get("qa_by_claim", []) or []):
# # #             for qt in ["mcq", "true_false", "fill_blank", "assertion_reason"]:
# # #                 items = claim_obj.get(qt, []) or []
# # #                 n = len(items)
# # #                 if n == 0:
# # #                     continue

# # #                 groups_total += 1
# # #                 half = n // 2
# # #                 q1_n = half
# # #                 q2_n = n - half

# # #                 q1_counter[qt] += q1_n
# # #                 q2_counter[qt] += q2_n

# # #                 if n % 2 != 0:
# # #                     odd_groups.append({
# # #                         "pdf_name": pdf_name,
# # #                         "claim_index": claim_idx,
# # #                         "qtype": qt,
# # #                         "count": n,
# # #                     })

# # #     q1_counter["total"] = sum(q1_counter.values())
# # #     q2_counter["total"] = sum(q2_counter.values())

# # #     return {
# # #         "groups_total": groups_total,
# # #         "q1_counts": dict(q1_counter),
# # #         "q2_counts": dict(q2_counter),
# # #         "absolute_imbalance_total": abs(q1_counter["total"] - q2_counter["total"]),
# # #         "odd_groups_count": len(odd_groups),
# # #         "odd_groups_examples": odd_groups[:20],
# # #     }


# # # # ============================================================
# # # # Rejected-file loading / filtering
# # # # ============================================================

# # # def load_optional_json_list(path: Path, name: str) -> List[Dict[str, Any]]:
# # #     if not path.exists():
# # #         return []
# # #     return ensure_list(load_json(path), name)


# # # def filter_rejected_records_by_common_ids(
# # #     records: List[Dict[str, Any]],
# # #     common_ids: Set[str],
# # # ) -> List[Dict[str, Any]]:
# # #     kept = []
# # #     for rec in records:
# # #         aid = anchor_id_from_record(rec)
# # #         if aid in common_ids:
# # #             kept.append(rec)
# # #     return kept


# # # # ============================================================
# # # # Dataset summaries
# # # # ============================================================

# # # def summarize_forget(
# # #     records: List[Dict[str, Any]],
# # #     rejected_records: List[Dict[str, Any]],
# # #     year_lookup: Dict[str, int],
# # # ) -> Dict[str, Any]:
# # #     year_counter = Counter()
# # #     qtype_counter = Counter()
# # #     rejected_counter = Counter()

# # #     claims_per_paper: List[int] = []
# # #     questions_per_paper: List[int] = []
# # #     verbatim_counts: List[int] = []
# # #     missing_year_ids: List[str] = []
# # #     duplicate_ids = []
# # #     seen = set()

# # #     total_claims = 0

# # #     for rec in records:
# # #         fid = forget_id_from_record(rec)
# # #         if fid in seen:
# # #             duplicate_ids.append(fid)
# # #         seen.add(fid)

# # #         year = year_lookup.get(fid)
# # #         if year is None:
# # #             missing_year_ids.append(fid)
# # #         else:
# # #             year_counter[str(year)] += 1

# # #         n_claims = total_claims_for_forget_or_retain_external(rec)
# # #         total_claims += n_claims
# # #         claims_per_paper.append(n_claims)

# # #         qcounts = count_questions_claim_based(rec)
# # #         questions_per_paper.append(qcounts["total"])

# # #         for qt in ["mcq", "true_false", "fill_blank", "assertion_reason"]:
# # #             qtype_counter[qt] += qcounts[qt]

# # #         verbatim_counts.append(len(rec.get("verbatim_claims", []) or []))

# # #     for rec in rejected_records:
# # #         rc = count_rejected_claim_based(rec)
# # #         for qt in ["mcq", "true_false", "fill_blank", "assertion_reason"]:
# # #             rejected_counter[qt] += rc[qt]

# # #     qtype_counter["total"] = sum(qtype_counter.values())
# # #     rejected_counter["total"] = sum(rejected_counter.values())

# # #     return {
# # #         "dataset": "forget",
# # #         "record_count": len(records),
# # #         "unique_paper_ids": len(seen),
# # #         "duplicate_ids_count": len(set(duplicate_ids)),
# # #         "duplicate_ids_examples": sorted(set(duplicate_ids))[:20],
# # #         "year_distribution": dict(year_counter),
# # #         "missing_year_ids_count": len(set(missing_year_ids)),
# # #         "missing_year_ids_examples": sorted(set(missing_year_ids))[:20],
# # #         "total_claims": total_claims,
# # #         "claims_per_paper": numeric_summary(claims_per_paper),
# # #         "verbatim_claims_per_paper": numeric_summary(verbatim_counts),
# # #         "questions_per_paper": numeric_summary(questions_per_paper),
# # #         "question_type_counts": dict(qtype_counter),
# # #         "rejected_question_type_counts": dict(rejected_counter),
# # #         "forget_pair_split": forget_pair_split_stats(records),
# # #     }


# # # def summarize_retain_external(
# # #     records: List[Dict[str, Any]],
# # #     rejected_records: List[Dict[str, Any]],
# # #     year_lookup: Dict[str, int],
# # # ) -> Dict[str, Any]:
# # #     anchor_year_counter = Counter()
# # #     selected_year_counter = Counter()
# # #     qtype_counter = Counter()
# # #     rejected_counter = Counter()

# # #     claims_per_paper: List[int] = []
# # #     questions_per_paper: List[int] = []
# # #     missing_year_ids: List[str] = []

# # #     survived_true = 0
# # #     survived_false = 0
# # #     total_claims = 0

# # #     for rec in records:
# # #         aid = anchor_id_from_record(rec)
# # #         year = year_lookup.get(aid)
# # #         if year is None:
# # #             missing_year_ids.append(aid)
# # #         else:
# # #             anchor_year_counter[str(year)] += 1

# # #         if rec.get("retain_survived", False):
# # #             survived_true += 1
# # #         else:
# # #             survived_false += 1

# # #         n_claims = total_claims_for_forget_or_retain_external(rec)
# # #         total_claims += n_claims
# # #         claims_per_paper.append(n_claims)

# # #         qcounts = count_questions_claim_based(rec)
# # #         questions_per_paper.append(qcounts["total"])
# # #         for qt in ["mcq", "true_false", "fill_blank", "assertion_reason"]:
# # #             qtype_counter[qt] += qcounts[qt]

# # #         sel = rec.get("selected_reference", {}) or {}
# # #         selected_corpus_id = (sel.get("corpusId") or "").strip()
# # #         selected_year = year_lookup.get(selected_corpus_id)
# # #         if selected_year is not None:
# # #             selected_year_counter[str(selected_year)] += 1

# # #     for rec in rejected_records:
# # #         rc = count_rejected_claim_based(rec)
# # #         for qt in ["mcq", "true_false", "fill_blank", "assertion_reason"]:
# # #             rejected_counter[qt] += rc[qt]

# # #     qtype_counter["total"] = sum(qtype_counter.values())
# # #     rejected_counter["total"] = sum(rejected_counter.values())

# # #     return {
# # #         "dataset": "retain_external",
# # #         "record_count": len(records),
# # #         "retain_survived_true": survived_true,
# # #         "retain_survived_false": survived_false,
# # #         "anchor_year_distribution": dict(anchor_year_counter),
# # #         "selected_reference_year_distribution": dict(selected_year_counter),
# # #         "missing_year_anchor_ids_count": len(set(missing_year_ids)),
# # #         "missing_year_anchor_ids_examples": sorted(set(missing_year_ids))[:20],
# # #         "total_claims": total_claims,
# # #         "claims_per_paper": numeric_summary(claims_per_paper),
# # #         "questions_per_paper": numeric_summary(questions_per_paper),
# # #         "question_type_counts": dict(qtype_counter),
# # #         "rejected_question_type_counts": dict(rejected_counter),
# # #     }


# # # def summarize_retain_internal(
# # #     records: List[Dict[str, Any]],
# # #     rejected_records: List[Dict[str, Any]],
# # #     year_lookup: Dict[str, int],
# # # ) -> Dict[str, Any]:
# # #     year_counter = Counter()
# # #     qtype_counter = Counter()
# # #     rejected_counter = Counter()

# # #     questions_per_paper: List[int] = []
# # #     missing_year_ids: List[str] = []

# # #     survived_true = 0
# # #     survived_false = 0

# # #     # retain internal is base-centric, so claims are inherited from forget paper_claims
# # #     total_claims = 0
# # #     claims_per_paper: List[int] = []

# # #     for rec in records:
# # #         aid = anchor_id_from_record(rec)
# # #         year = year_lookup.get(aid)
# # #         if year is None:
# # #             missing_year_ids.append(aid)
# # #         else:
# # #             year_counter[str(year)] += 1

# # #         survived = bool(rec.get("internal_retain_survived", False))
# # #         if survived:
# # #             survived_true += 1
# # #         else:
# # #             survived_false += 1

# # #         n_claims = len(rec.get("paper_claims", []) or [])
# # #         total_claims += n_claims
# # #         claims_per_paper.append(n_claims)

# # #         qcounts = count_questions_internal(rec)
# # #         questions_per_paper.append(qcounts["total"])
# # #         for qt in ["mcq", "true_false", "fill_blank", "assertion_reason"]:
# # #             qtype_counter[qt] += qcounts[qt]

# # #     for rec in rejected_records:
# # #         rc = count_rejected_internal(rec)
# # #         for qt in ["mcq", "true_false", "fill_blank", "assertion_reason"]:
# # #             rejected_counter[qt] += rc[qt]

# # #     qtype_counter["total"] = sum(qtype_counter.values())
# # #     rejected_counter["total"] = sum(rejected_counter.values())

# # #     return {
# # #         "dataset": "retain_internal",
# # #         "record_count": len(records),
# # #         "internal_retain_survived_true": survived_true,
# # #         "internal_retain_survived_false": survived_false,
# # #         "year_distribution": dict(year_counter),
# # #         "missing_year_anchor_ids_count": len(set(missing_year_ids)),
# # #         "missing_year_anchor_ids_examples": sorted(set(missing_year_ids))[:20],
# # #         "total_claims": total_claims,
# # #         "claims_per_paper": numeric_summary(claims_per_paper),
# # #         "questions_per_paper": numeric_summary(questions_per_paper),
# # #         "question_type_counts": dict(qtype_counter),
# # #         "rejected_question_type_counts": dict(rejected_counter),
# # #     }


# # # def summarize_derived(
# # #     records: List[Dict[str, Any]],
# # #     rejected_records: List[Dict[str, Any]],
# # #     year_lookup: Dict[str, int],
# # # ) -> Dict[str, Any]:
# # #     year_counter = Counter()
# # #     qtype_counter = Counter()
# # #     rejected_counter = Counter()

# # #     questions_per_paper: List[int] = []
# # #     source_questions_per_paper: List[int] = []
# # #     missing_year_ids: List[str] = []
# # #     source_type_counter = Counter()

# # #     survived_true = 0
# # #     survived_false = 0

# # #     # derived inherits paper_claims from forget
# # #     total_claims = 0
# # #     claims_per_paper: List[int] = []

# # #     for rec in records:
# # #         aid = anchor_id_from_record(rec)
# # #         year = year_lookup.get(aid)
# # #         if year is None:
# # #             missing_year_ids.append(aid)
# # #         else:
# # #             year_counter[str(year)] += 1

# # #         survived = bool(rec.get("derived_survived", False))
# # #         if survived:
# # #             survived_true += 1
# # #         else:
# # #             survived_false += 1

# # #         n_claims = len(rec.get("paper_claims", []) or [])
# # #         total_claims += n_claims
# # #         claims_per_paper.append(n_claims)

# # #         qcounts = count_questions_derived(rec)
# # #         questions_per_paper.append(qcounts["total"])
# # #         for qt in ["mcq", "true_false", "fill_blank", "assertion_reason"]:
# # #             qtype_counter[qt] += qcounts[qt]

# # #         sq = rec.get("source_questions", []) or []
# # #         source_questions_per_paper.append(len(sq))
# # #         for item in sq:
# # #             st = (item.get("source_type") or "").strip()
# # #             if st:
# # #                 source_type_counter[st] += 1

# # #     for rec in rejected_records:
# # #         rc = count_rejected_derived(rec)
# # #         for qt in ["mcq", "true_false", "fill_blank", "assertion_reason"]:
# # #             rejected_counter[qt] += rc[qt]

# # #     qtype_counter["total"] = sum(qtype_counter.values())
# # #     rejected_counter["total"] = sum(rejected_counter.values())

# # #     return {
# # #         "dataset": "derived",
# # #         "record_count": len(records),
# # #         "derived_survived_true": survived_true,
# # #         "derived_survived_false": survived_false,
# # #         "year_distribution": dict(year_counter),
# # #         "missing_year_anchor_ids_count": len(set(missing_year_ids)),
# # #         "missing_year_anchor_ids_examples": sorted(set(missing_year_ids))[:20],
# # #         "total_claims": total_claims,
# # #         "claims_per_paper": numeric_summary(claims_per_paper),
# # #         "questions_per_paper": numeric_summary(questions_per_paper),
# # #         "question_type_counts": dict(qtype_counter),
# # #         "rejected_question_type_counts": dict(rejected_counter),
# # #         "source_questions_per_paper": numeric_summary(source_questions_per_paper),
# # #         "source_question_type_distribution": dict(source_type_counter),
# # #     }


# # # # ============================================================
# # # # Cross-dataset consistency
# # # # ============================================================

# # # def compute_id_sets(
# # #     forget_records: List[Dict[str, Any]],
# # #     retain_external_records: List[Dict[str, Any]],
# # #     retain_internal_records: List[Dict[str, Any]],
# # #     derived_records: List[Dict[str, Any]],
# # # ) -> Dict[str, Set[str]]:
# # #     forget_ids = {forget_id_from_record(r) for r in forget_records if forget_id_from_record(r)}
# # #     retain_external_ids = {anchor_id_from_record(r) for r in retain_external_records if anchor_id_from_record(r)}
# # #     retain_internal_ids = {anchor_id_from_record(r) for r in retain_internal_records if anchor_id_from_record(r)}
# # #     derived_ids = {anchor_id_from_record(r) for r in derived_records if anchor_id_from_record(r)}

# # #     return {
# # #         "forget": forget_ids,
# # #         "retain_external": retain_external_ids,
# # #         "retain_internal": retain_internal_ids,
# # #         "derived": derived_ids,
# # #     }


# # # def cross_dataset_consistency(id_sets: Dict[str, Set[str]]) -> Dict[str, Any]:
# # #     names = list(id_sets.keys())
# # #     common_all = set.intersection(*(id_sets[n] for n in names)) if names else set()

# # #     pairwise = {}
# # #     for i, a in enumerate(names):
# # #         for b in names[i + 1:]:
# # #             inter = id_sets[a] & id_sets[b]
# # #             union = id_sets[a] | id_sets[b]
# # #             jaccard = len(inter) / len(union) if union else 0.0

# # #             pairwise[f"{a}__{b}"] = {
# # #                 "intersection": len(inter),
# # #                 "union": len(union),
# # #                 "jaccard": jaccard,
# # #                 "a_minus_b_examples": sorted(id_sets[a] - id_sets[b])[:20],
# # #                 "b_minus_a_examples": sorted(id_sets[b] - id_sets[a])[:20],
# # #             }

# # #     return {
# # #         "common_across_all_count": len(common_all),
# # #         "common_across_all_examples": sorted(common_all)[:50],
# # #         "pairwise": pairwise,
# # #     }


# # # # ============================================================
# # # # Markdown report
# # # # ============================================================

# # # def build_markdown_report(report: Dict[str, Any]) -> str:
# # #     lines: List[str] = []

# # #     lines.append("# Common-Dataset Evaluation Report")
# # #     lines.append("")
# # #     lines.append("## 1. Input Files")
# # #     for k, v in report["paths"].items():
# # #         lines.append(f"- **{k}**: `{v}`")

# # #     lines.append("")
# # #     lines.append("## 2. Common-ID Consistency")
# # #     c = report["cross_dataset_consistency"]
# # #     lines.append(f"- Common papers across all 4 JSONs: **{c['common_across_all_count']}**")

# # #     lines.append("")
# # #     lines.append("## 3. Dataset Summaries")

# # #     for key in ["forget", "retain_external", "retain_internal", "derived"]:
# # #         ds = report["datasets"][key]

# # #         lines.append(f"### {key}")
# # #         lines.append(f"- Record count: **{ds.get('record_count')}**")
# # #         lines.append(f"- Total claims: **{ds.get('total_claims', 0)}**")

# # #         q = ds.get("question_type_counts", {})
# # #         if q:
# # #             lines.append(
# # #                 f"- Question counts — "
# # #                 f"MCQ: **{q.get('mcq', 0)}**, "
# # #                 f"TF: **{q.get('true_false', 0)}**, "
# # #                 f"Fill: **{q.get('fill_blank', 0)}**, "
# # #                 f"AR: **{q.get('assertion_reason', 0)}**, "
# # #                 f"Total: **{q.get('total', 0)}**"
# # #             )

# # #         rq = ds.get("rejected_question_type_counts", {})
# # #         if rq:
# # #             lines.append(
# # #                 f"- Rejected QA counts — "
# # #                 f"MCQ: **{rq.get('mcq', 0)}**, "
# # #                 f"TF: **{rq.get('true_false', 0)}**, "
# # #                 f"Fill: **{rq.get('fill_blank', 0)}**, "
# # #                 f"AR: **{rq.get('assertion_reason', 0)}**, "
# # #                 f"Total: **{rq.get('total', 0)}**"
# # #             )

# # #         if "year_distribution" in ds:
# # #             lines.append(f"- Year coverage: `{ds['year_distribution']}`")
# # #         elif "anchor_year_distribution" in ds:
# # #             lines.append(f"- Anchor year coverage: `{ds['anchor_year_distribution']}`")

# # #         lines.append("")

# # #     lines.append("## 4. Forget Q1/Q2 Balance")
# # #     split = report["datasets"]["forget"].get("forget_pair_split", {})
# # #     lines.append(f"- Total QA groups inspected: **{split.get('groups_total', 0)}**")
# # #     lines.append(f"- Q1 counts: `{split.get('q1_counts', {})}`")
# # #     lines.append(f"- Q2 counts: `{split.get('q2_counts', {})}`")
# # #     lines.append(f"- Odd QA groups count: **{split.get('odd_groups_count', 0)}**")

# # #     return "\n".join(lines)


# # # # ============================================================
# # # # Main evaluation
# # # # ============================================================

# # # def evaluate_common_datasets(config: AppConfig) -> Dict[str, Any]:
# # #     # ----------------------------------------------------------
# # #     # Main common input files
# # #     # ----------------------------------------------------------
# # #     forget_path = Path(getattr(config, "common_forget_output_json", Path("forget_common.json")))
# # #     retain_external_path = Path(getattr(config, "common_retain_output_json", Path("retain_external_common.json")))
# # #     retain_internal_path = Path(getattr(config, "common_retain_internal_output_json", Path("retain_internal_common.json")))
# # #     derived_path = Path(getattr(config, "common_derived_output_json", Path("derived_common.json")))

# # #     for p in [forget_path, retain_external_path, retain_internal_path, derived_path]:
# # #         if not p.exists():
# # #             raise FileNotFoundError(f"Common dataset input file not found: {p}")

# # #     # ----------------------------------------------------------
# # #     # Optional rejected-questions files
# # #     # ----------------------------------------------------------
# # #     forget_rejected_path = Path(getattr(config, "forget_olmo_rejected_json", Path("forget_olmo_rejected.json")))
# # #     retain_external_rejected_path = Path(
# # #         getattr(config, "retain_external_olmo_rejected_json", Path("retain_external_olmo_rejected.json"))
# # #     )
# # #     retain_internal_rejected_path = Path(
# # #         getattr(config, "retain_internal_olmo_rejected_json", Path("retain_internal_olmo_rejected.json"))
# # #     )
# # #     derived_rejected_path = Path(
# # #         getattr(config, "derived_olmo_rejected_json", Path("derived_olmo_rejected.json"))
# # #     )

# # #     # ----------------------------------------------------------
# # #     # Load common datasets
# # #     # ----------------------------------------------------------
# # #     forget_records = ensure_list(load_json(forget_path), "forget common JSON")
# # #     retain_external_records = ensure_list(load_json(retain_external_path), "retain external common JSON")
# # #     retain_internal_records = ensure_list(load_json(retain_internal_path), "retain internal common JSON")
# # #     derived_records = ensure_list(load_json(derived_path), "derived common JSON")

# # #     # ----------------------------------------------------------
# # #     # Build common id set (from the already-common files this should match,
# # #     # but we still compute it explicitly and also use it to filter rejected files)
# # #     # ----------------------------------------------------------
# # #     id_sets = compute_id_sets(
# # #         forget_records,
# # #         retain_external_records,
# # #         retain_internal_records,
# # #         derived_records,
# # #     )
# # #     common_ids = set.intersection(*id_sets.values()) if id_sets else set()

# # #     # ----------------------------------------------------------
# # #     # Load and filter rejected files to the same common population
# # #     # ----------------------------------------------------------
# # #     forget_rejected_records = filter_rejected_records_by_common_ids(
# # #         load_optional_json_list(forget_rejected_path, "forget rejected JSON"),
# # #         common_ids,
# # #     )
# # #     retain_external_rejected_records = filter_rejected_records_by_common_ids(
# # #         load_optional_json_list(retain_external_rejected_path, "retain external rejected JSON"),
# # #         common_ids,
# # #     )
# # #     retain_internal_rejected_records = filter_rejected_records_by_common_ids(
# # #         load_optional_json_list(retain_internal_rejected_path, "retain internal rejected JSON"),
# # #         common_ids,
# # #     )
# # #     derived_rejected_records = filter_rejected_records_by_common_ids(
# # #         load_optional_json_list(derived_rejected_path, "derived rejected JSON"),
# # #         common_ids,
# # #     )

# # #     # ----------------------------------------------------------
# # #     # Year lookup
# # #     # ----------------------------------------------------------
# # #     year_dir = infer_year_index_dir(config)
# # #     year_lookup, year_file_counts = build_year_lookup(year_dir)

# # #     # ----------------------------------------------------------
# # #     # Build report
# # #     # ----------------------------------------------------------
# # #     report = {
# # #         "paths": {
# # #             "forget": str(forget_path),
# # #             "retain_external": str(retain_external_path),
# # #             "retain_internal": str(retain_internal_path),
# # #             "derived": str(derived_path),
# # #             "forget_rejected": str(forget_rejected_path),
# # #             "retain_external_rejected": str(retain_external_rejected_path),
# # #             "retain_internal_rejected": str(retain_internal_rejected_path),
# # #             "derived_rejected": str(derived_rejected_path),
# # #             "year_index_dir": str(year_dir),
# # #         },
# # #         "year_index_file_counts": year_file_counts,
# # #         "datasets": {
# # #             "forget": summarize_forget(forget_records, forget_rejected_records, year_lookup),
# # #             "retain_external": summarize_retain_external(retain_external_records, retain_external_rejected_records, year_lookup),
# # #             "retain_internal": summarize_retain_internal(retain_internal_records, retain_internal_rejected_records, year_lookup),
# # #             "derived": summarize_derived(derived_records, derived_rejected_records, year_lookup),
# # #         },
# # #         "cross_dataset_consistency": cross_dataset_consistency(id_sets),
# # #     }

# # #     # ----------------------------------------------------------
# # #     # Save outputs
# # #     # ----------------------------------------------------------
# # #     report_json_path = Path(
# # #         getattr(config, "common_eval_report_json", Path("common_dataset_evaluation_report.json"))
# # #     )
# # #     report_md_path = Path(
# # #         getattr(config, "common_eval_report_md", Path("common_dataset_evaluation_report.md"))
# # #     )

# # #     save_json(report, report_json_path)
# # #     report_md_path.write_text(build_markdown_report(report), encoding="utf-8")

# # #     # ----------------------------------------------------------
# # #     # Console summary
# # #     # ----------------------------------------------------------
# # #     print("\n" + "=" * 120)
# # #     print("COMMON DATASET EVALUATION SUMMARY")
# # #     print("=" * 120)
# # #     print(f"Forget records           : {report['datasets']['forget']['record_count']}")
# # #     print(f"Retain external records  : {report['datasets']['retain_external']['record_count']}")
# # #     print(f"Retain internal records  : {report['datasets']['retain_internal']['record_count']}")
# # #     print(f"Derived records          : {report['datasets']['derived']['record_count']}")
# # #     print("-" * 120)
# # #     print(f"Common ids across all 4  : {report['cross_dataset_consistency']['common_across_all_count']}")
# # #     print(f"Report JSON              : {report_json_path}")
# # #     print(f"Report Markdown          : {report_md_path}")

# # #     return report


# # # if __name__ == "__main__":
# # #     config = AppConfig()
# # #     evaluate_common_datasets(config)


# # from __future__ import annotations

# # import json
# # import re
# # import statistics as stats
# # from collections import Counter
# # from pathlib import Path
# # from typing import Any, Dict, List, Optional, Set, Tuple

# # from config import AppConfig
# # from utils.json_utils import load_json, save_json


# # # ============================================================
# # # Generic helpers
# # # ============================================================

# # QA_TYPES = ["mcq", "true_false", "fill_blank", "assertion_reason"]


# # def ensure_list(data: Any, name: str) -> List[Dict[str, Any]]:
# #     if isinstance(data, dict):
# #         data = [data]

# #     if not isinstance(data, list):
# #         raise ValueError(f"{name} must contain a JSON array (or a single dict).")

# #     out: List[Dict[str, Any]] = []
# #     for i, rec in enumerate(data):
# #         if not isinstance(rec, dict):
# #             raise ValueError(f"{name}[{i}] is not a JSON object.")
# #         out.append(rec)
# #     return out


# # def safe_mean(xs: List[float]) -> Optional[float]:
# #     return float(stats.mean(xs)) if xs else None


# # def safe_median(xs: List[float]) -> Optional[float]:
# #     return float(stats.median(xs)) if xs else None


# # def safe_stdev(xs: List[float]) -> Optional[float]:
# #     return float(stats.pstdev(xs)) if len(xs) > 1 else (0.0 if xs else None)


# # def numeric_summary(xs: List[float]) -> Dict[str, Optional[float]]:
# #     if not xs:
# #         return {
# #             "count": 0,
# #             "mean": None,
# #             "median": None,
# #             "std": None,
# #             "min": None,
# #             "max": None,
# #             "sum": 0,
# #         }

# #     return {
# #         "count": len(xs),
# #         "mean": safe_mean(xs),
# #         "median": safe_median(xs),
# #         "std": safe_stdev(xs),
# #         "min": float(min(xs)),
# #         "max": float(max(xs)),
# #         "sum": float(sum(xs)),
# #     }


# # def stem_id(name: str) -> str:
# #     return Path((name or "").strip()).stem.strip()


# # def forget_id_from_record(rec: Dict[str, Any]) -> str:
# #     """
# #     Forget JSON uses pdf_name like:
# #       119529102.pdf -> 119529102
# #     """
# #     return stem_id(rec.get("pdf_name", ""))


# # def anchor_id_from_record(rec: Dict[str, Any]) -> str:
# #     """
# #     Retain / derived records usually identify the forget anchor via one of:
# #       1. anchor_forget_paper_id
# #       2. anchor_corpus_id
# #       3. anchor_forget_pdf_name
# #       4. fallback: pdf_name
# #     """
# #     for key in ("anchor_forget_paper_id", "anchor_corpus_id", "anchor_forget_pdf_name", "pdf_name"):
# #         val = (rec.get(key) or "").strip()
# #         if val:
# #             return stem_id(val)
# #     return ""


# # def empty_qtype_counter() -> Dict[str, int]:
# #     return {
# #         "mcq": 0,
# #         "true_false": 0,
# #         "fill_blank": 0,
# #         "assertion_reason": 0,
# #         "total": 0,
# #     }


# # def normalize_qtype_counter(counts: Dict[str, int]) -> Dict[str, int]:
# #     counts["total"] = (
# #         counts.get("mcq", 0)
# #         + counts.get("true_false", 0)
# #         + counts.get("fill_blank", 0)
# #         + counts.get("assertion_reason", 0)
# #     )
# #     return counts


# # # ============================================================
# # # Year lookup
# # # ============================================================

# # def infer_year_index_dir(config: AppConfig) -> Path:
# #     """
# #     Infer the directory containing year-wise corpus-id files.
# #     """
# #     for attr in ("year_wise_corpus_ids_dir", "year_corpus_ids_dir", "corpus_ids_dir"):
# #         p = getattr(config, attr, None)
# #         if p:
# #             return Path(p)

# #     return Path("year_wise_corpus_ids")


# # def build_year_lookup(year_dir: Path) -> Tuple[Dict[str, int], Dict[str, int]]:
# #     """
# #     Build corpus-id -> year from files like corpus_ids_2020.txt.
# #     """
# #     id_to_year: Dict[str, int] = {}
# #     file_counts_by_year: Dict[str, int] = {}

# #     if not year_dir.exists():
# #         return id_to_year, file_counts_by_year

# #     for path in sorted(year_dir.glob("*.txt")):
# #         m = re.search(r"(19|20)\d{2}", path.stem)
# #         if not m:
# #             continue

# #         year = int(m.group(0))
# #         count = 0

# #         with path.open("r", encoding="utf-8") as f:
# #             for line in f:
# #                 cid = line.strip()
# #                 if not cid:
# #                     continue

# #                 id_to_year[cid] = year
# #                 count += 1

# #         file_counts_by_year[str(year)] = count

# #     return id_to_year, file_counts_by_year


# # # ============================================================
# # # Question / claim counters
# # # ============================================================

# # def count_questions_claim_based(rec: Dict[str, Any]) -> Dict[str, int]:
# #     """
# #     For:
# #       - forget
# #       - retain external
# #     where questions live inside qa_by_claim
# #     """
# #     counts = {"mcq": 0, "true_false": 0, "fill_blank": 0, "assertion_reason": 0}

# #     for claim_obj in rec.get("qa_by_claim", []) or []:
# #         for qt in QA_TYPES:
# #             counts[qt] += len(claim_obj.get(qt, []) or [])

# #     counts["total"] = sum(counts.values())
# #     return counts


# # def count_questions_internal(rec: Dict[str, Any]) -> Dict[str, int]:
# #     """
# #     For retain internal:
# #       qa_by_base = { mcq: [...], ... }
# #     """
# #     counts = {"mcq": 0, "true_false": 0, "fill_blank": 0, "assertion_reason": 0}
# #     qa_by_base = rec.get("qa_by_base", {}) or {}

# #     for qt in QA_TYPES:
# #         counts[qt] = len(qa_by_base.get(qt, []) or [])

# #     counts["total"] = sum(counts.values())
# #     return counts


# # def count_questions_derived(rec: Dict[str, Any]) -> Dict[str, int]:
# #     """
# #     For derived:
# #       derived_qa = { mcq: [...], ... }
# #     """
# #     counts = {"mcq": 0, "true_false": 0, "fill_blank": 0, "assertion_reason": 0}
# #     derived_qa = rec.get("derived_qa", {}) or {}

# #     for qt in QA_TYPES:
# #         counts[qt] = len(derived_qa.get(qt, []) or [])

# #     counts["total"] = sum(counts.values())
# #     return counts


# # def total_claims_from_record(rec: Dict[str, Any]) -> int:
# #     """
# #     All dataset families still carry paper_claims.
# #     """
# #     return len(rec.get("paper_claims", []) or [])


# # # ============================================================
# # # Rejected-QA helpers (schema-robust)
# # # ============================================================

# # def extract_claim_level_rejected_groups(rec: Dict[str, Any]) -> List[Dict[str, Any]]:
# #     """
# #     For forget / retain external rejected files, support:
# #       - rejected_qa_by_claim
# #       - qa_by_claim
# #     """
# #     groups = rec.get("rejected_qa_by_claim", None)
# #     if isinstance(groups, list):
# #         return groups

# #     groups = rec.get("qa_by_claim", None)
# #     if isinstance(groups, list):
# #         return groups

# #     return []


# # def extract_base_level_rejected_group(rec: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
# #     """
# #     For retain internal rejected files, support:
# #       - rejected_qa_by_base
# #       - qa_by_base
# #     """
# #     groups = rec.get("rejected_qa_by_base", None)
# #     if isinstance(groups, dict):
# #         return groups

# #     groups = rec.get("qa_by_base", None)
# #     if isinstance(groups, dict):
# #         return groups

# #     return {}


# # def extract_derived_level_rejected_group(rec: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
# #     """
# #     For derived rejected files, support:
# #       - rejected_derived_qa
# #       - derived_qa
# #     """
# #     groups = rec.get("rejected_derived_qa", None)
# #     if isinstance(groups, dict):
# #         return groups

# #     groups = rec.get("derived_qa", None)
# #     if isinstance(groups, dict):
# #         return groups

# #     return {}


# # def count_rejected_claim_based_record(rec: Dict[str, Any]) -> Dict[str, int]:
# #     counts = empty_qtype_counter()
# #     groups = extract_claim_level_rejected_groups(rec)

# #     for claim_obj in groups:
# #         if not isinstance(claim_obj, dict):
# #             continue

# #         for qt in QA_TYPES:
# #             counts[qt] += len(claim_obj.get(qt, []) or [])

# #     return normalize_qtype_counter(counts)


# # def count_rejected_internal_record(rec: Dict[str, Any]) -> Dict[str, int]:
# #     counts = empty_qtype_counter()
# #     groups = extract_base_level_rejected_group(rec)

# #     for qt in QA_TYPES:
# #         counts[qt] += len(groups.get(qt, []) or [])

# #     return normalize_qtype_counter(counts)


# # def count_rejected_derived_record(rec: Dict[str, Any]) -> Dict[str, int]:
# #     counts = empty_qtype_counter()
# #     groups = extract_derived_level_rejected_group(rec)

# #     for qt in QA_TYPES:
# #         counts[qt] += len(groups.get(qt, []) or [])

# #     return normalize_qtype_counter(counts)


# # def aggregate_rejected_counts(
# #     rejected_records: List[Dict[str, Any]],
# #     mode: str,
# # ) -> Tuple[Dict[str, int], Dict[str, Optional[float]]]:
# #     """
# #     Aggregate rejected QA stats over a rejected-record list.

# #     Returns:
# #       - rejected_question_type_counts
# #       - rejected_questions_per_paper summary
# #     """
# #     total_counts = empty_qtype_counter()
# #     per_paper_totals: List[float] = []

# #     for rec in rejected_records:
# #         if mode == "claim":
# #             rc = count_rejected_claim_based_record(rec)
# #         elif mode == "internal":
# #             rc = count_rejected_internal_record(rec)
# #         elif mode == "derived":
# #             rc = count_rejected_derived_record(rec)
# #         else:
# #             raise ValueError(f"Unknown rejected counting mode: {mode}")

# #         for qt in QA_TYPES:
# #             total_counts[qt] += rc[qt]

# #         total_counts["total"] += rc["total"]
# #         per_paper_totals.append(rc["total"])

# #     return total_counts, numeric_summary(per_paper_totals)


# # def load_optional_json_list(path: Path, name: str) -> List[Dict[str, Any]]:
# #     if not path.exists():
# #         return []
# #     return ensure_list(load_json(path), name)


# # # def filter_rejected_records_by_common_ids(
# # #     records: List[Dict[str, Any]],
# # #     common_ids: Set[str],
# # # ) -> List[Dict[str, Any]]:
# # #     kept = []
# # #     for rec in records:
# # #         aid = anchor_id_from_record(rec)
# # #         if aid in common_ids:
# # #             kept.append(rec)
# # #     return kept

# # def filter_forget_rejected_records_by_common_ids(
# #     records: List[Dict[str, Any]],
# #     common_ids: Set[str],
# # ) -> List[Dict[str, Any]]:
# #     """
# #     Forget rejected file should be matched using pdf_name stem,
# #     because it behaves like the forget dataset itself.
# #     """
# #     kept = []

# #     for rec in records:
# #         fid = forget_id_from_record(rec)
# #         if fid in common_ids:
# #             kept.append(rec)

# #     return kept


# # def filter_anchor_rejected_records_by_common_ids(
# #     records: List[Dict[str, Any]],
# #     common_ids: Set[str],
# # ) -> List[Dict[str, Any]]:
# #     """
# #     Retain external / retain internal / derived rejected files should be matched
# #     using anchor-style identifiers.
# #     """
# #     kept = []

# #     for rec in records:
# #         aid = anchor_id_from_record(rec)
# #         if aid in common_ids:
# #             kept.append(rec)

# #     return kept



# # # ============================================================
# # # Forget q1 / q2 split diagnostics
# # # ============================================================

# # def forget_pair_split_stats(forget_records: List[Dict[str, Any]]) -> Dict[str, Any]:
# #     """
# #     The forget set is later divided into q1 / q2 by splitting each QA-type group equally.
# #     This computes the implied q1 / q2 balance directly from the common forget JSON.
# #     """
# #     q1_counter = Counter()
# #     q2_counter = Counter()
# #     odd_groups = []
# #     groups_total = 0

# #     for rec in forget_records:
# #         pdf_name = rec.get("pdf_name", "")

# #         for claim_idx, claim_obj in enumerate(rec.get("qa_by_claim", []) or []):
# #             for qt in QA_TYPES:
# #                 items = claim_obj.get(qt, []) or []
# #                 n = len(items)
# #                 if n == 0:
# #                     continue

# #                 groups_total += 1
# #                 half = n // 2
# #                 q1_n = half
# #                 q2_n = n - half

# #                 q1_counter[qt] += q1_n
# #                 q2_counter[qt] += q2_n

# #                 if n % 2 != 0:
# #                     odd_groups.append({
# #                         "pdf_name": pdf_name,
# #                         "claim_index": claim_idx,
# #                         "qtype": qt,
# #                         "count": n,
# #                     })

# #     q1_counter["total"] = sum(q1_counter.values())
# #     q2_counter["total"] = sum(q2_counter.values())

# #     return {
# #         "groups_total": groups_total,
# #         "q1_counts": dict(q1_counter),
# #         "q2_counts": dict(q2_counter),
# #         "absolute_imbalance_total": abs(q1_counter["total"] - q2_counter["total"]),
# #         "odd_groups_count": len(odd_groups),
# #         "odd_groups_examples": odd_groups[:20],
# #     }


# # # ============================================================
# # # Dataset summaries
# # # ============================================================

# # def summarize_forget(
# #     records: List[Dict[str, Any]],
# #     rejected_records: List[Dict[str, Any]],
# #     year_lookup: Dict[str, int],
# # ) -> Dict[str, Any]:
# #     year_counter = Counter()
# #     qtype_counter = Counter()

# #     claims_per_paper: List[int] = []
# #     questions_per_paper: List[int] = []
# #     verbatim_counts: List[int] = []
# #     missing_year_ids: List[str] = []
# #     duplicate_ids = []
# #     seen = set()

# #     total_claims = 0

# #     for rec in records:
# #         fid = forget_id_from_record(rec)

# #         if fid in seen:
# #             duplicate_ids.append(fid)
# #         seen.add(fid)

# #         year = year_lookup.get(fid)
# #         if year is None:
# #             missing_year_ids.append(fid)
# #         else:
# #             year_counter[str(year)] += 1

# #         n_claims = total_claims_from_record(rec)
# #         total_claims += n_claims
# #         claims_per_paper.append(n_claims)

# #         qcounts = count_questions_claim_based(rec)
# #         questions_per_paper.append(qcounts["total"])
# #         for qt in QA_TYPES:
# #             qtype_counter[qt] += qcounts[qt]

# #         verbatim_counts.append(len(rec.get("verbatim_claims", []) or []))

# #     rejected_counter, rejected_per_paper_summary = aggregate_rejected_counts(
# #         rejected_records,
# #         mode="claim",
# #     )

# #     qtype_counter["total"] = sum(qtype_counter.values())

# #     return {
# #         "dataset": "forget",
# #         "record_count": len(records),
# #         "unique_paper_ids": len(seen),
# #         "duplicate_ids_count": len(set(duplicate_ids)),
# #         "duplicate_ids_examples": sorted(set(duplicate_ids))[:20],
# #         "year_distribution": dict(year_counter),
# #         "missing_year_ids_count": len(set(missing_year_ids)),
# #         "missing_year_ids_examples": sorted(set(missing_year_ids))[:20],
# #         "total_claims": total_claims,
# #         "claims_per_paper": numeric_summary(claims_per_paper),
# #         "verbatim_claims_per_paper": numeric_summary(verbatim_counts),
# #         "questions_per_paper": numeric_summary(questions_per_paper),
# #         "question_type_counts": dict(qtype_counter),
# #         "rejected_question_type_counts": dict(rejected_counter),
# #         "rejected_questions_per_paper": rejected_per_paper_summary,
# #         "forget_pair_split": forget_pair_split_stats(records),
# #     }


# # def summarize_retain_external(
# #     records: List[Dict[str, Any]],
# #     rejected_records: List[Dict[str, Any]],
# #     year_lookup: Dict[str, int],
# # ) -> Dict[str, Any]:
# #     anchor_year_counter = Counter()
# #     selected_year_counter = Counter()
# #     qtype_counter = Counter()

# #     claims_per_paper: List[int] = []
# #     questions_per_paper: List[int] = []
# #     missing_year_ids: List[str] = []

# #     survived_true = 0
# #     survived_false = 0
# #     total_claims = 0

# #     for rec in records:
# #         aid = anchor_id_from_record(rec)
# #         year = year_lookup.get(aid)
# #         if year is None:
# #             missing_year_ids.append(aid)
# #         else:
# #             anchor_year_counter[str(year)] += 1

# #         if rec.get("retain_survived", False):
# #             survived_true += 1
# #         else:
# #             survived_false += 1

# #         n_claims = total_claims_from_record(rec)
# #         total_claims += n_claims
# #         claims_per_paper.append(n_claims)

# #         qcounts = count_questions_claim_based(rec)
# #         questions_per_paper.append(qcounts["total"])
# #         for qt in QA_TYPES:
# #             qtype_counter[qt] += qcounts[qt]

# #         sel = rec.get("selected_reference", {}) or {}
# #         selected_corpus_id = (sel.get("corpusId") or "").strip()
# #         selected_year = year_lookup.get(selected_corpus_id)
# #         if selected_year is not None:
# #             selected_year_counter[str(selected_year)] += 1

# #     rejected_counter, rejected_per_paper_summary = aggregate_rejected_counts(
# #         rejected_records,
# #         mode="claim",
# #     )

# #     qtype_counter["total"] = sum(qtype_counter.values())

# #     return {
# #         "dataset": "retain_external",
# #         "record_count": len(records),
# #         "retain_survived_true": survived_true,
# #         "retain_survived_false": survived_false,
# #         "anchor_year_distribution": dict(anchor_year_counter),
# #         "selected_reference_year_distribution": dict(selected_year_counter),
# #         "missing_year_anchor_ids_count": len(set(missing_year_ids)),
# #         "missing_year_anchor_ids_examples": sorted(set(missing_year_ids))[:20],
# #         "total_claims": total_claims,
# #         "claims_per_paper": numeric_summary(claims_per_paper),
# #         "questions_per_paper": numeric_summary(questions_per_paper),
# #         "question_type_counts": dict(qtype_counter),
# #         "rejected_question_type_counts": dict(rejected_counter),
# #         "rejected_questions_per_paper": rejected_per_paper_summary,
# #     }


# # def summarize_retain_internal(
# #     records: List[Dict[str, Any]],
# #     rejected_records: List[Dict[str, Any]],
# #     year_lookup: Dict[str, int],
# # ) -> Dict[str, Any]:
# #     year_counter = Counter()
# #     qtype_counter = Counter()

# #     claims_per_paper: List[int] = []
# #     questions_per_paper: List[int] = []
# #     missing_year_ids: List[str] = []

# #     survived_true = 0
# #     survived_false = 0
# #     total_claims = 0

# #     for rec in records:
# #         aid = anchor_id_from_record(rec)
# #         year = year_lookup.get(aid)
# #         if year is None:
# #             missing_year_ids.append(aid)
# #         else:
# #             year_counter[str(year)] += 1

# #         if rec.get("internal_retain_survived", False):
# #             survived_true += 1
# #         else:
# #             survived_false += 1

# #         n_claims = total_claims_from_record(rec)
# #         total_claims += n_claims
# #         claims_per_paper.append(n_claims)

# #         qcounts = count_questions_internal(rec)
# #         questions_per_paper.append(qcounts["total"])
# #         for qt in QA_TYPES:
# #             qtype_counter[qt] += qcounts[qt]

# #     rejected_counter, rejected_per_paper_summary = aggregate_rejected_counts(
# #         rejected_records,
# #         mode="internal",
# #     )

# #     qtype_counter["total"] = sum(qtype_counter.values())

# #     return {
# #         "dataset": "retain_internal",
# #         "record_count": len(records),
# #         "internal_retain_survived_true": survived_true,
# #         "internal_retain_survived_false": survived_false,
# #         "year_distribution": dict(year_counter),
# #         "missing_year_anchor_ids_count": len(set(missing_year_ids)),
# #         "missing_year_anchor_ids_examples": sorted(set(missing_year_ids))[:20],
# #         "total_claims": total_claims,
# #         "claims_per_paper": numeric_summary(claims_per_paper),
# #         "questions_per_paper": numeric_summary(questions_per_paper),
# #         "question_type_counts": dict(qtype_counter),
# #         "rejected_question_type_counts": dict(rejected_counter),
# #         "rejected_questions_per_paper": rejected_per_paper_summary,
# #     }


# # def summarize_derived(
# #     records: List[Dict[str, Any]],
# #     rejected_records: List[Dict[str, Any]],
# #     year_lookup: Dict[str, int],
# # ) -> Dict[str, Any]:
# #     year_counter = Counter()
# #     qtype_counter = Counter()
# #     source_type_counter = Counter()

# #     claims_per_paper: List[int] = []
# #     questions_per_paper: List[int] = []
# #     source_questions_per_paper: List[int] = []
# #     missing_year_ids: List[str] = []

# #     survived_true = 0
# #     survived_false = 0
# #     total_claims = 0

# #     for rec in records:
# #         aid = anchor_id_from_record(rec)
# #         year = year_lookup.get(aid)
# #         if year is None:
# #             missing_year_ids.append(aid)
# #         else:
# #             year_counter[str(year)] += 1

# #         if rec.get("derived_survived", False):
# #             survived_true += 1
# #         else:
# #             survived_false += 1

# #         n_claims = total_claims_from_record(rec)
# #         total_claims += n_claims
# #         claims_per_paper.append(n_claims)

# #         qcounts = count_questions_derived(rec)
# #         questions_per_paper.append(qcounts["total"])
# #         for qt in QA_TYPES:
# #             qtype_counter[qt] += qcounts[qt]

# #         sq = rec.get("source_questions", []) or []
# #         source_questions_per_paper.append(len(sq))

# #         for item in sq:
# #             st = (item.get("source_type") or "").strip()
# #             if st:
# #                 source_type_counter[st] += 1

# #     rejected_counter, rejected_per_paper_summary = aggregate_rejected_counts(
# #         rejected_records,
# #         mode="derived",
# #     )

# #     qtype_counter["total"] = sum(qtype_counter.values())

# #     return {
# #         "dataset": "derived",
# #         "record_count": len(records),
# #         "derived_survived_true": survived_true,
# #         "derived_survived_false": survived_false,
# #         "year_distribution": dict(year_counter),
# #         "missing_year_anchor_ids_count": len(set(missing_year_ids)),
# #         "missing_year_anchor_ids_examples": sorted(set(missing_year_ids))[:20],
# #         "total_claims": total_claims,
# #         "claims_per_paper": numeric_summary(claims_per_paper),
# #         "questions_per_paper": numeric_summary(questions_per_paper),
# #         "question_type_counts": dict(qtype_counter),
# #         "rejected_question_type_counts": dict(rejected_counter),
# #         "rejected_questions_per_paper": rejected_per_paper_summary,
# #         "source_questions_per_paper": numeric_summary(source_questions_per_paper),
# #         "source_question_type_distribution": dict(source_type_counter),
# #     }


# # # ============================================================
# # # Cross-dataset consistency
# # # ============================================================

# # def compute_id_sets(
# #     forget_records: List[Dict[str, Any]],
# #     retain_external_records: List[Dict[str, Any]],
# #     retain_internal_records: List[Dict[str, Any]],
# #     derived_records: List[Dict[str, Any]],
# # ) -> Dict[str, Set[str]]:
# #     forget_ids = {forget_id_from_record(r) for r in forget_records if forget_id_from_record(r)}
# #     retain_external_ids = {anchor_id_from_record(r) for r in retain_external_records if anchor_id_from_record(r)}
# #     retain_internal_ids = {anchor_id_from_record(r) for r in retain_internal_records if anchor_id_from_record(r)}
# #     derived_ids = {anchor_id_from_record(r) for r in derived_records if anchor_id_from_record(r)}

# #     return {
# #         "forget": forget_ids,
# #         "retain_external": retain_external_ids,
# #         "retain_internal": retain_internal_ids,
# #         "derived": derived_ids,
# #     }


# # def cross_dataset_consistency(id_sets: Dict[str, Set[str]]) -> Dict[str, Any]:
# #     names = list(id_sets.keys())
# #     common_all = set.intersection(*(id_sets[n] for n in names)) if names else set()

# #     pairwise = {}
# #     for i, a in enumerate(names):
# #         for b in names[i + 1:]:
# #             inter = id_sets[a] & id_sets[b]
# #             union = id_sets[a] | id_sets[b]
# #             jaccard = len(inter) / len(union) if union else 0.0

# #             pairwise[f"{a}__{b}"] = {
# #                 "intersection": len(inter),
# #                 "union": len(union),
# #                 "jaccard": jaccard,
# #                 "a_minus_b_examples": sorted(id_sets[a] - id_sets[b])[:20],
# #                 "b_minus_a_examples": sorted(id_sets[b] - id_sets[a])[:20],
# #             }

# #     return {
# #         "common_across_all_count": len(common_all),
# #         "common_across_all_examples": sorted(common_all)[:50],
# #         "pairwise": pairwise,
# #     }


# # # ============================================================
# # # Markdown report
# # # ============================================================

# # def build_markdown_report(report: Dict[str, Any]) -> str:
# #     lines: List[str] = []

# #     lines.append("# Common-Dataset Evaluation Report")
# #     lines.append("")
# #     lines.append("## 1. Input Files")
# #     for k, v in report["paths"].items():
# #         lines.append(f"- **{k}**: `{v}`")

# #     lines.append("")
# #     lines.append("## 2. Common-ID Consistency")
# #     c = report["cross_dataset_consistency"]
# #     lines.append(f"- Common papers across all 4 JSONs: **{c['common_across_all_count']}**")

# #     lines.append("")
# #     lines.append("## 3. Dataset Summaries")

# #     for key in ["forget", "retain_external", "retain_internal", "derived"]:
# #         ds = report["datasets"][key]

# #         lines.append(f"### {key}")
# #         lines.append(f"- Record count: **{ds.get('record_count')}**")
# #         lines.append(f"- Total claims: **{ds.get('total_claims', 0)}**")

# #         q = ds.get("question_type_counts", {})
# #         if q:
# #             lines.append(
# #                 f"- Question counts — "
# #                 f"MCQ: **{q.get('mcq', 0)}**, "
# #                 f"TF: **{q.get('true_false', 0)}**, "
# #                 f"Fill: **{q.get('fill_blank', 0)}**, "
# #                 f"AR: **{q.get('assertion_reason', 0)}**, "
# #                 f"Total: **{q.get('total', 0)}**"
# #             )

# #         rq = ds.get("rejected_question_type_counts", {})
# #         if rq:
# #             lines.append(
# #                 f"- Rejected QA counts — "
# #                 f"MCQ: **{rq.get('mcq', 0)}**, "
# #                 f"TF: **{rq.get('true_false', 0)}**, "
# #                 f"Fill: **{rq.get('fill_blank', 0)}**, "
# #                 f"AR: **{rq.get('assertion_reason', 0)}**, "
# #                 f"Total: **{rq.get('total', 0)}**"
# #             )

# #         rpp = ds.get("rejected_questions_per_paper", {})
# #         if rpp:
# #             lines.append(
# #                 f"- Rejected questions per paper — "
# #                 f"Mean: **{rpp.get('mean')}**, "
# #                 f"Median: **{rpp.get('median')}**, "
# #                 f"Std: **{rpp.get('std')}**, "
# #                 f"Min: **{rpp.get('min')}**, "
# #                 f"Max: **{rpp.get('max')}**"
# #             )

# #         if "year_distribution" in ds:
# #             lines.append(f"- Year coverage: `{ds['year_distribution']}`")
# #         elif "anchor_year_distribution" in ds:
# #             lines.append(f"- Anchor year coverage: `{ds['anchor_year_distribution']}`")

# #         lines.append("")

# #     lines.append("## 4. Forget Q1/Q2 Balance")
# #     split = report["datasets"]["forget"].get("forget_pair_split", {})
# #     lines.append(f"- Total QA groups inspected: **{split.get('groups_total', 0)}**")
# #     lines.append(f"- Q1 counts: `{split.get('q1_counts', {})}`")
# #     lines.append(f"- Q2 counts: `{split.get('q2_counts', {})}`")
# #     lines.append(f"- Odd QA groups count: **{split.get('odd_groups_count', 0)}**")

# #     return "\n".join(lines)


# # # ============================================================
# # # Main evaluation
# # # ============================================================

# # def evaluate_common_datasets(config: AppConfig) -> Dict[str, Any]:
# #     # ----------------------------------------------------------
# #     # Main common input files
# #     # ----------------------------------------------------------
# #     forget_path = Path(getattr(config, "common_forget_output_json", Path("forget_common.json")))
# #     retain_external_path = Path(getattr(config, "common_retain_output_json", Path("retain_external_common.json")))
# #     retain_internal_path = Path(getattr(config, "common_retain_internal_output_json", Path("retain_internal_common.json")))
# #     derived_path = Path(getattr(config, "common_derived_output_json", Path("derived_common.json")))

# #     for p in [forget_path, retain_external_path, retain_internal_path, derived_path]:
# #         if not p.exists():
# #             raise FileNotFoundError(f"Common dataset input file not found: {p}")

# #     # ----------------------------------------------------------
# #     # Optional rejected-question files
# #     # ----------------------------------------------------------
# #     forget_rejected_path = Path(getattr(config, "forget_olmo_rejected_json", Path("forget_olmo_rejected.json")))
# #     retain_external_rejected_path = Path(
# #         getattr(config, "retain_external_olmo_rejected_json", Path("retain_external_olmo_rejected.json"))
# #     )
# #     retain_internal_rejected_path = Path(
# #         getattr(config, "retain_internal_olmo_rejected_json", Path("retain_internal_olmo_rejected.json"))
# #     )
# #     derived_rejected_path = Path(
# #         getattr(config, "derived_olmo_rejected_json", Path("derived_olmo_rejected.json"))
# #     )

# #     # ----------------------------------------------------------
# #     # Load common datasets
# #     # ----------------------------------------------------------
# #     forget_records = ensure_list(load_json(forget_path), "forget common JSON")
# #     retain_external_records = ensure_list(load_json(retain_external_path), "retain external common JSON")
# #     retain_internal_records = ensure_list(load_json(retain_internal_path), "retain internal common JSON")
# #     derived_records = ensure_list(load_json(derived_path), "derived common JSON")

# #     # ----------------------------------------------------------
# #     # Build common id set (explicitly)
# #     # ----------------------------------------------------------
# #     id_sets = compute_id_sets(
# #         forget_records,
# #         retain_external_records,
# #         retain_internal_records,
# #         derived_records,
# #     )
# #     common_ids = set.intersection(*id_sets.values()) if id_sets else set()

# #     # ----------------------------------------------------------
# #     # Load rejected files and filter them to same common ids
# #     # ----------------------------------------------------------
# #     forget_rejected_records = filter_forget_rejected_records_by_common_ids(
# #         load_optional_json_list(forget_rejected_path, "forget rejected JSON"),
# #         common_ids,
# #     )
# #     retain_external_rejected_records = filter_anchor_rejected_records_by_common_ids(
# #         load_optional_json_list(retain_external_rejected_path, "retain external rejected JSON"),
# #         common_ids,
# #     )
# #     retain_internal_rejected_records = filter_anchor_rejected_records_by_common_ids(
# #         load_optional_json_list(retain_internal_rejected_path, "retain internal rejected JSON"),
# #         common_ids,
# #     )
# #     derived_rejected_records = filter_anchor_rejected_records_by_common_ids(
# #         load_optional_json_list(derived_rejected_path, "derived rejected JSON"),
# #         common_ids,
# #     )

# #     # ----------------------------------------------------------
# #     # Year lookup
# #     # ----------------------------------------------------------
# #     year_dir = infer_year_index_dir(config)
# #     year_lookup, year_file_counts = build_year_lookup(year_dir)

# #     # ----------------------------------------------------------
# #     # Build report
# #     # ----------------------------------------------------------
# #     report = {
# #         "paths": {
# #             "forget": str(forget_path),
# #             "retain_external": str(retain_external_path),
# #             "retain_internal": str(retain_internal_path),
# #             "derived": str(derived_path),
# #             "forget_rejected": str(forget_rejected_path),
# #             "retain_external_rejected": str(retain_external_rejected_path),
# #             "retain_internal_rejected": str(retain_internal_rejected_path),
# #             "derived_rejected": str(derived_rejected_path),
# #             "year_index_dir": str(year_dir),
# #         },
# #         "year_index_file_counts": year_file_counts,
# #         "datasets": {
# #             "forget": summarize_forget(forget_records, forget_rejected_records, year_lookup),
# #             "retain_external": summarize_retain_external(retain_external_records, retain_external_rejected_records, year_lookup),
# #             "retain_internal": summarize_retain_internal(retain_internal_records, retain_internal_rejected_records, year_lookup),
# #             "derived": summarize_derived(derived_records, derived_rejected_records, year_lookup),
# #         },
# #         "cross_dataset_consistency": cross_dataset_consistency(id_sets),
# #     }

# #     # ----------------------------------------------------------
# #     # Save outputs
# #     # ----------------------------------------------------------
# #     report_json_path = Path(
# #         getattr(config, "common_eval_report_json", Path("common_dataset_evaluation_report_1.json"))
# #     )
# #     report_md_path = Path(
# #         getattr(config, "common_eval_report_md", Path("common_dataset_evaluation_report_1.md"))
# #     )

# #     save_json(report, report_json_path)
# #     report_md_path.write_text(build_markdown_report(report), encoding="utf-8")

# #     # ----------------------------------------------------------
# #     # Console summary
# #     # ----------------------------------------------------------
# #     print("\n" + "=" * 120)
# #     print("COMMON DATASET EVALUATION SUMMARY")
# #     print("=" * 120)
# #     print(f"Forget records           : {report['datasets']['forget']['record_count']}")
# #     print(f"Retain external records  : {report['datasets']['retain_external']['record_count']}")
# #     print(f"Retain internal records  : {report['datasets']['retain_internal']['record_count']}")
# #     print(f"Derived records          : {report['datasets']['derived']['record_count']}")
# #     print("-" * 120)
# #     print(f"Common ids across all 4  : {report['cross_dataset_consistency']['common_across_all_count']}")
# #     print(f"Report JSON              : {report_json_path}")
# #     print(f"Report Markdown          : {report_md_path}")

# #     return report


# # if __name__ == "__main__":
# #     config = AppConfig()
# #     evaluate_common_datasets(config)



# from __future__ import annotations

# import re
# import statistics as stats
# from collections import Counter
# from pathlib import Path
# from typing import Any, Dict, List, Optional, Set, Tuple

# from config import AppConfig
# from semantic_scholar.client import fetch_metadata
# from utils.json_utils import load_json, save_json


# # ============================================================
# # Generic helpers
# # ============================================================

# QA_TYPES = ["mcq", "true_false", "fill_blank", "assertion_reason"]


# def ensure_list(data: Any, name: str) -> List[Dict[str, Any]]:
#     if isinstance(data, dict):
#         data = [data]

#     if not isinstance(data, list):
#         raise ValueError(f"{name} must contain a JSON array (or a single dict).")

#     out: List[Dict[str, Any]] = []
#     for i, rec in enumerate(data):
#         if not isinstance(rec, dict):
#             raise ValueError(f"{name}[{i}] is not a JSON object.")
#         out.append(rec)
#     return out


# def safe_mean(xs: List[float]) -> Optional[float]:
#     return float(stats.mean(xs)) if xs else None


# def safe_median(xs: List[float]) -> Optional[float]:
#     return float(stats.median(xs)) if xs else None


# def safe_stdev(xs: List[float]) -> Optional[float]:
#     return float(stats.pstdev(xs)) if len(xs) > 1 else (0.0 if xs else None)


# def numeric_summary(xs: List[float]) -> Dict[str, Optional[float]]:
#     if not xs:
#         return {
#             "count": 0,
#             "mean": None,
#             "median": None,
#             "std": None,
#             "min": None,
#             "max": None,
#             "sum": 0,
#         }

#     return {
#         "count": len(xs),
#         "mean": safe_mean(xs),
#         "median": safe_median(xs),
#         "std": safe_stdev(xs),
#         "min": float(min(xs)),
#         "max": float(max(xs)),
#         "sum": float(sum(xs)),
#     }


# def stem_id(name: str) -> str:
#     return Path((name or "").strip()).stem.strip()


# def forget_id_from_record(rec: Dict[str, Any]) -> str:
#     """
#     Forget JSON uses pdf_name like:
#       119529102.pdf -> 119529102
#     """
#     return stem_id(rec.get("pdf_name", ""))


# def anchor_id_from_record(rec: Dict[str, Any]) -> str:
#     """
#     Retain / derived records usually identify the forget anchor via one of:
#       1. anchor_forget_paper_id
#       2. anchor_corpus_id
#       3. anchor_forget_pdf_name
#       4. fallback: pdf_name
#     """
#     for key in ("anchor_forget_paper_id", "anchor_corpus_id", "anchor_forget_pdf_name", "pdf_name"):
#         val = (rec.get(key) or "").strip()
#         if val:
#             return stem_id(val)
#     return ""


# def empty_qtype_counter() -> Dict[str, int]:
#     return {
#         "mcq": 0,
#         "true_false": 0,
#         "fill_blank": 0,
#         "assertion_reason": 0,
#         "total": 0,
#     }


# def normalize_qtype_counter(counts: Dict[str, int]) -> Dict[str, int]:
#     counts["total"] = (
#         counts.get("mcq", 0)
#         + counts.get("true_false", 0)
#         + counts.get("fill_blank", 0)
#         + counts.get("assertion_reason", 0)
#     )
#     return counts


# # ============================================================
# # Year lookup for forget / internal / derived
# # ============================================================

# def infer_year_index_dir(config: AppConfig) -> Path:
#     """
#     Infer the directory containing year-wise corpus-id files.
#     """
#     for attr in ("year_wise_corpus_ids_dir", "year_corpus_ids_dir", "corpus_ids_dir"):
#         p = getattr(config, attr, None)
#         if p:
#             return Path(p)

#     return Path("year_wise_corpus_ids")


# def build_year_lookup(year_dir: Path) -> Tuple[Dict[str, int], Dict[str, int]]:
#     """
#     Build corpus-id -> year from files like corpus_ids_2020.txt.
#     """
#     id_to_year: Dict[str, int] = {}
#     file_counts_by_year: Dict[str, int] = {}

#     if not year_dir.exists():
#         return id_to_year, file_counts_by_year

#     for path in sorted(year_dir.glob("*.txt")):
#         m = re.search(r"(19|20)\d{2}", path.stem)
#         if not m:
#             continue

#         year = int(m.group(0))
#         count = 0

#         with path.open("r", encoding="utf-8") as f:
#             for line in f:
#                 cid = line.strip()
#                 if not cid:
#                     continue
#                 id_to_year[cid] = year
#                 count += 1

#         file_counts_by_year[str(year)] = count

#     return id_to_year, file_counts_by_year


# # ============================================================
# # External retain year lookup from Semantic Scholar
# # ============================================================

# def build_semantic_scholar_year_lookup_for_retain_external(
#     records: List[Dict[str, Any]],
#     config: AppConfig,
# ) -> Tuple[Dict[str, int], List[str]]:
#     """
#     For retain external, fetch year for the retain papers themselves using Semantic Scholar.

#     Uses:
#       rec["pdf_name"] -> stem -> corpus_id

#     Example:
#       "254827565.pdf" -> "254827565"

#     Returns:
#       - retain_corpus_id -> year
#       - retain_corpus_ids that could not be resolved
#     """
#     id_to_year: Dict[str, int] = {}
#     unresolved: List[str] = []
#     seen: Set[str] = set()

#     for rec in records:
#         retain_corpus_id = stem_id(rec.get("pdf_name", ""))

#         if not retain_corpus_id or retain_corpus_id in seen:
#             continue

#         seen.add(retain_corpus_id)

#         try:
#             meta = fetch_metadata(corpus_id=retain_corpus_id, config=config)
#         except Exception:
#             meta = None

#         year = None
#         if isinstance(meta, dict):
#             year = meta.get("year")

#         if isinstance(year, int):
#             id_to_year[retain_corpus_id] = year
#         else:
#             unresolved.append(retain_corpus_id)

#     return id_to_year, unresolved


# def compute_forget_vs_retain_external_paper_overlap(
#     forget_records: List[Dict[str, Any]],
#     retain_external_records: List[Dict[str, Any]],
# ) -> Dict[str, Any]:
#     """
#     Check whether any forget paper ID overlaps with any external retain paper ID.

#     IMPORTANT:
#     For retain external, we use the retain paper's own corpus id from:
#       rec["pdf_name"] -> stem

#     Example:
#       "254827565.pdf" -> "254827565"
#     """
#     forget_ids = {
#         forget_id_from_record(rec)
#         for rec in forget_records
#         if forget_id_from_record(rec)
#     }

#     retain_external_paper_ids = {
#         stem_id(rec.get("pdf_name", ""))
#         for rec in retain_external_records
#         if stem_id(rec.get("pdf_name", ""))
#     }

#     overlap_ids = forget_ids & retain_external_paper_ids

#     return {
#         "forget_paper_count": len(forget_ids),
#         "retain_external_paper_count": len(retain_external_paper_ids),
#         "overlap_count": len(overlap_ids),
#         "overlap_ids_examples": sorted(overlap_ids)[:50],
#         "has_overlap": len(overlap_ids) > 0,
#     }

# # ============================================================
# # Question / claim counters
# # ============================================================

# def count_questions_claim_based(rec: Dict[str, Any]) -> Dict[str, int]:
#     """
#     For:
#       - forget
#       - retain external
#     where questions live inside qa_by_claim
#     """
#     counts = {"mcq": 0, "true_false": 0, "fill_blank": 0, "assertion_reason": 0}

#     for claim_obj in rec.get("qa_by_claim", []) or []:
#         for qt in QA_TYPES:
#             counts[qt] += len(claim_obj.get(qt, []) or [])

#     counts["total"] = sum(counts.values())
#     return counts


# def count_questions_internal(rec: Dict[str, Any]) -> Dict[str, int]:
#     """
#     For retain internal:
#       qa_by_base = { mcq: [...], ... }
#     """
#     counts = {"mcq": 0, "true_false": 0, "fill_blank": 0, "assertion_reason": 0}
#     qa_by_base = rec.get("qa_by_base", {}) or {}

#     for qt in QA_TYPES:
#         counts[qt] = len(qa_by_base.get(qt, []) or [])

#     counts["total"] = sum(counts.values())
#     return counts


# def count_questions_derived(rec: Dict[str, Any]) -> Dict[str, int]:
#     """
#     For derived:
#       derived_qa = { mcq: [...], ... }
#     """
#     counts = {"mcq": 0, "true_false": 0, "fill_blank": 0, "assertion_reason": 0}
#     derived_qa = rec.get("derived_qa", {}) or {}

#     for qt in QA_TYPES:
#         counts[qt] = len(derived_qa.get(qt, []) or [])

#     counts["total"] = sum(counts.values())
#     return counts


# def total_claims_from_record(rec: Dict[str, Any]) -> int:
#     """
#     All dataset families still carry paper_claims.
#     """
#     return len(rec.get("paper_claims", []) or [])


# # ============================================================
# # Rejected-QA helpers (schema-robust)
# # ============================================================

# def extract_claim_level_rejected_groups(rec: Dict[str, Any]) -> List[Dict[str, Any]]:
#     """
#     For forget / retain external rejected files, support:
#       - rejected_qa_by_claim
#       - qa_by_claim
#     """
#     groups = rec.get("rejected_qa_by_claim", None)
#     if isinstance(groups, list):
#         return groups

#     groups = rec.get("qa_by_claim", None)
#     if isinstance(groups, list):
#         return groups

#     return []


# def extract_base_level_rejected_group(rec: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
#     """
#     For retain internal rejected files, support:
#       - rejected_qa_by_base
#       - qa_by_base
#     """
#     groups = rec.get("rejected_qa_by_base", None)
#     if isinstance(groups, dict):
#         return groups

#     groups = rec.get("qa_by_base", None)
#     if isinstance(groups, dict):
#         return groups

#     return {}


# def extract_derived_level_rejected_group(rec: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
#     """
#     For derived rejected files, support:
#       - rejected_derived_qa
#       - derived_qa
#     """
#     groups = rec.get("rejected_derived_qa", None)
#     if isinstance(groups, dict):
#         return groups

#     groups = rec.get("derived_qa", None)
#     if isinstance(groups, dict):
#         return groups

#     return {}


# def count_rejected_claim_based_record(rec: Dict[str, Any]) -> Dict[str, int]:
#     counts = empty_qtype_counter()
#     groups = extract_claim_level_rejected_groups(rec)

#     for claim_obj in groups:
#         if not isinstance(claim_obj, dict):
#             continue

#         for qt in QA_TYPES:
#             counts[qt] += len(claim_obj.get(qt, []) or [])

#     return normalize_qtype_counter(counts)


# def count_rejected_internal_record(rec: Dict[str, Any]) -> Dict[str, int]:
#     counts = empty_qtype_counter()
#     groups = extract_base_level_rejected_group(rec)

#     for qt in QA_TYPES:
#         counts[qt] += len(groups.get(qt, []) or [])

#     return normalize_qtype_counter(counts)


# def count_rejected_derived_record(rec: Dict[str, Any]) -> Dict[str, int]:
#     counts = empty_qtype_counter()
#     groups = extract_derived_level_rejected_group(rec)

#     for qt in QA_TYPES:
#         counts[qt] += len(groups.get(qt, []) or [])

#     return normalize_qtype_counter(counts)


# def aggregate_rejected_counts(
#     rejected_records: List[Dict[str, Any]],
#     mode: str,
# ) -> Tuple[Dict[str, int], Dict[str, Optional[float]]]:
#     """
#     Aggregate rejected QA stats over a rejected-record list.

#     Returns:
#       - rejected_question_type_counts
#       - rejected_questions_per_paper summary
#     """
#     total_counts = empty_qtype_counter()
#     per_paper_totals: List[float] = []

#     for rec in rejected_records:
#         if mode == "claim":
#             rc = count_rejected_claim_based_record(rec)
#         elif mode == "internal":
#             rc = count_rejected_internal_record(rec)
#         elif mode == "derived":
#             rc = count_rejected_derived_record(rec)
#         else:
#             raise ValueError(f"Unknown rejected counting mode: {mode}")

#         for qt in QA_TYPES:
#             total_counts[qt] += rc[qt]

#         total_counts["total"] += rc["total"]
#         per_paper_totals.append(rc["total"])

#     return total_counts, numeric_summary(per_paper_totals)


# def load_optional_json_list(path: Path, name: str) -> List[Dict[str, Any]]:
#     if not path.exists():
#         return []
#     return ensure_list(load_json(path), name)


# def filter_forget_rejected_records_by_common_ids(
#     records: List[Dict[str, Any]],
#     common_ids: Set[str],
# ) -> List[Dict[str, Any]]:
#     """
#     Forget rejected file should be matched using pdf_name stem.
#     """
#     kept = []
#     for rec in records:
#         fid = forget_id_from_record(rec)
#         if fid in common_ids:
#             kept.append(rec)
#     return kept


# def filter_anchor_rejected_records_by_common_ids(
#     records: List[Dict[str, Any]],
#     common_ids: Set[str],
# ) -> List[Dict[str, Any]]:
#     """
#     Retain external / retain internal / derived rejected files should be matched
#     using anchor-style identifiers.
#     """
#     kept = []
#     for rec in records:
#         aid = anchor_id_from_record(rec)
#         if aid in common_ids:
#             kept.append(rec)
#     return kept


# # ============================================================
# # Forget q1 / q2 split diagnostics
# # ============================================================

# def forget_pair_split_stats(forget_records: List[Dict[str, Any]]) -> Dict[str, Any]:
#     """
#     The forget set is later divided into q1 / q2 by splitting each QA-type group equally.
#     This computes the implied q1 / q2 balance directly from the common forget JSON.
#     """
#     q1_counter = Counter()
#     q2_counter = Counter()
#     odd_groups = []
#     groups_total = 0

#     for rec in forget_records:
#         pdf_name = rec.get("pdf_name", "")

#         for claim_idx, claim_obj in enumerate(rec.get("qa_by_claim", []) or []):
#             for qt in QA_TYPES:
#                 items = claim_obj.get(qt, []) or []
#                 n = len(items)
#                 if n == 0:
#                     continue

#                 groups_total += 1
#                 half = n // 2
#                 q1_n = half
#                 q2_n = n - half

#                 q1_counter[qt] += q1_n
#                 q2_counter[qt] += q2_n

#                 if n % 2 != 0:
#                     odd_groups.append({
#                         "pdf_name": pdf_name,
#                         "claim_index": claim_idx,
#                         "qtype": qt,
#                         "count": n,
#                     })

#     q1_counter["total"] = sum(q1_counter.values())
#     q2_counter["total"] = sum(q2_counter.values())

#     return {
#         "groups_total": groups_total,
#         "q1_counts": dict(q1_counter),
#         "q2_counts": dict(q2_counter),
#         "absolute_imbalance_total": abs(q1_counter["total"] - q2_counter["total"]),
#         "odd_groups_count": len(odd_groups),
#         "odd_groups_examples": odd_groups[:20],
#     }


# # ============================================================
# # Internal helper only: compute common ids for rejected filtering
# # ============================================================

# def compute_common_ids(
#     forget_records: List[Dict[str, Any]],
#     retain_external_records: List[Dict[str, Any]],
#     retain_internal_records: List[Dict[str, Any]],
#     derived_records: List[Dict[str, Any]],
# ) -> Set[str]:
#     """
#     Internal helper only: compute common ids across the four already-common JSONs.
#     This is used ONLY to filter rejected files consistently.
#     It is NOT written into the output report.
#     """
#     forget_ids = {forget_id_from_record(r) for r in forget_records if forget_id_from_record(r)}
#     retain_external_ids = {anchor_id_from_record(r) for r in retain_external_records if anchor_id_from_record(r)}
#     retain_internal_ids = {anchor_id_from_record(r) for r in retain_internal_records if anchor_id_from_record(r)}
#     derived_ids = {anchor_id_from_record(r) for r in derived_records if anchor_id_from_record(r)}

#     return forget_ids & retain_external_ids & retain_internal_ids & derived_ids


# # ============================================================
# # Dataset summaries
# # ============================================================

# def summarize_forget(
#     records: List[Dict[str, Any]],
#     rejected_records: List[Dict[str, Any]],
#     year_lookup: Dict[str, int],
# ) -> Dict[str, Any]:
#     year_counter = Counter()
#     qtype_counter = Counter()

#     claims_per_paper: List[int] = []
#     questions_per_paper: List[int] = []
#     verbatim_counts: List[int] = []
#     missing_year_ids: List[str] = []
#     duplicate_ids = []
#     seen = set()

#     total_claims = 0

#     for rec in records:
#         fid = forget_id_from_record(rec)

#         if fid in seen:
#             duplicate_ids.append(fid)
#         seen.add(fid)

#         year = year_lookup.get(fid)
#         if year is None:
#             missing_year_ids.append(fid)
#         else:
#             year_counter[str(year)] += 1

#         n_claims = total_claims_from_record(rec)
#         total_claims += n_claims
#         claims_per_paper.append(n_claims)

#         qcounts = count_questions_claim_based(rec)
#         questions_per_paper.append(qcounts["total"])
#         for qt in QA_TYPES:
#             qtype_counter[qt] += qcounts[qt]

#         verbatim_counts.append(len(rec.get("verbatim_claims", []) or []))

#     rejected_counter, rejected_per_paper_summary = aggregate_rejected_counts(
#         rejected_records,
#         mode="claim",
#     )

#     qtype_counter["total"] = sum(qtype_counter.values())

#     return {
#         "dataset": "forget",
#         "record_count": len(records),
#         "unique_paper_ids": len(seen),
#         "duplicate_ids_count": len(set(duplicate_ids)),
#         "duplicate_ids_examples": sorted(set(duplicate_ids))[:20],
#         "year_distribution": dict(year_counter),
#         "missing_year_ids_count": len(set(missing_year_ids)),
#         "missing_year_ids_examples": sorted(set(missing_year_ids))[:20],
#         "total_claims": total_claims,
#         "claims_per_paper": numeric_summary(claims_per_paper),
#         "verbatim_claims_per_paper": numeric_summary(verbatim_counts),
#         "questions_per_paper": numeric_summary(questions_per_paper),
#         "question_type_counts": dict(qtype_counter),
#         "rejected_question_type_counts": dict(rejected_counter),
#         "rejected_questions_per_paper": rejected_per_paper_summary,
#         "forget_pair_split": forget_pair_split_stats(records),
#     }


# def summarize_retain_external(
#     records: List[Dict[str, Any]],
#     rejected_records: List[Dict[str, Any]],
#     retain_year_lookup: Dict[str, int],
#     unresolved_retain_ids: List[str],
# ) -> Dict[str, Any]:
#     """
#     Summarize retain external statistics.

#     IMPORTANT:
#     Year distribution here is computed from the retain external papers themselves,
#     using:
#       rec["pdf_name"] -> corpus_id -> Semantic Scholar metadata -> year
#     """
#     retain_paper_year_counter = Counter()
#     qtype_counter = Counter()

#     claims_per_paper: List[int] = []
#     questions_per_paper: List[int] = []

#     survived_true = 0
#     survived_false = 0
#     total_claims = 0

#     for rec in records:
#         if rec.get("retain_survived", False):
#             survived_true += 1
#         else:
#             survived_false += 1

#         n_claims = total_claims_from_record(rec)
#         total_claims += n_claims
#         claims_per_paper.append(n_claims)

#         qcounts = count_questions_claim_based(rec)
#         questions_per_paper.append(qcounts["total"])
#         for qt in QA_TYPES:
#             qtype_counter[qt] += qcounts[qt]

#         retain_corpus_id = stem_id(rec.get("pdf_name", ""))
#         retain_year = retain_year_lookup.get(retain_corpus_id)

#         if retain_year is not None:
#             retain_paper_year_counter[str(retain_year)] += 1

#     rejected_counter, rejected_per_paper_summary = aggregate_rejected_counts(
#         rejected_records,
#         mode="claim",
#     )

#     qtype_counter["total"] = sum(qtype_counter.values())

#     return {
#         "dataset": "retain_external",
#         "record_count": len(records),
#         "retain_survived_true": survived_true,
#         "retain_survived_false": survived_false,
#         "retain_paper_year_distribution": dict(retain_paper_year_counter),
#         "retain_paper_years_unresolved_count": len(set(unresolved_retain_ids)),
#         "retain_paper_years_unresolved_examples": sorted(set(unresolved_retain_ids))[:20],
#         "total_claims": total_claims,
#         "claims_per_paper": numeric_summary(claims_per_paper),
#         "questions_per_paper": numeric_summary(questions_per_paper),
#         "question_type_counts": dict(qtype_counter),
#         "rejected_question_type_counts": dict(rejected_counter),
#         "rejected_questions_per_paper": rejected_per_paper_summary,
#     }



# def summarize_retain_internal(
#     records: List[Dict[str, Any]],
#     rejected_records: List[Dict[str, Any]],
#     year_lookup: Dict[str, int],
# ) -> Dict[str, Any]:
#     year_counter = Counter()
#     qtype_counter = Counter()

#     claims_per_paper: List[int] = []
#     questions_per_paper: List[int] = []
#     missing_year_ids: List[str] = []

#     survived_true = 0
#     survived_false = 0
#     total_claims = 0

#     for rec in records:
#         aid = anchor_id_from_record(rec)
#         year = year_lookup.get(aid)
#         if year is None:
#             missing_year_ids.append(aid)
#         else:
#             year_counter[str(year)] += 1

#         if rec.get("internal_retain_survived", False):
#             survived_true += 1
#         else:
#             survived_false += 1

#         n_claims = total_claims_from_record(rec)
#         total_claims += n_claims
#         claims_per_paper.append(n_claims)

#         qcounts = count_questions_internal(rec)
#         questions_per_paper.append(qcounts["total"])
#         for qt in QA_TYPES:
#             qtype_counter[qt] += qcounts[qt]

#     rejected_counter, rejected_per_paper_summary = aggregate_rejected_counts(
#         rejected_records,
#         mode="internal",
#     )

#     qtype_counter["total"] = sum(qtype_counter.values())

#     return {
#         "dataset": "retain_internal",
#         "record_count": len(records),
#         "internal_retain_survived_true": survived_true,
#         "internal_retain_survived_false": survived_false,
#         "year_distribution": dict(year_counter),
#         "missing_year_anchor_ids_count": len(set(missing_year_ids)),
#         "missing_year_anchor_ids_examples": sorted(set(missing_year_ids))[:20],
#         "total_claims": total_claims,
#         "claims_per_paper": numeric_summary(claims_per_paper),
#         "questions_per_paper": numeric_summary(questions_per_paper),
#         "question_type_counts": dict(qtype_counter),
#         "rejected_question_type_counts": dict(rejected_counter),
#         "rejected_questions_per_paper": rejected_per_paper_summary,
#     }


# def summarize_derived(
#     records: List[Dict[str, Any]],
#     rejected_records: List[Dict[str, Any]],
#     year_lookup: Dict[str, int],
# ) -> Dict[str, Any]:
#     year_counter = Counter()
#     qtype_counter = Counter()
#     source_type_counter = Counter()

#     claims_per_paper: List[int] = []
#     questions_per_paper: List[int] = []
#     source_questions_per_paper: List[int] = []
#     missing_year_ids: List[str] = []

#     survived_true = 0
#     survived_false = 0
#     total_claims = 0

#     for rec in records:
#         aid = anchor_id_from_record(rec)
#         year = year_lookup.get(aid)
#         if year is None:
#             missing_year_ids.append(aid)
#         else:
#             year_counter[str(year)] += 1

#         if rec.get("derived_survived", False):
#             survived_true += 1
#         else:
#             survived_false += 1

#         n_claims = total_claims_from_record(rec)
#         total_claims += n_claims
#         claims_per_paper.append(n_claims)

#         qcounts = count_questions_derived(rec)
#         questions_per_paper.append(qcounts["total"])
#         for qt in QA_TYPES:
#             qtype_counter[qt] += qcounts[qt]

#         sq = rec.get("source_questions", []) or []
#         source_questions_per_paper.append(len(sq))

#         for item in sq:
#             st = (item.get("source_type") or "").strip()
#             if st:
#                 source_type_counter[st] += 1

#     rejected_counter, rejected_per_paper_summary = aggregate_rejected_counts(
#         rejected_records,
#         mode="derived",
#     )

#     qtype_counter["total"] = sum(qtype_counter.values())

#     return {
#         "dataset": "derived",
#         "record_count": len(records),
#         "derived_survived_true": survived_true,
#         "derived_survived_false": survived_false,
#         "year_distribution": dict(year_counter),
#         "missing_year_anchor_ids_count": len(set(missing_year_ids)),
#         "missing_year_anchor_ids_examples": sorted(set(missing_year_ids))[:20],
#         "total_claims": total_claims,
#         "claims_per_paper": numeric_summary(claims_per_paper),
#         "questions_per_paper": numeric_summary(questions_per_paper),
#         "question_type_counts": dict(qtype_counter),
#         "rejected_question_type_counts": dict(rejected_counter),
#         "rejected_questions_per_paper": rejected_per_paper_summary,
#         "source_questions_per_paper": numeric_summary(source_questions_per_paper),
#         "source_question_type_distribution": dict(source_type_counter),
#     }


# # ============================================================
# # Markdown report
# # ============================================================

# def build_markdown_report(report: Dict[str, Any]) -> str:
#     lines: List[str] = []

#     lines.append("# Common-Dataset Evaluation Report")
#     lines.append("")
#     lines.append("## 1. Input Files")
#     for k, v in report["paths"].items():
#         lines.append(f"- **{k}**: `{v}`")

#     lines.append("")
#     lines.append("## 2. Dataset Summaries")

#     for key in ["forget", "retain_external", "retain_internal", "derived"]:
#         ds = report["datasets"][key]

#         lines.append(f"### {key}")
#         lines.append(f"- Record count: **{ds.get('record_count')}**")
#         lines.append(f"- Total claims: **{ds.get('total_claims', 0)}**")

#         q = ds.get("question_type_counts", {})
#         if q:
#             lines.append(
#                 f"- Question counts — "
#                 f"MCQ: **{q.get('mcq', 0)}**, "
#                 f"TF: **{q.get('true_false', 0)}**, "
#                 f"Fill: **{q.get('fill_blank', 0)}**, "
#                 f"AR: **{q.get('assertion_reason', 0)}**, "
#                 f"Total: **{q.get('total', 0)}**"
#             )

#         rq = ds.get("rejected_question_type_counts", {})
#         if rq:
#             lines.append(
#                 f"- Rejected QA counts — "
#                 f"MCQ: **{rq.get('mcq', 0)}**, "
#                 f"TF: **{rq.get('true_false', 0)}**, "
#                 f"Fill: **{rq.get('fill_blank', 0)}**, "
#                 f"AR: **{rq.get('assertion_reason', 0)}**, "
#                 f"Total: **{rq.get('total', 0)}**"
#             )

#         rpp = ds.get("rejected_questions_per_paper", {})
#         if rpp:
#             lines.append(
#                 f"- Rejected questions per paper — "
#                 f"Mean: **{rpp.get('mean')}**, "
#                 f"Median: **{rpp.get('median')}**, "
#                 f"Std: **{rpp.get('std')}**, "
#                 f"Min: **{rpp.get('min')}**, "
#                 f"Max: **{rpp.get('max')}**"
#             )

#         if "year_distribution" in ds:
#             lines.append(f"- Year coverage: `{ds['year_distribution']}`")

#         if "retain_paper_year_distribution" in ds:
#             lines.append(
#                 f"- Retain paper year coverage (Semantic Scholar): "
#                 f"`{ds['retain_paper_year_distribution']}`"
#             )

#         lines.append("")

#     lines.append("## 3. Forget vs External Retain Paper Overlap")
#     overlap = report.get("forget_vs_retain_external_paper_overlap", {})
#     lines.append(f"- Forget paper count: **{overlap.get('forget_paper_count', 0)}**")
#     lines.append(f"- External retain paper count: **{overlap.get('retain_external_paper_count', 0)}**")
#     lines.append(f"- Overlap count: **{overlap.get('overlap_count', 0)}**")
#     lines.append(f"- Has overlap: **{overlap.get('has_overlap', False)}**")
#     lines.append(f"- Overlap ID examples: `{overlap.get('overlap_ids_examples', [])}`")
#     lines.append("")

#     lines.append("## 4. Forget Q1/Q2 Balance")
#     split = report["datasets"]["forget"].get("forget_pair_split", {})
#     lines.append(f"- Total QA groups inspected: **{split.get('groups_total', 0)}**")
#     lines.append(f"- Q1 counts: `{split.get('q1_counts', {})}`")
#     lines.append(f"- Q2 counts: `{split.get('q2_counts', {})}`")
#     lines.append(f"- Odd QA groups count: **{split.get('odd_groups_count', 0)}**")

#     return "\n".join(lines)


# # ============================================================
# # Main evaluation
# # ============================================================

# def evaluate_common_datasets(config: AppConfig) -> Dict[str, Any]:
#     # ----------------------------------------------------------
#     # Main common input files
#     # ----------------------------------------------------------
#     forget_path = Path(getattr(config, "common_forget_output_json", Path("forget_common.json")))
#     retain_external_path = Path(getattr(config, "common_retain_output_json", Path("retain_external_common.json")))
#     retain_internal_path = Path(getattr(config, "common_retain_internal_output_json", Path("retain_internal_common.json")))
#     derived_path = Path(getattr(config, "common_derived_output_json", Path("derived_common.json")))

#     for p in [forget_path, retain_external_path, retain_internal_path, derived_path]:
#         if not p.exists():
#             raise FileNotFoundError(f"Common dataset input file not found: {p}")

#     # ----------------------------------------------------------
#     # Optional rejected-question files
#     # ----------------------------------------------------------
#     forget_rejected_path = Path(getattr(config, "forget_olmo_rejected_json", Path("forget_olmo_rejected.json")))
#     retain_external_rejected_path = Path(
#         getattr(config, "retain_external_olmo_rejected_json", Path("retain_external_olmo_rejected.json"))
#     )
#     retain_internal_rejected_path = Path(
#         getattr(config, "retain_internal_olmo_rejected_json", Path("retain_internal_olmo_rejected.json"))
#     )
#     derived_rejected_path = Path(
#         getattr(config, "derived_olmo_rejected_json", Path("derived_olmo_rejected.json"))
#     )

#     # ----------------------------------------------------------
#     # Load common datasets
#     # ----------------------------------------------------------
#     forget_records = ensure_list(load_json(forget_path), "forget common JSON")
#     retain_external_records = ensure_list(load_json(retain_external_path), "retain external common JSON")
#     retain_internal_records = ensure_list(load_json(retain_internal_path), "retain internal common JSON")
#     derived_records = ensure_list(load_json(derived_path), "derived common JSON")

#     # ----------------------------------------------------------
#     # Internal common-id set for rejected-file filtering only
#     # ----------------------------------------------------------
#     common_ids = compute_common_ids(
#         forget_records,
#         retain_external_records,
#         retain_internal_records,
#         derived_records,
#     )

#     # ----------------------------------------------------------
#     # Load rejected files and filter them to same common ids
#     # ----------------------------------------------------------
#     forget_rejected_records = filter_forget_rejected_records_by_common_ids(
#         load_optional_json_list(forget_rejected_path, "forget rejected JSON"),
#         common_ids,
#     )
#     retain_external_rejected_records = filter_anchor_rejected_records_by_common_ids(
#         load_optional_json_list(retain_external_rejected_path, "retain external rejected JSON"),
#         common_ids,
#     )
#     retain_internal_rejected_records = filter_anchor_rejected_records_by_common_ids(
#         load_optional_json_list(retain_internal_rejected_path, "retain internal rejected JSON"),
#         common_ids,
#     )
#     derived_rejected_records = filter_anchor_rejected_records_by_common_ids(
#         load_optional_json_list(derived_rejected_path, "derived rejected JSON"),
#         common_ids,
#     )

#     # ----------------------------------------------------------
#     # Local year lookup (for forget / internal / derived)
#     # ----------------------------------------------------------
#     year_dir = infer_year_index_dir(config)
#     year_lookup, year_file_counts = build_year_lookup(year_dir)

#     # ----------------------------------------------------------
#     # Semantic Scholar year lookup for external retain papers
#     # ----------------------------------------------------------
#     retain_year_lookup, unresolved_retain_ids = build_semantic_scholar_year_lookup_for_retain_external(
#         retain_external_records,
#         config,
#     )

#     # ----------------------------------------------------------
#     # Build report
#     # ----------------------------------------------------------
#     report = {
#         "paths": {
#             "forget": str(forget_path),
#             "retain_external": str(retain_external_path),
#             "retain_internal": str(retain_internal_path),
#             "derived": str(derived_path),
#             "forget_rejected": str(forget_rejected_path),
#             "retain_external_rejected": str(retain_external_rejected_path),
#             "retain_internal_rejected": str(retain_internal_rejected_path),
#             "derived_rejected": str(derived_rejected_path),
#             "year_index_dir": str(year_dir),
#         },
#         "year_index_file_counts": year_file_counts,
#         "datasets": {
#             "forget": summarize_forget(
#                 forget_records,
#                 forget_rejected_records,
#                 year_lookup,
#             ),
#             "retain_external": summarize_retain_external(
#                 retain_external_records,
#                 retain_external_rejected_records,
#                 retain_year_lookup,
#                 unresolved_retain_ids,
#             ),
#             "retain_internal": summarize_retain_internal(
#                 retain_internal_records,
#                 retain_internal_rejected_records,
#                 year_lookup,
#             ),
#             "derived": summarize_derived(
#                 derived_records,
#                 derived_rejected_records,
#                 year_lookup,
#             ),
#         },
#     }

#     report["forget_vs_retain_external_paper_overlap"] = compute_forget_vs_retain_external_paper_overlap(
#         forget_records,
#         retain_external_records,
#     )


#     # ----------------------------------------------------------
#     # Save outputs
#     # ----------------------------------------------------------
#     report_json_path = Path(
#         getattr(config, "common_eval_report_json", Path("common_dataset_evaluation_report_1.json"))
#     )
#     report_md_path = Path(
#         getattr(config, "common_eval_report_md", Path("common_dataset_evaluation_report_1.md"))
#     )

#     save_json(report, report_json_path)
#     report_md_path.write_text(build_markdown_report(report), encoding="utf-8")

#     # ----------------------------------------------------------
#     # Console summary
#     # ----------------------------------------------------------
#     print("\n" + "=" * 120)
#     print("COMMON DATASET EVALUATION SUMMARY")
#     print("=" * 120)
#     print(f"Forget records           : {report['datasets']['forget']['record_count']}")
#     print(f"Retain external records  : {report['datasets']['retain_external']['record_count']}")
#     print(f"Retain internal records  : {report['datasets']['retain_internal']['record_count']}")
#     print(f"Derived records          : {report['datasets']['derived']['record_count']}")
#     print("-" * 120)
#     print(f"Report JSON              : {report_json_path}")
#     print(f"Report Markdown          : {report_md_path}")

#     overlap = report["forget_vs_retain_external_paper_overlap"]
#     print("-" * 120)
#     print(f"Forget vs external retain overlap count : {overlap['overlap_count']}")
#     if overlap["has_overlap"]:
#         print(f"Overlap examples                    : {overlap['overlap_ids_examples'][:10]}")

#     return report


# if __name__ == "__main__":
#     config = AppConfig()
#     evaluate_common_datasets(config)


from __future__ import annotations

import json
import math
import re
import statistics as stats
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from config import AppConfig
from semantic_scholar.client import fetch_metadata
from utils.json_utils import load_json, save_json


# ============================================================
# Generic helpers
# ============================================================

QA_TYPES = ["mcq", "true_false", "fill_blank", "assertion_reason"]


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


def safe_mean(xs: List[float]) -> Optional[float]:
    return float(stats.mean(xs)) if xs else None


def safe_median(xs: List[float]) -> Optional[float]:
    return float(stats.median(xs)) if xs else None


def safe_stdev(xs: List[float]) -> Optional[float]:
    return float(stats.pstdev(xs)) if len(xs) > 1 else (0.0 if xs else None)


def numeric_summary(xs: List[float]) -> Dict[str, Optional[float]]:
    if not xs:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
            "sum": 0,
        }

    return {
        "count": len(xs),
        "mean": safe_mean(xs),
        "median": safe_median(xs),
        "std": safe_stdev(xs),
        "min": float(min(xs)),
        "max": float(max(xs)),
        "sum": float(sum(xs)),
    }


def percentile(xs: List[float], q: float) -> Optional[float]:
    if not xs:
        return None

    xs_sorted = sorted(xs)
    pos = (len(xs_sorted) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)

    if lo == hi:
        return float(xs_sorted[lo])

    frac = pos - lo
    return float(xs_sorted[lo] * (1 - frac) + xs_sorted[hi] * frac)


def cost_summary(xs: List[float]) -> Dict[str, Optional[float]]:
    if not xs:
        return {
            "count": 0,
            "sum": 0.0,
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
            "p25": None,
            "p75": None,
            "p90": None,
            "p95": None,
        }

    return {
        "count": len(xs),
        "sum": float(sum(xs)),
        "mean": safe_mean(xs),
        "median": safe_median(xs),
        "std": safe_stdev(xs),
        "min": float(min(xs)),
        "max": float(max(xs)),
        "p25": percentile(xs, 0.25),
        "p75": percentile(xs, 0.75),
        "p90": percentile(xs, 0.90),
        "p95": percentile(xs, 0.95),
    }


def safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def stem_id(name: str) -> str:
    return Path((name or "").strip()).stem.strip()


def forget_id_from_record(rec: Dict[str, Any]) -> str:
    """
    Forget JSON uses pdf_name like:
      119529102.pdf -> 119529102
    """
    return stem_id(rec.get("pdf_name", ""))


def anchor_id_from_record(rec: Dict[str, Any]) -> str:
    """
    Retain / derived records usually identify the forget anchor via one of:
      1. anchor_forget_paper_id
      2. anchor_corpus_id
      3. anchor_forget_pdf_name
      4. fallback: pdf_name
    """
    for key in ("anchor_forget_paper_id", "anchor_corpus_id", "anchor_forget_pdf_name", "pdf_name"):
        val = (rec.get(key) or "").strip()
        if val:
            return stem_id(val)
    return ""


def empty_qtype_counter() -> Dict[str, int]:
    return {
        "mcq": 0,
        "true_false": 0,
        "fill_blank": 0,
        "assertion_reason": 0,
        "total": 0,
    }


def normalize_qtype_counter(counts: Dict[str, int]) -> Dict[str, int]:
    counts["total"] = (
        counts.get("mcq", 0)
        + counts.get("true_false", 0)
        + counts.get("fill_blank", 0)
        + counts.get("assertion_reason", 0)
    )
    return counts


# ============================================================
# Year lookup for forget / internal / derived
# ============================================================

def infer_year_index_dir(config: AppConfig) -> Path:
    for attr in ("year_wise_corpus_ids_dir", "year_corpus_ids_dir", "corpus_ids_dir"):
        p = getattr(config, attr, None)
        if p:
            return Path(p)

    return Path("year_wise_corpus_ids")


def build_year_lookup(year_dir: Path) -> Tuple[Dict[str, int], Dict[str, int]]:
    id_to_year: Dict[str, int] = {}
    file_counts_by_year: Dict[str, int] = {}

    if not year_dir.exists():
        return id_to_year, file_counts_by_year

    for path in sorted(year_dir.glob("*.txt")):
        m = re.search(r"(19|20)\d{2}", path.stem)
        if not m:
            continue

        year = int(m.group(0))
        count = 0

        with path.open("r", encoding="utf-8") as f:
            for line in f:
                cid = line.strip()
                if not cid:
                    continue
                id_to_year[cid] = year
                count += 1

        file_counts_by_year[str(year)] = count

    return id_to_year, file_counts_by_year


# ============================================================
# External retain year lookup from Semantic Scholar
# ============================================================

def build_semantic_scholar_year_lookup_for_retain_external(
    records: List[Dict[str, Any]],
    config: AppConfig,
) -> Tuple[Dict[str, int], List[str]]:
    """
    For retain external, fetch year for the retain papers themselves using Semantic Scholar.

    Uses:
      rec["pdf_name"] -> stem -> corpus_id
    """
    id_to_year: Dict[str, int] = {}
    unresolved: List[str] = []
    seen: Set[str] = set()

    for rec in records:
        retain_corpus_id = stem_id(rec.get("pdf_name", ""))

        if not retain_corpus_id or retain_corpus_id in seen:
            continue

        seen.add(retain_corpus_id)

        try:
            meta = fetch_metadata(corpus_id=retain_corpus_id, config=config)
        except Exception:
            meta = None

        year = None
        if isinstance(meta, dict):
            year = meta.get("year")

        if isinstance(year, int):
            id_to_year[retain_corpus_id] = year
        else:
            unresolved.append(retain_corpus_id)

    return id_to_year, unresolved


def compute_forget_vs_retain_external_paper_overlap(
    forget_records: List[Dict[str, Any]],
    retain_external_records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Check whether any forget paper ID overlaps with any external retain paper ID.

    For retain external, we use the retain paper's own corpus id from:
      rec["pdf_name"] -> stem
    """
    forget_ids = {
        forget_id_from_record(rec)
        for rec in forget_records
        if forget_id_from_record(rec)
    }

    retain_external_paper_ids = {
        stem_id(rec.get("pdf_name", ""))
        for rec in retain_external_records
        if stem_id(rec.get("pdf_name", ""))
    }

    overlap_ids = forget_ids & retain_external_paper_ids

    return {
        "forget_paper_count": len(forget_ids),
        "retain_external_paper_count": len(retain_external_paper_ids),
        "overlap_count": len(overlap_ids),
        "overlap_ids_examples": sorted(overlap_ids)[:50],
        "has_overlap": len(overlap_ids) > 0,
    }


# ============================================================
# Question / claim counters
# ============================================================

def count_questions_claim_based(rec: Dict[str, Any]) -> Dict[str, int]:
    counts = {"mcq": 0, "true_false": 0, "fill_blank": 0, "assertion_reason": 0}

    for claim_obj in rec.get("qa_by_claim", []) or []:
        for qt in QA_TYPES:
            counts[qt] += len(claim_obj.get(qt, []) or [])

    counts["total"] = sum(counts.values())
    return counts


def count_questions_internal(rec: Dict[str, Any]) -> Dict[str, int]:
    counts = {"mcq": 0, "true_false": 0, "fill_blank": 0, "assertion_reason": 0}
    qa_by_base = rec.get("qa_by_base", {}) or {}

    for qt in QA_TYPES:
        counts[qt] = len(qa_by_base.get(qt, []) or [])

    counts["total"] = sum(counts.values())
    return counts


def count_questions_derived(rec: Dict[str, Any]) -> Dict[str, int]:
    counts = {"mcq": 0, "true_false": 0, "fill_blank": 0, "assertion_reason": 0}
    derived_qa = rec.get("derived_qa", {}) or {}

    for qt in QA_TYPES:
        counts[qt] = len(derived_qa.get(qt, []) or [])

    counts["total"] = sum(counts.values())
    return counts


def total_claims_from_record(rec: Dict[str, Any]) -> int:
    return len(rec.get("paper_claims", []) or [])


# ============================================================
# Rejected-QA helpers (schema-robust)
# ============================================================

def extract_claim_level_rejected_groups(rec: Dict[str, Any]) -> List[Dict[str, Any]]:
    groups = rec.get("rejected_qa_by_claim", None)
    if isinstance(groups, list):
        return groups

    groups = rec.get("qa_by_claim", None)
    if isinstance(groups, list):
        return groups

    return []


def extract_base_level_rejected_group(rec: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    groups = rec.get("rejected_qa_by_base", None)
    if isinstance(groups, dict):
        return groups

    groups = rec.get("qa_by_base", None)
    if isinstance(groups, dict):
        return groups

    return {}


def extract_derived_level_rejected_group(rec: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    groups = rec.get("rejected_derived_qa", None)
    if isinstance(groups, dict):
        return groups

    groups = rec.get("derived_qa", None)
    if isinstance(groups, dict):
        return groups

    return {}


def count_rejected_claim_based_record(rec: Dict[str, Any]) -> Dict[str, int]:
    counts = empty_qtype_counter()
    groups = extract_claim_level_rejected_groups(rec)

    for claim_obj in groups:
        if not isinstance(claim_obj, dict):
            continue

        for qt in QA_TYPES:
            counts[qt] += len(claim_obj.get(qt, []) or [])

    return normalize_qtype_counter(counts)


def count_rejected_internal_record(rec: Dict[str, Any]) -> Dict[str, int]:
    counts = empty_qtype_counter()
    groups = extract_base_level_rejected_group(rec)

    for qt in QA_TYPES:
        counts[qt] += len(groups.get(qt, []) or [])

    return normalize_qtype_counter(counts)


def count_rejected_derived_record(rec: Dict[str, Any]) -> Dict[str, int]:
    counts = empty_qtype_counter()
    groups = extract_derived_level_rejected_group(rec)

    for qt in QA_TYPES:
        counts[qt] += len(groups.get(qt, []) or [])

    return normalize_qtype_counter(counts)


def aggregate_rejected_counts(
    rejected_records: List[Dict[str, Any]],
    mode: str,
) -> Tuple[Dict[str, int], Dict[str, Optional[float]]]:
    total_counts = empty_qtype_counter()
    per_paper_totals: List[float] = []

    for rec in rejected_records:
        if mode == "claim":
            rc = count_rejected_claim_based_record(rec)
        elif mode == "internal":
            rc = count_rejected_internal_record(rec)
        elif mode == "derived":
            rc = count_rejected_derived_record(rec)
        else:
            raise ValueError(f"Unknown rejected counting mode: {mode}")

        for qt in QA_TYPES:
            total_counts[qt] += rc[qt]

        total_counts["total"] += rc["total"]
        per_paper_totals.append(rc["total"])

    return total_counts, numeric_summary(per_paper_totals)


def load_optional_json_list(path: Path, name: str) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return ensure_list(load_json(path), name)


def filter_forget_rejected_records_by_common_ids(
    records: List[Dict[str, Any]],
    common_ids: Set[str],
) -> List[Dict[str, Any]]:
    kept = []
    for rec in records:
        fid = forget_id_from_record(rec)
        if fid in common_ids:
            kept.append(rec)
    return kept


def filter_anchor_rejected_records_by_common_ids(
    records: List[Dict[str, Any]],
    common_ids: Set[str],
) -> List[Dict[str, Any]]:
    kept = []
    for rec in records:
        aid = anchor_id_from_record(rec)
        if aid in common_ids:
            kept.append(rec)
    return kept


# ============================================================
# Internal helper only: compute common ids for rejected filtering
# ============================================================

def compute_common_ids(
    forget_records: List[Dict[str, Any]],
    retain_external_records: List[Dict[str, Any]],
    retain_internal_records: List[Dict[str, Any]],
    derived_records: List[Dict[str, Any]],
) -> Set[str]:
    forget_ids = {forget_id_from_record(r) for r in forget_records if forget_id_from_record(r)}
    retain_external_ids = {anchor_id_from_record(r) for r in retain_external_records if anchor_id_from_record(r)}
    retain_internal_ids = {anchor_id_from_record(r) for r in retain_internal_records if anchor_id_from_record(r)}
    derived_ids = {anchor_id_from_record(r) for r in derived_records if anchor_id_from_record(r)}

    return forget_ids & retain_external_ids & retain_internal_ids & derived_ids


# ============================================================
# Cost statistics helpers
# ============================================================

COST_TYPE_KEYS = ["type", "scenario", "stage", "task", "pipeline"]
TOTAL_COST_KEYS = ["total_cost", "totalCost", "cost"]
PROMPT_COST_KEYS = ["prompt_cost", "promptCost"]
COMPLETION_COST_KEYS = ["completion_cost", "completionCost"]


def parse_cost_log_line(line: str) -> Optional[Dict[str, Any]]:
    """
    Expect one JSON object per line.

    Example:
      {"type": "claim extraction", "total_cost": 0.012, "prompt_cost": 0.008, "completion_cost": 0.004}
    """
    line = line.strip()
    if not line:
        return None

    try:
        obj = json.loads(line)
    except Exception:
        return None

    if not isinstance(obj, dict):
        return None

    type_value = None
    for key in COST_TYPE_KEYS:
        if key in obj and obj[key] is not None:
            type_value = str(obj[key]).strip()
            if type_value:
                break

    total_cost = None
    for key in TOTAL_COST_KEYS:
        if key in obj:
            total_cost = safe_float(obj.get(key))
            if total_cost is not None:
                break

    prompt_cost = None
    for key in PROMPT_COST_KEYS:
        if key in obj:
            prompt_cost = safe_float(obj.get(key))
            if prompt_cost is not None:
                break

    completion_cost = None
    for key in COMPLETION_COST_KEYS:
        if key in obj:
            completion_cost = safe_float(obj.get(key))
            if completion_cost is not None:
                break

    if type_value is None and total_cost is None and prompt_cost is None and completion_cost is None:
        return None

    return {
        "type": type_value or "unknown",
        "total_cost": total_cost,
        "prompt_cost": prompt_cost,
        "completion_cost": completion_cost,
        "raw": obj,
    }


def load_cost_log_records(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []

    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            rec = parse_cost_log_line(line)
            if rec is not None:
                records.append(rec)

    return records


def normalize_cost_type(type_value: str) -> str:
    """
    Normalize type strings to make matching robust.
    """
    t = (type_value or "").strip().lower()

    # collapse repeated whitespace
    t = re.sub(r"\s+", " ", t)

    aliases = {
        "cs paper filter": "cs filter",
        "paper type filter": "paper type",
        "retain internal generation": "retain internal",
        "derived question generation": "derived qa generation",
    }

    return aliases.get(t, t)


def infer_dataset_from_cost_type(type_value: str, source: str) -> str:
    """
    source:
      - "forget_log"
      - "retain_log"

    Correct dataset mapping based on BOTH source file and type.

    cost_log:
      forget:
        - cs filter
        - paper type
        - claim extraction
        - qa generation
        - verbatim claim extraction
      retain_internal:
        - retain internal
      derived:
        - derived qa generation

    cost_log_retain:
      retain_external:
        - paper type
        - claim extraction
        - qa generation

    If a type appears in retain_log, treat it as retain_external.
    """
    t = normalize_cost_type(type_value)

    if source == "retain_log":
        return "retain_external"

    # source == forget_log
    if t == "retain internal":
        return "retain_internal"

    if t == "derived qa generation":
        return "derived"

    # everything else in cost_log belongs to forget
    return "forget"


def aggregate_cost_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_costs = [r["total_cost"] for r in rows if r.get("total_cost") is not None]
    prompt_costs = [r["prompt_cost"] for r in rows if r.get("prompt_cost") is not None]
    completion_costs = [r["completion_cost"] for r in rows if r.get("completion_cost") is not None]

    return {
        "event_count": len(rows),
        "total_cost_summary": cost_summary(total_costs),
        "prompt_cost_summary": cost_summary(prompt_costs),
        "completion_cost_summary": cost_summary(completion_costs),
    }


def summarize_cost_logs(config: AppConfig) -> Dict[str, Any]:
    """
    Build cost statistics from:
      - config.cost_log_path
      - config.cost_log_retain_path
    """
    forget_log_path = Path(getattr(config, "cost_log_path", Path("logs_2/cost_log.jsonl")))
    retain_log_path = Path(getattr(config, "cost_log_retain_path", Path("logs_2/cost_log_retain.jsonl")))

    forget_rows_raw = load_cost_log_records(forget_log_path)
    retain_rows_raw = load_cost_log_records(retain_log_path)

    rows: List[Dict[str, Any]] = []

    for rec in forget_rows_raw:
        rows.append({
            **rec,
            "type": normalize_cost_type(rec.get("type", "")),
            "dataset": infer_dataset_from_cost_type(rec.get("type", ""), source="forget_log"),
            "source_file": str(forget_log_path),
        })

    for rec in retain_rows_raw:
        rows.append({
            **rec,
            "type": normalize_cost_type(rec.get("type", "")),
            "dataset": infer_dataset_from_cost_type(rec.get("type", ""), source="retain_log"),
            "source_file": str(retain_log_path),
        })

    # ----------------------------------------------------------
    # Overall
    # ----------------------------------------------------------
    overall = aggregate_cost_rows(rows)

    # ----------------------------------------------------------
    # By dataset
    # ----------------------------------------------------------
    by_dataset_rows: Dict[str, List[Dict[str, Any]]] = {
        "forget": [],
        "retain_external": [],
        "retain_internal": [],
        "derived": [],
    }

    for r in rows:
        ds = r.get("dataset", "unknown")
        by_dataset_rows.setdefault(ds, []).append(r)

    by_dataset = {
        ds: aggregate_cost_rows(ds_rows)
        for ds, ds_rows in by_dataset_rows.items()
    }

    # ----------------------------------------------------------
    # By dataset -> by type
    # ----------------------------------------------------------
    by_dataset_by_type: Dict[str, Dict[str, Any]] = {}

    for ds, ds_rows in by_dataset_rows.items():
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for r in ds_rows:
            t = r.get("type", "unknown")
            grouped.setdefault(t, []).append(r)

        by_dataset_by_type[ds] = {
            t: aggregate_cost_rows(t_rows)
            for t, t_rows in sorted(grouped.items())
        }

    return {
        "paths": {
            "cost_log": str(forget_log_path),
            "cost_log_retain": str(retain_log_path),
        },
        "overall": overall,
        "by_dataset": by_dataset,
        "by_dataset_by_type": by_dataset_by_type,
    }




# ============================================================
# Forget q1 / q2 split diagnostics
# ============================================================

def forget_pair_split_stats(forget_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    q1_counter = Counter()
    q2_counter = Counter()
    odd_groups = []
    groups_total = 0

    for rec in forget_records:
        pdf_name = rec.get("pdf_name", "")

        for claim_idx, claim_obj in enumerate(rec.get("qa_by_claim", []) or []):
            for qt in QA_TYPES:
                items = claim_obj.get(qt, []) or []
                n = len(items)
                if n == 0:
                    continue

                groups_total += 1
                half = n // 2
                q1_n = half
                q2_n = n - half

                q1_counter[qt] += q1_n
                q2_counter[qt] += q2_n

                if n % 2 != 0:
                    odd_groups.append({
                        "pdf_name": pdf_name,
                        "claim_index": claim_idx,
                        "qtype": qt,
                        "count": n,
                    })

    q1_counter["total"] = sum(q1_counter.values())
    q2_counter["total"] = sum(q2_counter.values())

    return {
        "groups_total": groups_total,
        "q1_counts": dict(q1_counter),
        "q2_counts": dict(q2_counter),
        "absolute_imbalance_total": abs(q1_counter["total"] - q2_counter["total"]),
        "odd_groups_count": len(odd_groups),
        "odd_groups_examples": odd_groups[:20],
    }


# ============================================================
# Dataset summaries
# ============================================================

def summarize_forget(
    records: List[Dict[str, Any]],
    rejected_records: List[Dict[str, Any]],
    year_lookup: Dict[str, int],
) -> Dict[str, Any]:
    year_counter = Counter()
    qtype_counter = Counter()

    claims_per_paper: List[int] = []
    questions_per_paper: List[int] = []
    verbatim_counts: List[int] = []
    missing_year_ids: List[str] = []
    duplicate_ids = []
    seen = set()

    total_claims = 0

    for rec in records:
        fid = forget_id_from_record(rec)

        if fid in seen:
            duplicate_ids.append(fid)
        seen.add(fid)

        year = year_lookup.get(fid)
        if year is None:
            missing_year_ids.append(fid)
        else:
            year_counter[str(year)] += 1

        n_claims = total_claims_from_record(rec)
        total_claims += n_claims
        claims_per_paper.append(n_claims)

        qcounts = count_questions_claim_based(rec)
        questions_per_paper.append(qcounts["total"])
        for qt in QA_TYPES:
            qtype_counter[qt] += qcounts[qt]

        verbatim_counts.append(len(rec.get("verbatim_claims", []) or []))

    rejected_counter, rejected_per_paper_summary = aggregate_rejected_counts(
        rejected_records,
        mode="claim",
    )

    qtype_counter["total"] = sum(qtype_counter.values())

    return {
        "dataset": "forget",
        "record_count": len(records),
        "unique_paper_ids": len(seen),
        "duplicate_ids_count": len(set(duplicate_ids)),
        "duplicate_ids_examples": sorted(set(duplicate_ids))[:20],
        "year_distribution": dict(year_counter),
        "missing_year_ids_count": len(set(missing_year_ids)),
        "missing_year_ids_examples": sorted(set(missing_year_ids))[:20],
        "total_claims": total_claims,
        "claims_per_paper": numeric_summary(claims_per_paper),
        "verbatim_claims_per_paper": numeric_summary(verbatim_counts),
        "questions_per_paper": numeric_summary(questions_per_paper),
        "question_type_counts": dict(qtype_counter),
        "rejected_question_type_counts": dict(rejected_counter),
        "rejected_questions_per_paper": rejected_per_paper_summary,
        "forget_pair_split": forget_pair_split_stats(records),
    }


def summarize_retain_external(
    records: List[Dict[str, Any]],
    rejected_records: List[Dict[str, Any]],
    retain_year_lookup: Dict[str, int],
    unresolved_retain_ids: List[str],
) -> Dict[str, Any]:
    retain_paper_year_counter = Counter()
    qtype_counter = Counter()

    claims_per_paper: List[int] = []
    questions_per_paper: List[int] = []

    survived_true = 0
    survived_false = 0
    total_claims = 0

    for rec in records:
        if rec.get("retain_survived", False):
            survived_true += 1
        else:
            survived_false += 1

        n_claims = total_claims_from_record(rec)
        total_claims += n_claims
        claims_per_paper.append(n_claims)

        qcounts = count_questions_claim_based(rec)
        questions_per_paper.append(qcounts["total"])
        for qt in QA_TYPES:
            qtype_counter[qt] += qcounts[qt]

        retain_corpus_id = stem_id(rec.get("pdf_name", ""))
        retain_year = retain_year_lookup.get(retain_corpus_id)

        if retain_year is not None:
            retain_paper_year_counter[str(retain_year)] += 1

    rejected_counter, rejected_per_paper_summary = aggregate_rejected_counts(
        rejected_records,
        mode="claim",
    )

    qtype_counter["total"] = sum(qtype_counter.values())

    return {
        "dataset": "retain_external",
        "record_count": len(records),
        "retain_survived_true": survived_true,
        "retain_survived_false": survived_false,
        "retain_paper_year_distribution": dict(retain_paper_year_counter),
        "retain_paper_years_unresolved_count": len(set(unresolved_retain_ids)),
        "retain_paper_years_unresolved_examples": sorted(set(unresolved_retain_ids))[:20],
        "total_claims": total_claims,
        "claims_per_paper": numeric_summary(claims_per_paper),
        "questions_per_paper": numeric_summary(questions_per_paper),
        "question_type_counts": dict(qtype_counter),
        "rejected_question_type_counts": dict(rejected_counter),
        "rejected_questions_per_paper": rejected_per_paper_summary,
    }


def summarize_retain_internal(
    records: List[Dict[str, Any]],
    rejected_records: List[Dict[str, Any]],
    year_lookup: Dict[str, int],
) -> Dict[str, Any]:
    year_counter = Counter()
    qtype_counter = Counter()

    claims_per_paper: List[int] = []
    questions_per_paper: List[int] = []
    missing_year_ids: List[str] = []

    survived_true = 0
    survived_false = 0
    total_claims = 0

    for rec in records:
        aid = anchor_id_from_record(rec)
        year = year_lookup.get(aid)
        if year is None:
            missing_year_ids.append(aid)
        else:
            year_counter[str(year)] += 1

        if rec.get("internal_retain_survived", False):
            survived_true += 1
        else:
            survived_false += 1

        n_claims = total_claims_from_record(rec)
        total_claims += n_claims
        claims_per_paper.append(n_claims)

        qcounts = count_questions_internal(rec)
        questions_per_paper.append(qcounts["total"])
        for qt in QA_TYPES:
            qtype_counter[qt] += qcounts[qt]

    rejected_counter, rejected_per_paper_summary = aggregate_rejected_counts(
        rejected_records,
        mode="internal",
    )

    qtype_counter["total"] = sum(qtype_counter.values())

    return {
        "dataset": "retain_internal",
        "record_count": len(records),
        "internal_retain_survived_true": survived_true,
        "internal_retain_survived_false": survived_false,
        "year_distribution": dict(year_counter),
        "missing_year_anchor_ids_count": len(set(missing_year_ids)),
        "missing_year_anchor_ids_examples": sorted(set(missing_year_ids))[:20],
        "total_claims": total_claims,
        "claims_per_paper": numeric_summary(claims_per_paper),
        "questions_per_paper": numeric_summary(questions_per_paper),
        "question_type_counts": dict(qtype_counter),
        "rejected_question_type_counts": dict(rejected_counter),
        "rejected_questions_per_paper": rejected_per_paper_summary,
    }


def summarize_derived(
    records: List[Dict[str, Any]],
    rejected_records: List[Dict[str, Any]],
    year_lookup: Dict[str, int],
) -> Dict[str, Any]:
    year_counter = Counter()
    qtype_counter = Counter()
    source_type_counter = Counter()

    claims_per_paper: List[int] = []
    questions_per_paper: List[int] = []
    source_questions_per_paper: List[int] = []
    missing_year_ids: List[str] = []

    survived_true = 0
    survived_false = 0
    total_claims = 0

    for rec in records:
        aid = anchor_id_from_record(rec)
        year = year_lookup.get(aid)
        if year is None:
            missing_year_ids.append(aid)
        else:
            year_counter[str(year)] += 1

        if rec.get("derived_survived", False):
            survived_true += 1
        else:
            survived_false += 1

        n_claims = total_claims_from_record(rec)
        total_claims += n_claims
        claims_per_paper.append(n_claims)

        qcounts = count_questions_derived(rec)
        questions_per_paper.append(qcounts["total"])
        for qt in QA_TYPES:
            qtype_counter[qt] += qcounts[qt]

        sq = rec.get("source_questions", []) or []
        source_questions_per_paper.append(len(sq))

        for item in sq:
            st = (item.get("source_type") or "").strip()
            if st:
                source_type_counter[st] += 1

    rejected_counter, rejected_per_paper_summary = aggregate_rejected_counts(
        rejected_records,
        mode="derived",
    )

    qtype_counter["total"] = sum(qtype_counter.values())

    return {
        "dataset": "derived",
        "record_count": len(records),
        "derived_survived_true": survived_true,
        "derived_survived_false": survived_false,
        "year_distribution": dict(year_counter),
        "missing_year_anchor_ids_count": len(set(missing_year_ids)),
        "missing_year_anchor_ids_examples": sorted(set(missing_year_ids))[:20],
        "total_claims": total_claims,
        "claims_per_paper": numeric_summary(claims_per_paper),
        "questions_per_paper": numeric_summary(questions_per_paper),
        "question_type_counts": dict(qtype_counter),
        "rejected_question_type_counts": dict(rejected_counter),
        "rejected_questions_per_paper": rejected_per_paper_summary,
        "source_questions_per_paper": numeric_summary(source_questions_per_paper),
        "source_question_type_distribution": dict(source_type_counter),
    }


# ============================================================
# Markdown report
# ============================================================

def build_markdown_report(report: Dict[str, Any]) -> str:
    lines: List[str] = []

    lines.append("# Common-Dataset Evaluation Report")
    lines.append("")
    lines.append("## 1. Input Files")
    for k, v in report["paths"].items():
        lines.append(f"- **{k}**: `{v}`")

    lines.append("")
    lines.append("## 2. Dataset Summaries")

    for key in ["forget", "retain_external", "retain_internal", "derived"]:
        ds = report["datasets"][key]

        lines.append(f"### {key}")
        lines.append(f"- Record count: **{ds.get('record_count')}**")
        lines.append(f"- Total claims: **{ds.get('total_claims', 0)}**")

        q = ds.get("question_type_counts", {})
        if q:
            lines.append(
                f"- Question counts — "
                f"MCQ: **{q.get('mcq', 0)}**, "
                f"TF: **{q.get('true_false', 0)}**, "
                f"Fill: **{q.get('fill_blank', 0)}**, "
                f"AR: **{q.get('assertion_reason', 0)}**, "
                f"Total: **{q.get('total', 0)}**"
            )

        rq = ds.get("rejected_question_type_counts", {})
        if rq:
            lines.append(
                f"- Rejected QA counts — "
                f"MCQ: **{rq.get('mcq', 0)}**, "
                f"TF: **{rq.get('true_false', 0)}**, "
                f"Fill: **{rq.get('fill_blank', 0)}**, "
                f"AR: **{rq.get('assertion_reason', 0)}**, "
                f"Total: **{rq.get('total', 0)}**"
            )

        rpp = ds.get("rejected_questions_per_paper", {})
        if rpp:
            lines.append(
                f"- Rejected questions per paper — "
                f"Mean: **{rpp.get('mean')}**, "
                f"Median: **{rpp.get('median')}**, "
                f"Std: **{rpp.get('std')}**, "
                f"Min: **{rpp.get('min')}**, "
                f"Max: **{rpp.get('max')}**"
            )

        if "year_distribution" in ds:
            lines.append(f"- Year coverage: `{ds['year_distribution']}`")

        if "retain_paper_year_distribution" in ds:
            lines.append(
                f"- Retain paper year coverage (Semantic Scholar): "
                f"`{ds['retain_paper_year_distribution']}`"
            )

        lines.append("")

    lines.append("## 3. Forget vs External Retain Paper Overlap")
    overlap = report.get("forget_vs_retain_external_paper_overlap", {})
    lines.append(f"- Forget paper count: **{overlap.get('forget_paper_count', 0)}**")
    lines.append(f"- External retain paper count: **{overlap.get('retain_external_paper_count', 0)}**")
    lines.append(f"- Overlap count: **{overlap.get('overlap_count', 0)}**")
    lines.append(f"- Has overlap: **{overlap.get('has_overlap', False)}**")
    lines.append(f"- Overlap ID examples: `{overlap.get('overlap_ids_examples', [])}`")
    lines.append("")

    lines.append("## 4. Cost Statistics")
    cost_stats = report.get("cost_statistics", {})
    overall_cost = cost_stats.get("overall", {})
    if overall_cost:
        tc = overall_cost.get("total_cost_summary", {})
        pc = overall_cost.get("prompt_cost_summary", {})
        cc = overall_cost.get("completion_cost_summary", {})
        lines.append("### Overall")
        lines.append(f"- Event count: **{overall_cost.get('event_count', 0)}**")
        lines.append(f"- Total cost sum: **{tc.get('sum')}** | mean: **{tc.get('mean')}** | median: **{tc.get('median')}** | std: **{tc.get('std')}**")
        lines.append(f"- Prompt cost sum: **{pc.get('sum')}** | mean: **{pc.get('mean')}**")
        lines.append(f"- Completion cost sum: **{cc.get('sum')}** | mean: **{cc.get('mean')}**")
        lines.append("")

    by_dataset = cost_stats.get("by_dataset", {})
    by_dataset_by_type = cost_stats.get("by_dataset_by_type", {})

    for ds_name in ["forget", "retain_external", "retain_internal", "derived"]:
        ds_payload = by_dataset.get(ds_name, {})
        if not ds_payload:
            continue

        tc = ds_payload.get("total_cost_summary", {})
        pc = ds_payload.get("prompt_cost_summary", {})
        cc = ds_payload.get("completion_cost_summary", {})

        lines.append(f"### {ds_name}")
        lines.append(f"- Event count: **{ds_payload.get('event_count', 0)}**")
        lines.append(f"- Total cost sum: **{tc.get('sum')}** | mean: **{tc.get('mean')}** | median: **{tc.get('median')}** | std: **{tc.get('std')}**")
        lines.append(f"- Prompt cost sum: **{pc.get('sum')}** | mean: **{pc.get('mean')}**")
        lines.append(f"- Completion cost sum: **{cc.get('sum')}** | mean: **{cc.get('mean')}**")

        type_map = by_dataset_by_type.get(ds_name, {})
        if type_map:
            lines.append("- By type:")
            for type_name, type_payload in type_map.items():
                ttc = type_payload.get("total_cost_summary", {})
                lines.append(
                    f"  - `{type_name}` -> count: **{type_payload.get('event_count', 0)}**, "
                    f"sum: **{ttc.get('sum')}**, mean: **{ttc.get('mean')}**, median: **{ttc.get('median')}**"
                )

        lines.append("")

    lines.append("## 5. Forget Q1/Q2 Balance")
    split = report["datasets"]["forget"].get("forget_pair_split", {})
    lines.append(f"- Total QA groups inspected: **{split.get('groups_total', 0)}**")
    lines.append(f"- Q1 counts: `{split.get('q1_counts', {})}`")
    lines.append(f"- Q2 counts: `{split.get('q2_counts', {})}`")
    lines.append(f"- Odd QA groups count: **{split.get('odd_groups_count', 0)}**")

    return "\n".join(lines)


# ============================================================
# Main evaluation
# ============================================================

def evaluate_common_datasets(config: AppConfig) -> Dict[str, Any]:
    forget_path = Path(getattr(config, "common_forget_output_json", Path("forget_common.json")))
    retain_external_path = Path(getattr(config, "common_retain_output_json", Path("retain_external_common.json")))
    retain_internal_path = Path(getattr(config, "common_retain_internal_output_json", Path("retain_internal_common.json")))
    derived_path = Path(getattr(config, "common_derived_output_json", Path("derived_common.json")))

    for p in [forget_path, retain_external_path, retain_internal_path, derived_path]:
        if not p.exists():
            raise FileNotFoundError(f"Common dataset input file not found: {p}")

    forget_rejected_path = Path(getattr(config, "forget_olmo_rejected_json", Path("forget_olmo_rejected.json")))
    retain_external_rejected_path = Path(
        getattr(config, "retain_external_olmo_rejected_json", Path("retain_external_olmo_rejected.json"))
    )
    retain_internal_rejected_path = Path(
        getattr(config, "retain_internal_olmo_rejected_json", Path("retain_internal_olmo_rejected.json"))
    )
    derived_rejected_path = Path(
        getattr(config, "derived_olmo_rejected_json", Path("derived_olmo_rejected.json"))
    )

    forget_records = ensure_list(load_json(forget_path), "forget common JSON")
    retain_external_records = ensure_list(load_json(retain_external_path), "retain external common JSON")
    retain_internal_records = ensure_list(load_json(retain_internal_path), "retain internal common JSON")
    derived_records = ensure_list(load_json(derived_path), "derived common JSON")

    common_ids = compute_common_ids(
        forget_records,
        retain_external_records,
        retain_internal_records,
        derived_records,
    )

    forget_rejected_records = filter_forget_rejected_records_by_common_ids(
        load_optional_json_list(forget_rejected_path, "forget rejected JSON"),
        common_ids,
    )
    retain_external_rejected_records = filter_anchor_rejected_records_by_common_ids(
        load_optional_json_list(retain_external_rejected_path, "retain external rejected JSON"),
        common_ids,
    )
    retain_internal_rejected_records = filter_anchor_rejected_records_by_common_ids(
        load_optional_json_list(retain_internal_rejected_path, "retain internal rejected JSON"),
        common_ids,
    )
    derived_rejected_records = filter_anchor_rejected_records_by_common_ids(
        load_optional_json_list(derived_rejected_path, "derived rejected JSON"),
        common_ids,
    )

    year_dir = infer_year_index_dir(config)
    year_lookup, year_file_counts = build_year_lookup(year_dir)

    retain_year_lookup, unresolved_retain_ids = build_semantic_scholar_year_lookup_for_retain_external(
        retain_external_records,
        config,
    )

    report = {
        "paths": {
            "forget": str(forget_path),
            "retain_external": str(retain_external_path),
            "retain_internal": str(retain_internal_path),
            "derived": str(derived_path),
            "forget_rejected": str(forget_rejected_path),
            "retain_external_rejected": str(retain_external_rejected_path),
            "retain_internal_rejected": str(retain_internal_rejected_path),
            "derived_rejected": str(derived_rejected_path),
            "year_index_dir": str(year_dir),
        },
        "year_index_file_counts": year_file_counts,
        "datasets": {
            "forget": summarize_forget(
                forget_records,
                forget_rejected_records,
                year_lookup,
            ),
            "retain_external": summarize_retain_external(
                retain_external_records,
                retain_external_rejected_records,
                retain_year_lookup,
                unresolved_retain_ids,
            ),
            "retain_internal": summarize_retain_internal(
                retain_internal_records,
                retain_internal_rejected_records,
                year_lookup,
            ),
            "derived": summarize_derived(
                derived_records,
                derived_rejected_records,
                year_lookup,
            ),
        },
    }

    report["forget_vs_retain_external_paper_overlap"] = compute_forget_vs_retain_external_paper_overlap(
        forget_records,
        retain_external_records,
    )
    report["cost_statistics"] = summarize_cost_logs(config)

    report_json_path = Path(
        getattr(config, "common_eval_report_json", Path("common_dataset_evaluation_report_2.json"))
    )
    report_md_path = Path(
        getattr(config, "common_eval_report_md", Path("common_dataset_evaluation_report_2.md"))
    )

    save_json(report, report_json_path)
    report_md_path.write_text(build_markdown_report(report), encoding="utf-8")

    print("\n" + "=" * 120)
    print("COMMON DATASET EVALUATION SUMMARY")
    print("=" * 120)
    print(f"Forget records           : {report['datasets']['forget']['record_count']}")
    print(f"Retain external records  : {report['datasets']['retain_external']['record_count']}")
    print(f"Retain internal records  : {report['datasets']['retain_internal']['record_count']}")
    print(f"Derived records          : {report['datasets']['derived']['record_count']}")

    overlap = report["forget_vs_retain_external_paper_overlap"]
    print("-" * 120)
    print(f"Forget vs external retain overlap count : {overlap['overlap_count']}")
    if overlap["has_overlap"]:
        print(f"Overlap examples                    : {overlap['overlap_ids_examples'][:10]}")

    cost_stats = report.get("cost_statistics", {})
    overall_cost = cost_stats.get("overall", {})
    total_cost_summary = overall_cost.get("total_cost_summary", {})
    print("-" * 120)
    print(f"Overall logged cost sum   : {total_cost_summary.get('sum')}")
    print(f"Overall logged cost mean  : {total_cost_summary.get('mean')}")
    print(f"Overall logged cost median: {total_cost_summary.get('median')}")

    print("-" * 120)
    print(f"Report JSON              : {report_json_path}")
    print(f"Report Markdown          : {report_md_path}")

    return report


if __name__ == "__main__":
    config = AppConfig()
    evaluate_common_datasets(config)