# # from pathlib import Path
# # from typing import Dict, Any, List, Optional

# # from config import AppConfig
# # from utils.json_utils import load_json, save_json
# # from utils.pdf_utils import extract_text_from_pdf
# # from llm_client.azure_gpt5_client import call_llm_internal_retain_questions
# # from model.olmo_runner import OLMORunner
# # from evaluation.qa_filter import evaluate_pair


# # def load_forget_records(config: AppConfig) -> List[Dict[str, Any]]:
# #     """
# #     Load final forget-side accepted records from QA_final_covered.json.
# #     """
# #     if not config.claim_output_json.exists():
# #         raise FileNotFoundError(f"Forget-side final JSON not found: {config.claim_output_json}")

# #     data = load_json(config.claim_output_json)
# #     if not isinstance(data, list):
# #         raise ValueError("QA_final_covered.json must contain a JSON array.")

# #     return data


# # def enrich_qa_by_base_with_olmo(
# #     qa_by_base: Dict[str, List[Dict[str, Any]]],
# #     olmo_runner: OLMORunner,
# # ) -> Dict[str, List[Dict[str, Any]]]:
# #     """
# #     Add olmo_answer to every QA item in qa_by_base.
# #     """
# #     for item in qa_by_base.get("mcq", []) or []:
# #         item["olmo_answer"] = olmo_runner.answer_item("mcq", item.get("question", ""))

# #     for item in qa_by_base.get("true_false", []) or []:
# #         item["olmo_answer"] = olmo_runner.answer_item("true_false", item.get("question", ""))

# #     for item in qa_by_base.get("fill_blank", []) or []:
# #         item["olmo_answer"] = olmo_runner.answer_item("fill_blank", item.get("question", ""))

# #     for item in qa_by_base.get("assertion_reason", []) or []:
# #         item["olmo_answer"] = olmo_runner.answer_item("assertion_reason", item.get("question", ""))

# #     return qa_by_base


# # def filter_qa_by_base_with_olmo(
# #     qa_by_base: Dict[str, List[Dict[str, Any]]],
# #     config: AppConfig,
# # ) -> Dict[str, List[Dict[str, Any]]]:
# #     """
# #     Keep only QA types whose matched items form a good pair.
# #     Reuses evaluate_pair() from evaluation.qa_filter.
# #     """
# #     filtered: Dict[str, List[Dict[str, Any]]] = {}
# #     good_pairs_count = 0

# #     for qtype in config.all_qa_types:
# #         items = qa_by_base.get(qtype, []) or []
# #         kept_items, pair_good = evaluate_pair(items, qtype, config)

# #         if pair_good:
# #             filtered[qtype] = kept_items
# #             good_pairs_count += 1

# #     if good_pairs_count < config.min_good_pairs_per_base:
# #         return {}

# #     return filtered


# # def process_one_forget_paper(
# #     rec: Dict[str, Any],
# #     config: AppConfig,
# #     olmo_runner: Optional[OLMORunner] = None,
# # ) -> Dict[str, Any]:
# #     """
# #     Build one internal-retain record from one forget-side paper.
# #     """
# #     pdf_name = rec.get("pdf_name", "")
# #     paper_title = rec.get("paper_title", "") or "(unknown title)"
# #     paper_claims = rec.get("paper_claims", []) or []

# #     pdf_path = config.download_dir / pdf_name
# #     if not pdf_path.exists():
# #         return {
# #             "pdf_name": pdf_name,
# #             "paper_title": paper_title,
# #             "paper_claims": paper_claims,
# #             "error": f"PDF not found at {pdf_path}",
# #         }

# #     text = extract_text_from_pdf(pdf_path)
# #     if len(text) < config.min_text_chars_for_claims:
# #         return {
# #             "pdf_name": pdf_name,
# #             "paper_title": paper_title,
# #             "paper_claims": paper_claims,
# #             "error": "Extracted text too short for internal retain generation",
# #         }

# #     try:
# #         payload = call_llm_internal_retain_questions(
# #             text=text,
# #             paper_title=paper_title,
# #             paper_claims=paper_claims,
# #             config=config,
# #         )
# #     except Exception as e:
# #         return {
# #             "pdf_name": pdf_name,
# #             "paper_title": paper_title,
# #             "paper_claims": paper_claims,
# #             "error": f"Internal retain generation failed: {e}",
# #         }

# #     paper_base = payload.get("paper_base", {}) or {}
# #     qa_by_base = payload.get("qa_by_base", {}) or {}

# #     # OLMo enrichment + filter
# #     if olmo_runner is not None:
# #         try:
# #             qa_by_base = enrich_qa_by_base_with_olmo(qa_by_base, olmo_runner)
# #             print(f"  ✅ OLMo answers generated for internal retain paper {pdf_name}")

# #             qa_by_base = filter_qa_by_base_with_olmo(qa_by_base, config)
# #             if qa_by_base:
# #                 print(f"  ✅ Internal retain QA passed OLMo matching for {pdf_name}")
# #             else:
# #                 print(f"  ❌ Internal retain QA failed OLMo survival for {pdf_name}")
# #         except Exception as e:
# #             return {
# #                 "pdf_name": pdf_name,
# #                 "paper_title": paper_title,
# #                 "paper_claims": paper_claims,
# #                 "error": f"OLMo internal retain check failed: {e}",
# #             }

# #     if not qa_by_base:
# #         return {
# #             "pdf_name": pdf_name,
# #             "paper_title": paper_title,
# #             "paper_claims": paper_claims,
# #             "paper_base": {
# #                 "topic": paper_base.get("topic", ""),
# #                 "paper_type": paper_base.get("paper_type", ""),
# #                 "task_or_problem": paper_base.get("task_or_problem", ""),
# #                 "method_family": paper_base.get("method_family", ""),
# #                 "data_or_domain": paper_base.get("data_or_domain", ""),
# #                 "core_concept": paper_base.get("core_concept", ""),
# #             },
# #             "qa_by_base": {},
# #             "error": "Internal retain paper did not survive OLMo filtering",
# #         }

# #     return {
# #         "pdf_name": pdf_name,
# #         "paper_title": paper_title,
# #         "paper_claims": paper_claims,
# #         "paper_base": {
# #             "topic": paper_base.get("topic", ""),
# #             "paper_type": paper_base.get("paper_type", ""),
# #             "task_or_problem": paper_base.get("task_or_problem", ""),
# #             "method_family": paper_base.get("method_family", ""),
# #             "data_or_domain": paper_base.get("data_or_domain", ""),
# #             "core_concept": paper_base.get("core_concept", ""),
# #         },
# #         "qa_by_base": qa_by_base,
# #     }


# # def run_retain_internal_pipeline(
# #     config: AppConfig,
# #     olmo_runner: Optional[OLMORunner] = None,
# # ) -> None:
# #     """
# #     Generate retain_set_internal.json from forget-side final accepted papers.
# #     Only keep papers that survive OLMo matching.
# #     """
# #     records = load_forget_records(config)

# #     all_records: List[Dict[str, Any]] = []

# #     for idx, rec in enumerate(records, start=1):
# #         pdf_name = rec.get("pdf_name", "")
# #         print("\n" + "=" * 100)
# #         print(f"[INFO] Processing internal retain {idx}/{len(records)} | pdf={pdf_name}")
# #         print("=" * 100)

# #         try:
# #             out = process_one_forget_paper(rec, config, olmo_runner=olmo_runner)
# #             if out.get("qa_by_base"):
# #                 all_records.append(out)
# #             else:
# #                 print(f"[INFO] Skipping internal retain output for {pdf_name} because it did not survive.")
# #         except Exception as e:
# #             print(f"[WARN] Internal retain processing failed for {pdf_name}: {e}")

# #     save_json(all_records, config.retain_internal_output_json)
# #     print(f"\n[INFO] Wrote retain internal records to: {config.retain_internal_output_json}")


# from pathlib import Path
# from typing import Dict, Any, List, Optional, Set
# import shutil

# from config import AppConfig
# from utils.json_utils import load_json, save_json
# from utils.pdf_utils import extract_text_from_pdf
# from llm_client.azure_gpt5_client import call_llm_internal_retain_questions
# from model.olmo_runner import OLMORunner
# from evaluation.qa_filter import (
#     normalize_mcq,
#     normalize_tf,
#     normalize_fill,
#     normalize_ar,
#     rouge_l_f1,
# )


# def load_forget_records(config: AppConfig) -> List[Dict[str, Any]]:
#     """
#     Load final forget-side accepted records from the forget JSON.
#     """
#     if not config.claim_output_json.exists():
#         raise FileNotFoundError(f"Forget-side final JSON not found: {config.claim_output_json}")

#     data = load_json(config.claim_output_json)
#     if not isinstance(data, list):
#         raise ValueError("Forget JSON must contain a JSON array.")

#     return data


# def attach_anchor_metadata(
#     record: Dict[str, Any],
#     pdf_name: str,
#     survived: bool,
# ) -> Dict[str, Any]:
#     """
#     Attach forget-anchor metadata so this file can later be used
#     to prune the forget JSON.
#     """
#     forget_id = Path(pdf_name).stem.strip()
#     record["anchor_forget_paper_id"] = forget_id
#     record["anchor_forget_pdf_name"] = pdf_name
#     record["internal_retain_survived"] = survived
#     return record


# def enrich_qa_by_base_with_olmo(
#     qa_by_base: Dict[str, List[Dict[str, Any]]],
#     olmo_runner: OLMORunner,
# ) -> Dict[str, List[Dict[str, Any]]]:
#     """
#     Add olmo_answer to every QA item in qa_by_base.
#     """
#     for item in qa_by_base.get("mcq", []) or []:
#         item["olmo_answer"] = olmo_runner.answer_item("mcq", item.get("question", ""))

#     for item in qa_by_base.get("true_false", []) or []:
#         item["olmo_answer"] = olmo_runner.answer_item("true_false", item.get("question", ""))

#     for item in qa_by_base.get("fill_blank", []) or []:
#         item["olmo_answer"] = olmo_runner.answer_item("fill_blank", item.get("question", ""))

#     for item in qa_by_base.get("assertion_reason", []) or []:
#         item["olmo_answer"] = olmo_runner.answer_item("assertion_reason", item.get("question", ""))

#     return qa_by_base


# def score_single_item(
#     expected: str,
#     modelout: str,
#     qtype: str,
#     config: AppConfig,
# ) -> Dict[str, Any]:
#     """
#     Score one question individually against OLMo output.
#     """
#     if qtype == "mcq":
#         exp_n = normalize_mcq(expected)
#         out_n = normalize_mcq(modelout)
#     elif qtype == "true_false":
#         exp_n = normalize_tf(expected)
#         out_n = normalize_tf(modelout)
#     elif qtype == "fill_blank":
#         exp_n = normalize_fill(expected)
#         out_n = normalize_fill(modelout)
#     elif qtype == "assertion_reason":
#         exp_n = normalize_ar(expected)
#         out_n = normalize_ar(modelout)
#     else:
#         exp_n = str(expected).strip()
#         out_n = str(modelout).strip()

#     match = exp_n == out_n
#     rouge = 1.0 if match else rouge_l_f1(exp_n, out_n)

#     if not match and rouge >= config.rouge_threshold:
#         match = True

#     return {
#         "expected_normalized": exp_n,
#         "olmo_normalized": out_n,
#         "rouge": round(rouge, 4),
#         "match": bool(match),
#     }


# def filter_qa_by_base_with_olmo(
#     qa_by_base: Dict[str, List[Dict[str, Any]]],
#     config: AppConfig,
# ) -> Dict[str, List[Dict[str, Any]]]:
#     """
#     Keep ALL individually correct questions across all QA types.

#     New survival rule:
#     - evaluate each question independently
#     - keep all correct questions
#     - if total kept questions > 2, the internal retain sample survives
#     """
#     filtered: Dict[str, List[Dict[str, Any]]] = {}
#     total_correct_questions = 0

#     for qtype in config.all_qa_types:
#         items = qa_by_base.get(qtype, []) or []
#         kept_items: List[Dict[str, Any]] = []

#         for item in items:
#             expected = item.get("answer", "")
#             modelout = item.get("olmo_answer", "")

#             score_info = score_single_item(expected, modelout, qtype, config)
#             item.update(score_info)

#             if item["match"]:
#                 kept_items.append(item)
#                 total_correct_questions += 1

#         if kept_items:
#             filtered[qtype] = kept_items

#     if total_correct_questions <= 2:
#         return {}

#     return filtered


# def process_one_forget_paper(
#     rec: Dict[str, Any],
#     config: AppConfig,
#     olmo_runner: Optional[OLMORunner] = None,
# ) -> Dict[str, Any]:
#     """
#     Build one internal-retain record from one forget-side paper.
#     """
#     pdf_name = rec.get("pdf_name", "")
#     paper_title = rec.get("paper_title", "") or "(unknown title)"
#     paper_claims = rec.get("paper_claims", []) or []

#     pdf_path = config.download_dir / pdf_name
#     if not pdf_path.exists():
#         return attach_anchor_metadata(
#             {
#                 "pdf_name": pdf_name,
#                 "paper_title": paper_title,
#                 "paper_claims": paper_claims,
#                 "error": f"PDF not found at {pdf_path}",
#             },
#             pdf_name,
#             False,
#         )

#     text = extract_text_from_pdf(pdf_path)
#     if len(text) < config.min_text_chars_for_claims:
#         return attach_anchor_metadata(
#             {
#                 "pdf_name": pdf_name,
#                 "paper_title": paper_title,
#                 "paper_claims": paper_claims,
#                 "error": "Extracted text too short for internal retain generation",
#             },
#             pdf_name,
#             False,
#         )

#     try:
#         payload = call_llm_internal_retain_questions(
#             text=text,
#             paper_title=paper_title,
#             paper_claims=paper_claims,
#             config=config,
#         )
#     except Exception as e:
#         return attach_anchor_metadata(
#             {
#                 "pdf_name": pdf_name,
#                 "paper_title": paper_title,
#                 "paper_claims": paper_claims,
#                 "error": f"Internal retain generation failed: {e}",
#             },
#             pdf_name,
#             False,
#         )

#     paper_base = payload.get("paper_base", {}) or {}
#     qa_by_base = payload.get("qa_by_base", {}) or {}

#     # OLMo enrichment + filtering
#     if olmo_runner is not None:
#         try:
#             qa_by_base = enrich_qa_by_base_with_olmo(qa_by_base, olmo_runner)
#             print(f"  ✅ OLMo answers generated for internal retain paper {pdf_name}")

#             qa_by_base = filter_qa_by_base_with_olmo(qa_by_base, config)

#             if qa_by_base:
#                 total_kept = sum(len(v) for v in qa_by_base.values())
#                 print(f"  ✅ Internal retain survived for {pdf_name} | kept_questions={total_kept}")
#             else:
#                 print(f"  ❌ Internal retain failed for {pdf_name} | kept_questions<=2")

#         except Exception as e:
#             return attach_anchor_metadata(
#                 {
#                     "pdf_name": pdf_name,
#                     "paper_title": paper_title,
#                     "paper_claims": paper_claims,
#                     "error": f"OLMo internal retain check failed: {e}",
#                 },
#                 pdf_name,
#                 False,
#             )

#     if not qa_by_base:
#         return attach_anchor_metadata(
#             {
#                 "pdf_name": pdf_name,
#                 "paper_title": paper_title,
#                 "paper_claims": paper_claims,
#                 "paper_base": {
#                     "topic": paper_base.get("topic", ""),
#                     "paper_type": paper_base.get("paper_type", ""),
#                     "task_or_problem": paper_base.get("task_or_problem", ""),
#                     "method_family": paper_base.get("method_family", ""),
#                     "data_or_domain": paper_base.get("data_or_domain", ""),
#                     "core_concept": paper_base.get("core_concept", ""),
#                 },
#                 "qa_by_base": {},
#                 "error": "Internal retain paper did not survive OLMo filtering",
#             },
#             pdf_name,
#             False,
#         )

#     return attach_anchor_metadata(
#         {
#             "pdf_name": pdf_name,
#             "paper_title": paper_title,
#             "paper_claims": paper_claims,
#             "paper_base": {
#                 "topic": paper_base.get("topic", ""),
#                 "paper_type": paper_base.get("paper_type", ""),
#                 "task_or_problem": paper_base.get("task_or_problem", ""),
#                 "method_family": paper_base.get("method_family", ""),
#                 "data_or_domain": paper_base.get("data_or_domain", ""),
#                 "core_concept": paper_base.get("core_concept", ""),
#             },
#             "qa_by_base": qa_by_base,
#         },
#         pdf_name,
#         True,
#     )


# def get_successful_anchor_ids_from_internal_records(
#     internal_records: List[Dict[str, Any]],
# ) -> Set[str]:
#     """
#     Collect forget paper IDs that successfully produced an internal retain sample.
#     """
#     successful_ids: Set[str] = set()

#     for rec in internal_records:
#         anchor_id = (rec.get("anchor_forget_paper_id") or "").strip()
#         survived = bool(rec.get("internal_retain_survived", False))

#         if anchor_id and survived:
#             successful_ids.add(anchor_id)

#     return successful_ids


# def prune_forget_records_by_internal_survival(
#     forget_records: List[Dict[str, Any]],
#     successful_anchor_ids: Set[str],
# ) -> List[Dict[str, Any]]:
#     """
#     Keep only forget records whose pdf_name stem is in successful_anchor_ids.
#     """
#     kept: List[Dict[str, Any]] = []

#     for rec in forget_records:
#         pdf_name = rec.get("pdf_name", "")
#         forget_id = Path(pdf_name).stem.strip()

#         if forget_id in successful_anchor_ids:
#             kept.append(rec)

#     return kept


# def prune_forget_json_after_internal_retain(
#     config: AppConfig,
#     forget_records: List[Dict[str, Any]],
#     internal_records: List[Dict[str, Any]],
# ) -> Dict[str, Any]:
#     """
#     Prune the forget JSON so that only forget papers that successfully produced
#     an internal retain sample are kept.

#     Safe defaults are used if these config attributes do not exist:
#     - prune_forget_after_retain -> True
#     - pruned_forget_output_json -> <claim_output_json stem>_pruned.json
#     - overwrite_forget_json_after_prune -> False
#     - backup_forget_json_before_overwrite -> True
#     """
#     prune_enabled = getattr(config, "prune_forget_after_retain", True)
#     if not prune_enabled:
#         return {
#             "original_count": len(forget_records),
#             "pruned_count": len(forget_records),
#             "removed_count": 0,
#             "final_output_path": str(config.claim_output_json),
#             "pruning_skipped": True,
#         }

#     successful_anchor_ids = get_successful_anchor_ids_from_internal_records(internal_records)
#     pruned_forget_records = prune_forget_records_by_internal_survival(
#         forget_records,
#         successful_anchor_ids,
#     )

#     original_count = len(forget_records)
#     pruned_count = len(pruned_forget_records)
#     removed_count = original_count - pruned_count

#     default_pruned_path = config.claim_output_json.with_name(
#         f"{config.claim_output_json.stem}_pruned{config.claim_output_json.suffix}"
#     )
#     final_output_path = Path(getattr(config, "pruned_forget_output_json", default_pruned_path))

#     save_json(pruned_forget_records, final_output_path)

#     # overwrite_original = bool(getattr(config, "overwrite_forget_json_after_prune", False))
#     # backup_before_overwrite = bool(getattr(config, "backup_forget_json_before_overwrite", True))

#     # if overwrite_original:
#     #     if backup_before_overwrite and config.claim_output_json.exists():
#     #         backup_path = config.claim_output_json.with_suffix(
#     #             config.claim_output_json.suffix + ".bak"
#     #         )
#     #         shutil.copy2(config.claim_output_json, backup_path)
#     #         print(f"[PRUNE][INTERNAL] Backup created: {backup_path}")

#     #     save_json(pruned_forget_records, config.claim_output_json)
#     #     final_output_path = config.claim_output_json
#     #     print(f"[PRUNE][INTERNAL] Overwrote original forget JSON: {config.claim_output_json}")

#     print(f"[PRUNE][INTERNAL] Forget records before pruning: {original_count}")
#     print(f"[PRUNE][INTERNAL] Forget records after pruning : {pruned_count}")
#     print(f"[PRUNE][INTERNAL] Forget records removed       : {removed_count}")
#     print(f"[PRUNE][INTERNAL] Final pruned forget JSON    : {final_output_path}")

#     return {
#         "original_count": original_count,
#         "pruned_count": pruned_count,
#         "removed_count": removed_count,
#         "successful_anchor_ids": sorted(successful_anchor_ids),
#         "final_output_path": str(final_output_path),
#         "pruning_skipped": False,
#     }


# def run_retain_internal_pipeline(
#     config: AppConfig,
#     olmo_runner: Optional[OLMORunner] = None,
# ) -> None:
#     """
#     Generate retain_set_internal.json from forget-side final accepted papers.

#     IMPORTANT:
#     - We save BOTH survivors and failures.
#     - Then we prune the forget JSON based on which forget papers successfully
#       produced internal retain samples.
#     """
#     records = load_forget_records(config)
#     all_records: List[Dict[str, Any]] = []

#     for idx, rec in enumerate(records, start=1):
#         pdf_name = rec.get("pdf_name", "")
#         print("\n" + "=" * 100)
#         print(f"[INFO] Processing internal retain {idx}/{len(records)} | pdf={pdf_name}")
#         print("=" * 100)

#         try:
#             out = process_one_forget_paper(rec, config, olmo_runner=olmo_runner)

#             if out.get("internal_retain_survived", False):
#                 print(f"[INFO] Internal retain survived for {pdf_name}")
#             else:
#                 print(f"[INFO] Internal retain did NOT survive for {pdf_name}")

#             all_records.append(out)

#         except Exception as e:
#             fail_record = attach_anchor_metadata(
#                 {
#                     "pdf_name": pdf_name,
#                     "paper_title": rec.get("paper_title", ""),
#                     "paper_claims": rec.get("paper_claims", []) or [],
#                     "error": f"Unexpected failure: {e}",
#                 },
#                 pdf_name,
#                 False,
#             )
#             all_records.append(fail_record)
#             print(f"[WARN] Internal retain processing failed for {pdf_name}: {e}")

#     save_json(all_records, config.retain_internal_output_json)
#     print(f"\n[INFO] Wrote retain internal records to: {config.retain_internal_output_json}")

#     # ----------------------------------------------------------
#     # NEW: prune forget JSON based on internal retain survival
#     # ----------------------------------------------------------
#     try:
#         prune_summary = prune_forget_json_after_internal_retain(
#             config=config,
#             forget_records=records,
#             internal_records=all_records,
#         )
#         print("[SUMMARY] Forget JSON pruning after internal retain completed successfully.")
#         print(f"  Original forget records : {prune_summary['original_count']}")
#         print(f"  Pruned forget records   : {prune_summary['pruned_count']}")
#         print(f"  Removed forget records  : {prune_summary['removed_count']}")
#         print(f"  Final output path       : {prune_summary['final_output_path']}")
#     except Exception as e:
#         print(f"[WARN] Forget JSON pruning after internal retain failed: {e}")



from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Tuple
import shutil

from config import AppConfig
from utils.json_utils import load_json, save_json
from utils.pdf_utils import extract_text_from_pdf
from llm_client.azure_gpt5_client import call_llm_internal_retain_questions
from model.olmo_runner import OLMORunner
from evaluation.qa_filter import (
    normalize_mcq,
    normalize_tf,
    normalize_fill,
    normalize_ar,
    rouge_l_f1,
)

import re
from functools import lru_cache

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer


def load_forget_records(config: AppConfig) -> List[Dict[str, Any]]:
    """
    Load final forget-side accepted records from the forget JSON.
    """
    if not config.claim_output_json.exists():
        raise FileNotFoundError(f"Forget-side final JSON not found: {config.claim_output_json}")

    data = load_json(config.claim_output_json)
    if not isinstance(data, list):
        raise ValueError("Forget JSON must contain a JSON array.")

    return data


def attach_anchor_metadata(
    record: Dict[str, Any],
    pdf_name: str,
    survived: bool,
) -> Dict[str, Any]:
    """
    Attach forget-anchor metadata so this file can later be used
    to prune the forget JSON.
    """
    forget_id = Path(pdf_name).stem.strip()
    record["anchor_forget_paper_id"] = forget_id
    record["anchor_forget_pdf_name"] = pdf_name
    record["internal_retain_survived"] = survived
    return record


def enrich_qa_by_base_with_olmo(
    qa_by_base: Dict[str, List[Dict[str, Any]]],
    olmo_runner: OLMORunner,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Add olmo_answer to every QA item in qa_by_base.
    """
    for item in qa_by_base.get("mcq", []) or []:
        item["olmo_answer"] = olmo_runner.answer_item("mcq", item.get("question", ""))

    for item in qa_by_base.get("true_false", []) or []:
        item["olmo_answer"] = olmo_runner.answer_item("true_false", item.get("question", ""))

    for item in qa_by_base.get("fill_blank", []) or []:
        item["olmo_answer"] = olmo_runner.answer_item("fill_blank", item.get("question", ""))

    for item in qa_by_base.get("assertion_reason", []) or []:
        item["olmo_answer"] = olmo_runner.answer_item("assertion_reason", item.get("question", ""))

    return qa_by_base


def normalize_text_for_semantic(text: str) -> str:
    """
    Light normalization before semantic similarity.
    """
    text = (text or "").strip().lower()

    # normalize separators
    text = text.replace("/", " ")
    text = text.replace("-", " ")

    # collapse whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


@lru_cache(maxsize=2)
def get_semantic_model(model_name: str) -> SentenceTransformer:
    """
    Load and cache the semantic similarity model only once.
    """
    return SentenceTransformer(model_name)


def semantic_similarity_text(
    expected: str,
    predicted: str,
    model_name: str,
) -> float:
    """
    Compute cosine semantic similarity between two short text strings.
    """
    exp_norm = normalize_text_for_semantic(expected)
    pred_norm = normalize_text_for_semantic(predicted)

    if not exp_norm and not pred_norm:
        return 1.0
    if not exp_norm or not pred_norm:
        return 0.0

    model = get_semantic_model(model_name)

    embeddings = model.encode(
        [exp_norm, pred_norm],
        convert_to_tensor=True,
        normalize_embeddings=True,
    )

    sim = F.cosine_similarity(
        embeddings[0].unsqueeze(0),
        embeddings[1].unsqueeze(0)
    ).item()

    return float(sim)


# def score_single_item(
#     expected: str,
#     modelout: str,
#     qtype: str,
#     config: AppConfig,
# ) -> Dict[str, Any]:
#     """
#     Score one question individually against OLMo output.
#     """
#     if qtype == "mcq":
#         exp_n = normalize_mcq(expected)
#         out_n = normalize_mcq(modelout)
#     elif qtype == "true_false":
#         exp_n = normalize_tf(expected)
#         out_n = normalize_tf(modelout)
#     elif qtype == "fill_blank":
#         exp_n = normalize_fill(expected)
#         out_n = normalize_fill(modelout)
#     elif qtype == "assertion_reason":
#         exp_n = normalize_ar(expected)
#         out_n = normalize_ar(modelout)
#     else:
#         exp_n = str(expected).strip()
#         out_n = str(modelout).strip()

#     match = exp_n == out_n
#     rouge = 1.0 if match else rouge_l_f1(exp_n, out_n)

#     if not match and rouge >= config.rouge_threshold:
#         match = True

#     return {
#         "expected_normalized": exp_n,
#         "olmo_normalized": out_n,
#         "rouge": round(rouge, 4),
#         "match": bool(match),
#     }


def score_single_item(
    expected: str,
    modelout: str,
    qtype: str,
    config: AppConfig,
) -> Dict[str, Any]:
    """
    Score one question individually against OLMo output.

    For every question type, compute:
      - exact normalized match
      - ROUGE score
      - semantic similarity score
    """
    # ------------------------------------------------------
    # Type-specific normalization
    # ------------------------------------------------------
    if qtype == "mcq":
        exp_n = normalize_mcq(expected)
        out_n = normalize_mcq(modelout)
    elif qtype == "true_false":
        exp_n = normalize_tf(expected)
        out_n = normalize_tf(modelout)
    elif qtype == "fill_blank":
        exp_n = normalize_fill(expected)
        out_n = normalize_fill(modelout)
    elif qtype == "assertion_reason":
        exp_n = normalize_ar(expected)
        out_n = normalize_ar(modelout)
    else:
        exp_n = str(expected).strip()
        out_n = str(modelout).strip()

    # ------------------------------------------------------
    # Exact match
    # ------------------------------------------------------
    exact_match = (exp_n == out_n)

    # ------------------------------------------------------
    # ROUGE score
    # ------------------------------------------------------
    rouge = 1.0 if exact_match else rouge_l_f1(exp_n, out_n)

    # ------------------------------------------------------
    # Semantic similarity score
    # ------------------------------------------------------
    semantic_similarity = semantic_similarity_text(
        expected=exp_n,
        predicted=out_n,
        model_name=config.semantic_model_name,
    )

    # ------------------------------------------------------
    # Final match rule
    # ------------------------------------------------------
    match = (
        exact_match
        or rouge >= config.rouge_threshold
        or semantic_similarity >= config.semantic_similarity_threshold
    )

    return {
        "expected_normalized": exp_n,
        "olmo_normalized": out_n,
        "exact_match": bool(exact_match),
        "rouge": round(float(rouge), 4),
        "semantic_similarity": round(float(semantic_similarity), 4),
        "match": bool(match),
    }


def filter_qa_by_base_with_olmo(
    qa_by_base: Dict[str, List[Dict[str, Any]]],
    config: AppConfig,
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, List[Dict[str, Any]]]]:
    """
    Keep ALL individually correct questions across all QA types, and separately
    store all rejected / omitted questions.

    Survival rule:
    - evaluate each question independently
    - keep all correct questions
    - if total kept questions > 2, the internal retain sample survives
    """
    filtered: Dict[str, List[Dict[str, Any]]] = {}
    rejected: Dict[str, List[Dict[str, Any]]] = {}
    total_correct_questions = 0

    for qtype in config.all_qa_types:
        items = qa_by_base.get(qtype, []) or []
        kept_items: List[Dict[str, Any]] = []
        rejected_items: List[Dict[str, Any]] = []

        for item in items:
            expected = item.get("answer", "")
            modelout = item.get("olmo_answer", "")

            score_info = score_single_item(expected, modelout, qtype, config)
            item.update(score_info)

            if item["match"]:
                kept_items.append(item)
                total_correct_questions += 1
            else:
                item["rejection_reason"] = "olmo_mismatch"
                rejected_items.append(item)

        if kept_items:
            filtered[qtype] = kept_items

        if rejected_items:
            rejected[qtype] = rejected_items

    if total_correct_questions <= 2:
        return {}, rejected

    return filtered, rejected


def has_rejected_content(rejected_qa_by_base: Dict[str, List[Dict[str, Any]]]) -> bool:
    """
    Check whether the rejected-by-OLMo structure actually contains anything.
    """
    if not rejected_qa_by_base:
        return False

    for qtype in ["mcq", "true_false", "fill_blank", "assertion_reason"]:
        items = rejected_qa_by_base.get(qtype, []) or []
        if items:
            return True

    return False


def process_one_forget_paper(
    rec: Dict[str, Any],
    config: AppConfig,
    olmo_runner: Optional[OLMORunner] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Build one internal-retain record from one forget-side paper.

    Returns:
      (main_internal_record, rejected_record)
    """
    pdf_name = rec.get("pdf_name", "")
    paper_title = rec.get("paper_title", "") or "(unknown title)"
    paper_claims = rec.get("paper_claims", []) or []

    pdf_path = config.download_dir / pdf_name
    if not pdf_path.exists():
        fail_record = attach_anchor_metadata(
            {
                "pdf_name": pdf_name,
                "paper_title": paper_title,
                "paper_claims": paper_claims,
                "error": f"PDF not found at {pdf_path}",
            },
            pdf_name,
            False,
        )
        rejected_record = attach_anchor_metadata(
            {
                "pdf_name": pdf_name,
                "paper_title": paper_title,
                "paper_claims": paper_claims,
                "rejected_qa_by_base": {},
            },
            pdf_name,
            False,
        )
        return fail_record, rejected_record

    text = extract_text_from_pdf(pdf_path)
    if len(text) < config.min_text_chars_for_claims:
        fail_record = attach_anchor_metadata(
            {
                "pdf_name": pdf_name,
                "paper_title": paper_title,
                "paper_claims": paper_claims,
                "error": "Extracted text too short for internal retain generation",
            },
            pdf_name,
            False,
        )
        rejected_record = attach_anchor_metadata(
            {
                "pdf_name": pdf_name,
                "paper_title": paper_title,
                "paper_claims": paper_claims,
                "rejected_qa_by_base": {},
            },
            pdf_name,
            False,
        )
        return fail_record, rejected_record

    try:
        payload = call_llm_internal_retain_questions(
            text=text,
            paper_title=paper_title,
            paper_claims=paper_claims,
            config=config,
        )
    except Exception as e:
        fail_record = attach_anchor_metadata(
            {
                "pdf_name": pdf_name,
                "paper_title": paper_title,
                "paper_claims": paper_claims,
                "error": f"Internal retain generation failed: {e}",
            },
            pdf_name,
            False,
        )
        rejected_record = attach_anchor_metadata(
            {
                "pdf_name": pdf_name,
                "paper_title": paper_title,
                "paper_claims": paper_claims,
                "rejected_qa_by_base": {},
            },
            pdf_name,
            False,
        )
        return fail_record, rejected_record

    paper_base = payload.get("paper_base", {}) or {}
    qa_by_base = payload.get("qa_by_base", {}) or {}

    rejected_qa_by_base: Dict[str, List[Dict[str, Any]]] = {}

    # OLMo enrichment + filtering
    if olmo_runner is not None:
        try:
            qa_by_base = enrich_qa_by_base_with_olmo(qa_by_base, olmo_runner)
            print(f"  ✅ OLMo answers generated for internal retain paper {pdf_name}")

            qa_by_base, rejected_qa_by_base = filter_qa_by_base_with_olmo(qa_by_base, config)

            if qa_by_base:
                total_kept = sum(len(v) for v in qa_by_base.values())
                print(f"  ✅ Internal retain survived for {pdf_name} | kept_questions={total_kept}")
            else:
                print(f"  ❌ Internal retain failed for {pdf_name} | kept_questions<=2")

        except Exception as e:
            fail_record = attach_anchor_metadata(
                {
                    "pdf_name": pdf_name,
                    "paper_title": paper_title,
                    "paper_claims": paper_claims,
                    "error": f"OLMo internal retain check failed: {e}",
                },
                pdf_name,
                False,
            )
            rejected_record = attach_anchor_metadata(
                {
                    "pdf_name": pdf_name,
                    "paper_title": paper_title,
                    "paper_claims": paper_claims,
                    "rejected_qa_by_base": {},
                },
                pdf_name,
                False,
            )
            return fail_record, rejected_record

    if not qa_by_base:
        fail_record = attach_anchor_metadata(
            {
                "pdf_name": pdf_name,
                "paper_title": paper_title,
                "paper_claims": paper_claims,
                "paper_base": {
                    "topic": paper_base.get("topic", ""),
                    "paper_type": paper_base.get("paper_type", ""),
                    "task_or_problem": paper_base.get("task_or_problem", ""),
                    "method_family": paper_base.get("method_family", ""),
                    "data_or_domain": paper_base.get("data_or_domain", ""),
                    "core_concept": paper_base.get("core_concept", ""),
                },
                "qa_by_base": {},
                "error": "Internal retain paper did not survive OLMo filtering",
            },
            pdf_name,
            False,
        )

        rejected_record = attach_anchor_metadata(
            {
                "pdf_name": pdf_name,
                "paper_title": paper_title,
                "paper_claims": paper_claims,
                "paper_base": {
                    "topic": paper_base.get("topic", ""),
                    "paper_type": paper_base.get("paper_type", ""),
                    "task_or_problem": paper_base.get("task_or_problem", ""),
                    "method_family": paper_base.get("method_family", ""),
                    "data_or_domain": paper_base.get("data_or_domain", ""),
                    "core_concept": paper_base.get("core_concept", ""),
                },
                "rejected_qa_by_base": rejected_qa_by_base,
            },
            pdf_name,
            False,
        )

        return fail_record, rejected_record

    success_record = attach_anchor_metadata(
        {
            "pdf_name": pdf_name,
            "paper_title": paper_title,
            "paper_claims": paper_claims,
            "paper_base": {
                "topic": paper_base.get("topic", ""),
                "paper_type": paper_base.get("paper_type", ""),
                "task_or_problem": paper_base.get("task_or_problem", ""),
                "method_family": paper_base.get("method_family", ""),
                "data_or_domain": paper_base.get("data_or_domain", ""),
                "core_concept": paper_base.get("core_concept", ""),
            },
            "qa_by_base": qa_by_base,
        },
        pdf_name,
        True,
    )

    rejected_record = attach_anchor_metadata(
        {
            "pdf_name": pdf_name,
            "paper_title": paper_title,
            "paper_claims": paper_claims,
            "paper_base": {
                "topic": paper_base.get("topic", ""),
                "paper_type": paper_base.get("paper_type", ""),
                "task_or_problem": paper_base.get("task_or_problem", ""),
                "method_family": paper_base.get("method_family", ""),
                "data_or_domain": paper_base.get("data_or_domain", ""),
                "core_concept": paper_base.get("core_concept", ""),
            },
            "rejected_qa_by_base": rejected_qa_by_base,
        },
        pdf_name,
        False,
    )

    return success_record, rejected_record


def get_successful_anchor_ids_from_internal_records(
    internal_records: List[Dict[str, Any]],
) -> Set[str]:
    """
    Collect forget paper IDs that successfully produced an internal retain sample.
    """
    successful_ids: Set[str] = set()

    for rec in internal_records:
        anchor_id = (rec.get("anchor_forget_paper_id") or "").strip()
        survived = bool(rec.get("internal_retain_survived", False))

        if anchor_id and survived:
            successful_ids.add(anchor_id)

    return successful_ids


def prune_forget_records_by_internal_survival(
    forget_records: List[Dict[str, Any]],
    successful_anchor_ids: Set[str],
) -> List[Dict[str, Any]]:
    """
    Keep only forget records whose pdf_name stem is in successful_anchor_ids.
    """
    kept: List[Dict[str, Any]] = []

    for rec in forget_records:
        pdf_name = rec.get("pdf_name", "")
        forget_id = Path(pdf_name).stem.strip()

        if forget_id in successful_anchor_ids:
            kept.append(rec)

    return kept


def prune_forget_json_after_internal_retain(
    config: AppConfig,
    forget_records: List[Dict[str, Any]],
    internal_records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Prune the forget JSON so that only forget papers that successfully produced
    an internal retain sample are kept.
    """
    prune_enabled = getattr(config, "prune_forget_after_retain", True)
    if not prune_enabled:
        return {
            "original_count": len(forget_records),
            "pruned_count": len(forget_records),
            "removed_count": 0,
            "final_output_path": str(config.claim_output_json),
            "pruning_skipped": True,
        }

    successful_anchor_ids = get_successful_anchor_ids_from_internal_records(internal_records)
    pruned_forget_records = prune_forget_records_by_internal_survival(
        forget_records,
        successful_anchor_ids,
    )

    original_count = len(forget_records)
    pruned_count = len(pruned_forget_records)
    removed_count = original_count - pruned_count

    default_pruned_path = config.claim_output_json.with_name(
        f"{config.claim_output_json.stem}_pruned{config.claim_output_json.suffix}"
    )
    final_output_path = Path(getattr(config, "pruned_forget_output_json", default_pruned_path))

    save_json(pruned_forget_records, final_output_path)

    print(f"[PRUNE][INTERNAL] Forget records before pruning: {original_count}")
    print(f"[PRUNE][INTERNAL] Forget records after pruning : {pruned_count}")
    print(f"[PRUNE][INTERNAL] Forget records removed       : {removed_count}")
    print(f"[PRUNE][INTERNAL] Final pruned forget JSON    : {final_output_path}")

    return {
        "original_count": original_count,
        "pruned_count": pruned_count,
        "removed_count": removed_count,
        "successful_anchor_ids": sorted(successful_anchor_ids),
        "final_output_path": str(final_output_path),
        "pruning_skipped": False,
    }


def run_retain_internal_pipeline(
    config: AppConfig,
    olmo_runner: Optional[OLMORunner] = None,
) -> None:
    """
    Generate retain_set_internal.json from forget-side final accepted papers.

    IMPORTANT:
    - We save BOTH survivors and failures.
    - We also save OLMo-rejected questions in a separate JSON.
    - Then we prune the forget JSON based on which forget papers successfully
      produced internal retain samples.
    """
    records = load_forget_records(config)
    all_records: List[Dict[str, Any]] = []
    all_rejected_records: List[Dict[str, Any]] = []

    for idx, rec in enumerate(records, start=1):
        pdf_name = rec.get("pdf_name", "")
        print("\n" + "=" * 100)
        print(f"[INFO] Processing internal retain {idx}/{len(records)} | pdf={pdf_name}")
        print("=" * 100)

        try:
            out, rejected_record = process_one_forget_paper(rec, config, olmo_runner=olmo_runner)

            if out.get("internal_retain_survived", False):
                print(f"[INFO] Internal retain survived for {pdf_name}")
            else:
                print(f"[INFO] Internal retain did NOT survive for {pdf_name}")

            all_records.append(out)

            if has_rejected_content(rejected_record.get("rejected_qa_by_base", {})):
                all_rejected_records.append(rejected_record)

        except Exception as e:
            fail_record = attach_anchor_metadata(
                {
                    "pdf_name": pdf_name,
                    "paper_title": rec.get("paper_title", ""),
                    "paper_claims": rec.get("paper_claims", []) or [],
                    "error": f"Unexpected failure: {e}",
                },
                pdf_name,
                False,
            )
            all_records.append(fail_record)
            print(f"[WARN] Internal retain processing failed for {pdf_name}: {e}")

    save_json(all_records, config.retain_internal_output_json)
    print(f"\n[INFO] Wrote retain internal records to: {config.retain_internal_output_json}")

    rejected_out_path = getattr(
        config,
        "retain_internal_olmo_rejected_json",
        Path("retain_internal_olmo_rejected.json"),
    )
    save_json(all_rejected_records, rejected_out_path)
    print(f"[INFO] Wrote retain internal OLMo-rejected questions to: {rejected_out_path}")

    # ----------------------------------------------------------
    # prune forget JSON based on internal retain survival
    # ----------------------------------------------------------
    try:
        prune_summary = prune_forget_json_after_internal_retain(
            config=config,
            forget_records=records,
            internal_records=all_records,
        )
        print("[SUMMARY] Forget JSON pruning after internal retain completed successfully.")
        print(f"  Original forget records : {prune_summary['original_count']}")
        print(f"  Pruned forget records   : {prune_summary['pruned_count']}")
        print(f"  Removed forget records  : {prune_summary['removed_count']}")
        print(f"  Final output path       : {prune_summary['final_output_path']}")
    except Exception as e:
        print(f"[WARN] Forget JSON pruning after internal retain failed: {e}")