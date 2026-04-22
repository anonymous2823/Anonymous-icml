# # from typing import Any, Dict, List, Optional

# # from config import AppConfig
# # from utils.json_utils import load_json, save_json
# # from llm_client.azure_gpt5_client import call_llm_derived_questions
# # from model.olmo_runner import OLMORunner
# # from evaluation.qa_filter import (
# #     normalize_mcq,
# #     normalize_tf,
# #     normalize_fill,
# #     normalize_ar,
# #     rouge_l_f1,
# # )


# # def load_forget_records(config: AppConfig) -> List[Dict[str, Any]]:
# #     if not config.claim_output_json.exists():
# #         raise FileNotFoundError(f"Forget-side final JSON not found: {config.claim_output_json}")

# #     data = load_json(config.claim_output_json)
# #     if not isinstance(data, list):
# #         raise ValueError("QA_final_covered.json must contain a JSON array.")
# #     return data


# # def collect_source_questions(rec: Dict[str, Any], config: AppConfig) -> List[Dict[str, Any]]:
# #     """
# #     Collect up to 8 atomic source questions from the forget-side QA pool.
# #     Keeps ordering stable by claim, then QA type, then item order.
# #     """
# #     qa_by_claim = rec.get("qa_by_claim", []) or []
# #     source_items: List[Dict[str, Any]] = []

# #     for claim_idx, claim_obj in enumerate(qa_by_claim):
# #         claim_text = claim_obj.get("claim", "")
# #         for qtype in config.all_qa_types:
# #             items = claim_obj.get(qtype, []) or []
# #             for item_idx, item in enumerate(items, start=1):
# #                 q = (item.get("question") or "").strip()
# #                 a = (item.get("answer") or "").strip()
# #                 if not q or not a:
# #                     continue

# #                 source_items.append({
# #                     "claim_index": claim_idx,
# #                     "claim": claim_text,
# #                     "source_type": qtype,
# #                     "question_index": item_idx,
# #                     "question": q,
# #                     "answer": a,
# #                 })

# #                 if len(source_items) >= config.derived_max_source_questions:
# #                     return source_items

# #     return source_items


# # def _score_single_item(expected: str, modelout: str, qtype: str, config: AppConfig) -> Dict[str, Any]:
# #     if qtype == "mcq":
# #         exp_n = normalize_mcq(expected)
# #         out_n = normalize_mcq(modelout)
# #     elif qtype == "true_false":
# #         exp_n = normalize_tf(expected)
# #         out_n = normalize_tf(modelout)
# #     elif qtype == "fill_blank":
# #         exp_n = normalize_fill(expected)
# #         out_n = normalize_fill(modelout)
# #     elif qtype == "assertion_reason":
# #         exp_n = normalize_ar(expected)
# #         out_n = normalize_ar(modelout)
# #     else:
# #         exp_n = str(expected).strip()
# #         out_n = str(modelout).strip()

# #     match = exp_n == out_n
# #     score = 1.0 if match else rouge_l_f1(exp_n, out_n)

# #     if not match and score >= config.rouge_threshold:
# #         match = True

# #     return {
# #         "expected_normalized": exp_n,
# #         "olmo_normalized": out_n,
# #         "rouge": round(score, 4),
# #         "match": bool(match),
# #     }


# # def enrich_and_filter_with_olmo(
# #     derived_qa: Dict[str, List[Dict[str, Any]]],
# #     olmo_runner: OLMORunner,
# #     config: AppConfig,
# # ) -> Dict[str, List[Dict[str, Any]]]:
# #     """
# #     Generate OLMo answers for the 4 derived questions and keep only matched items.
# #     Since there is exactly one intended question per type, survival typically means all 4 match.
# #     """
# #     kept: Dict[str, List[Dict[str, Any]]] = {}
# #     matched_count = 0

# #     for qtype in config.all_qa_types:
# #         items = derived_qa.get(qtype, []) or []
# #         if not items:
# #             continue

# #         item = items[0]
# #         question = item.get("question", "")
# #         answer = item.get("answer", "")

# #         olmo_answer = olmo_runner.answer_item(qtype, question)
# #         item["olmo_answer"] = olmo_answer

# #         score_info = _score_single_item(answer, olmo_answer, qtype, config)
# #         item.update(score_info)

# #         if item["match"]:
# #             kept[qtype] = [item]
# #             matched_count += 1


# #     if config.derived_require_all_four_match:
# #         return kept if matched_count == 4 else {}

# #     return kept if matched_count >= config.derived_min_matched_questions else {}


# # def process_one_record(
# #     rec: Dict[str, Any],
# #     config: AppConfig,
# #     olmo_runner: Optional[OLMORunner] = None,
# # ) -> Dict[str, Any]:
# #     pdf_name = rec.get("pdf_name", "")
# #     paper_title = rec.get("paper_title", "")
# #     paper_claims = rec.get("paper_claims", []) or []

# #     source_questions = collect_source_questions(rec, config)
# #     if not source_questions:
# #         return {
# #             "pdf_name": pdf_name,
# #             "paper_title": paper_title,
# #             "paper_claims": paper_claims,
# #             "source_questions": [],
# #             "derived_qa": {},
# #             "error": "No source questions available for derivation",
# #         }

# #     try:
# #         payload = call_llm_derived_questions(
# #             source_questions=source_questions,
# #             source_claims=paper_claims,
# #             config=config,
# #         )
# #     except Exception as e:
# #         return {
# #             "pdf_name": pdf_name,
# #             "paper_title": paper_title,
# #             "paper_claims": paper_claims,
# #             "source_questions": source_questions,
# #             "derived_qa": {},
# #             "error": f"Derived question generation failed: {e}",
# #         }

# #     derived_qa = payload.get("derived_qa", {}) or {}

# #     if olmo_runner is not None:
# #         try:
# #             derived_qa = enrich_and_filter_with_olmo(derived_qa, olmo_runner, config)
# #         except Exception as e:
# #             return {
# #                 "pdf_name": pdf_name,
# #                 "paper_title": paper_title,
# #                 "paper_claims": paper_claims,
# #                 "source_questions": source_questions,
# #                 "derived_qa": {},
# #                 "error": f"Derived OLMo validation failed: {e}",
# #             }

# #     return {
# #         "pdf_name": pdf_name,
# #         "paper_title": paper_title,
# #         "paper_claims": paper_claims,
# #         "source_questions": source_questions,
# #         "derived_qa": derived_qa,
# #     }


# # def run_derived_set_pipeline(
# #     config: AppConfig,
# #     olmo_runner: Optional[OLMORunner] = None,
# # ) -> None:
# #     records = load_forget_records(config)
# #     out_records: List[Dict[str, Any]] = []

# #     for idx, rec in enumerate(records, start=1):
# #         pdf_name = rec.get("pdf_name", "")
# #         print("\n" + "=" * 100)
# #         print(f"[INFO] Processing derived-set {idx}/{len(records)} | pdf={pdf_name}")
# #         print("=" * 100)

# #         try:
# #             out = process_one_record(rec, config, olmo_runner=olmo_runner)
# #             if out.get("derived_qa"):
# #                 out_records.append(out)
# #             else:
# #                 print(f"[INFO] Skipping {pdf_name} because derived QA did not survive.")
# #         except Exception as e:
# #             print(f"[WARN] Derived-set generation failed for {pdf_name}: {e}")

# #     save_json(out_records, config.derived_output_json)
# #     print(f"\n[INFO] Wrote derived-set records to: {config.derived_output_json}")


# from pathlib import Path
# from typing import Any, Dict, List, Optional

# from config import AppConfig
# from utils.json_utils import load_json, save_json
# from llm_client.azure_gpt5_client import call_llm_derived_questions
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
#     Load forget-side accepted records from the current forget JSON.
#     """
#     if not config.claim_output_json.exists():
#         raise FileNotFoundError(f"Forget-side final JSON not found: {config.claim_output_json}")

#     data = load_json(config.claim_output_json)
#     if not isinstance(data, list):
#         raise ValueError("Forget JSON must contain a JSON array.")

#     return data


# def attach_derived_anchor_metadata(
#     record: Dict[str, Any],
#     pdf_name: str,
#     survived: bool,
# ) -> Dict[str, Any]:
#     """
#     Attach forget-anchor metadata so later pruning can use derived-set survival.
#     """
#     forget_id = Path(pdf_name).stem.strip()
#     record["anchor_forget_paper_id"] = forget_id
#     record["anchor_forget_pdf_name"] = pdf_name
#     record["derived_survived"] = survived
#     return record


# def collect_source_questions(rec: Dict[str, Any], config: AppConfig) -> List[Dict[str, Any]]:
#     """
#     Collect up to config.derived_max_source_questions atomic source questions
#     from the forget-side QA pool.
#     Keeps ordering stable by claim, then QA type, then item order.
#     """
#     qa_by_claim = rec.get("qa_by_claim", []) or []
#     source_items: List[Dict[str, Any]] = []

#     for claim_idx, claim_obj in enumerate(qa_by_claim):
#         claim_text = claim_obj.get("claim", "")

#         for qtype in config.all_qa_types:
#             items = claim_obj.get(qtype, []) or []

#             for item_idx, item in enumerate(items, start=1):
#                 q = (item.get("question") or "").strip()
#                 a = (item.get("answer") or "").strip()

#                 if not q or not a:
#                     continue

#                 source_items.append({
#                     "claim_index": claim_idx,
#                     "claim": claim_text,
#                     "source_type": qtype,
#                     "question_index": item_idx,
#                     "question": q,
#                     "answer": a,
#                 })

#                 if len(source_items) >= config.derived_max_source_questions:
#                     return source_items

#     return source_items


# def _score_single_item(
#     expected: str,
#     modelout: str,
#     qtype: str,
#     config: AppConfig,
# ) -> Dict[str, Any]:
#     """
#     Score one derived question against OLMo output.
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
#     score = 1.0 if match else rouge_l_f1(exp_n, out_n)

#     if not match and score >= config.rouge_threshold:
#         match = True

#     return {
#         "expected_normalized": exp_n,
#         "olmo_normalized": out_n,
#         "rouge": round(score, 4),
#         "match": bool(match),
#     }


# def enrich_and_filter_with_olmo(
#     derived_qa: Dict[str, List[Dict[str, Any]]],
#     olmo_runner: OLMORunner,
#     config: AppConfig,
# ) -> Dict[str, List[Dict[str, Any]]]:
#     """
#     Generate OLMo answers for the derived questions and keep only individually matched items.

#     Since the derived set is typically:
#     - 1 mcq
#     - 1 true_false
#     - 1 fill_blank
#     - 1 assertion_reason

#     survival is determined by how many of these survive.

#     If config.derived_require_all_four_match == True:
#         all 4 must survive

#     else:
#         at least config.derived_min_matched_questions must survive
#     """
#     kept: Dict[str, List[Dict[str, Any]]] = {}
#     matched_count = 0

#     for qtype in config.all_qa_types:
#         items = derived_qa.get(qtype, []) or []
#         if not items:
#             continue

#         # We expect at most one item per type, but this logic is safe either way
#         kept_items_for_type: List[Dict[str, Any]] = []

#         for item in items:
#             question = item.get("question", "")
#             answer = item.get("answer", "")

#             olmo_answer = olmo_runner.answer_item(qtype, question)
#             item["olmo_answer"] = olmo_answer

#             score_info = _score_single_item(answer, olmo_answer, qtype, config)
#             item.update(score_info)

#             if item["match"]:
#                 kept_items_for_type.append(item)
#                 matched_count += 1

#         if kept_items_for_type:
#             kept[qtype] = kept_items_for_type

#     if config.derived_require_all_four_match:
#         return kept if matched_count == 4 else {}

#     return kept if matched_count >= config.derived_min_matched_questions else {}


# def process_one_record(
#     rec: Dict[str, Any],
#     config: AppConfig,
#     olmo_runner: Optional[OLMORunner] = None,
# ) -> Dict[str, Any]:
#     """
#     Build one derived-set record from one forget-side paper.
#     """
#     pdf_name = rec.get("pdf_name", "")
#     paper_title = rec.get("paper_title", "")
#     paper_claims = rec.get("paper_claims", []) or []

#     source_questions = collect_source_questions(rec, config)
#     if not source_questions:
#         return attach_derived_anchor_metadata(
#             {
#                 "pdf_name": pdf_name,
#                 "paper_title": paper_title,
#                 "paper_claims": paper_claims,
#                 "source_questions": [],
#                 "derived_qa": {},
#                 "error": "No source questions available for derivation",
#             },
#             pdf_name,
#             False,
#         )

#     try:
#         payload = call_llm_derived_questions(
#             source_questions=source_questions,
#             source_claims=paper_claims,
#             config=config,
#         )
#     except Exception as e:
#         return attach_derived_anchor_metadata(
#             {
#                 "pdf_name": pdf_name,
#                 "paper_title": paper_title,
#                 "paper_claims": paper_claims,
#                 "source_questions": source_questions,
#                 "derived_qa": {},
#                 "error": f"Derived question generation failed: {e}",
#             },
#             pdf_name,
#             False,
#         )

#     derived_qa = payload.get("derived_qa", {}) or {}

#     if olmo_runner is not None:
#         try:
#             derived_qa = enrich_and_filter_with_olmo(derived_qa, olmo_runner, config)
#         except Exception as e:
#             return attach_derived_anchor_metadata(
#                 {
#                     "pdf_name": pdf_name,
#                     "paper_title": paper_title,
#                     "paper_claims": paper_claims,
#                     "source_questions": source_questions,
#                     "derived_qa": {},
#                     "error": f"Derived OLMo validation failed: {e}",
#                 },
#                 pdf_name,
#                 False,
#             )

#     if not derived_qa:
#         return attach_derived_anchor_metadata(
#             {
#                 "pdf_name": pdf_name,
#                 "paper_title": paper_title,
#                 "paper_claims": paper_claims,
#                 "source_questions": source_questions,
#                 "derived_qa": {},
#                 "error": "Derived questions did not survive OLMo validation",
#             },
#             pdf_name,
#             False,
#         )

#     return attach_derived_anchor_metadata(
#         {
#             "pdf_name": pdf_name,
#             "paper_title": paper_title,
#             "paper_claims": paper_claims,
#             "source_questions": source_questions,
#             "derived_qa": derived_qa,
#         },
#         pdf_name,
#         True,
#     )


# def run_derived_set_pipeline(
#     config: AppConfig,
#     olmo_runner: Optional[OLMORunner] = None,
# ) -> None:
#     """
#     Generate derived_set.json from forget-side accepted papers.

#     IMPORTANT:
#     We save BOTH survivors and failures, because later we want to prune:
#     - forget JSON
#     - retain external JSON
#     - retain internal JSON
#     using derived-set survival.
#     """
#     records = load_forget_records(config)
#     out_records: List[Dict[str, Any]] = []

#     for idx, rec in enumerate(records, start=1):
#         pdf_name = rec.get("pdf_name", "")
#         print("\n" + "=" * 100)
#         print(f"[INFO] Processing derived-set {idx}/{len(records)} | pdf={pdf_name}")
#         print("=" * 100)

#         try:
#             out = process_one_record(rec, config, olmo_runner=olmo_runner)

#             if out.get("derived_survived", False):
#                 print(f"[INFO] Derived set survived for {pdf_name}")
#             else:
#                 print(f"[INFO] Derived set did NOT survive for {pdf_name}")

#             out_records.append(out)

#         except Exception as e:
#             fail_record = attach_derived_anchor_metadata(
#                 {
#                     "pdf_name": pdf_name,
#                     "paper_title": rec.get("paper_title", ""),
#                     "paper_claims": rec.get("paper_claims", []) or [],
#                     "source_questions": [],
#                     "derived_qa": {},
#                     "error": f"Unexpected failure: {e}",
#                 },
#                 pdf_name,
#                 False,
#             )
#             out_records.append(fail_record)
#             print(f"[WARN] Derived-set generation failed for {pdf_name}: {e}")

#     save_json(out_records, config.derived_output_json)
#     print(f"\n[INFO] Wrote derived-set records to: {config.derived_output_json}")


from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config import AppConfig
from utils.json_utils import load_json, save_json
from llm_client.azure_gpt5_client import call_llm_derived_questions
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
    Load forget-side accepted records from the current forget JSON.
    """
    if not config.claim_output_json.exists():
        raise FileNotFoundError(f"Forget-side final JSON not found: {config.claim_output_json}")

    data = load_json(config.claim_output_json)
    if not isinstance(data, list):
        raise ValueError("Forget JSON must contain a JSON array.")

    return data


def attach_derived_anchor_metadata(
    record: Dict[str, Any],
    pdf_name: str,
    survived: bool,
) -> Dict[str, Any]:
    """
    Attach forget-anchor metadata so later pruning can use derived-set survival.
    """
    forget_id = Path(pdf_name).stem.strip()
    record["anchor_forget_paper_id"] = forget_id
    record["anchor_forget_pdf_name"] = pdf_name
    record["derived_survived"] = survived
    return record


def collect_source_questions(rec: Dict[str, Any], config: AppConfig) -> List[Dict[str, Any]]:
    """
    Collect up to config.derived_max_source_questions atomic source questions
    from the forget-side QA pool.
    Keeps ordering stable by claim, then QA type, then item order.
    """
    qa_by_claim = rec.get("qa_by_claim", []) or []
    source_items: List[Dict[str, Any]] = []

    for claim_idx, claim_obj in enumerate(qa_by_claim):
        claim_text = claim_obj.get("claim", "")

        for qtype in config.all_qa_types:
            items = claim_obj.get(qtype, []) or []

            for item_idx, item in enumerate(items, start=1):
                q = (item.get("question") or "").strip()
                a = (item.get("answer") or "").strip()

                if not q or not a:
                    continue

                source_items.append({
                    "claim_index": claim_idx,
                    "claim": claim_text,
                    "source_type": qtype,
                    "question_index": item_idx,
                    "question": q,
                    "answer": a,
                })

                if len(source_items) >= config.derived_max_source_questions:
                    return source_items

    return source_items


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


# def _score_single_item(
#     expected: str,
#     modelout: str,
#     qtype: str,
#     config: AppConfig,
# ) -> Dict[str, Any]:
#     """
#     Score one derived question against OLMo output.
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
#     score = 1.0 if match else rouge_l_f1(exp_n, out_n)

#     if not match and score >= config.rouge_threshold:
#         match = True

#     return {
#         "expected_normalized": exp_n,
#         "olmo_normalized": out_n,
#         "rouge": round(score, 4),
#         "match": bool(match),
#     }



def _score_single_item(
    expected: str,
    modelout: str,
    qtype: str,
    config: AppConfig,
) -> Dict[str, Any]:
    """
    Score one derived question against OLMo output.

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


def enrich_and_filter_with_olmo(
    derived_qa: Dict[str, List[Dict[str, Any]]],
    olmo_runner: OLMORunner,
    config: AppConfig,
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, List[Dict[str, Any]]]]:
    """
    Generate OLMo answers for the derived questions and split them into:
    - kept (survived)
    - rejected (omitted due to OLMo mismatch)

    Since the derived set is typically:
      - 1 mcq
      - 1 true_false
      - 1 fill_blank
      - 1 assertion_reason

    survival is determined by how many of these survive.

    If config.derived_require_all_four_match == True:
        all 4 must survive

    else:
        at least config.derived_min_matched_questions must survive
    """
    kept: Dict[str, List[Dict[str, Any]]] = {}
    rejected: Dict[str, List[Dict[str, Any]]] = {}
    matched_count = 0

    for qtype in config.all_qa_types:
        items = derived_qa.get(qtype, []) or []
        if not items:
            continue

        kept_items_for_type: List[Dict[str, Any]] = []
        rejected_items_for_type: List[Dict[str, Any]] = []

        for item in items:
            question = item.get("question", "")
            answer = item.get("answer", "")

            olmo_answer = olmo_runner.answer_item(qtype, question)
            item["olmo_answer"] = olmo_answer

            score_info = _score_single_item(answer, olmo_answer, qtype, config)
            item.update(score_info)

            if item["match"]:
                kept_items_for_type.append(item)
                matched_count += 1
            else:
                item["rejection_reason"] = "olmo_mismatch"
                rejected_items_for_type.append(item)

        if kept_items_for_type:
            kept[qtype] = kept_items_for_type

        if rejected_items_for_type:
            rejected[qtype] = rejected_items_for_type

    if config.derived_require_all_four_match:
        return (kept, rejected) if matched_count == 4 else ({}, rejected)

    return (kept, rejected) if matched_count >= config.derived_min_matched_questions else ({}, rejected)


def has_rejected_content(rejected_derived_qa: Dict[str, List[Dict[str, Any]]]) -> bool:
    """
    Check whether the rejected-by-OLMo derived structure actually contains any omitted questions.
    """
    if not rejected_derived_qa:
        return False

    for qtype in ["mcq", "true_false", "fill_blank", "assertion_reason"]:
        items = rejected_derived_qa.get(qtype, []) or []
        if items:
            return True

    return False


def process_one_record(
    rec: Dict[str, Any],
    config: AppConfig,
    olmo_runner: Optional[OLMORunner] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Build one derived-set record from one forget-side paper.

    Returns:
      (main_record, rejected_record)
    """
    pdf_name = rec.get("pdf_name", "")
    paper_title = rec.get("paper_title", "")
    paper_claims = rec.get("paper_claims", []) or []

    source_questions = collect_source_questions(rec, config)
    if not source_questions:
        fail_record = attach_derived_anchor_metadata(
            {
                "pdf_name": pdf_name,
                "paper_title": paper_title,
                "paper_claims": paper_claims,
                "source_questions": [],
                "derived_qa": {},
                "error": "No source questions available for derivation",
            },
            pdf_name,
            False,
        )

        rejected_record = attach_derived_anchor_metadata(
            {
                "pdf_name": pdf_name,
                "paper_title": paper_title,
                "paper_claims": paper_claims,
                "source_questions": [],
                "rejected_derived_qa": {},
            },
            pdf_name,
            False,
        )

        return fail_record, rejected_record

    try:
        payload = call_llm_derived_questions(
            source_questions=source_questions,
            source_claims=paper_claims,
            config=config,
        )
    except Exception as e:
        fail_record = attach_derived_anchor_metadata(
            {
                "pdf_name": pdf_name,
                "paper_title": paper_title,
                "paper_claims": paper_claims,
                "source_questions": source_questions,
                "derived_qa": {},
                "error": f"Derived question generation failed: {e}",
            },
            pdf_name,
            False,
        )

        rejected_record = attach_derived_anchor_metadata(
            {
                "pdf_name": pdf_name,
                "paper_title": paper_title,
                "paper_claims": paper_claims,
                "source_questions": source_questions,
                "rejected_derived_qa": {},
            },
            pdf_name,
            False,
        )

        return fail_record, rejected_record

    derived_qa = payload.get("derived_qa", {}) or {}
    rejected_derived_qa: Dict[str, List[Dict[str, Any]]] = {}

    if olmo_runner is not None:
        try:
            derived_qa, rejected_derived_qa = enrich_and_filter_with_olmo(
                derived_qa,
                olmo_runner,
                config,
            )
        except Exception as e:
            fail_record = attach_derived_anchor_metadata(
                {
                    "pdf_name": pdf_name,
                    "paper_title": paper_title,
                    "paper_claims": paper_claims,
                    "source_questions": source_questions,
                    "derived_qa": {},
                    "error": f"Derived OLMo validation failed: {e}",
                },
                pdf_name,
                False,
            )

            rejected_record = attach_derived_anchor_metadata(
                {
                    "pdf_name": pdf_name,
                    "paper_title": paper_title,
                    "paper_claims": paper_claims,
                    "source_questions": source_questions,
                    "rejected_derived_qa": {},
                },
                pdf_name,
                False,
            )

            return fail_record, rejected_record

    if not derived_qa:
        fail_record = attach_derived_anchor_metadata(
            {
                "pdf_name": pdf_name,
                "paper_title": paper_title,
                "paper_claims": paper_claims,
                "source_questions": source_questions,
                "derived_qa": {},
                "error": "Derived questions did not survive OLMo validation",
            },
            pdf_name,
            False,
        )

        rejected_record = attach_derived_anchor_metadata(
            {
                "pdf_name": pdf_name,
                "paper_title": paper_title,
                "paper_claims": paper_claims,
                "source_questions": source_questions,
                "rejected_derived_qa": rejected_derived_qa,
            },
            pdf_name,
            False,
        )

        return fail_record, rejected_record

    success_record = attach_derived_anchor_metadata(
        {
            "pdf_name": pdf_name,
            "paper_title": paper_title,
            "paper_claims": paper_claims,
            "source_questions": source_questions,
            "derived_qa": derived_qa,
        },
        pdf_name,
        True,
    )

    rejected_record = attach_derived_anchor_metadata(
        {
            "pdf_name": pdf_name,
            "paper_title": paper_title,
            "paper_claims": paper_claims,
            "source_questions": source_questions,
            "rejected_derived_qa": rejected_derived_qa,
        },
        pdf_name,
        False,
    )

    return success_record, rejected_record


def run_derived_set_pipeline(
    config: AppConfig,
    olmo_runner: Optional[OLMORunner] = None,
) -> None:
    """
    Generate derived_set.json from forget-side accepted papers.

    IMPORTANT:
    - We save BOTH survivors and failures.
    - We also save OLMo-rejected derived questions in a separate JSON.
    - Later pruning can use derived-set survival.
    """
    records = load_forget_records(config)
    out_records: List[Dict[str, Any]] = []
    all_rejected_records: List[Dict[str, Any]] = []

    for idx, rec in enumerate(records, start=1):
        pdf_name = rec.get("pdf_name", "")
        print("\n" + "=" * 100)
        print(f"[INFO] Processing derived-set {idx}/{len(records)} | pdf={pdf_name}")
        print("=" * 100)

        try:
            out, rejected_record = process_one_record(rec, config, olmo_runner=olmo_runner)

            if out.get("derived_survived", False):
                print(f"[INFO] Derived set survived for {pdf_name}")
            else:
                print(f"[INFO] Derived set did NOT survive for {pdf_name}")

            out_records.append(out)

            if has_rejected_content(rejected_record.get("rejected_derived_qa", {})):
                all_rejected_records.append(rejected_record)

        except Exception as e:
            fail_record = attach_derived_anchor_metadata(
                {
                    "pdf_name": pdf_name,
                    "paper_title": rec.get("paper_title", ""),
                    "paper_claims": rec.get("paper_claims", []) or [],
                    "source_questions": [],
                    "derived_qa": {},
                    "error": f"Unexpected failure: {e}",
                },
                pdf_name,
                False,
            )
            out_records.append(fail_record)
            print(f"[WARN] Derived-set generation failed for {pdf_name}: {e}")

    save_json(out_records, config.derived_output_json)
    print(f"\n[INFO] Wrote derived-set records to: {config.derived_output_json}")

    rejected_out_path = getattr(
        config,
        "derived_olmo_rejected_json",
        Path("derived_olmo_rejected.json"),
    )
    save_json(all_rejected_records, rejected_out_path)
    print(f"[INFO] Wrote derived OLMo-rejected questions to: {rejected_out_path}")