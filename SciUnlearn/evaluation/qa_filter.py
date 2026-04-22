import re
from typing import Any, Dict, List, Tuple

from config import AppConfig

from evaluation.olmo_rejection_utils import split_items_by_olmo_match

import re
from functools import lru_cache

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer


ALLOWED_AR = [
    "A is True, R is True, and R explains A",
    "A is True, R is True, but R does not explain A",
    "A is True, R is False",
    "A is False, R is True",
    "A is False, R is False",
]


# ---------------------- Normalizers ----------------------------

def normalize_mcq(s: str) -> str:
    """Normalize MCQ answers to A/B/C/D."""
    if not isinstance(s, str):
        return ""

    m = re.search(r"\b([ABCD])\b", s.strip(), flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()

    t = s.strip().lower()
    for k, v in {"a": "A", "b": "B", "c": "C", "d": "D"}.items():
        if t.startswith(k):
            return v

    return (s.strip()[:1] or "").upper()


def normalize_tf(s: str) -> str:
    """Normalize True/False answers to exactly 'True' or 'False'."""
    if not isinstance(s, str):
        return ""

    t = s.strip().lower()
    if "true" in t and "false" not in t:
        return "True"
    if "false" in t and "true" not in t:
        return "False"

    tok = re.split(r"\s+", s.strip())[0] if s.strip() else ""
    return "True" if tok.lower().startswith("t") else "False"


def normalize_fill(s: str) -> str:
    """Normalize fill-in answers to a compact lowercase single-line string."""
    if not isinstance(s, str):
        return ""

    ans = s.strip().strip('"\'').strip()
    ans = re.sub(r"\s+", " ", ans)
    return ans.lower()


def normalize_ar(s: str) -> str:
    """Normalize assertion-reason labels to one of the allowed labels."""
    if not isinstance(s, str):
        return ""

    t = s.strip()

    for label in ALLOWED_AR:
        if t == label:
            return label

    tl = t.lower()
    for label in ALLOWED_AR:
        if label.lower() in tl:
            return label

    return ALLOWED_AR[0]


# ---------------------- ROUGE-L (F1) ---------------------------

def _tokenize(s: str) -> List[str]:
    """Simple tokenization: split on non-alphanumeric."""
    if not s:
        return []
    return [t for t in re.split(r"[^A-Za-z0-9]+", s.strip()) if t]


def _lcs_len(xs: List[str], ys: List[str]) -> int:
    """Length of the Longest Common Subsequence over tokens."""
    n, m = len(xs), len(ys)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n):
        xi = xs[i]
        for j in range(m):
            if xi == ys[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])

    return dp[n][m]


def rouge_l_f1(a: str, b: str) -> float:
    """ROUGE-L F1 between strings a and b using token LCS."""
    xs, ys = _tokenize(a), _tokenize(b)
    if not xs or not ys:
        return 0.0

    lcs = _lcs_len(xs, ys)
    prec = lcs / len(xs)
    rec = lcs / len(ys)

    if prec + rec == 0:
        return 0.0

    return 2 * prec * rec / (prec + rec)



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
    Load and cache the embedding model once.
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


# ---------------------- Pair evaluation ------------------------

# def evaluate_pair(
#     items: List[Dict[str, Any]],
#     qtype: str,
#     config: AppConfig,
# ) -> Tuple[List[Dict[str, Any]], bool]:
#     """
#     Evaluate one QA type for one claim.

#     Returns:
#         kept_items: only items that matched
#         pair_good: True if at least 2 matched items remain
#     """
#     kept: List[Dict[str, Any]] = []

#     for it in items or []:
#         expected = it.get("answer", "")
#         modelout = it.get("olmo_answer", "")

#         if qtype == "mcq":
#             exp_n = normalize_mcq(expected)
#             out_n = normalize_mcq(modelout)
#         elif qtype == "true_false":
#             exp_n = normalize_tf(expected)
#             out_n = normalize_tf(modelout)
#         elif qtype == "fill_blank":
#             exp_n = normalize_fill(expected)
#             out_n = normalize_fill(modelout)
#         elif qtype == "assertion_reason":
#             exp_n = normalize_ar(expected)
#             out_n = normalize_ar(modelout)
#         else:
#             exp_n = str(expected).strip()
#             out_n = str(modelout).strip()

#         match = (exp_n == out_n)
#         score = 1.0 if match else rouge_l_f1(exp_n, out_n)

#         if not match and score >= config.rouge_threshold:
#             match = True

#         it["rouge"] = round(score, 4)
#         it["match"] = bool(match)

#         if match:
#             kept.append(it)

#     pair_good = len(kept) >= 2
#     return kept, pair_good


def evaluate_pair(
    items: List[Dict[str, Any]],
    qtype: str,
    config: AppConfig,
) -> Tuple[List[Dict[str, Any]], bool]:
    """
    Evaluate one QA type for one claim.

    For every question item, compute:
      - exact normalized match
      - ROUGE score
      - semantic similarity score

    Returns:
        kept_items: only items that matched
        pair_good: True if at least 2 matched items remain
    """
    kept: List[Dict[str, Any]] = []

    for it in items or []:
        expected = it.get("answer", "")
        modelout = it.get("olmo_answer", "")

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
        rouge_score = 1.0 if exact_match else rouge_l_f1(exp_n, out_n)

        # ------------------------------------------------------
        # Semantic similarity score
        # ------------------------------------------------------
        semantic_score = semantic_similarity_text(
            expected=exp_n,
            predicted=out_n,
            model_name=config.semantic_model_name,
        )

        # ------------------------------------------------------
        # Final match rule
        # ------------------------------------------------------
        match = (
            exact_match
            or rouge_score >= config.rouge_threshold
            or semantic_score >= config.semantic_similarity_threshold
        )

        # ------------------------------------------------------
        # Save diagnostics
        # ------------------------------------------------------
        it["expected_normalized"] = exp_n
        it["olmo_normalized"] = out_n
        it["exact_match"] = bool(exact_match)
        it["rouge"] = round(float(rouge_score), 4)
        it["semantic_similarity"] = round(float(semantic_score), 4)
        it["match"] = bool(match)

        if match:
            kept.append(it)

    pair_good = len(kept) >= 2
    return kept, pair_good


# ---------------------- Record-level filtering -------------------------

# def filter_record_by_olmo(
#     record: Dict[str, Any],
#     config: AppConfig,
# ) -> Dict[str, Any]:
#     """
#     Filter one per-paper record after OLMo enrichment.

#     Rule:
#     - For each claim:
#       - evaluate all four QA types
#       - keep claim only if >= min_good_pairs_per_claim pairs are good
#     - For each kept type:
#       - keep only matched items
#     """
#     qa_by_claim = record.get("qa_by_claim", []) or []

#     kept_claim_objs: List[Dict[str, Any]] = []
#     kept_claim_texts: List[str] = []

#     for claim_obj in qa_by_claim:
#         claim_text = claim_obj.get("claim", "")
#         results_by_type: Dict[str, Dict[str, Any]] = {}
#         good_pairs_count = 0

#         for qtype in config.all_qa_types:
#             items = claim_obj.get(qtype, []) or []
#             kept_items, pair_good = evaluate_pair(items, qtype, config)

#             results_by_type[qtype] = {
#                 "kept_items": kept_items,
#                 "pair_good": pair_good,
#             }

#             if pair_good:
#                 good_pairs_count += 1

#         if good_pairs_count >= config.min_good_pairs_per_claim:
#             kept_claim_texts.append(claim_text)
#             kept_claim_objs.append({
#                 "claim": claim_text,
#                 **{
#                     qtype: results_by_type[qtype]["kept_items"]
#                     for qtype in config.all_qa_types
#                     if results_by_type[qtype]["pair_good"]
#                 }
#             })

#     return {
#         "pdf_name": record.get("pdf_name"),
#         "paper_title": record.get("paper_title"),
#         "paper_claims": kept_claim_texts,
#         "verbatim_claims": record.get("verbatim_claims", []),
#         "qa_by_claim": kept_claim_objs,
#     }


def filter_record_by_olmo_with_rejected(
    record: Dict[str, Any],
    config: AppConfig,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    qa_by_claim = record.get("qa_by_claim", []) or []

    kept_claim_objs: List[Dict[str, Any]] = []
    kept_claim_texts: List[str] = []

    rejected_claim_objs: List[Dict[str, Any]] = []

    for claim_obj in qa_by_claim:
        claim_text = claim_obj.get("claim", "")
        kept_types = {}
        rejected_types = {}
        good_pairs_count = 0

        for qtype in config.all_qa_types:
            items = claim_obj.get(qtype, []) or []

            kept_items, rejected_items = split_items_by_olmo_match(items, qtype, config)

            if rejected_items:
                rejected_types[qtype] = rejected_items

            # preserve existing pair rule
            pair_good = len(kept_items) >= 2
            if pair_good:
                kept_types[qtype] = kept_items
                good_pairs_count += 1

        if rejected_types:
            rejected_claim_objs.append({
                "claim": claim_text,
                **rejected_types,
            })

        if good_pairs_count >= config.min_good_pairs_per_claim:
            kept_claim_texts.append(claim_text)
            kept_claim_objs.append({
                "claim": claim_text,
                **kept_types,
            })

    filtered_record = {
        "pdf_name": record.get("pdf_name"),
        "paper_title": record.get("paper_title"),
        "paper_claims": kept_claim_texts,
        "verbatim_claims": record.get("verbatim_claims", []),
        "qa_by_claim": kept_claim_objs,
    }

    rejected_record = {
        "pdf_name": record.get("pdf_name"),
        "paper_title": record.get("paper_title"),
        "paper_claims": record.get("paper_claims", []),
        "rejected_qa_by_claim": rejected_claim_objs,
    }

    return filtered_record, rejected_record


def filter_records_by_olmo(
    records: List[Dict[str, Any]],
    config: AppConfig,
) -> List[Dict[str, Any]]:
    """
    Batch-filter a list of records.
    Useful if you want to reuse this module for existing JSON files later.
    """
    return [filter_record_by_olmo(record, config) for record in records]