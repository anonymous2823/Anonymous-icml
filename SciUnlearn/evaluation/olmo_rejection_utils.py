from typing import Dict, Any, List, Tuple
import re

from config import AppConfig


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
    Score one QA item against OLMo output.

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


def split_items_by_olmo_match(
    items: List[Dict[str, Any]],
    qtype: str,
    config: AppConfig,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split items into:
    - kept_items
    - rejected_items

    Assumes each item already has:
      - answer
      - olmo_answer
      - (optional) olmo_is_consistent, olmo_consistency_score, olmo_all_answers (for consistency check)
    """
    kept_items: List[Dict[str, Any]] = []
    rejected_items: List[Dict[str, Any]] = []

    for item in items:
        expected = item.get("answer", "")
        modelout = item.get("olmo_answer", "")

        score_info = score_single_item(expected, modelout, qtype, config)
        item.update(score_info)

        # Check consistency if enabled
        should_reject_for_inconsistency = False
        if config.enable_olmo_consistency_check:
            is_consistent = item.get("olmo_is_consistent", True)
            if not is_consistent:
                should_reject_for_inconsistency = True
                item["rejection_reason"] = "olmo_inconsistency"

        if not should_reject_for_inconsistency and item["match"]:
            kept_items.append(item)
        else:
            if not should_reject_for_inconsistency:
                item["rejection_reason"] = "olmo_mismatch"
            rejected_items.append(item)

    return kept_items, rejected_items
