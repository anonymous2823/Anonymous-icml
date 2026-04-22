import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from config import AppConfig

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    from sklearn.feature_extraction.text import TfidfVectorizer


def load_json_file(path: str | Path) -> Any:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def flatten_claims_from_qa_records(data: Any) -> List[Dict[str, Any]]:
    """
    Flatten claims from QA_final_covered.json.
    """
    flattened = []

    if isinstance(data, dict):
        data = [data]

    if not isinstance(data, list):
        raise ValueError("QA_final_covered.json must contain either a list or a single record dict.")

    for idx, rec in enumerate(data):
        if not isinstance(rec, dict):
            continue

        paper_title = rec.get("paper_title")
        pdf_name = rec.get("pdf_name")

        if not paper_title and isinstance(rec.get("selected_reference"), dict):
            paper_title = rec["selected_reference"].get("title")

        paper_claims = rec.get("paper_claims", [])
        if not isinstance(paper_claims, list):
            continue

        for claim_idx, claim in enumerate(paper_claims):
            if isinstance(claim, str) and claim.strip():
                flattened.append({
                    "record_index": idx,
                    "claim_index": claim_idx,
                    "paper_title": paper_title,
                    "pdf_name": pdf_name,
                    "claim": " ".join(claim.split()),
                })

    return flattened


def build_semantic_embeddings(texts: List[str], config: AppConfig) -> np.ndarray:
    """
    Preferred route: sentence-transformers.
    Fallback: TF-IDF.
    """
    if not texts:
        return np.zeros((0, 0), dtype=float)

    if SENTENCE_TRANSFORMERS_AVAILABLE:
        print(f"[INFO] Using sentence-transformers model: {config.retain_semantic_model_name}")
        model = SentenceTransformer(config.retain_semantic_model_name)
        embeddings = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings

    print("[WARN] sentence-transformers not available. Falling back to TF-IDF cosine similarity.")
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    matrix = vectorizer.fit_transform(texts)
    return matrix.toarray()


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=float)

    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True)

    a_safe = a / np.clip(a_norm, 1e-12, None)
    b_safe = b / np.clip(b_norm, 1e-12, None)

    return np.matmul(a_safe, b_safe.T)


def rank_retain_claims_against_qa_claims(
    retain_claims: List[str],
    qa_claim_records: List[Dict[str, Any]],
    config: AppConfig,
    top_n: int,
) -> List[Dict[str, Any]]:
    if not retain_claims or not qa_claim_records:
        return []

    qa_claim_texts = [x["claim"] for x in qa_claim_records]
    all_texts = retain_claims + qa_claim_texts

    embeddings = build_semantic_embeddings(all_texts, config)

    retain_emb = embeddings[:len(retain_claims)]
    qa_emb = embeddings[len(retain_claims):]

    sim_matrix = cosine_similarity_matrix(retain_emb, qa_emb)

    results = []
    for i, retain_claim in enumerate(retain_claims):
        sims = sim_matrix[i]
        ranked_indices = np.argsort(-sims)

        ranked_matches = []
        for rank_pos, j in enumerate(ranked_indices[:top_n], start=1):
            rec = qa_claim_records[int(j)]
            ranked_matches.append({
                "rank": rank_pos,
                "similarity_score": float(sims[j]),
                "matched_claim": rec["claim"],
                "paper_title": rec.get("paper_title"),
                "pdf_name": rec.get("pdf_name"),
                "record_index": rec.get("record_index"),
                "claim_index": rec.get("claim_index"),
            })

        results.append({
            "retain_claim": retain_claim,
            "top_matches": ranked_matches,
        })

    return results

def print_similarity_results(similarity_results: List[Dict[str, Any]], top_n: int) -> None:
    if not similarity_results:
        print("[INFO] No similarity results to display.")
        return

    print("\n" + "=" * 100)
    print(f"[INFO] Semantic similarity ranking (top {top_n})")
    print("=" * 100)

    for block_idx, item in enumerate(similarity_results, start=1):
        print(f"\n[RETAIN CLAIM {block_idx}]")
        print(item["retain_claim"])
        print("-" * 100)

        for match in item["top_matches"]:
            print(
                f"rank={match['rank']:>2} | "
                f"score={match['similarity_score']:.6f} | "
                f"title={match.get('paper_title')} | "
                f"pdf={match.get('pdf_name')} | "
                f"claim={match['matched_claim']}"
            )