import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import AppConfig
from utils.http_utils import request_with_retries
from semantic_scholar.downloader import download_pdf
from semantic_scholar.filters import is_computer_science_paper, is_experimental_original_research_paper


def cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return float("-inf")

    dot = 0.0
    na = 0.0
    nb = 0.0

    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y

    if na == 0.0 or nb == 0.0:
        return float("-inf")

    return dot / (math.sqrt(na) * math.sqrt(nb))


def extract_embedding(paper_obj: Dict[str, Any]) -> Optional[List[float]]:
    if not paper_obj:
        return None

    emb = paper_obj.get("embedding")
    if isinstance(emb, dict):
        vec = emb.get("vector")
        if isinstance(vec, list):
            return vec

        specter = emb.get("specter_v2")
        if isinstance(specter, dict):
            vec2 = specter.get("vector")
            if isinstance(vec2, list):
                return vec2

    specter_root = paper_obj.get("specter_v2")
    if isinstance(specter_root, dict):
        vec3 = specter_root.get("vector")
        if isinstance(vec3, list):
            return vec3

    return None


def get_anchor_paper(corpus_id: str, config: AppConfig) -> Optional[Dict[str, Any]]:
    url = f"{config.semantic_scholar_api_base}/CorpusId:{corpus_id}"
    params = {
        "fields": (
            "paperId,abstract,corpusId,title,year,url,fieldsOfStudy,s2FieldsOfStudy,"
            "references.paperId,references.title,"
            "embedding.specter_v2"
        )
    }

    headers = {}
    if config.semantic_scholar_api_key:
        headers["x-api-key"] = config.semantic_scholar_api_key

    resp = request_with_retries(
        "GET",
        url,
        params=params,
        headers=headers,
        retries=config.retries,
        backoff=config.backoff,
    )
    return resp.json() if resp else None


def get_reference_paper_ids(anchor: Dict[str, Any]) -> List[str]:
    refs = anchor.get("references", []) or []
    out = []

    for ref in refs:
        pid = ref.get("paperId")
        if pid:
            out.append(pid)

    return out


def batch_fetch_reference_papers(reference_ids: List[str], config: AppConfig) -> List[Dict[str, Any]]:
    if not reference_ids:
        return []

    url = f"{config.semantic_scholar_api_base}/batch"
    params = {
        "fields": (
            "paperId,corpusId,title,abstract,year,url,fieldsOfStudy,s2FieldsOfStudy,"
            "isOpenAccess,openAccessPdf,externalIds,"
            "embedding.specter_v2"
        )
    }

    headers = {}
    if config.semantic_scholar_api_key:
        headers["x-api-key"] = config.semantic_scholar_api_key

    batch_size = 500
    all_results = []

    for i in range(0, len(reference_ids), batch_size):
        batch = reference_ids[i:i + batch_size]

        resp = request_with_retries(
            "POST",
            url,
            params=params,
            json_body={"ids": batch},
            headers=headers,
            retries=config.retries,
            backoff=config.backoff,
        )

        if resp:
            data = resp.json()
            if isinstance(data, list):
                all_results.extend([x for x in data if x])

    return all_results


def rank_references_by_similarity(corpus_id: str, config: AppConfig, top_k: int) -> List[Dict[str, Any]]:
    anchor = get_anchor_paper(corpus_id, config)
    if not anchor:
        print(f"[MISS] Anchor paper not found for CorpusId:{corpus_id}")
        return []

    anchor_title = anchor.get("title")
    anchor_paper_id = anchor.get("paperId")
    anchor_embedding = extract_embedding(anchor)

    print(f"[INFO] Anchor title: {anchor_title}")
    print(f"[INFO] Anchor paperId: {anchor_paper_id}")

    if anchor_embedding is None:
        print("[ERROR] Anchor embedding not returned by API.")
        return []

    reference_ids = get_reference_paper_ids(anchor)
    print(f"[INFO] Found {len(reference_ids)} references with paperId")

    if not reference_ids:
        return []

    ref_papers = batch_fetch_reference_papers(reference_ids, config)
    print(f"[INFO] Retrieved {len(ref_papers)} reference paper records")

    scored = []
    for ref in ref_papers:
        ref_embedding = extract_embedding(ref)
        if ref_embedding is None:
            continue

        score = cosine_similarity(anchor_embedding, ref_embedding)
        if score == float("-inf"):
            continue

        scored.append({
            "paperId": ref.get("paperId"),
            "corpusId": ref.get("corpusId"),
            "title": ref.get("title"),
            "abstract": ref.get("abstract"),
            "year": ref.get("year"),
            "url": ref.get("url"),
            "isOpenAccess": ref.get("isOpenAccess"),
            "openAccessPdf": ref.get("openAccessPdf"),
            "externalIds": ref.get("externalIds"),
            "similarity": score,
            "fieldsOfStudy": ref.get("fieldsOfStudy"),
            "s2FieldsOfStudy": ref.get("s2FieldsOfStudy"),
        })

    scored.sort(key=lambda x: x["similarity"], reverse=True)

    print(f"\n[INFO] Top {min(top_k, len(scored))} most similar references:\n")
    for rank, item in enumerate(scored[:top_k], start=1):
        print(
            f"{rank}. similarity={item['similarity']:.6f} | "
            f"year={item.get('year')} | "
            f"paperId={item.get('paperId')} | "
            f"corpusId={item.get('corpusId')} | "
            f"title={item.get('title')}"
        )

    return scored[:top_k]


def get_pdf_url(paper: Dict[str, Any]) -> Optional[str]:
    oa = paper.get("openAccessPdf")
    if isinstance(oa, dict):
        url = oa.get("url")
        if url:
            return url

    external_ids = paper.get("externalIds") or {}
    arxiv_id = external_ids.get("ArXiv")
    if arxiv_id:
        return f"https://arxiv.org/pdf/{arxiv_id}.pdf"

    url = paper.get("url")
    if isinstance(url, str) and url.lower().endswith(".pdf"):
        return url

    return None


def safe_filename(text: str) -> str:
    text = re.sub(r"[^\w\s.-]", "", text, flags=re.UNICODE)
    text = re.sub(r"\s+", "_", text).strip("._")
    return text[:150] if text else "paper"


def download_reference_candidate(
    paper: Dict[str, Any],
    rank: int,
    download_dir: Path,
    config: AppConfig
) -> Optional[Dict[str, Any]]:
    """
    Download one ranked candidate reference paper.
    Returns candidate metadata if successful, otherwise None.
    """
    if not is_computer_science_paper(paper, config):
        print(f"[SKIP] Rank {rank}: '{paper.get('title')}' is not classified as Computer Science")
        return None

    if not is_experimental_original_research_paper(paper, config):
        print(f"[SKIP] Rank {rank}: '{paper.get('title')}' is not classified as experimental original research paper type")
        return None

    pdf_url = get_pdf_url(paper)
    if not pdf_url:
        print(f"[SKIP] Rank {rank}: no downloadable PDF URL for '{paper.get('title')}'")
        return None

    corpus_id = paper.get("corpusId")
    paper_id = paper.get("paperId")
    title = paper.get("title") or "paper"

    stem = str(corpus_id or paper_id or safe_filename(title))
    out_path = download_dir / f"{stem}.pdf"

    print(f"[INFO] Trying rank {rank}: {title}")
    print(f"[INFO] Candidate PDF URL: {pdf_url}")

    ok = download_pdf(pdf_url, out_path)
    if not ok:
        return None

    return {
        "rank": rank,
        "paperId": paper_id,
        "corpusId": corpus_id,
        "title": title,
        "year": paper.get("year"),
        "url": paper.get("url"),
        "pdf_url": pdf_url,
        "similarity": paper.get("similarity"),
        "pdf_path": str(out_path),
    }