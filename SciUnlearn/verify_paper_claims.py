from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urljoin

import litellm
from litellm import completion
import requests

from config import AppConfig
from semantic_scholar.client import fetch_metadata
from semantic_scholar.downloader import download_pdf
from utils.env_utils import validate_required_env_vars
from utils.json_utils import load_json, save_json
from utils.pdf_utils import extract_text_from_pdf


DEFAULT_FORGET_JSON = Path("forget_common_3.json")
DEFAULT_RETAIN_JSON = Path("retain_external_common_3.json")
DEFAULT_OUTPUT_JSON = Path("claim_metric_verification_all_papers.json")
DEFAULT_DOWNLOAD_DIR = Path("claim_verification_pdfs")

# Drop unsupported provider/model params automatically (e.g., temperature for some GPT-5 routes).
litellm.drop_params = True


SYSTEM_PROMPT = """You are an expert scientific-claim evaluator.

You will receive: paper title, one paper claim, and paper text.
Score the claim on:
1) Accuracy (0-5)
2) Independence (0-5)
3) Clarity (0-5)

Definitions:
- Accuracy: faithfulness to the paper's findings/scope, no contradiction/exaggeration.
- Independence: self-contained claim, not dependent on hidden assumptions.
- Clarity: precise and unambiguous phrasing.

Scoring rubric for Accuracy:
- 5: Fully accurate. Explicitly or unambiguously supported by the paper; scope and qualifiers preserved.
- 4: Mostly accurate. Supported, with minor simplification that does not change meaning.
- 3: Partially accurate. Loosely supported; notable omissions or mild overgeneralization.
- 2: Weakly accurate. Related topic but misstates key details (method, population, strength, condition).
- 1: Minimally accurate. Mentions topic but substantially misrepresents the finding.
- 0: Inaccurate. Not supported, contradicted, or invented conclusion.

Scoring rubric for Independence:
- 5: Fully independent. Single self-contained idea; evaluable without extra context.
- 4: Mostly independent. Small implicit assumptions but still largely self-contained.
- 3: Moderately independent. Bundles related ideas; still evaluable with some interpretation.
- 2: Weakly independent. Multiple distinct claims bundled together and should be split.
- 1: Minimally independent. Heavily dependent on unstated background assumptions.
- 0: Not independent. Cannot be meaningfully evaluated on its own.

Scoring rubric for Clarity:
- 5: Very clear. Precise, unambiguous, and easy to interpret in one way.
- 4: Clear. Understandable with only minor ambiguity.
- 3: Moderately clear. Understandable but contains vague/abstract wording.
- 2: Unclear. Meaning is difficult to pin down due to broad or vague phrasing.
- 1: Very unclear. Confusing and unreliable to interpret.
- 0: Incoherent. Unintelligible or internally inconsistent.

Return EXACTLY one JSON object with this EASY schema:
{
    "accuracy_score": 0,
    "accuracy_reason": "...",
    "independence_score": 0,
    "independence_reason": "...",
    "clarity_score": 0,
    "clarity_reason": "...",
    "overall_assessment": "..."
}

Hard requirements:
- Output must never be empty.
- Output must be valid JSON only (no markdown, no prose outside JSON).
- Scores must be integers in [0,5].
- If evidence is insufficient, still return the JSON object with conservative scores and explain uncertainty in reasons.
"""


RECOVERY_PROMPT = """Return ONLY valid JSON, never empty.
Schema (exact keys required):
{
    "accuracy_score": 0,
    "accuracy_reason": "...",
    "independence_score": 0,
    "independence_reason": "...",
    "clarity_score": 0,
    "clarity_reason": "...",
    "overall_assessment": "..."
}
If uncertain, provide conservative scores but still return the JSON object.
"""


SIMPLE_PROMPT = """You are scoring one claim against paper text.
Return EXACTLY one line in this format (no extra text):
ACC=<0-5>|IND=<0-5>|CLR=<0-5>|ACC_R=<short reason>|IND_R=<short reason>|CLR_R=<short reason>|OVERALL=<short summary>
Never return empty output.
"""


@dataclass
class PaperRecord:
    side: str
    anchor_id: str
    pdf_name: str
    paper_title: str
    paper_claims: List[str]
    raw_record: Dict[str, Any]


def normalize_pdf_name(pdf_name: str) -> str:
    return Path((pdf_name or "").strip()).stem.strip()


def load_records(path: Path) -> List[Dict[str, Any]]:
    data = load_json(path)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {path}")
    return [rec for rec in data if isinstance(rec, dict)]


def build_retain_lookup(records: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    lookup: Dict[str, Dict[str, Any]] = {}
    for record in records:
        anchor_id = str(record.get("anchor_forget_paper_id") or record.get("anchor_corpus_id") or "").strip()
        if anchor_id and anchor_id not in lookup:
            lookup[anchor_id] = record
    return lookup


def select_paired_records(
    forget_records: Sequence[Dict[str, Any]],
    retain_lookup: Dict[str, Dict[str, Any]],
    sample_count: int,
) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []

    for forget_record in forget_records:
        anchor_id = normalize_pdf_name(str(forget_record.get("pdf_name") or ""))
        retain_record = retain_lookup.get(anchor_id)
        if retain_record is None:
            continue

        pairs.append((forget_record, retain_record))
        # sample_count <= 0 means: evaluate all available paired papers.
        if sample_count > 0 and len(pairs) >= sample_count:
            break

    return pairs


def ensure_pdf(
    corpus_id: str,
    pdf_name: str,
    side: str,
    download_dir: Path,
    config: AppConfig,
) -> Tuple[Optional[Path], Dict[str, Any]]:
    download_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = download_dir / f"{corpus_id}.pdf"

    metadata: Dict[str, Any] = {}

    # 1) Prefer existing PDFs from the project dataset folders.
    local_candidates = candidate_local_dirs(side, config)
    local_hit = find_existing_pdf(corpus_id=corpus_id, pdf_name=pdf_name, roots=local_candidates)
    if local_hit is not None:
        metadata["download_status"] = "dataset_local_cache"
        metadata["pdf_path"] = str(local_hit)
        return local_hit, metadata

    # 2) Fallback to verifier cache directory.
    if pdf_path.exists() and pdf_path.stat().st_size > 0:
        metadata["download_status"] = "local_cache"
        metadata["pdf_path"] = str(pdf_path)
        return pdf_path, metadata

    fetched = fetch_metadata(corpus_id=int(corpus_id), config=config)
    if not fetched:
        metadata["download_status"] = "metadata_unavailable"
        return None, metadata

    metadata = {
        "download_status": "metadata_fetched",
        "title": fetched.get("title"),
        "year": fetched.get("year"),
        "is_open_access": fetched.get("isOpenAccess"),
        "pdf_url": (fetched.get("openAccessPdf") or {}).get("url"),
        "source_url": fetched.get("url"),
    }

    # Retry metadata with richer fields to find alternate download identifiers.
    metadata_rich = fetch_metadata_rich(corpus_id=corpus_id, config=config)
    if metadata_rich:
        metadata["external_ids"] = metadata_rich.get("externalIds") or {}
        if not metadata.get("pdf_url"):
            metadata["pdf_url"] = (metadata_rich.get("openAccessPdf") or {}).get("url") or ""

    pdf_url = metadata.get("pdf_url")
    if not pdf_url:
        alt_url = discover_fallback_pdf_url(metadata)
        if alt_url:
            pdf_url = alt_url
            metadata["pdf_url"] = alt_url
            metadata["download_status"] = "fallback_pdf_url_discovered"
        else:
            metadata["download_status"] = "no_open_access_pdf"
            return None, metadata

    if not download_pdf(pdf_url, pdf_path):
        metadata["download_status"] = "download_failed"
        return None, metadata

    metadata["download_status"] = "downloaded"
    metadata["pdf_path"] = str(pdf_path)
    return pdf_path, metadata


def candidate_local_dirs(side: str, config: AppConfig) -> List[Path]:
    dirs: List[Path] = []

    if side == "forget":
        dirs.append(Path("Downloaded_paper_forget"))
        dirs.append(Path("Downloaded_paper_forget"))
        dirs.append(Path(config.download_dir))
    else:
        dirs.append(Path("Downloaded_paper_retain"))
        dirs.append(Path("Downloaded_paper_retain"))
        dirs.append(Path(config.retain_download_dir))

    # Deduplicate while preserving order.
    deduped: List[Path] = []
    seen: set[str] = set()
    for d in dirs:
        key = str(d)
        if key not in seen:
            seen.add(key)
            deduped.append(d)

    return deduped


def find_existing_pdf(corpus_id: str, pdf_name: str, roots: Sequence[Path]) -> Optional[Path]:
    names = [f"{corpus_id}.pdf"]
    if pdf_name and pdf_name not in names:
        names.append(pdf_name)

    for root in roots:
        for name in names:
            candidate = root / name
            if candidate.exists() and candidate.stat().st_size > 0:
                return candidate

    return None


def fetch_metadata_rich(corpus_id: str, config: AppConfig) -> Optional[Dict[str, Any]]:
    url = f"{config.semantic_scholar_api_base}/CorpusId:{corpus_id}"
    params = {
        "fields": "title,year,isOpenAccess,openAccessPdf,url,externalIds"
    }
    headers: Dict[str, str] = {}
    if config.semantic_scholar_api_key:
        headers["x-api-key"] = config.semantic_scholar_api_key

    try:
        resp = requests.get(url, params=params, headers=headers, timeout=30)
        if resp.ok:
            return resp.json()
    except requests.RequestException:
        return None

    return None


def discover_fallback_pdf_url(metadata: Dict[str, Any]) -> str:
    source_url = str(metadata.get("source_url") or "").strip()
    external_ids = metadata.get("external_ids") or {}

    # ArXiv fallback is usually reliable.
    arxiv_id = str(external_ids.get("ArXiv") or "").strip()
    if arxiv_id:
        return f"https://arxiv.org/pdf/{arxiv_id}.pdf"

    # Scrape source page for likely PDF links.
    if source_url:
        try:
            resp = requests.get(source_url, timeout=30)
            if resp.ok and resp.text:
                html = resp.text

                # meta citation PDF links
                meta_match = re.search(r'citation_pdf_url"\s*content="([^"]+)"', html, flags=re.IGNORECASE)
                if meta_match:
                    return meta_match.group(1).strip()

                hrefs = re.findall(r'href="([^"]+)"', html, flags=re.IGNORECASE)
                for href in hrefs:
                    if ".pdf" in href.lower():
                        return urljoin(source_url, href)
        except requests.RequestException:
            pass

    # DOI fallback.
    doi = str(external_ids.get("DOI") or "").strip()
    if doi:
        return f"https://doi.org/{doi}"

    return ""


def extract_paper_text(pdf_path: Path) -> str:
    return extract_text_from_pdf(pdf_path)


def build_evaluation_prompt(
    paper_title: str,
    paper_claim: str,
    paper_text: str,
    recovery_mode: bool = False,
) -> List[Dict[str, str]]:
    system_prompt = RECOVERY_PROMPT if recovery_mode else SYSTEM_PROMPT

    user_content = f"""Paper title:
{paper_title}

Paper claim:
{paper_claim}

Paper text:
{paper_text}
"""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def parse_model_json(raw: str) -> Dict[str, Any]:
    text = raw.strip()

    if text.startswith("```"):
        text = text.removeprefix("```json").removeprefix("```").strip()
        if text.endswith("```"):
            text = text[:-3].strip()

    if not text:
        raise ValueError("Model returned empty content")

    # Allow extra prose around JSON by extracting the first JSON object span.
    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and last > first:
        text = text[first : last + 1]

    return json.loads(text)


def content_to_text(content: Any) -> str:
    if content is None:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue

            if isinstance(item, dict):
                txt = item.get("text")
                if isinstance(txt, str):
                    parts.append(txt)

        return "\n".join(parts)

    return str(content)


def normalize_metrics_shape(parsed: Dict[str, Any]) -> Dict[str, Any]:
    """
    Accept either:
    - flat schema (accuracy_score, accuracy_reason, ...)
    - nested schema (accuracy: {score, reason}, ...)
    and normalize to the nested schema used in output files.
    """
    if all(k in parsed for k in (
        "accuracy_score",
        "accuracy_reason",
        "independence_score",
        "independence_reason",
        "clarity_score",
        "clarity_reason",
    )):
        return {
            "accuracy": {
                "score": parsed.get("accuracy_score"),
                "reason": parsed.get("accuracy_reason"),
            },
            "independence": {
                "score": parsed.get("independence_score"),
                "reason": parsed.get("independence_reason"),
            },
            "clarity": {
                "score": parsed.get("clarity_score"),
                "reason": parsed.get("clarity_reason"),
            },
            "overall_assessment": parsed.get("overall_assessment", ""),
        }

    return parsed


def parse_simple_line(raw: str) -> Dict[str, Any]:
    text = raw.strip()
    if not text:
        raise ValueError("Simple fallback response is empty")

    parts = [p.strip() for p in text.split("|") if p.strip()]
    kv: Dict[str, str] = {}
    for p in parts:
        if "=" not in p:
            continue
        k, v = p.split("=", 1)
        kv[k.strip().upper()] = v.strip()

    def safe_score(k: str) -> int:
        try:
            v = int(kv.get(k, "2"))
        except ValueError:
            v = 2
        return min(5, max(0, v))

    return {
        "accuracy": {
            "score": safe_score("ACC"),
            "reason": kv.get("ACC_R", "Short fallback reason from simple format."),
        },
        "independence": {
            "score": safe_score("IND"),
            "reason": kv.get("IND_R", "Short fallback reason from simple format."),
        },
        "clarity": {
            "score": safe_score("CLR"),
            "reason": kv.get("CLR_R", "Short fallback reason from simple format."),
        },
        "overall_assessment": kv.get("OVERALL", "Simple fallback summary."),
    }


def call_gpt5_for_claim(
    paper_title: str,
    paper_claim: str,
    paper_text: str,
    config: AppConfig,
) -> Dict[str, Any]:
    last_error: Optional[Exception] = None
    last_content_preview = ""

    for attempt in range(1, 4):
        try:
            # Progressively shorten context and tighten instruction style on retries.
            if attempt == 1:
                target_chars = min(len(paper_text), 24000)
                recovery_mode = False
            elif attempt == 2:
                target_chars = min(len(paper_text), 12000)
                recovery_mode = True
            else:
                target_chars = min(len(paper_text), 6000)
                recovery_mode = True

            truncated_text = paper_text[:target_chars]
            messages = build_evaluation_prompt(
                paper_title=paper_title,
                paper_claim=paper_claim,
                paper_text=truncated_text,
                recovery_mode=recovery_mode,
            )

            response = completion(
                model="azure/gpt-4o",
                messages=messages,
                temperature=config.llm_temperature,
                # response_format={"type": "json_object"},
            )

            print(f"[DEBUG] Attempt {attempt} raw response: {response.choices[0].message.content}")

            raw_content = extract_response_text(response)
            last_content_preview = (raw_content or "")[:300]
            parsed = normalize_metrics_shape(parse_model_json(raw_content))

            for key in ("accuracy", "independence", "clarity"):
                if key not in parsed or not isinstance(parsed.get(key), dict):
                    raise ValueError(f"Model response missing '{key}' block")
                score = parsed[key].get("score")
                if not isinstance(score, int) or score < 0 or score > 5:
                    raise ValueError(f"Model response contains invalid score for '{key}': {score}")

            return parsed

        except Exception as exc:
            last_error = exc
            if attempt < 3:
                continue

    # Final fallback: ask for an even simpler, single-line format.
    try:
        compact_text = paper_text[:4000]
        simple_messages = [
            {"role": "system", "content": SIMPLE_PROMPT},
            {
                "role": "user",
                "content": (
                    f"TITLE: {paper_title}\n"
                    f"CLAIM: {paper_claim}\n"
                    f"TEXT: {compact_text}"
                ),
            },
        ]

        simple_response = completion(
            model="azure/gpt-4o",
            messages=simple_messages,
            max_tokens=500,
        )
        simple_raw = extract_response_text(simple_response)
        if simple_raw.strip():
            return parse_simple_line(simple_raw)
    except Exception:
        pass

    raise ValueError(
        "Failed to parse model output in both JSON and simple-line modes. "
        f"Last error: {last_error}. Content preview: {last_content_preview!r}"
    )


def extract_response_text(response: Any) -> str:
    try:
        choice0 = response.choices[0]
    except Exception:
        return ""

    message = getattr(choice0, "message", None)
    if message is not None:
        content = getattr(message, "content", None)
        text = content_to_text(content)
        if text.strip():
            return text

        # Some providers expose content as dict-like objects.
        try:
            content_dict = message.get("content")  # type: ignore[attr-defined]
            text = content_to_text(content_dict)
            if text.strip():
                return text
        except Exception:
            pass

    # Fallback for completion-style providers.
    text_field = getattr(choice0, "text", None)
    if isinstance(text_field, str):
        return text_field

    return ""


def score_one_claim(
    record: PaperRecord,
    claim_index: int,
    claim_text: str,
    paper_text: str,
    config: AppConfig,
    text_truncated: bool,
) -> Dict[str, Any]:
    if text_truncated:
        print(
            f"[WARN] {record.side}:{record.anchor_id} claim {claim_index} used truncated text input "
            f"({len(paper_text)} chars)."
        )

    model_result = call_gpt5_for_claim(record.paper_title, claim_text, paper_text, config)

    return {
        "claim_index": claim_index,
        "claim": claim_text,
        "metrics": model_result,
    }


def evaluate_record(
    record: PaperRecord,
    download_dir: Path,
    config: AppConfig,
) -> Dict[str, Any]:
    corpus_id = normalize_pdf_name(record.pdf_name)
    pdf_path, download_meta = ensure_pdf(
        corpus_id=corpus_id,
        pdf_name=record.pdf_name,
        side=record.side,
        download_dir=download_dir,
        config=config,
    )

    result: Dict[str, Any] = {
        "side": record.side,
        "anchor_id": record.anchor_id,
        "pdf_name": record.pdf_name,
        "paper_title": record.paper_title,
        "corpus_id": corpus_id,
        "download": download_meta,
        "claims": [],
    }

    if pdf_path is None:
        result["status"] = "skipped"
        result["reason"] = download_meta.get("download_status", "download_unavailable")
        return result

    paper_text = extract_paper_text(pdf_path)
    if not paper_text.strip():
        result["status"] = "skipped"
        result["reason"] = "empty_extracted_text"
        result["download"]["pdf_path"] = str(pdf_path)
        return result

    text_limit = max(1, int(config.max_chars_for_model))
    text_for_model = paper_text
    text_truncated = False
    if len(paper_text) > text_limit:
        text_for_model = paper_text[:text_limit]
        text_truncated = True

    result["status"] = "evaluated"
    result["download"]["pdf_path"] = str(pdf_path)
    result["paper_text_chars"] = len(paper_text)
    result["paper_text_used_chars"] = len(text_for_model)
    result["paper_text_truncated"] = text_truncated

    for claim_index, claim_text in enumerate(record.paper_claims):
        try:
            claim_result = score_one_claim(
                record=record,
                claim_index=claim_index,
                claim_text=claim_text,
                paper_text=text_for_model,
                config=config,
                text_truncated=text_truncated,
            )
        except Exception as exc:
            claim_result = {
                "claim_index": claim_index,
                "claim": claim_text,
                "error": str(exc),
            }

        result["claims"].append(claim_result)

    return result


def build_paper_record(side: str, record: Dict[str, Any], anchor_id: str) -> PaperRecord:
    return PaperRecord(
        side=side,
        anchor_id=anchor_id,
        pdf_name=str(record.get("pdf_name") or ""),
        paper_title=str(record.get("paper_title") or ""),
        paper_claims=[str(claim) for claim in (record.get("paper_claims") or []) if str(claim).strip()],
        raw_record=record,
    )


def _empty_bucket() -> Dict[str, Any]:
    return {
        "accuracy": [],
        "independence": [],
        "clarity": [],
        "claims_total": 0,
        "claims_scored": 0,
        "claims_error": 0,
    }


def _mean(values: List[int]) -> Optional[float]:
    if not values:
        return None
    return round(sum(values) / len(values), 4)


def _distribution(values: List[int]) -> Dict[str, int]:
    dist = {str(i): 0 for i in range(6)}
    for v in values:
        if 0 <= v <= 5:
            dist[str(v)] += 1
    return dist


def _summarize_bucket(bucket: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "claims_total": bucket["claims_total"],
        "claims_scored": bucket["claims_scored"],
        "claims_error": bucket["claims_error"],
        "scored_fraction": (
            round(bucket["claims_scored"] / bucket["claims_total"], 4)
            if bucket["claims_total"] > 0
            else None
        ),
        "accuracy": {
            "mean": _mean(bucket["accuracy"]),
            "distribution": _distribution(bucket["accuracy"]),
        },
        "independence": {
            "mean": _mean(bucket["independence"]),
            "distribution": _distribution(bucket["independence"]),
        },
        "clarity": {
            "mean": _mean(bucket["clarity"]),
            "distribution": _distribution(bucket["clarity"]),
        },
    }


def build_combined_statistics(evaluations: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    overall = _empty_bucket()
    by_side: Dict[str, Dict[str, Any]] = {
        "forget": _empty_bucket(),
        "retain": _empty_bucket(),
    }

    for pair in evaluations:
        for side in ("forget", "retain"):
            section = pair.get(side, {}) or {}
            claims = section.get("claims", []) or []

            for claim_obj in claims:
                overall["claims_total"] += 1
                by_side[side]["claims_total"] += 1

                metrics = claim_obj.get("metrics") or {}
                if not isinstance(metrics, dict):
                    overall["claims_error"] += 1
                    by_side[side]["claims_error"] += 1
                    continue

                try:
                    acc = int(((metrics.get("accuracy") or {}).get("score")))
                    ind = int(((metrics.get("independence") or {}).get("score")))
                    clr = int(((metrics.get("clarity") or {}).get("score")))
                except Exception:
                    overall["claims_error"] += 1
                    by_side[side]["claims_error"] += 1
                    continue

                if not all(0 <= v <= 5 for v in (acc, ind, clr)):
                    overall["claims_error"] += 1
                    by_side[side]["claims_error"] += 1
                    continue

                overall["claims_scored"] += 1
                by_side[side]["claims_scored"] += 1

                overall["accuracy"].append(acc)
                overall["independence"].append(ind)
                overall["clarity"].append(clr)

                by_side[side]["accuracy"].append(acc)
                by_side[side]["independence"].append(ind)
                by_side[side]["clarity"].append(clr)

    return {
        "overall": _summarize_bucket(overall),
        "by_side": {
            "forget": _summarize_bucket(by_side["forget"]),
            "retain": _summarize_bucket(by_side["retain"]),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify paper claims from forget_common_3.json and retain_external_common_3.json using GPT-5."
    )
    parser.add_argument("--forget-json", type=Path, default=DEFAULT_FORGET_JSON)
    parser.add_argument("--retain-json", type=Path, default=DEFAULT_RETAIN_JSON)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--download-dir", type=Path, default=DEFAULT_DOWNLOAD_DIR)
    parser.add_argument(
        "--sample-count",
        type=int,
        default=0,
        help="Number of paired papers to evaluate. Use 0 (default) to evaluate all paired papers.",
    )
    parser.add_argument("--max-chars", type=int, default=40000)
    args = parser.parse_args()

    required = ["AZURE_API_KEY", "AZURE_API_BASE", "AZURE_API_VERSION"]
    missing = validate_required_env_vars(required)
    if missing:
        raise RuntimeError(f"Missing required environment variable(s): {', '.join(missing)}")

    config = AppConfig()
    config.max_chars_for_model = args.max_chars

    forget_records = load_records(args.forget_json)
    retain_records = load_records(args.retain_json)
    retain_lookup = build_retain_lookup(retain_records)
    pairs = select_paired_records(forget_records, retain_lookup, args.sample_count)

    if not pairs:
        raise RuntimeError("No paired forget/retain records were found.")

    if args.sample_count > 0:
        print(f"[INFO] Selected {len(pairs)} paired papers for evaluation (limit={args.sample_count}).")
    else:
        print(f"[INFO] Selected all {len(pairs)} paired papers for evaluation.")

    evaluations: List[Dict[str, Any]] = []
    for pair_index, (forget_record, retain_record) in enumerate(pairs, start=1):
        anchor_id = normalize_pdf_name(str(forget_record.get("pdf_name") or ""))
        print(f"[INFO] Evaluating pair {pair_index}/{len(pairs)} for anchor {anchor_id}")

        forget_paper = build_paper_record("forget", forget_record, anchor_id)
        retain_anchor = str(retain_record.get("anchor_forget_paper_id") or retain_record.get("anchor_corpus_id") or "").strip()
        retain_paper = build_paper_record("retain", retain_record, retain_anchor)

        pair_result = {
            "anchor_id": anchor_id,
            "forget": evaluate_record(forget_paper, args.download_dir / "forget", config),
            "retain": evaluate_record(retain_paper, args.download_dir / "retain", config),
        }
        evaluations.append(pair_result)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "forget_json": str(args.forget_json),
        "retain_json": str(args.retain_json),
        "sample_count_requested": args.sample_count,
        "sample_count_evaluated": len(evaluations),
        "model": "azure/gpt-4o",
        "evaluations": evaluations,
        "summary_statistics": build_combined_statistics(evaluations),
    }

    save_json(report, args.output_json)
    print(f"[DONE] Wrote verification report to {args.output_json}")


if __name__ == "__main__":
    main()