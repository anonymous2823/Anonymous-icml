import json
import re
from typing import Any, Dict, List

from config import AppConfig
from llm_client.azure_gpt5_client import call_llm_json


DECOMP_SYSTEM = """
You are a precise scientific reader.

Task:
Decompose the following one-sentence claim into its minimal semantic components:
- Preconditions/conditions
- Actions/methods
- Targets/objects
- Purposes/outcomes
- Key constraints or scope qualifiers (if present)

Rules:
- Return ONLY valid JSON in the following schema:
  { "components": ["...", "..."] }
- Components must be concise, non-overlapping, and together fully reconstruct the claim.
- Do not add information not present in the claim.
- No commentary; JSON only.
""".strip()


COVERAGE_SYSTEM = """
You are a precise exam-auditor.

Goal:
Determine if the given question requires understanding of the ENTIRE claim to answer correctly.

You are given:
1) The original claim (one sentence).
2) A set of minimal components that jointly reconstruct the claim.
3) One question generated to test the claim.

Rules:
- For each component, decide if a student must understand that component to answer the question correctly: "Yes" or "No".
- The question "covers the whole claim" ONLY IF every component is "Yes".
- Return ONLY valid JSON in this exact schema:
  {
    "components": [
      { "text": "<component_1>", "entailed": "Yes" },
      { "text": "<component_2>", "entailed": "No" }
    ],
    "covers_claim": true
  }
- No commentary; JSON only.
""".strip()


def _strip_code_fences(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(
            r"^```(?:json)?\s*|\s*```$",
            "",
            raw,
            flags=re.IGNORECASE | re.DOTALL,
        ).strip()
    return raw


def decompose_claim_to_components(claim: str, config: AppConfig) -> List[str]:
    messages = [
        {"role": "system", "content": DECOMP_SYSTEM},
        {"role": "user", "content": f"Claim:\n{claim}"},
    ]

    raw = call_llm_json(
        messages=messages,
        config=config,
        task_type="coverage_decomposition",
        model_name=config.coverage_model_name,
    )
    raw = _strip_code_fences(raw)
    data = json.loads(raw)

    out: List[str] = []
    seen = set()

    for c in data.get("components", []):
        if isinstance(c, str):
            t = " ".join(c.split()).strip()
            if t and t.lower() not in seen:
                out.append(t)
                seen.add(t.lower())

    return out


def check_question_covers_claim(
    claim: str,
    components: List[str],
    question: str,
    config: AppConfig,
) -> Dict[str, Any]:
    comp_json = json.dumps({"components": components}, ensure_ascii=False)

    user = (
        f"Claim:\n{claim}\n\n"
        f"Components JSON:\n{comp_json}\n\n"
        f"Question:\n{question}\n"
    )

    messages = [
        {"role": "system", "content": COVERAGE_SYSTEM},
        {"role": "user", "content": user},
    ]

    raw = call_llm_json(
        messages=messages,
        config=config,
        task_type="coverage_check",
        model_name=config.coverage_model_name,
    )
    raw = _strip_code_fences(raw)
    data = json.loads(raw)

    comps_out = []
    for i, c in enumerate(components):
        ent = False
        if "components" in data and i < len(data["components"]):
            e = data["components"][i].get("entailed", "")
            ent = isinstance(e, str) and e.strip().lower().startswith("y")

        comps_out.append({
            "text": c,
            "entailed": ent,
        })

    covers = bool(data.get("covers_claim", False))
    covers = covers and all(d["entailed"] for d in comps_out)

    return {
        "components": comps_out,
        "covers_claim": covers,
    }


def pair_is_good(items: List[Dict[str, Any]]) -> bool:
    if not items or len(items) < 2:
        return False

    for it in items:
        cov = it.get("coverage", {})
        if not isinstance(cov, dict) or not cov.get("covers_claim", False):
            return False

    return True


def apply_coverage_filter(
    record: Dict[str, Any],
    config: AppConfig,
) -> Dict[str, Any]:
    """
    Apply claim-level coverage filtering to one paper record.

    Keeps a claim only if at least `min_coverage_good_pairs_per_claim`
    QA types are fully coverage-valid.
    """
    qa_by_claim = record.get("qa_by_claim", []) or []

    kept_claim_objs: List[Dict[str, Any]] = []
    kept_claim_texts: List[str] = []

    for claim_obj in qa_by_claim:
        claim_text = (claim_obj.get("claim") or "").strip()
        if not claim_text:
            continue

        # 1) Decompose claim once
        try:
            components = decompose_claim_to_components(claim_text, config)
        except Exception as e:
            print(f"[WARN] Claim decomposition failed: {e}")
            continue

        if not components:
            continue

        # 2) Evaluate coverage of every question
        enriched_by_type: Dict[str, List[Dict[str, Any]]] = {}

        for qtype in config.all_qa_types:
            items = claim_obj.get(qtype, []) or []
            new_items = []

            for it in items:
                qtxt = (it.get("question") or "").strip()
                if not qtxt:
                    continue

                try:
                    cov = check_question_covers_claim(
                        claim=claim_text,
                        components=components,
                        question=qtxt,
                        config=config,
                    )
                except Exception:
                    cov = {
                        "components": [{"text": c, "entailed": False} for c in components],
                        "covers_claim": False,
                    }

                it["coverage"] = cov
                new_items.append(it)

            enriched_by_type[qtype] = new_items

        # 3) Evaluate which QA-type pairs are fully good
        good_pairs = {
            qt: pair_is_good(enriched_by_type[qt])
            for qt in config.all_qa_types
        }

        num_good = sum(1 for _, ok in good_pairs.items() if ok)

        # 4) Keep the claim only if enough pairs fully cover it
        if num_good >= config.min_coverage_good_pairs_per_claim:
            kept_types = {
                qt: enriched_by_type[qt]
                for qt, ok in good_pairs.items()
                if ok
            }

            kept_claim_objs.append({
                "claim": claim_text,
                "components": components,
                **kept_types,
            })
            kept_claim_texts.append(claim_text)

    return {
        "pdf_name": record.get("pdf_name"),
        "paper_title": record.get("paper_title"),
        "paper_claims": kept_claim_texts,
        "verbatim_claims": record.get("verbatim_claims", []),
        "qa_by_claim": kept_claim_objs,
    }