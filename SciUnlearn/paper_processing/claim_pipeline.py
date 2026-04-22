# import json
# from pathlib import Path
# from typing import Optional, TextIO, Tuple

# from config import AppConfig
# from utils.pdf_utils import extract_text_from_pdf
# from utils.claim_utils import parse_claims_json
# from utils.verbatim_claim_utils import verify_verbatim_claims
# from llm_client.azure_gpt5_client import (
#     call_llm_single_pass,
#     call_llm_questions,
#     call_llm_verbatim_claims,
# )
# from model.olmo_runner import OLMORunner
# from evaluation.qa_filter import filter_record_by_olmo_with_rejected
# from evaluation.coverage_check import apply_coverage_filter
# from utils.json_utils import load_json, save_json


# def process_downloaded_pdf(
#     pdf_path: Path,
#     corpus_id: int,
#     title: str,
#     out: TextIO,
#     first_record: bool,
#     config: AppConfig,
#     olmo_runner: Optional[OLMORunner] = None,
# ) -> Tuple[bool, bool]:
#     """
#     Process one downloaded PDF immediately:
#     - extract full text
#     - generate paraphrased claims
#     - generate verbatim claims from full paper text
#       (GPT identifies abstract/introduction/conclusion itself)
#     - verify verbatim claims against paper text
#     - generate QA for paraphrased claims
#     - optionally answer each QA item with OLMo
#     - optionally filter QA by OLMo agreement
#     - optionally apply coverage check
#     - append one JSON record only if final filtered content survives

#     Returns:
#         (first_record, record_written)
#     """
#     try:
#         text = extract_text_from_pdf(pdf_path)
#     except Exception as e:
#         print(f"[ERROR] Failed to extract text from {pdf_path.name}: {e}")
#         return first_record, False

#     # ----------------------------------------------------------
#     # Existing: paraphrased claim extraction
#     # ----------------------------------------------------------
#     if len(text) < config.min_text_chars_for_claims:
#         claims = []
#     else:
#         try:
#             claim_raw = call_llm_single_pass(text, corpus_id, config)
#             claims = parse_claims_json(claim_raw)
#         except Exception as e:
#             print(f"[WARN] Claim extraction failed for {pdf_path.name}: {e}")
#             claims = []

#     # ----------------------------------------------------------
#     # New: verbatim claim extraction directly from full text
#     # GPT decides whether a claim belongs to abstract/introduction/conclusion
#     # ----------------------------------------------------------
#     verbatim_claims = []
#     try:
#         verbatim_payload = call_llm_verbatim_claims(
#             paper_title=title,
#             full_text=text,
#             config=config,
#         )
#         raw_verbatim = verbatim_payload.get("verbatim_claims", []) or []

#         if isinstance(raw_verbatim, list):
#             verbatim_claims = verify_verbatim_claims(
#                 claims=raw_verbatim,
#                 full_text=text,
#             )
#     except Exception as e:
#         print(f"[WARN] Verbatim claim extraction failed for {pdf_path.name}: {e}")
#         verbatim_claims = []

#     # ----------------------------------------------------------
#     # Existing: QA generation for paraphrased claims
#     # ----------------------------------------------------------
#     qa_by_claim = []
#     for claim in claims:
#         try:
#             qa = call_llm_questions(claim, config)
#             qa_by_claim.append({
#                 "claim": claim,
#                 **qa,
#             })
#         except Exception as e:
#             qa_by_claim.append({
#                 "claim": claim,
#                 "mcq": [],
#                 "true_false": [],
#                 "fill_blank": [],
#                 "assertion_reason": [],
#                 "error": f"QA generation failed: {e}",
#             })

#     # OLMo enrichment
#     if olmo_runner is not None and qa_by_claim:
#         try:
#             qa_by_claim = olmo_runner.enrich_qa_by_claim(qa_by_claim)
#             print(f"  ✅ OLMo answers generated for {pdf_path.name}")
#         except Exception as e:
#             print(f"[WARN] OLMo enrichment failed for {pdf_path.name}: {e}")

#     # Build raw record
#     record = {
#         "pdf_name": pdf_path.name,
#         "paper_title": title,
#         "paper_claims": claims,              # paraphrased claims
#         "verbatim_claims": verbatim_claims,  # exact copied claims + verification flags
#         "qa_by_claim": qa_by_claim,
#     }

#     # OLMo agreement filtering
#     if config.enable_qa_filtering and olmo_runner is not None:
#         try:
#             # record = filter_record_by_olmo(record, config)
#             record, rejected_record = filter_record_by_olmo_with_rejected(record, config)
#             save_json(rejected_record, config.forget_olmo_rejected_json)
#             print(f"  ✅ QA filtered using OLMo agreement for {pdf_path.name}")
#         except Exception as e:
#             print(f"[WARN] QA filtering failed for {pdf_path.name}: {e}")

#     # Coverage check filtering
#     if config.enable_coverage_check:
#         try:
#             record = apply_coverage_filter(record, config)
#             print(f"  ✅ Coverage check applied for {pdf_path.name}")
#         except Exception as e:
#             print(f"[WARN] Coverage check failed for {pdf_path.name}: {e}")

#     # ----------------------------------------------------------
#     # Survival check
#     # ----------------------------------------------------------
#     has_final_content = bool(record.get("qa_by_claim")) and bool(record.get("paper_claims"))

#     if not has_final_content:
#         print(f"  ❌ {pdf_path.name} did not survive final filtering; not added to output JSON")
#         return first_record, False

#     if not first_record:
#         out.write(",\n")

#     json.dump(record, out, ensure_ascii=False, indent=2)
#       # Save the OLMo-rejected record for analysis
    
#     out.flush()

#     print(f"  ✅ Final record saved for {pdf_path.name}")
#     return False, True




import json
from pathlib import Path
from typing import Optional, TextIO, Tuple, Dict, Any

from config import AppConfig
from utils.pdf_utils import extract_text_from_pdf
from utils.claim_utils import parse_claims_json
from utils.verbatim_claim_utils import verify_verbatim_claims
from llm_client.azure_gpt5_client import (
    call_llm_single_pass,
    call_llm_questions,
    call_llm_verbatim_claims,
)
from model.olmo_runner import OLMORunner
from evaluation.qa_filter import filter_record_by_olmo_with_rejected
from evaluation.coverage_check import apply_coverage_filter


def has_rejected_forget_content(rejected_record: Dict[str, Any]) -> bool:
    """
    Check whether the forget rejected record actually contains any omitted QA items.
    Structure expected:
      rejected_record["rejected_qa_by_claim"] = [
        {
          "claim": "...",
          "mcq": [...],
          "true_false": [...],
          "fill_blank": [...],
          "assertion_reason": [...]
        },
        ...
      ]
    """
    groups = rejected_record.get("rejected_qa_by_claim", []) or []
    if not groups:
        return False

    for claim_obj in groups:
        if not isinstance(claim_obj, dict):
            continue

        for qtype in ["mcq", "true_false", "fill_blank", "assertion_reason"]:
            items = claim_obj.get(qtype, []) or []
            if items:
                return True

    return False


def process_downloaded_pdf(
    pdf_path: Path,
    corpus_id: int,
    title: str,
    out: TextIO,
    first_record: bool,
    config: AppConfig,
    olmo_runner: Optional[OLMORunner] = None,
) -> Tuple[bool, bool, Optional[Dict[str, Any]]]:
    """
    Process one downloaded PDF immediately:
    - extract full text
    - generate paraphrased claims
    - generate verbatim claims from full paper text
    - verify verbatim claims against paper text
    - generate QA for paraphrased claims
    - optionally answer each QA item with OLMo
    - optionally filter QA by OLMo agreement
    - optionally apply coverage check
    - append one JSON record only if final filtered content survives

    Returns:
        (first_record, record_written, rejected_record)

    rejected_record is returned separately so the caller can accumulate all rejected
    records and save them once at the end.
    """
    try:
        text = extract_text_from_pdf(pdf_path)
    except Exception as e:
        print(f"[ERROR] Failed to extract text from {pdf_path.name}: {e}")
        return first_record, False, None

    # ----------------------------------------------------------
    # Paraphrased claim extraction
    # ----------------------------------------------------------
    if len(text) < config.min_text_chars_for_claims:
        claims = []
    else:
        try:
            claim_raw = call_llm_single_pass(text, corpus_id, config)
            claims = parse_claims_json(claim_raw)
        except Exception as e:
            print(f"[WARN] Claim extraction failed for {pdf_path.name}: {e}")
            claims = []

    # ----------------------------------------------------------
    # Verbatim claim extraction from full text
    # ----------------------------------------------------------
    verbatim_claims = []
    try:
        verbatim_payload = call_llm_verbatim_claims(
            paper_title=title,
            full_text=text,
            config=config,
        )
        raw_verbatim = verbatim_payload.get("verbatim_claims", []) or []

        if isinstance(raw_verbatim, list):
            verbatim_claims = verify_verbatim_claims(
                claims=raw_verbatim,
                full_text=text,
            )
    except Exception as e:
        print(f"[WARN] Verbatim claim extraction failed for {pdf_path.name}: {e}")
        verbatim_claims = []

    # ----------------------------------------------------------
    # QA generation for paraphrased claims
    # ----------------------------------------------------------
    qa_by_claim = []
    for claim in claims:
        try:
            qa = call_llm_questions(claim, config)
            qa_by_claim.append({
                "claim": claim,
                **qa,
            })
        except Exception as e:
            qa_by_claim.append({
                "claim": claim,
                "mcq": [],
                "true_false": [],
                "fill_blank": [],
                "assertion_reason": [],
                "error": f"QA generation failed: {e}",
            })

    # ----------------------------------------------------------
    # OLMo enrichment
    # ----------------------------------------------------------
    if olmo_runner is not None and qa_by_claim:
        try:
            qa_by_claim = olmo_runner.enrich_qa_by_claim(
                qa_by_claim,
                use_consistency_check=config.enable_olmo_consistency_check,
                num_runs=config.olmo_consistency_runs,
            )
            print(f"  ✅ OLMo answers generated for {pdf_path.name}")
        except Exception as e:
            print(f"[WARN] OLMo enrichment failed for {pdf_path.name}: {e}")

    # ----------------------------------------------------------
    # Build raw record
    # ----------------------------------------------------------
    record = {
        "pdf_name": pdf_path.name,
        "paper_title": title,
        "paper_claims": claims,              # paraphrased claims
        "verbatim_claims": verbatim_claims,  # exact copied claims
        "qa_by_claim": qa_by_claim,
    }

    rejected_record: Optional[Dict[str, Any]] = None

    # ----------------------------------------------------------
    # OLMo agreement filtering
    # ----------------------------------------------------------
    if config.enable_qa_filtering and olmo_runner is not None:
        try:
            record, rejected_record = filter_record_by_olmo_with_rejected(record, config)
            print(f"  ✅ QA filtered using OLMo agreement for {pdf_path.name}")
        except Exception as e:
            print(f"[WARN] QA filtering failed for {pdf_path.name}: {e}")
            rejected_record = None

    # ----------------------------------------------------------
    # Coverage check filtering
    # ----------------------------------------------------------
    if config.enable_coverage_check:
        try:
            record = apply_coverage_filter(record, config)
            print(f"  ✅ Coverage check applied for {pdf_path.name}")
        except Exception as e:
            print(f"[WARN] Coverage check failed for {pdf_path.name}: {e}")

    # ----------------------------------------------------------
    # Survival check
    # ----------------------------------------------------------
    has_final_content = bool(record.get("qa_by_claim")) and bool(record.get("paper_claims"))

    if not has_final_content:
        print(f"  ❌ {pdf_path.name} did not survive final filtering; not added to output JSON")
        return first_record, False, rejected_record

    # ----------------------------------------------------------
    # Write surviving main record
    # ----------------------------------------------------------
    if not first_record:
        out.write(",\n")

    json.dump(record, out, ensure_ascii=False, indent=2)
    out.flush()

    print(f"  ✅ Final record saved for {pdf_path.name}")
    return False, True, rejected_record
