# from pathlib import Path
# from typing import Any, Dict, List, Optional

# from config import AppConfig
# from utils.json_utils import save_json
# from utils.pdf_utils import extract_text_from_pdf
# from utils.claim_utils import parse_claims_json
# from llm_client.azure_gpt5_client import call_llm_single_pass, call_llm_questions
# from model.olmo_runner import OLMORunner
# from evaluation.qa_filter import filter_record_by_olmo_with_rejected
# from evaluation.coverage_check import apply_coverage_filter
# from evaluation.survival import record_survives
# from retain_set.reference_selector import (
#     rank_references_by_similarity,
#     download_reference_candidate,
# )
# from retain_set.similarity import (
#     load_json_file,
#     flatten_claims_from_qa_records,
#     rank_retain_claims_against_qa_claims,
#     print_similarity_results,
# )


# def extract_claims_from_pdf(pdf_path: Path, corpus_id: str, config: AppConfig) -> List[str]:
#     text = extract_text_from_pdf(pdf_path)
#     if len(text) < config.min_text_chars_for_claims:
#         print(f"[WARN] Extracted text too short from {pdf_path.name}")
#         return []

#     raw = call_llm_single_pass(text, corpus_id=corpus_id, config=config)
#     try:
#         return parse_claims_json(raw)
#     except Exception as e:
#         print(f"[WARN] Failed to parse claim JSON for {pdf_path.name}: {e}")
#         return []


# def load_anchor_corpus_ids_from_qa_final(path: Path) -> List[str]:
#     data = load_json_file(path)

#     if isinstance(data, dict):
#         data = [data]

#     if not isinstance(data, list):
#         raise ValueError("QA_final_covered.json must be a list or a single dict.")

#     corpus_ids = []
#     seen = set()

#     for rec in data:
#         if not isinstance(rec, dict):
#             continue

#         pdf_name = rec.get("pdf_name")
#         if not pdf_name or not isinstance(pdf_name, str):
#             continue

#         corpus_id = Path(pdf_name).stem.strip()
#         if corpus_id and corpus_id not in seen:
#             seen.add(corpus_id)
#             corpus_ids.append(corpus_id)

#     return corpus_ids

# def attach_anchor_metadata(
#     record: Dict[str, Any],
#     anchor_corpus_id: str,
#     retain_survived: bool,
# ) -> Dict[str, Any]:
#     """
#     Ensure every retain record consistently stores:
#     - the source forget paper id
#     - the source forget pdf name
#     - whether the retain sample survived
#     """
#     record["anchor_corpus_id"] = anchor_corpus_id
#     record["anchor_forget_paper_id"] = anchor_corpus_id
#     record["anchor_forget_pdf_name"] = f"{anchor_corpus_id}.pdf"
#     record["retain_survived"] = retain_survived
#     return record


# def build_retain_record_for_candidate(
#     anchor_corpus_id: str,
#     selected: Dict[str, Any],
#     config: AppConfig,
#     olmo_runner: Optional[OLMORunner] = None,
# ) -> Dict[str, Any]:
#     """
#     Run the full retain pipeline for one already-downloaded candidate reference.
#     """
#     pdf_path = Path(selected["pdf_path"])

#     # Step 1: claims
#     claims = extract_claims_from_pdf(
#         pdf_path,
#         corpus_id=str(selected.get("corpusId") or ""),
#         config=config,
#     )

#     # Step 2: QA generation
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

#     # Step 3: OLMo answers
#     if olmo_runner is not None and qa_by_claim:
#         try:
#             qa_by_claim = olmo_runner.enrich_qa_by_claim(qa_by_claim)
#             print(f"  ✅ OLMo answers generated for retain paper {pdf_path.name}")
#         except Exception as e:
#             print(f"[WARN] OLMo enrichment failed for retain paper {pdf_path.name}: {e}")

#     # record = {
#     #     "anchor_corpus_id": anchor_corpus_id,
#     #     "selected_reference": {
#     #         "rank_within_top_k": selected.get("rank"),
#     #         "paperId": selected.get("paperId"),
#     #         "abstract": selected.get("abstract"),
#     #         "corpusId": selected.get("corpusId"),
#     #         "title": selected.get("title"),
#     #         "year": selected.get("year"),
#     #         "url": selected.get("url"),
#     #         "pdf_url": selected.get("pdf_url"),
#     #         "similarity": selected.get("similarity"),
#     #         "pdf_path": selected.get("pdf_path"),
#     #     },
#     #     "paper_title": selected.get("title"),
#     #     "pdf_name": Path(selected.get("pdf_path", "")).name,
#     #     "paper_claims": claims,
#     #     "qa_by_claim": qa_by_claim,
#     # }

#     record = {
#         "selected_reference": {
#             "rank_within_top_k": selected.get("rank"),
#             "paperId": selected.get("paperId"),
#             "abstract": selected.get("abstract"),
#             "corpusId": selected.get("corpusId"),
#             "title": selected.get("title"),
#             "year": selected.get("year"),
#             "url": selected.get("url"),
#             "pdf_url": selected.get("pdf_url"),
#             "similarity": selected.get("similarity"),
#             "pdf_path": selected.get("pdf_path"),
#         },
#         "paper_title": selected.get("title"),
#         "pdf_name": Path(selected.get("pdf_path", "")).name,
#         "paper_claims": claims,
#         "qa_by_claim": qa_by_claim,
#     }

#     record = attach_anchor_metadata(
#         record=record,
#         anchor_corpus_id=anchor_corpus_id,
#         retain_survived=True,   # provisional, final survival checked later
#     )

#     # Step 4: OLMo agreement filtering
#     if config.enable_qa_filtering and olmo_runner is not None:
#         try:
#             record = filter_record_by_olmo(record, config)
#             # record, rejected_record = filter_record_by_olmo_with_rejected(record, config)
#             record = attach_anchor_metadata(
#                 record=record,
#                 anchor_corpus_id=anchor_corpus_id,
#                 retain_survived=True,
#             )
#             print(f"  ✅ Retain QA filtered using OLMo agreement for {pdf_path.name}")
#         except Exception as e:
#             print(f"[WARN] Retain QA filtering failed for {pdf_path.name}: {e}")

#     # Step 5: coverage filtering
#     if config.enable_coverage_check:
#         try:
#             record = apply_coverage_filter(record, config)
#             record = attach_anchor_metadata(
#                 record=record,
#                 anchor_corpus_id=anchor_corpus_id,
#                 retain_survived=True,
#             )
#             print(f"  ✅ Retain coverage check applied for {pdf_path.name}")
#         except Exception as e:
#             print(f"[WARN] Retain coverage check failed for {pdf_path.name}: {e}")

#     return record


# def process_one_anchor_corpus(
#     anchor_corpus_id: str,
#     config: AppConfig,
#     olmo_runner: Optional[OLMORunner] = None,
# ) -> Dict[str, Any]:
#     """
#     For one anchor paper:
#     - rank top-K references
#     - try candidates in order
#     - keep the first candidate that SURVIVES after all filtering
#     """
#     top_refs = rank_references_by_similarity(
#         anchor_corpus_id,
#         config=config,
#         top_k=config.retain_top_k,
#     )

#     print("Top References:", top_refs)
#     if not top_refs:
#         return attach_anchor_metadata(
#             record={
#                 "error": "No ranked references available.",
#             },
#             anchor_corpus_id=anchor_corpus_id,
#             retain_survived=False,
#         )

#     # Try ranked candidates one by one
#     for rank_idx, paper in enumerate(top_refs, start=1):
#         selected = download_reference_candidate(
#             paper,
#             rank=rank_idx,
#             download_dir=config.retain_download_dir,
#             config=config,
#         )
#         if not selected:
#             continue

#         print(f"[INFO] Downloaded retain candidate at rank {rank_idx} for anchor {anchor_corpus_id}")

#         # Build record through full survival pipeline
#         record = build_retain_record_for_candidate(
#             anchor_corpus_id=anchor_corpus_id,
#             selected=selected,
#             config=config,
#             olmo_runner=olmo_runner,
#         )

#         # Survival check
#         if not record_survives(record):
#             print(
#                 f"[INFO] Retain candidate rank {rank_idx} failed survival for anchor {anchor_corpus_id}. "
#                 f"Trying next similar reference."
#             )
#             continue

#         print(
#             f"[INFO] Retain candidate rank {rank_idx} survived for anchor {anchor_corpus_id}. "
#             f"Using this paper."
#         )

#         # Only compute semantic similarity after survival succeeds
#         final_claims = record.get("paper_claims", []) or []
#         similarity_results = []

#         qa_path = config.claim_output_json
#         # qa_path = Path("sample.json")  # For testing with sample data
#         if qa_path.exists() and final_claims:
#             try:
#                 qa_data = load_json_file(qa_path)
#                 qa_claim_records = flatten_claims_from_qa_records(qa_data)

#                 if qa_claim_records:
#                     similarity_results = rank_retain_claims_against_qa_claims(
#                         retain_claims=final_claims,
#                         qa_claim_records=qa_claim_records,
#                         config=config,
#                         top_n=config.retain_top_similar_to_show,
#                     )
#                     print_similarity_results(
#                         similarity_results,
#                         top_n=config.retain_top_similar_to_show,
#                     )
#             except Exception as e:
#                 print(f"[WARN] Failed to compute semantic similarity for anchor {anchor_corpus_id}: {e}")


        
#         record["similarity_results"] = similarity_results
#         record = attach_anchor_metadata(
#             record=record,
#             anchor_corpus_id=anchor_corpus_id,
#             retain_survived=True,
#         )
#         return record


#     # If none survived
#     return attach_anchor_metadata(
#         record={
#             "error": "No candidate reference survived retain filtering.",
#         },
#         anchor_corpus_id=anchor_corpus_id,
#         retain_survived=False,
#     )


# def run_retain_pipeline(
#     config: AppConfig,
#     olmo_runner: Optional[OLMORunner] = None,
# ) -> None:
#     """
#     Orchestrate retain-set generation over all anchor papers.
#     """
#     qa_path = config.claim_output_json
#     # qa_path = Path("sample.json")  # For testing with sample data
#     if not qa_path.exists():
#         print(f"[ERROR] {qa_path} not found.")
#         return

#     config.retain_download_dir.mkdir(parents=True, exist_ok=True)

#     try:
#         anchor_corpus_ids = load_anchor_corpus_ids_from_qa_final(qa_path)
#     except Exception as e:
#         print(f"[ERROR] Failed to load anchor corpus IDs from {qa_path}: {e}")
#         return

#     if not anchor_corpus_ids:
#         print(f"[ERROR] No valid corpus IDs found in {qa_path} via pdf_name.")
#         return

#     print(f"[INFO] Found {len(anchor_corpus_ids)} anchor corpus IDs in {qa_path}")

#     all_records: List[Dict[str, Any]] = []

#     for idx, corpus_id in enumerate(anchor_corpus_ids, start=1):
#         print("\n" + "=" * 120)
#         print(f"[INFO] Processing {idx}/{len(anchor_corpus_ids)} | anchor_corpus_id={corpus_id}")
#         print("=" * 120)

#         try:
#             record = process_one_anchor_corpus(
#                 corpus_id,
#                 config=config,
#                 olmo_runner=olmo_runner,
#             )

#             all_records.append(record)
#         except Exception as e:
#             print(f"[ERROR] Unexpected failure while processing corpus_id={corpus_id}: {e}")
#             all_records.append(
#                 attach_anchor_metadata(
#                     record={
#                         "error": f"Unexpected failure: {e}",
#                     },
#                     anchor_corpus_id=corpus_id,
#                     retain_survived=False,
#                 )
#             )

#     save_json(all_records, config.retain_output_json)
#     print(f"\n[INFO] Wrote all retain records to: {config.retain_output_json}")


from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config import AppConfig
from utils.json_utils import save_json
from utils.pdf_utils import extract_text_from_pdf
from utils.claim_utils import parse_claims_json
from llm_client.azure_gpt5_client import call_llm_single_pass, call_llm_questions
from model.olmo_runner import OLMORunner
from evaluation.qa_filter import filter_record_by_olmo_with_rejected
from evaluation.coverage_check import apply_coverage_filter
from evaluation.survival import record_survives
from retain_set.reference_selector import (
    rank_references_by_similarity,
    download_reference_candidate,
)
from retain_set.similarity import (
    load_json_file,
    flatten_claims_from_qa_records,
    rank_retain_claims_against_qa_claims,
    print_similarity_results,
)


def extract_claims_from_pdf(pdf_path: Path, corpus_id: str, config: AppConfig) -> List[str]:
    text = extract_text_from_pdf(pdf_path)
    if len(text) < config.min_text_chars_for_claims:
        print(f"[WARN] Extracted text too short from {pdf_path.name}")
        return []

    raw = call_llm_single_pass(text, corpus_id=corpus_id, config=config)
    try:
        return parse_claims_json(raw)
    except Exception as e:
        print(f"[WARN] Failed to parse claim JSON for {pdf_path.name}: {e}")
        return []


def load_anchor_corpus_ids_from_qa_final(path: Path) -> List[str]:
    data = load_json_file(path)

    if isinstance(data, dict):
        data = [data]

    if not isinstance(data, list):
        raise ValueError("QA_final_covered.json must be a list or a single dict.")

    corpus_ids = []
    seen = set()

    for rec in data:
        if not isinstance(rec, dict):
            continue

        pdf_name = rec.get("pdf_name")
        if not pdf_name or not isinstance(pdf_name, str):
            continue

        corpus_id = Path(pdf_name).stem.strip()
        if corpus_id and corpus_id not in seen:
            seen.add(corpus_id)
            corpus_ids.append(corpus_id)

    return corpus_ids


def attach_anchor_metadata(
    record: Dict[str, Any],
    anchor_corpus_id: str,
    retain_survived: bool,
) -> Dict[str, Any]:
    """
    Ensure every retain record consistently stores:
    - the source forget paper id
    - the source forget pdf name
    - whether the retain sample survived
    """
    record["anchor_corpus_id"] = anchor_corpus_id
    record["anchor_forget_paper_id"] = anchor_corpus_id
    record["anchor_forget_pdf_name"] = f"{anchor_corpus_id}.pdf"
    record["retain_survived"] = retain_survived
    return record


def has_rejected_content(rejected_record: Dict[str, Any]) -> bool:
    """
    Check whether a rejected record actually contains any omitted questions.
    """
    rejected_claims = rejected_record.get("rejected_qa_by_claim", []) or []
    if not rejected_claims:
        return False

    for claim_obj in rejected_claims:
        for qtype in ["mcq", "true_false", "fill_blank", "assertion_reason"]:
            items = claim_obj.get(qtype, []) or []
            if items:
                return True

    return False


def build_retain_record_for_candidate(
    anchor_corpus_id: str,
    selected: Dict[str, Any],
    config: AppConfig,
    olmo_runner: Optional[OLMORunner] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Run the full retain pipeline for one already-downloaded candidate reference.

    Returns:
      (filtered_record, rejected_record_from_olmo)
    """
    pdf_path = Path(selected["pdf_path"])

    # Step 1: claims
    claims = extract_claims_from_pdf(
        pdf_path,
        corpus_id=str(selected.get("corpusId") or ""),
        config=config,
    )

    # Step 2: QA generation
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

    # Step 3: OLMo answers
    if olmo_runner is not None and qa_by_claim:
        try:
            qa_by_claim = olmo_runner.enrich_qa_by_claim(
                qa_by_claim,
                use_consistency_check=config.enable_olmo_consistency_check,
                num_runs=config.olmo_consistency_runs,
            )
            print(f"  ✅ OLMo answers generated for retain paper {pdf_path.name}")
        except Exception as e:
            print(f"[WARN] OLMo enrichment failed for retain paper {pdf_path.name}: {e}")

    record = {
        "selected_reference": {
            "rank_within_top_k": selected.get("rank"),
            "paperId": selected.get("paperId"),
            "abstract": selected.get("abstract"),
            "corpusId": selected.get("corpusId"),
            "title": selected.get("title"),
            "year": selected.get("year"),
            "url": selected.get("url"),
            "pdf_url": selected.get("pdf_url"),
            "similarity": selected.get("similarity"),
            "pdf_path": selected.get("pdf_path"),
        },
        "paper_title": selected.get("title"),
        "pdf_name": Path(selected.get("pdf_path", "")).name,
        "paper_claims": claims,
        "qa_by_claim": qa_by_claim,
    }

    record = attach_anchor_metadata(
        record=record,
        anchor_corpus_id=anchor_corpus_id,
        retain_survived=True,  # provisional
    )

    # Default empty rejected record
    rejected_record = attach_anchor_metadata(
        record={
            "paper_title": selected.get("title"),
            "pdf_name": Path(selected.get("pdf_path", "")).name,
            "paper_claims": claims,
            "selected_reference": {
                "rank_within_top_k": selected.get("rank"),
                "paperId": selected.get("paperId"),
                "abstract": selected.get("abstract"),
                "corpusId": selected.get("corpusId"),
                "title": selected.get("title"),
                "year": selected.get("year"),
                "url": selected.get("url"),
                "pdf_url": selected.get("pdf_url"),
                "similarity": selected.get("similarity"),
                "pdf_path": selected.get("pdf_path"),
            },
            "rejected_qa_by_claim": [],
        },
        anchor_corpus_id=anchor_corpus_id,
        retain_survived=False,
    )

    # Step 4: OLMo agreement filtering
    if config.enable_qa_filtering and olmo_runner is not None:
        try:
            record, rejected_record = filter_record_by_olmo_with_rejected(record, config)

            record = attach_anchor_metadata(
                record=record,
                anchor_corpus_id=anchor_corpus_id,
                retain_survived=True,
            )

            rejected_record = attach_anchor_metadata(
                record=rejected_record,
                anchor_corpus_id=anchor_corpus_id,
                retain_survived=False,
            )

            print(f"  ✅ Retain QA filtered using OLMo agreement for {pdf_path.name}")
        except Exception as e:
            print(f"[WARN] Retain QA filtering failed for {pdf_path.name}: {e}")

    # Step 5: coverage filtering (applies only to kept/surviving record)
    if config.enable_coverage_check:
        try:
            record = apply_coverage_filter(record, config)
            record = attach_anchor_metadata(
                record=record,
                anchor_corpus_id=anchor_corpus_id,
                retain_survived=True,
            )
            print(f"  ✅ Retain coverage check applied for {pdf_path.name}")
        except Exception as e:
            print(f"[WARN] Retain coverage check failed for {pdf_path.name}: {e}")

    return record, rejected_record


def process_one_anchor_corpus(
    anchor_corpus_id: str,
    config: AppConfig,
    olmo_runner: Optional[OLMORunner] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    For one anchor paper:
    - rank top-K references
    - try candidates in order
    - keep the first candidate that SURVIVES after all filtering

    Returns:
      (final_retain_record, rejected_records_for_this_anchor)
    """
    top_refs = rank_references_by_similarity(
        anchor_corpus_id,
        config=config,
        top_k=config.retain_top_k,
    )

    print("Top References:", top_refs)

    if not top_refs:
        return (
            attach_anchor_metadata(
                record={
                    "error": "No ranked references available.",
                },
                anchor_corpus_id=anchor_corpus_id,
                retain_survived=False,
            ),
            [],
        )

    rejected_records_for_anchor: List[Dict[str, Any]] = []

    # Try ranked candidates one by one
    for rank_idx, paper in enumerate(top_refs, start=1):
        selected = download_reference_candidate(
            paper,
            rank=rank_idx,
            download_dir=config.retain_download_dir,
            config=config,
        )
        if not selected:
            continue

        print(f"[INFO] Downloaded retain candidate at rank {rank_idx} for anchor {anchor_corpus_id}")

        # Build record through full survival pipeline
        record, rejected_record = build_retain_record_for_candidate(
            anchor_corpus_id=anchor_corpus_id,
            selected=selected,
            config=config,
            olmo_runner=olmo_runner,
        )

        # Save rejected OLMo questions for analysis
        if has_rejected_content(rejected_record):
            rejected_records_for_anchor.append(rejected_record)

        # Survival check
        if not record_survives(record):
            print(
                f"[INFO] Retain candidate rank {rank_idx} failed survival for anchor {anchor_corpus_id}. "
                f"Trying next similar reference."
            )
            continue

        print(
            f"[INFO] Retain candidate rank {rank_idx} survived for anchor {anchor_corpus_id}. "
            f"Using this paper."
        )

        # Only compute semantic similarity after survival succeeds
        final_claims = record.get("paper_claims", []) or []
        similarity_results = []

        qa_path = config.claim_output_json
        if qa_path.exists() and final_claims:
            try:
                qa_data = load_json_file(qa_path)
                qa_claim_records = flatten_claims_from_qa_records(qa_data)

                if qa_claim_records:
                    similarity_results = rank_retain_claims_against_qa_claims(
                        retain_claims=final_claims,
                        qa_claim_records=qa_claim_records,
                        config=config,
                        top_n=config.retain_top_similar_to_show,
                    )
                    print_similarity_results(
                        similarity_results,
                        top_n=config.retain_top_similar_to_show,
                    )
            except Exception as e:
                print(f"[WARN] Failed to compute semantic similarity for anchor {anchor_corpus_id}: {e}")

        record["similarity_results"] = similarity_results
        record = attach_anchor_metadata(
            record=record,
            anchor_corpus_id=anchor_corpus_id,
            retain_survived=True,
        )
        return record, rejected_records_for_anchor

    # If none survived
    return (
        attach_anchor_metadata(
            record={
                "error": "No candidate reference survived retain filtering.",
            },
            anchor_corpus_id=anchor_corpus_id,
            retain_survived=False,
        ),
        rejected_records_for_anchor,
    )


def run_retain_pipeline(
    config: AppConfig,
    olmo_runner: Optional[OLMORunner] = None,
) -> None:
    """
    Orchestrate retain-set generation over all anchor papers.
    Also saves OLMo-rejected (no-knowledge) questions into a separate JSON.
    """
    qa_path = config.claim_output_json
    if not qa_path.exists():
        print(f"[ERROR] {qa_path} not found.")
        return

    config.retain_download_dir.mkdir(parents=True, exist_ok=True)

    try:
        anchor_corpus_ids = load_anchor_corpus_ids_from_qa_final(qa_path)
    except Exception as e:
        print(f"[ERROR] Failed to load anchor corpus IDs from {qa_path}: {e}")
        return

    if not anchor_corpus_ids:
        print(f"[ERROR] No valid corpus IDs found in {qa_path} via pdf_name.")
        return

    print(f"[INFO] Found {len(anchor_corpus_ids)} anchor corpus IDs in {qa_path}")

    all_records: List[Dict[str, Any]] = []
    all_rejected_records: List[Dict[str, Any]] = []

    for idx, corpus_id in enumerate(anchor_corpus_ids, start=1):
        print("\n" + "=" * 120)
        print(f"[INFO] Processing {idx}/{len(anchor_corpus_ids)} | anchor_corpus_id={corpus_id}")
        print("=" * 120)

        try:
            record, rejected_records = process_one_anchor_corpus(
                corpus_id,
                config=config,
                olmo_runner=olmo_runner,
            )

            all_records.append(record)

            if rejected_records:
                all_rejected_records.extend(rejected_records)

        except Exception as e:
            print(f"[ERROR] Unexpected failure while processing corpus_id={corpus_id}: {e}")
            all_records.append(
                attach_anchor_metadata(
                    record={
                        "error": f"Unexpected failure: {e}",
                    },
                    anchor_corpus_id=corpus_id,
                    retain_survived=False,
                )
            )

    save_json(all_records, config.retain_output_json)
    print(f"\n[INFO] Wrote all retain records to: {config.retain_output_json}")

    rejected_out_path = getattr(
        config,
        "retain_external_olmo_rejected_json",
        Path("retain_external_olmo_rejected.json"),
    )
    save_json(all_rejected_records, rejected_out_path)
    print(f"[INFO] Wrote retain external OLMo-rejected questions to: {rejected_out_path}")