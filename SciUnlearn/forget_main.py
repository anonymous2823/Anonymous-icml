# import os
# import random
# import time
# from pathlib import Path
# from typing import Dict, Set

# from config import AppConfig
# from semantic_scholar.client import fetch_metadata
# from semantic_scholar.filters import is_computer_science_paper, is_experimental_original_research_paper
# from semantic_scholar.downloader import download_pdf
# from utils.file_utils import ensure_dir, load_corpus_ids
# from utils.json_utils import save_json
# from utils.env_utils import validate_required_env_vars
# from paper_processing.claim_pipeline import process_downloaded_pdf
# from dataset_export.forget_set_builder import build_forget_set
# from model.olmo_runner import OLMORunner


# def main() -> None:
#     """
#     Main orchestration entrypoint.

#     Flow:
#     - read year-wise corpus IDs
#     - fetch metadata
#     - filter papers
#     - download PDFs
#     - immediately extract claims and QA after each successful download
#     - optionally generate OLMo answers for each QA item
#     - save title mapping JSON
#     - save claims/QA JSON
#     """
#     config = AppConfig()

#     print("=== Semantic Scholar PDF batch checker (CS-only, year-wise separated corpus IDs) ===")

#     ensure_dir(config.download_dir)

#     # Check Azure env vars for GPT-based claim/QA generation
#     needed = ["AZURE_API_KEY", "AZURE_API_BASE", "AZURE_API_VERSION"]
#     missing = validate_required_env_vars(needed)
#     if missing:
#         print(f"[ERROR] Missing environment variable(s): {', '.join(missing)}")
#         print("        Set them and rerun.")
#         return

#     # Initialize OLMo once, only if enabled
#     olmo_runner = None
#     if config.enable_olmo_validation:
#         try:
#             olmo_runner = OLMORunner(config)
#         except Exception as e:
#             print(f"[WARN] OLMo initialization failed. Continuing without OLMo. Error: {e}")
#             olmo_runner = None

#     print(f"[INFO] Reading year-separated corpus ID files from: {config.year_ids_dir}")
#     print(f"[INFO] Target plan:")
#     for year in config.target_years:
#         print(f"       - {config.per_year_limit} downloadable Computer Science papers from {year}")

#     checked = 0
#     total_downloaded = 0
#     year_download_counts: Dict[int, int] = {year: 0 for year in config.target_years}
#     downloaded_ids: Set[int] = set()
#     corpus_title_json = {}

#     with open(config.claim_output_json, "w", encoding="utf-8") as out:
#         out.write("[\n")
#         first_record = True

#         for target_year in config.target_years:
#             print(f"\n{'=' * 72}")
#             print(f"[YEAR START] Collecting {config.per_year_limit} downloadable CS papers from {target_year}")
#             print(f"{'=' * 72}")

#             year_file = Path(config.year_ids_dir) / f"corpus_ids_{target_year}.txt"

#             if not year_file.exists():
#                 print(f"[WARN] Year file not found: {year_file}. Skipping {target_year}.")
#                 continue

#             try:
#                 year_ids = load_corpus_ids(str(year_file))
#             except Exception as e:
#                 print(f"[WARN] Could not load IDs for {target_year}: {e}")
#                 continue

#             print(f"[INFO] Loaded {len(year_ids)} corpus IDs for {target_year}")

#             random.shuffle(year_ids)

#             for corpus_id in year_ids:
#                 if year_download_counts[target_year] >= config.per_year_limit:
#                     print(f"[YEAR DONE] Reached {config.per_year_limit} downloadable papers for {target_year}.")
#                     break

#                 if corpus_id in downloaded_ids:
#                     continue

#                 checked += 1
#                 print(f"\n[CHECK {checked}] CorpusId={corpus_id} | TargetYear={target_year}")

#                 meta = fetch_metadata(corpus_id=corpus_id, config=config)
#                 if not meta:
#                     time.sleep(config.sleep_secs)
#                     continue

#                 year = meta.get("year")
#                 if year != target_year:
#                     print(f"  ❌ Skipped (metadata year={year}, expected={target_year})")
#                     time.sleep(config.sleep_secs)
#                     continue

#                 if not is_computer_science_paper(meta, config):
#                     print("  ❌ Skipped (not Computer Science domain)")
#                     time.sleep(config.sleep_secs)
#                     continue

#                 if not is_experimental_original_research_paper(meta, config):
#                     print("  ❌ Skipped (not experimental original research paper type)")
#                     time.sleep(config.sleep_secs)
#                     continue

#                 title = meta.get("title") or "(unknown title)"
#                 is_open = meta.get("isOpenAccess", False)
#                 pdf_url = (meta.get("openAccessPdf") or {}).get("url")

#                 print(f"  ✅ Year: {year}")
#                 print(f"  Title: {title}")
#                 print(f"  Open Access: {is_open}")
#                 print(f"  Domain: Computer Science")
#                 print(f"  PDF: {pdf_url or 'None'}")

#                 if not pdf_url:
#                     print("  ❌ No open-access PDF")
#                     time.sleep(config.sleep_secs)
#                     continue

#                 pdf_path = config.download_dir / f"{corpus_id}.pdf"
#                 ok = download_pdf(pdf_url, pdf_path)

#                 if ok:
#                     corpus_title_json[str(corpus_id)] = {
#                         "title": title,
#                         "year": year,
#                     }
#                     downloaded_ids.add(corpus_id)

#                     print(f"  ✅ PDF downloaded: {pdf_path.name}")

#                     first_record, record_written = process_downloaded_pdf(
#                         pdf_path=pdf_path,
#                         corpus_id=corpus_id,
#                         title=title,
#                         out=out,
#                         first_record=first_record,
#                         config=config,
#                         olmo_runner=olmo_runner,
#                     )

#                     # ----------------------------------------------------------
#                     # Count only if the paper was actually written to final JSON
#                     # ----------------------------------------------------------
#                     if record_written:
#                         year_download_counts[target_year] += 1
#                         total_downloaded += 1
#                         print(f"  ✅ Counted ({year_download_counts[target_year]}/{config.per_year_limit} for {target_year})")
#                     else:
#                         print("  ⚠️ PDF downloaded but not counted (did not survive final checks)")
#                 else:
#                     print("  ❌ Download failed")
#                 time.sleep(config.sleep_secs)

#             if year_download_counts[target_year] < config.per_year_limit:
#                 print(
#                     f"[WARN] Could only download {year_download_counts[target_year]}/{config.per_year_limit} "
#                     f"Computer Science papers for {target_year} after exhausting all IDs in {year_file}."
#                 )

#         out.write("\n]\n")

#     save_json(corpus_title_json, config.mapping_file)

#     print(f"\n[SUMMARY]")
#     print(f"  Checked: {checked}")
#     for year in config.target_years:
#         print(f"  Downloaded from {year}: {year_download_counts[year]}")
#     print(f"  Total downloaded: {total_downloaded}")
#     print(f"  Claims/QA JSON written to: {config.claim_output_json}")

    
#     # ----------------------------------------------------------
#     if config.enable_forget_set_export:
#         try:
#             export_summary = build_forget_set(config)
#             print("[SUMMARY] Forget set export completed successfully.")
#             print(f"  Papers used: {export_summary['papers']}")
#             print(f"  Q1 rows: {export_summary['q1_rows']}")
#             print(f"  Q2 rows: {export_summary['q2_rows']}")
#             print(f"  Output directory: {config.forget_set_out_dir}")
#         except Exception as e:
#             print(f"[WARN] Forget set export failed: {e}")



# if __name__ == "__main__":
#     main()




import os
import random
import time
from pathlib import Path
from typing import Dict, Set

from config import AppConfig
from semantic_scholar.client import fetch_metadata
from semantic_scholar.filters import is_computer_science_paper, is_experimental_original_research_paper
from semantic_scholar.downloader import download_pdf
from utils.file_utils import ensure_dir, load_corpus_ids
from utils.json_utils import save_json
from utils.env_utils import validate_required_env_vars
from paper_processing.claim_pipeline import process_downloaded_pdf
from dataset_export.forget_set_builder import build_forget_set
from model.olmo_runner import OLMORunner


def has_rejected_forget_content(rejected_record: dict) -> bool:
    """
    Check whether the forget rejected record actually contains omitted QA items.
    Expected structure:
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


def main() -> None:
    """
    Main orchestration entrypoint.

    Flow:
    - read year-wise corpus IDs
    - fetch metadata
    - filter papers
    - download PDFs
    - immediately extract claims and QA after each successful download
    - optionally generate OLMo answers for each QA item
    - save title mapping JSON
    - save claims/QA JSON
    - save rejected-by-OLMo forget QA in a separate JSON
    """
    config = AppConfig()

    print("=== Semantic Scholar PDF batch checker (CS-only, year-wise separated corpus IDs) ===")

    ensure_dir(config.download_dir)

    # Check Azure env vars for GPT-based claim/QA generation
    needed = ["AZURE_API_KEY", "AZURE_API_BASE", "AZURE_API_VERSION"]
    missing = validate_required_env_vars(needed)
    if missing:
        print(f"[ERROR] Missing environment variable(s): {', '.join(missing)}")
        print("        Set them and rerun.")
        return

    # Initialize OLMo once, only if enabled
    olmo_runner = None
    if config.enable_olmo_validation:
        try:
            olmo_runner = OLMORunner(config)
        except Exception as e:
            print(f"[WARN] OLMo initialization failed. Continuing without OLMo. Error: {e}")
            olmo_runner = None

    print(f"[INFO] Reading year-separated corpus ID files from: {config.year_ids_dir}")
    print(f"[INFO] Target plan:")
    for year in config.target_years:
        print(f"       - {config.per_year_limit} downloadable Computer Science papers from {year}")

    checked = 0
    total_downloaded = 0
    year_download_counts: Dict[int, int] = {year: 0 for year in config.target_years}
    downloaded_ids: Set[int] = set()
    corpus_title_json = {}
    all_rejected_records = []

    with open(config.claim_output_json, "w", encoding="utf-8") as out:
        out.write("[\n")
        first_record = True

        for target_year in config.target_years:
            print(f"\n{'=' * 72}")
            print(f"[YEAR START] Collecting {config.per_year_limit} downloadable CS papers from {target_year}")
            print(f"{'=' * 72}")

            year_file = Path(config.year_ids_dir) / f"corpus_ids_{target_year}.txt"

            if not year_file.exists():
                print(f"[WARN] Year file not found: {year_file}. Skipping {target_year}.")
                continue

            try:
                year_ids = load_corpus_ids(str(year_file))
            except Exception as e:
                print(f"[WARN] Could not load IDs for {target_year}: {e}")
                continue

            print(f"[INFO] Loaded {len(year_ids)} corpus IDs for {target_year}")

            random.shuffle(year_ids)

            for corpus_id in year_ids:
                if year_download_counts[target_year] >= config.per_year_limit:
                    print(f"[YEAR DONE] Reached {config.per_year_limit} downloadable papers for {target_year}.")
                    break

                if corpus_id in downloaded_ids:
                    continue

                checked += 1
                print(f"\n[CHECK {checked}] CorpusId={corpus_id} | TargetYear={target_year}")

                meta = fetch_metadata(corpus_id=corpus_id, config=config)
                if not meta:
                    time.sleep(config.sleep_secs)
                    continue

                year = meta.get("year")
                if year != target_year:
                    print(f"  ❌ Skipped (metadata year={year}, expected={target_year})")
                    time.sleep(config.sleep_secs)
                    continue

                if not is_computer_science_paper(meta, config):
                    print("  ❌ Skipped (not Computer Science domain)")
                    time.sleep(config.sleep_secs)
                    continue

                if not is_experimental_original_research_paper(meta, config):
                    print("  ❌ Skipped (not experimental original research paper type)")
                    time.sleep(config.sleep_secs)
                    continue

                title = meta.get("title") or "(unknown title)"
                is_open = meta.get("isOpenAccess", False)
                pdf_url = (meta.get("openAccessPdf") or {}).get("url")

                print(f"  ✅ Year: {year}")
                print(f"  Title: {title}")
                print(f"  Open Access: {is_open}")
                print(f"  Domain: Computer Science")
                print(f"  PDF: {pdf_url or 'None'}")

                if not pdf_url:
                    print("  ❌ No open-access PDF")
                    time.sleep(config.sleep_secs)
                    continue

                pdf_path = config.download_dir / f"{corpus_id}.pdf"
                ok = download_pdf(pdf_url, pdf_path)

                if ok:
                    corpus_title_json[str(corpus_id)] = {
                        "title": title,
                        "year": year,
                    }
                    downloaded_ids.add(corpus_id)

                    print(f"  ✅ PDF downloaded: {pdf_path.name}")

                    first_record, record_written, rejected_record = process_downloaded_pdf(
                        pdf_path=pdf_path,
                        corpus_id=corpus_id,
                        title=title,
                        out=out,
                        first_record=first_record,
                        config=config,
                        olmo_runner=olmo_runner,
                    )

                    if rejected_record is not None and has_rejected_forget_content(rejected_record):
                        all_rejected_records.append(rejected_record)
                        print(f"  ✅ Rejected QA captured for {pdf_path.name}")

                    # ----------------------------------------------------------
                    # Count only if the paper was actually written to final JSON
                    # ----------------------------------------------------------
                    if record_written:
                        year_download_counts[target_year] += 1
                        total_downloaded += 1
                        print(f"  ✅ Counted ({year_download_counts[target_year]}/{config.per_year_limit} for {target_year})")
                    else:
                        print("  ⚠️ PDF downloaded but not counted (did not survive final checks)")
                else:
                    print("  ❌ Download failed")

                time.sleep(config.sleep_secs)

            if year_download_counts[target_year] < config.per_year_limit:
                print(
                    f"[WARN] Could only download {year_download_counts[target_year]}/{config.per_year_limit} "
                    f"Computer Science papers for {target_year} after exhausting all IDs in {year_file}."
                )

        out.write("\n]\n")

    # Save main mapping JSON
    save_json(corpus_title_json, config.mapping_file)

    # Save all rejected forget QA once at the end
    save_json(all_rejected_records, config.forget_olmo_rejected_json)
    print(f"[INFO] Forget OLMo-rejected JSON written to: {config.forget_olmo_rejected_json}")

    print(f"\n[SUMMARY]")
    print(f"  Checked: {checked}")
    for year in config.target_years:
        print(f"  Downloaded from {year}: {year_download_counts[year]}")
    print(f"  Total downloaded: {total_downloaded}")
    print(f"  Claims/QA JSON written to: {config.claim_output_json}")
    print(f"  Rejected forget QA count: {len(all_rejected_records)}")

    # ----------------------------------------------------------
    # Optional forget-set export
    # ----------------------------------------------------------
    if config.enable_forget_set_export:
        try:
            export_summary = build_forget_set(config)
            print("[SUMMARY] Forget set export completed successfully.")
            print(f"  Papers used: {export_summary['papers']}")
            print(f"  Q1 rows: {export_summary['q1_rows']}")
            print(f"  Q2 rows: {export_summary['q2_rows']}")
            print(f"  Output directory: {config.forget_set_out_dir}")
        except Exception as e:
            print(f"[WARN] Forget set export failed: {e}")


if __name__ == "__main__":
    main()