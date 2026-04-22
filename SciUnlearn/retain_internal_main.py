from config import AppConfig
from utils.env_utils import validate_required_env_vars
from model.olmo_runner import OLMORunner
from retain_set_internal.pipeline import run_retain_internal_pipeline
from dataset_export.retain_set_internal_builder import build_retain_internal_export
from dataset_sync.prune_forget_and_retain_external import prune_after_internal_retain


def main() -> None:
    """
    Internal retain-set generation entrypoint.

    Flow:
    - read forget-side accepted papers from QA_final_covered.json
    - open the original forget PDFs
    - generate broad paper-base QA using GPT-5
    - generate OLMo answers and filter by answer agreement
    - keep only surviving papers
    - write retain_set_internal.json
    - export to parquet + jsonl
    """
    config = AppConfig()

    needed = ["AZURE_API_KEY", "AZURE_API_BASE", "AZURE_API_VERSION"]
    missing = validate_required_env_vars(needed)
    if missing:
        print(f"[ERROR] Missing environment variable(s): {', '.join(missing)}")
        print("        Set them and rerun.")
        return

    olmo_runner = None
    if config.enable_olmo_validation:
        try:
            olmo_runner = OLMORunner(config)
            print("[INFO] OLMo initialized for internal retain pipeline.")
        except Exception as e:
            print(f"[WARN] OLMo initialization failed for internal retain pipeline. Continuing without OLMo. Error: {e}")
            olmo_runner = None

    run_retain_internal_pipeline(config, olmo_runner=olmo_runner)

    try:
        sync_summary = prune_after_internal_retain(config)
        print("[SUMMARY] Joint pruning after internal retain completed successfully.")
        print(f"  Forget original count         : {sync_summary['forget_original_count']}")
        print(f"  Forget pruned count           : {sync_summary['forget_pruned_count']}")
        print(f"  Forget removed count          : {sync_summary['forget_removed_count']}")
        print(f"  Retain external original count: {sync_summary['retain_external_original_count']}")
        print(f"  Retain external pruned count  : {sync_summary['retain_external_pruned_count']}")
        print(f"  Retain external removed count : {sync_summary['retain_external_removed_count']}")
        print(f"  Pruned forget output          : {sync_summary['pruned_forget_output']}")
        print(f"  Pruned retain output          : {sync_summary['pruned_retain_output']}")
    except Exception as e:
        print(f"[WARN] Joint pruning after internal retain failed: {e}")


    if config.enable_retain_internal_export:
        try:
            export_summary = build_retain_internal_export(config)
            print("[SUMMARY] Retain internal export completed successfully.")
            print(f"  Records used: {export_summary['records_total']}")
            print(f"  Records survived: {export_summary['records_survived']}")
            print(f"  Records failed: {export_summary['records_failed']}")
            print(f"  Rows written: {export_summary['rows']}")
            print(f"  Output directory: {config.retain_internal_export_out_dir}")
        except Exception as e:
            print(f"[WARN] Retain internal export failed: {e}")


if __name__ == "__main__":
    main()