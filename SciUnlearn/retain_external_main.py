from config import AppConfig
from utils.env_utils import validate_required_env_vars
from model.olmo_runner import OLMORunner
from retain_set.pipeline import run_retain_pipeline
from dataset_export.retain_set_builder import build_retain_set_export
from dataset_sync.prune_forget_by_retain import prune_forget_json_by_retain


def main() -> None:
    """
    Retain-set generation entrypoint.

    Flow:
    - read anchor corpus IDs from QA_final_covered.json
    - find top-K similar references
    - download first valid retain reference PDF
    - extract claims + generate QA
    - generate OLMo answers
    - filter by OLMo agreement
    - filter by coverage
    - compute semantic similarity against forget-set claims
    - write retain_paper_claims.json
    """
    config = AppConfig()

    needed = ["AZURE_API_KEY", "AZURE_API_BASE", "AZURE_API_VERSION"]
    missing = validate_required_env_vars(needed)
    if missing:
        print(f"[ERROR] Missing environment variable(s): {', '.join(missing)}")
        print("        Set them and rerun.")
        return

    # Optional: use a separate log file for retain
    config.cost_log_file = config.retain_cost_log_file

    olmo_runner = None
    if config.enable_olmo_validation:
        try:
            olmo_runner = OLMORunner(config)
            print("[INFO] OLMo initialized for retain pipeline.")
        except Exception as e:
            print(f"[WARN] OLMo initialization failed for retain pipeline. Continuing without OLMo. Error: {e}")
            olmo_runner = None

    run_retain_pipeline(config, olmo_runner=olmo_runner)

    if config.prune_forget_after_retain:
        try:
            prune_summary = prune_forget_json_by_retain(config)
            print("[SUMMARY] Forget JSON pruning completed successfully.")
            print(f"  Original forget records : {prune_summary['original_count']}")
            print(f"  Pruned forget records   : {prune_summary['pruned_count']}")
            print(f"  Removed forget records  : {prune_summary['removed_count']}")
            print(f"  Final output path       : {prune_summary['final_output_path']}")
        except Exception as e:
            print(f"[WARN] Forget JSON pruning after retain failed: {e}")

    
    if config.enable_retain_set_export:
        try:
            export_summary = build_retain_set_export(config)
            print("[SUMMARY] Retain export completed successfully.")
            print(f"  Records used: {export_summary['records']}")
            print(f"  Rows written: {export_summary['rows']}")
            print(f"  Output directory: {config.retain_export_out_dir}")
        except Exception as e:
            print(f"[WARN] Retain export failed: {e}")



if __name__ == "__main__":
    main()