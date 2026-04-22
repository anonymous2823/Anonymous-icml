from config import AppConfig
from utils.env_utils import validate_required_env_vars
from model.olmo_runner import OLMORunner
from derived_set.pipeline import run_derived_set_pipeline
from dataset_export.derived_set_builder import build_derived_set_export
from dataset_sync.prune_by_derived_set import prune_all_by_derived_set


def main() -> None:
    """
    Derived-question set generation entrypoint.

    Flow:
    - read final forget-side records from QA_final_covered.json
    - collect up to 8 source atomic questions per paper
    - generate exactly 4 derived questions total (1 MCQ, 1 TF, 1 Fill, 1 AR)
      that are grounded in the source question pool but conceptually more distant
      and not paraphrases
    - run OLMo on the 4 derived questions
    - keep only papers whose derived questions pass OLMo validation
    - write derived_set.json
    - export to one parquet and one jsonl
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
            print("[INFO] OLMo initialized for derived-question pipeline.")
        except Exception as e:
            print(f"[WARN] OLMo initialization failed for derived-question pipeline. Continuing without OLMo. Error: {e}")
            olmo_runner = None

    run_derived_set_pipeline(config, olmo_runner=olmo_runner)

    if config.prune_after_derived:
        try:
            sync_summary = prune_all_by_derived_set(config)
            print("[SUMMARY] Joint pruning after derived set completed successfully.")
            print(f"  Allowed anchor ids count        : {sync_summary['allowed_anchor_ids_count']}")
            print(f"  Forget original count           : {sync_summary['forget_original_count']}")
            print(f"  Forget pruned count             : {sync_summary['forget_pruned_count']}")
            print(f"  Forget removed count            : {sync_summary['forget_removed_count']}")
            print(f"  Retain external original count  : {sync_summary['retain_external_original_count']}")
            print(f"  Retain external pruned count    : {sync_summary['retain_external_pruned_count']}")
            print(f"  Retain external removed count   : {sync_summary['retain_external_removed_count']}")
            print(f"  Retain internal original count  : {sync_summary['retain_internal_original_count']}")
            print(f"  Retain internal pruned count    : {sync_summary['retain_internal_pruned_count']}")
            print(f"  Retain internal removed count   : {sync_summary['retain_internal_removed_count']}")
            print(f"  Pruned forget output            : {sync_summary['pruned_forget_output']}")
            print(f"  Pruned retain external output   : {sync_summary['pruned_retain_external_output']}")
            print(f"  Pruned retain internal output   : {sync_summary['pruned_retain_internal_output']}")
        except Exception as e:
            print(f"[WARN] Joint pruning after derived set failed: {e}")

    if config.enable_derived_export:
        try:
            export_summary = build_derived_set_export(config)
            print("[SUMMARY] Derived-set export completed successfully.")
            print(f"  Records used: {export_summary['records']}")
            print(f"  Rows written: {export_summary['rows']}")
            print(f"  Output directory: {config.derived_export_out_dir}")
        except Exception as e:
            print(f"[WARN] Derived-set export failed: {e}")


if __name__ == "__main__":
    main()