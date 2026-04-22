import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
load_dotenv()


@dataclass
class AppConfig:
    """
    Central application configuration.
    """

    semantic_scholar_api_base: str = "https://api.semanticscholar.org/graph/v1/paper"
    semantic_scholar_api_key: Optional[str] = os.getenv("S2_API_KEY")

    year_ids_dir: Path = Path("year_wise_corpus_ids")
    download_dir: Path = Path("Downloaded_paper_forget")
    mapping_file: Path = Path("corpus_title_mapping.json")

    sleep_secs: float = 1.0
    retries: int = 3
    backoff: float = 5.0

    target_years: List[int] = field(default_factory=lambda: [2022, 2021, 2020, 2019, 2018])
    # target_years: List[int] = field(default_factory=lambda: [2022])
    per_year_limit: int = 20

    min_text_chars_for_claims: int = 300

    claim_output_json: Path = Path("claims_output_forget.json")

    enable_gpt_cs_filter: bool = True

    enable_gpt_paper_type_filter: bool = True

    gpt_filter_title_only_fallback: bool = True

    required_paper_type: str = "experimental_original_research"  # e.g. "survey", "review", "empirical study"


    # Azure + LiteLLM model name
    llm_model_name: str = "azure/gpt-5"
    llm_temperature: float = 1.0
    max_chars_for_model: int = 40000

    # Azure env vars
    azure_api_key: Optional[str] = os.getenv("AZURE_API_KEY")
    azure_api_base: Optional[str] = os.getenv("AZURE_API_BASE")
    azure_api_version: Optional[str] = os.getenv("AZURE_API_VERSION")

    # Cost logging
    cost_log_dir: Path = Path("logs")
    cost_log_file: Path = Path("logs/cost_log.jsonl")

    # Pricing per 1K tokens
    price_per_1k_prompt: float = 0.00125
    price_per_1k_completion: float = 0.01


    # ---------------- OLMo ----------------
    enable_olmo_validation: bool = True
    olmo_model_name: str = "allenai/Olmo-3-7B-Instruct"
    olmo_device_map: str = "auto"
    olmo_dtype: str = "float16"   # float16 | bfloat16 | float32
    olmo_max_new_tokens: int = 12
    olmo_do_sample: bool = False
    olmo_temperature: float = 0.1
    olmo_top_p: float = 0.9
    
    # Consistency check: validate answer by running the question multiple times
    enable_olmo_consistency_check: bool = True
    olmo_consistency_runs: int = 3  # Number of times to run each question for consistency

    
    # ---------------- QA filtering / evaluation ----------------
    enable_qa_filtering: bool = True
    rouge_threshold: float = 0.70

    
    semantic_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    semantic_similarity_threshold: float = 0.80

    min_good_pairs_per_claim: int = 2
    all_qa_types: tuple = ("mcq", "true_false", "fill_blank", "assertion_reason")


    # ---------------- Coverage check ----------------
    enable_coverage_check: bool = False
    min_coverage_good_pairs_per_claim: int = 2

    # Use GPT-5 here as well (same as previous pipeline)
    coverage_model_name: str = "azure/gpt-5"

    
    # ---------------- Forget set export ----------------
    enable_forget_set_export: bool = True
    expected_final_papers: int = 50

    forget_set_out_dir: Path = Path("export_data")
    forget_set_task_name: str = "ForgetSet"
    forget_set_split_name: str = "forget"

    forget_set_q1_parquet: str = "forget_set_q1.parquet"
    forget_set_q2_parquet: str = "forget_set_q2.parquet"
    forget_set_q1_jsonl: str = "forget_set_q1.jsonl"
    forget_set_q2_jsonl: str = "forget_set_q2.jsonl"

    # ---------------- Retain set ----------------
    retain_download_dir: Path = Path("Downloaded_paper_retain")
    retain_output_json: Path = Path("retain_paper_claims.json")

    retain_top_k: int = 20
    retain_top_similar_to_show: int = 3

    # semantic similarity backend
    retain_semantic_model_name: str = "all-MiniLM-L6-v2"

    # retain-specific logging
    retain_cost_log_file: Path = Path("logs/cost_log_retain.jsonl")

    # ---------------- Retain set export ----------------
    enable_retain_set_export: bool = True
    retain_export_out_dir: Path = Path("export_data")

    retain_export_parquet_name: str = "retain_set_external.parquet"
    retain_export_jsonl_name: str = "retain_set_external.jsonl"

    retain_export_task_name: str = "RetainSet"
    retain_export_split_name: str = "retain"

    

    # ---------------- Retain set internal ----------------
    retain_internal_output_json: Path = Path("retain_set_internal.json")

    enable_retain_internal_export: bool = True
    retain_internal_export_out_dir: Path = Path("export_data")
    retain_internal_export_parquet_name: str = "retain_set_internal.parquet"
    retain_internal_export_jsonl_name: str = "retain_set_internal.jsonl"

    retain_internal_task_name: str = "RetainSetInternal"
    retain_internal_split_name: str = "retain_internal"

    # how many good QA-type pairs must survive after OLMo matching
    min_good_pairs_per_base: int = 2


    # ---------------- Derived question set ----------------
    derived_output_json: Path = Path("derived_set.json")
    enable_derived_export: bool = True
    derived_export_out_dir: Path = Path("export_data")
    derived_export_parquet_name: str = "derived_set.parquet"
    derived_export_jsonl_name: str = "derived_set.jsonl"
    derived_task_name: str = "DerivedSet"
    derived_split_name: str = "derived"

    # Source pool: use up to 8 forget-side source questions per paper
    derived_max_source_questions: int = 8

    # OLMo survival rule for derived set
    # Since we generate exactly 4 derived questions total, default is: all 4 must match
    derived_require_all_four_match: bool = False
    derived_min_matched_questions: int = 1

    
    # ---------------- Forget pruning after retain generation ----------------
    prune_forget_after_retain: bool = True

    # safer default: write a new pruned file instead of overwriting immediately
    pruned_forget_output_json: Path = Path("claims_output_forget_pruned.json")

    pruned_retain_output_json: Path = Path("retain_paper_claims_pruned.json")

    
    prune_after_derived: bool = True

    pruned_retain_internal_output_json: Path = Path("retain_set_internal_pruned.json")


    # ---------------- OLMo rejected question outputs ----------------
    forget_olmo_rejected_json: Path = Path("forget_olmo_rejected.json")
    retain_external_olmo_rejected_json: Path = Path("retain_external_olmo_rejected.json")
    retain_internal_olmo_rejected_json: Path = Path("retain_internal_olmo_rejected.json")
    derived_olmo_rejected_json: Path = Path("derived_olmo_rejected.json")


    # ---------------- Common-intersection outputs ----------------
    common_forget_output_json: Path = Path("forget_common.json")
    common_retain_output_json: Path = Path("retain_external_common.json")
    common_retain_internal_output_json: Path = Path("retain_internal_common.json")
    common_derived_output_json: Path = Path("derived_common.json")


