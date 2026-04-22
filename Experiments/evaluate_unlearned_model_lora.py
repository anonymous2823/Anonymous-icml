"""
evaluate_unlearned_model_lora.py

Extension of the existing evaluator to support two loading modes:
1) Final model path loading (default behavior)
2) Base model + LoRA checkpoint loading (optional)

If --lora_checkpoint is provided, --base_model_path is required.
If --lora_checkpoint is not provided, --model_path is used as the final model.
"""

import argparse
import gc
import os
import shutil
import time
from typing import List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluate_unlearned_model import (
    ConsolidatedReport,
    build_report,
    evaluate_jsonl_split,
    run_lm_eval,
)


LM_EVAL_TASKS = {
    "mmlu": "acc,none",
    "arc_challenge": "acc_norm,none",
    "hellaswag": "acc_norm,none",
}

BENCHMARK_ALIASES = {
    "acc_challenge": "arc_challenge",
}


class EvalLoRALinear(torch.nn.Module):
    """Minimal LoRA wrapper compatible with the training checkpoint format (i.A / i.B)."""

    def __init__(self, original: torch.nn.Linear, rank: int):
        super().__init__()
        self.original = original
        self.A = torch.nn.Linear(
            original.in_features,
            rank,
            bias=False,
            device=original.weight.device,
            dtype=original.weight.dtype,
        )
        self.B = torch.nn.Linear(
            rank,
            original.out_features,
            bias=False,
            device=original.weight.device,
            dtype=original.weight.dtype,
        )
        self._merged = False

        torch.nn.init.xavier_normal_(self.A.weight)
        torch.nn.init.zeros_(self.B.weight)

    def forward(self, x):
        if self._merged:
            return self.original(x)
        return self.original(x) + self.B(self.A(x))

    def merge_and_disable(self):
        if not self._merged:
            self.original.weight.data += self.B.weight @ self.A.weight
            self._merged = True


def _attach_custom_lora_modules(
    model: AutoModelForCausalLM,
    rank: int,
    target_modules: List[str],
) -> List[EvalLoRALinear]:
    lora_layers: List[EvalLoRALinear] = []
    modules = list(model.named_modules())

    for name, module in modules:
        if isinstance(module, torch.nn.Linear) and any(t in name for t in target_modules):
            parent_name, attr_name = name.rsplit(".", 1)
            parent_module = model.get_submodule(parent_name)
            wrapped = EvalLoRALinear(module, rank)
            lora_layers.append(wrapped)
            setattr(parent_module, attr_name, wrapped)

    if not lora_layers:
        raise RuntimeError(
            "No target linear modules were adapted for custom LoRA loading. "
            f"Targets={target_modules}"
        )

    return lora_layers


def _load_custom_epoch_lora_state(
    checkpoint_path: str,
    lora_layers: List[EvalLoRALinear],
):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(ckpt, dict) and "lora_state" in ckpt:
        state = ckpt["lora_state"]
    elif isinstance(ckpt, dict):
        state = ckpt
    else:
        raise RuntimeError(f"Unsupported checkpoint structure in {checkpoint_path}")

    loaded = 0
    loaded_indices = set()
    for i, layer in enumerate(lora_layers):
        a_key = f"{i}.A"
        b_key = f"{i}.B"
        if a_key not in state or b_key not in state:
            continue

        a_tensor = state[a_key]
        b_tensor = state[b_key]
        if a_tensor.shape != layer.A.weight.shape or b_tensor.shape != layer.B.weight.shape:
            raise RuntimeError(
                f"Shape mismatch for LoRA index {i}: "
                f"checkpoint A/B={tuple(a_tensor.shape)}/{tuple(b_tensor.shape)} vs "
                f"model A/B={tuple(layer.A.weight.shape)}/{tuple(layer.B.weight.shape)}"
            )

        layer.A.weight.data.copy_(a_tensor.to(layer.A.weight.device, dtype=layer.A.weight.dtype))
        layer.B.weight.data.copy_(b_tensor.to(layer.B.weight.device, dtype=layer.B.weight.dtype))
        loaded += 1
        loaded_indices.add(i)

    if loaded == 0:
        raise RuntimeError(
            "No LoRA layers were loaded from checkpoint. "
            "Expected keys like '0.A', '0.B', ... or a 'lora_state' mapping."
        )

    # Training saves LoRA tensors for every adapted layer index.
    # Enforce full coverage to avoid silent partial loads with very poor metrics.
    expected = len(lora_layers)
    if loaded != expected:
        missing = [str(i) for i in range(expected) if i not in loaded_indices]
        raise RuntimeError(
            "Partial custom LoRA load detected: "
            f"loaded {loaded}/{expected} layers from {checkpoint_path}. "
            f"Missing indices: {', '.join(missing[:20])}"
            + (" ..." if len(missing) > 20 else "")
        )

    print(f"[INFO] Loaded custom LoRA state for {loaded}/{len(lora_layers)} adapted layers.")


def _merge_and_strip_custom_lora(model: AutoModelForCausalLM) -> int:
    """Merge EvalLoRALinear deltas and restore original Linear modules."""
    restored = 0
    modules = list(model.named_modules())
    for name, module in modules:
        if isinstance(module, EvalLoRALinear):
            parent_name, attr_name = name.rsplit(".", 1)
            parent_module = model.get_submodule(parent_name)
            module.merge_and_disable()
            setattr(parent_module, attr_name, module.original)
            restored += 1
    return restored


def load_model_and_tokenizer_with_optional_lora(
    model_path: Optional[str],
    base_model_path: Optional[str],
    lora_checkpoint: Optional[str],
    device: str,
    merge_lora: bool,
    lora_rank: int,
    lora_target_modules: List[str],
) -> Tuple[AutoModelForCausalLM, AutoTokenizer, str]:
    """
    Load either:
    - A final model from model_path, or
    - A base model from base_model_path with LoRA adapter from lora_checkpoint.

    Returns: (model, tokenizer, report_model_label)
    """
    use_cuda = "cuda" in device
    dtype = torch.float16 if use_cuda else torch.float32

    if lora_checkpoint:
        if not base_model_path:
            raise ValueError("--base_model_path is required when --lora_checkpoint is provided.")

        print(f"[INFO] Loading base model from: {base_model_path}")
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=dtype,
            device_map="auto" if use_cuda else None,
            trust_remote_code=True,
        )

        print(f"[INFO] Attaching LoRA checkpoint from: {lora_checkpoint}")

        if os.path.isdir(lora_checkpoint):
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise ImportError(
                    "PEFT is required for adapter-directory LoRA loading. Install it with: pip install peft"
                ) from exc

            model = PeftModel.from_pretrained(base_model, lora_checkpoint)
            if merge_lora:
                print("[INFO] Merging PEFT LoRA weights into base model.")
                model = model.merge_and_unload()
        else:
            lora_layers = _attach_custom_lora_modules(base_model, lora_rank, lora_target_modules)
            _load_custom_epoch_lora_state(lora_checkpoint, lora_layers)
            if merge_lora:
                print("[INFO] Merging custom LoRA weights into base model.")
                restored = _merge_and_strip_custom_lora(base_model)
                print(f"[INFO] Restored {restored} custom LoRA-wrapped modules to plain Linear.")
            model = base_model

        if not use_cuda:
            model = model.to(device)
        model.eval()

        report_label = f"{base_model_path} + LoRA({lora_checkpoint})"
        print("[INFO] Model loaded successfully (base + LoRA).")
        return model, tokenizer, report_label

    if not model_path:
        raise ValueError("--model_path is required when --lora_checkpoint is not provided.")

    print(f"[INFO] Loading final model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto" if use_cuda else None,
        trust_remote_code=True,
    )
    if not use_cuda:
        model = model.to(device)
    model.eval()
    print("[INFO] Model loaded successfully (final model).")
    return model, tokenizer, model_path


def _prepare_temp_model_for_lm_eval(
    model,
    tokenizer: AutoTokenizer,
    output_dir: str,
) -> str:
    """Create a temporary merged model path that lm-eval can load via pretrained=..."""
    lm_eval_tmp_dir = os.path.join(output_dir, "lm_eval_temp_model")
    os.makedirs(lm_eval_tmp_dir, exist_ok=True)

    # PEFT adapter case: convert to a plain merged model before saving.
    try:
        from peft import PeftModel
    except ImportError:
        PeftModel = None

    model_to_save = model
    if PeftModel is not None and isinstance(model, PeftModel):
        print("[INFO] Merging PEFT adapter for lm-eval temporary model.")
        model_to_save = model.merge_and_unload()
    else:
        # Custom epoch_*.pt LoRA case: merge and strip wrappers to match fully saved model structure.
        restored = _merge_and_strip_custom_lora(model_to_save)
        if restored > 0:
            print(f"[INFO] Merged and restored {restored} custom LoRA modules for lm-eval save.")

    model_to_save.save_pretrained(lm_eval_tmp_dir)
    tokenizer.save_pretrained(lm_eval_tmp_dir)
    print(f"[INFO] Temporary lm-eval model written to: {lm_eval_tmp_dir}")
    return lm_eval_tmp_dir


def _flush_gpu_memory(device: str):
    """Release cached CUDA memory between evaluation phases when running on GPU."""
    if "cuda" not in device or not torch.cuda.is_available():
        return

    gc.collect()
    torch.cuda.empty_cache()
    # Safely collect inter-process cached allocations if available.
    if hasattr(torch.cuda, "ipc_collect"):
        torch.cuda.ipc_collect()
    print("[INFO] Flushed cached GPU allocations before benchmark phase.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate an unlearned LLM with optional LoRA checkpoint loading.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model_path",
        default=None,
        help=(
            "Path/ID of the final model to evaluate. "
            "Used when --lora_checkpoint is not provided. "
            "If --lora_checkpoint is provided, this can optionally be a merged model path for lm-eval benchmarks."
        ),
    )
    parser.add_argument(
        "--base_model_path",
        default=None,
        help="Base model path/ID required when using --lora_checkpoint.",
    )
    parser.add_argument(
        "--lora_checkpoint",
        default=None,
        help="Path to the LoRA checkpoint to attach to --base_model_path.",
    )
    parser.add_argument(
        "--merge_lora",
        action="store_true",
        help="Merge LoRA weights into the base model after loading.",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=8,
        help="LoRA rank for custom epoch_*.pt checkpoints (ignored for PEFT adapter directories).",
    )
    parser.add_argument(
        "--lora_target_modules",
        nargs="+",
        default=["q_proj", "v_proj"],
        help="Target module-name substrings for custom epoch_*.pt checkpoints.",
    )

    parser.add_argument("--forget_path_1", default=None, help="Path to forget split 1 (.jsonl)")
    parser.add_argument("--forget_path_2", default=None, help="Path to forget split 2 (.jsonl)")
    parser.add_argument("--retain_external", default=None, help="Path to external retain set (.jsonl)")
    parser.add_argument("--retain_internal", default=None, help="Path to internal retain set (.jsonl)")
    parser.add_argument("--derived", default=None, help="Path to derived split (.jsonl)")
    parser.add_argument("--output_dir", default="eval_results", help="Directory to save reports.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for generation.")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for inference (cuda / cpu).",
    )
    parser.add_argument("--max_new_tokens", type=int, default=64, help="Max new tokens to generate per example.")
    parser.add_argument("--no_benchmarks", action="store_true", help="Skip lm-eval benchmarks.")
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=list(LM_EVAL_TASKS.keys()),
        choices=list(LM_EVAL_TASKS.keys()),
        help="Which benchmarks to run via lm-eval.",
    )

    args = parser.parse_args()

    if args.lora_checkpoint and not args.base_model_path:
        parser.error("--base_model_path is required when --lora_checkpoint is provided.")

    if not args.lora_checkpoint and not args.model_path:
        parser.error("--model_path is required when --lora_checkpoint is not provided.")

    args.benchmarks = [BENCHMARK_ALIASES.get(task, task) for task in args.benchmarks]

    return args


def main():
    args = parse_args()
    t_start = time.time()
    temp_model_dir_for_lm_eval = None

    model, tokenizer, report_model_label = load_model_and_tokenizer_with_optional_lora(
        model_path=args.model_path,
        base_model_path=args.base_model_path,
        lora_checkpoint=args.lora_checkpoint,
        device=args.device,
        merge_lora=args.merge_lora,
        lora_rank=args.lora_rank,
        lora_target_modules=args.lora_target_modules,
    )

    report = ConsolidatedReport(
        model_path=report_model_label,
        evaluation_time_seconds=0.0,
    )

    splits_to_eval = [
        ("forget_1", args.forget_path_1),
        ("forget_2", args.forget_path_2),
        ("retain_external", args.retain_external),
        ("retain_internal", args.retain_internal),
        ("derived", args.derived),
    ]

    print("\n[PHASE 1] JSONL split evaluation")
    for split_name, path in splits_to_eval:
        result = evaluate_jsonl_split(
            split_name=split_name,
            path=path,
            model=model,
            tokenizer=tokenizer,
            device=args.device,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
        )
        if result is not None:
            report.split_results.append(result)

    _flush_gpu_memory(args.device)

    if not args.no_benchmarks:
        print("\n[PHASE 2] Standard benchmark evaluation (lm-eval)")

        try:
            import lm_eval  # noqa: F401
        except ImportError:
            msg = "lm-eval is not installed. Run: pip install lm-eval  Skipping benchmark phase."
            print(f"[WARN] {msg}")
            report.notes.append(msg)
            args.no_benchmarks = True

    if not args.no_benchmarks:
        selected_tasks = {k: v for k, v in LM_EVAL_TASKS.items() if k in args.benchmarks}

        model_path_for_benchmarks = args.model_path
        if args.lora_checkpoint and not args.model_path:
            print("[INFO] --model_path not provided in LoRA mode. Building temporary merged model for lm-eval.")
            temp_model_dir_for_lm_eval = _prepare_temp_model_for_lm_eval(
                model=model,
                tokenizer=tokenizer,
                output_dir=args.output_dir,
            )
            model_path_for_benchmarks = temp_model_dir_for_lm_eval

        bench_results = run_lm_eval(
            model_path=model_path_for_benchmarks,
            tasks=selected_tasks,
            output_dir=args.output_dir,
            device=args.device,
            batch_size=args.batch_size,
        )
        report.benchmark_results.extend(bench_results)
    else:
        report.notes.append("Benchmarks skipped (--no_benchmarks flag set or lm-eval not available).")

    report.evaluation_time_seconds = time.time() - t_start
    build_report(report, args.output_dir)

    if temp_model_dir_for_lm_eval and os.path.isdir(temp_model_dir_for_lm_eval):
        print(f"[INFO] Cleaning up temporary lm-eval model: {temp_model_dir_for_lm_eval}")
        shutil.rmtree(temp_model_dir_for_lm_eval, ignore_errors=True)


if __name__ == "__main__":
    main()
