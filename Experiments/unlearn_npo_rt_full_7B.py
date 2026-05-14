import gc
import json
import os
import shutil
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from unlearn_gd_full_7B import _atomic_write_text, _normalize_history, _render_series_svg


def _save_training_artifacts(history: list[dict], path: str) -> None:
    os.makedirs(path, exist_ok=True)

    history = _normalize_history(history)

    history_json_path = os.path.join(path, "training_metrics.json")
    history_csv_path = os.path.join(path, "training_metrics.csv")
    history_loss_svg_path = os.path.join(path, "training_loss.svg")
    history_task_loss_svg_path = os.path.join(path, "training_task_loss.svg")
    history_task_accuracy_svg_path = os.path.join(path, "training_task_accuracy.svg")

    history_json_tmp = f"{history_json_path}.tmp"
    with open(history_json_tmp, "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    os.replace(history_json_tmp, history_json_path)

    pd.DataFrame(history).to_csv(history_csv_path, index=False)

    _atomic_write_text(
        history_loss_svg_path,
        _render_series_svg(
            history,
            [{"key": "loss", "label": "Total loss", "color": "#38bdf8"}],
            title="Training Loss",
            y_label="Loss",
            no_data_message="No loss history available",
        ),
    )
    _atomic_write_text(
        history_task_loss_svg_path,
        _render_series_svg(
            history,
            [
                {"key": "npo_loss", "label": "NPO loss", "color": "#38bdf8"},
                {"key": "retain_loss", "label": "Retain loss", "color": "#f97316"},
            ],
            title="NPO vs Retain Loss",
            y_label="Token-level loss",
            no_data_message="No NPO/retain loss history available",
        ),
    )
    _atomic_write_text(
        history_task_accuracy_svg_path,
        _render_series_svg(
            history,
            [
                {"key": "forget_accuracy", "label": "Forget accuracy", "color": "#22c55e"},
                {"key": "retain_accuracy", "label": "Retain accuracy", "color": "#ef4444"},
            ],
            title="Forget vs Retain Accuracy",
            y_label="Token accuracy",
            no_data_message="No forget/retain accuracy history available",
        ),
    )


def unlearn(
    input_path_to_unlearning_candidate_model,
    output_path_to_write_unlearned_model,
    forget_set_path,
    retain_set_path,
):
    # Full fine-tuning wrapper for an LLM.
    class UnlearningModel(torch.nn.Module):
        def __init__(
            self,
            model: AutoModelForCausalLM,
            reference_model: AutoModelForCausalLM,
            tokenizer: AutoTokenizer,
            args: SimpleNamespace,
        ):
            super().__init__()
            self._device: torch.device = torch.device(
                args.device
                if args.device is not None
                else "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
            self._dtype: torch.dtype = torch.bfloat16
            is_bf16_supported = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
            if self._device.type == "cuda" and not is_bf16_supported:
                self._dtype = torch.float16
            elif self._device.type != "cuda":
                self._dtype = torch.float32

            self._args: SimpleNamespace = args
            self._llm: AutoModelForCausalLM = model
            self._reference_model: AutoModelForCausalLM = reference_model
            self._tokenizer = tokenizer

            if hasattr(self._llm, "config"):
                self._llm.config.use_cache = False
                if getattr(self._llm.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
                    self._llm.config.pad_token_id = tokenizer.pad_token_id

            if hasattr(self._reference_model, "config"):
                self._reference_model.config.use_cache = False
                if getattr(self._reference_model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
                    self._reference_model.config.pad_token_id = tokenizer.pad_token_id

            if self._device.type == "cuda":
                if hasattr(self._llm, "gradient_checkpointing_enable"):
                    self._llm.gradient_checkpointing_enable()
                if hasattr(self._llm, "enable_input_require_grads"):
                    self._llm.enable_input_require_grads()

            self._llm = self._llm.to(device=self._device, dtype=self._dtype)
            # Keep the frozen reference copy on CPU so it does not consume GPU memory.
            self._reference_model = self._reference_model.to(device="cpu", dtype=torch.float32)
            self._reference_model.requires_grad_(False)
            self._reference_model.eval()
            for param in self._llm.parameters():
                param.requires_grad = True

            self.logdir, self._writers = args.logdir, {}

            self._optimizer = torch.optim.Adam(
                self._llm.parameters(),
                lr=args.learning_rate,
            )
            self._optimizer.zero_grad(set_to_none=True)

        def train(self, mode: bool = True):
            super().train(mode)
            self._reference_model.eval()
            self._reference_model.requires_grad_(False)
            return self

        def _move_optimizer_state_to_device(self, optimizer: torch.optim.Optimizer, device: torch.device) -> None:
            def _move_value_to_device(value):
                if torch.is_tensor(value):
                    return value.to(device, non_blocking=True)
                if isinstance(value, dict):
                    return {key: _move_value_to_device(inner_value) for key, inner_value in value.items()}
                if isinstance(value, list):
                    return [_move_value_to_device(inner_value) for inner_value in value]
                if isinstance(value, tuple):
                    return tuple(_move_value_to_device(inner_value) for inner_value in value)
                return value

            for state in optimizer.state.values():
                for key, value in list(state.items()):
                    state[key] = _move_value_to_device(value)

        def _move_value_to_cpu(self, value):
            if torch.is_tensor(value):
                return value.detach().cpu()
            if isinstance(value, dict):
                return {key: self._move_value_to_cpu(inner_value) for key, inner_value in value.items()}
            if isinstance(value, list):
                return [self._move_value_to_cpu(inner_value) for inner_value in value]
            if isinstance(value, tuple):
                return tuple(self._move_value_to_cpu(inner_value) for inner_value in value)
            return value

        def extract_model(self) -> AutoModelForCausalLM:
            return self._llm

        def unlearn(
            self,
            tokenizer: AutoTokenizer,
            train_data: DataLoader,
            args: SimpleNamespace,
            save_path: str,
            start_time: float,
        ):
            checkpoint_path = os.path.join(save_path, "checkpoint_last.pt")
            history_path = os.path.join(save_path, "training_metrics.json")
            os.makedirs(save_path, exist_ok=True)

            history: list[dict] = []
            if os.path.exists(history_path):
                try:
                    with open(history_path, "r", encoding="utf-8") as handle:
                        loaded_history = json.load(handle)
                    if isinstance(loaded_history, list):
                        history = _normalize_history(loaded_history)
                except json.JSONDecodeError:
                    history = []

            resume_epoch = 0
            if os.path.exists(checkpoint_path):
                try:
                    ckpt = torch.load(checkpoint_path, map_location="cpu")
                    self._llm.load_state_dict(ckpt["model_state"])
                    if "optimizer_state" in ckpt:
                        self._optimizer.load_state_dict(ckpt["optimizer_state"])
                        self._move_optimizer_state_to_device(self._optimizer, self._device)
                    resume_epoch = int(ckpt.get("epoch", 0))
                    del ckpt
                    gc.collect()
                    if self._device.type == "cuda":
                        torch.cuda.empty_cache()
                    print(f"Resumed from {checkpoint_path} (epoch {resume_epoch})")
                except Exception as exc:
                    print(f"[WARN] Could not load checkpoint '{checkpoint_path}': {exc}. Starting fresh.")

            train_steps = 0
            for epoch in range(resume_epoch, args.epochs):
                print("epoch: ", epoch)
                self.train()
                epoch_message = f"Epoch={epoch + 1}/{args.epochs}"
                data_and_progress = tqdm(train_data, epoch_message, unit="batch", leave=False)

                npo_loss_sum = 0.0
                retain_loss_sum = 0.0
                kl_loss_sum = 0.0
                forget_token_count = 0
                retain_token_count = 0
                forget_correct_count = 0
                retain_correct_count = 0
                epoch_batches = 0

                for inputs, answer_mask, ranges, tasks in data_and_progress:
                    inputs.input_ids = inputs.input_ids.to(self._device, non_blocking=True)
                    inputs.attention_mask = inputs.attention_mask.to(self._device, non_blocking=True)
                    answer_mask = answer_mask.to(self._device, non_blocking=True)
                    ranges = ranges.to(self._device, non_blocking=True)
                    tasks = tasks.to(self._device, non_blocking=True)

                    metrics = self.train_step(inputs, answer_mask, tasks)

                    train_steps += 1
                    epoch_batches += 1
                    npo_loss_sum += metrics["npo_loss"] * metrics["forget_tokens"]
                    retain_loss_sum += metrics["retain_loss"] * metrics["retain_tokens"]
                    kl_loss_sum += metrics["kl_retain_loss"] * metrics["retain_tokens"]
                    forget_token_count += metrics["forget_tokens"]
                    retain_token_count += metrics["retain_tokens"]
                    forget_correct_count += metrics["forget_correct"]
                    retain_correct_count += metrics["retain_correct"]

                    running_npo_loss = npo_loss_sum / max(1, forget_token_count)
                    running_retain_loss = retain_loss_sum / max(1, retain_token_count)
                    running_kl_loss = kl_loss_sum / max(1, retain_token_count)
                    running_loss = (
                        self._args.npo_mult * running_npo_loss
                        + self._args.rt_mult * running_retain_loss
                        + self._args.kl_mult * running_kl_loss
                    )
                    running_forget_accuracy = forget_correct_count / max(1, forget_token_count)
                    running_retain_accuracy = retain_correct_count / max(1, retain_token_count)

                    print(
                        f"loss={metrics['total_loss']:.4f} | "
                        f"npo_loss={metrics['npo_loss']:.4f} | "
                        f"retain_loss={metrics['retain_loss']:.4f} | "
                        f"kl_retain_loss={metrics['kl_retain_loss']:.4f} | "
                        f"forget_acc={metrics['forget_accuracy']:.3f} | "
                        f"retain_acc={metrics['retain_accuracy']:.3f} | "
                        f"forget_tokens={metrics['forget_tokens']} | "
                        f"retain_tokens={metrics['retain_tokens']}"
                    )

                    data_and_progress.set_postfix(
                        {
                            "loss": running_loss,
                            "f_acc": running_forget_accuracy,
                            "r_acc": running_retain_accuracy,
                        }
                    )

                mean_npo_loss = npo_loss_sum / max(1, forget_token_count)
                mean_retain_loss = retain_loss_sum / max(1, retain_token_count)
                mean_kl_loss = kl_loss_sum / max(1, retain_token_count)
                mean_forget_accuracy = forget_correct_count / max(1, forget_token_count)
                mean_retain_accuracy = retain_correct_count / max(1, retain_token_count)
                mean_loss = (
                    self._args.npo_mult * mean_npo_loss
                    + self._args.rt_mult * mean_retain_loss
                    + self._args.kl_mult * mean_kl_loss
                )

                history.append(
                    {
                        "epoch": epoch + 1,
                        "objective": -mean_loss,
                        "loss": mean_loss,
                        "total_loss": mean_loss,
                        "npo_loss": mean_npo_loss,
                        "forget_loss": mean_npo_loss,
                        "retain_loss": mean_retain_loss,
                        "kl_retain_loss": mean_kl_loss,
                        "forget_accuracy": mean_forget_accuracy,
                        "retain_accuracy": mean_retain_accuracy,
                        "forget_tokens": forget_token_count,
                        "retain_tokens": retain_token_count,
                        "forget_correct": forget_correct_count,
                        "retain_correct": retain_correct_count,
                        "batches": epoch_batches,
                        "elapsed_seconds": time.time() - start_time,
                    }
                )

                _save_training_artifacts(history, save_path)

                model_state_cpu = self._move_value_to_cpu(self._llm.state_dict())
                optimizer_state_cpu = self._move_value_to_cpu(self._optimizer.state_dict())
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state": model_state_cpu,
                        "optimizer_state": optimizer_state_cpu,
                    },
                    checkpoint_path,
                    _use_new_zipfile_serialization=False,
                )
                print(f"Saved last checkpoint: {checkpoint_path}")

        def train_step(self, inputs, answer_mask, tasks):
            reference_output = None
            with torch.no_grad():
                ref_input_ids = inputs.input_ids.detach().to("cpu")
                ref_attention_mask = inputs.attention_mask.detach().to("cpu")
                reference_output = self._reference_model(
                    ref_input_ids,
                    attention_mask=ref_attention_mask,
                )

            outputs = self._llm(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
            )

            ref_logprob = F.log_softmax(
                reference_output.logits[:, :-1, :], dim=-1
            ).gather(2, ref_input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
            logprob = F.log_softmax(outputs.logits[:, :-1, :], dim=-1).gather(
                2, inputs.input_ids[:, 1:].unsqueeze(-1)
            ).squeeze(-1)
            ref_logprob = ref_logprob.to(logprob.device, non_blocking=True)

            token_mask = answer_mask[:, 1:] == 1
            forget_mask = (tasks == 1).unsqueeze(1).expand_as(token_mask) & token_mask
            retain_mask = (tasks == 0).unsqueeze(1).expand_as(token_mask) & token_mask

            device = logprob.device
            if forget_mask.any():
                forget_logprob = logprob[forget_mask]
                forget_ref_logprob = ref_logprob[forget_mask]
                npo_loss: torch.Tensor = (
                    -F.logsigmoid(
                        self._args.beta * (forget_ref_logprob - forget_logprob)
                    ).mean()
                    * 2
                    / self._args.beta
                )
                npo_loss = npo_loss.nan_to_num()
            else:
                npo_loss = torch.tensor(0.0, device=device)

            if retain_mask.any():
                retain_logprob = logprob[retain_mask]
                retain_ref_logprob = ref_logprob[retain_mask]
                retain_loss = -retain_logprob.mean()
                retain_loss = retain_loss.nan_to_num()

                kl_retain_loss = F.kl_div(
                    retain_logprob,
                    retain_ref_logprob,
                    reduction="batchmean",
                    log_target=True,
                ).nan_to_num()
            else:
                retain_loss = torch.tensor(0.0, device=device)
                kl_retain_loss = torch.tensor(0.0, device=device)

            loss = (
                self._args.npo_mult * npo_loss
                + self._args.rt_mult * retain_loss
                + self._args.kl_mult * kl_retain_loss
            )

            pred_ids = outputs.logits[:, :-1, :].argmax(dim=-1)
            next_token_ids = inputs.input_ids[:, 1:]

            if forget_mask.any():
                forget_correct = (pred_ids[forget_mask] == next_token_ids[forget_mask]).sum().item()
                forget_tokens = forget_mask.sum().item()
                forget_accuracy = forget_correct / max(1, forget_tokens)
            else:
                forget_correct = 0
                forget_tokens = 0
                forget_accuracy = 0.0

            if retain_mask.any():
                retain_correct = (pred_ids[retain_mask] == next_token_ids[retain_mask]).sum().item()
                retain_tokens = retain_mask.sum().item()
                retain_accuracy = retain_correct / max(1, retain_tokens)
            else:
                retain_correct = 0
                retain_tokens = 0
                retain_accuracy = 0.0

            loss.backward()
            self._optimizer.step()
            self._optimizer.zero_grad(set_to_none=True)

            return {
                "total_loss": loss.item(),
                "loss": loss.item(),
                "objective": -loss.item(),
                "npo_loss": npo_loss.item(),
                "retain_loss": retain_loss.item(),
                "kl_retain_loss": kl_retain_loss.item(),
                "forget_loss": npo_loss.item(),
                "forget_accuracy": forget_accuracy,
                "retain_accuracy": retain_accuracy,
                "forget_tokens": forget_tokens,
                "retain_tokens": retain_tokens,
                "forget_correct": forget_correct,
                "retain_correct": retain_correct,
                "forget_count": (tasks == 1).sum().item(),
                "retain_count": (tasks == 0).sum().item(),
            }

        def forward(self, x, **xs):
            return self._llm(x, **xs)

    # prepares the dataset for training/validation by tokenizing the strings
    # and taking note of the range of tokens that contains the output
    def prepare_data(
        tokenizer: AutoTokenizer,
        retain: pd.DataFrame,
        forget: pd.DataFrame,
        args: SimpleNamespace,
    ) -> pd.Series:
        def tokenize_function(example, forget: int):
            tokenized = tokenizer(
                example["input"],
                example["output"],
                return_tensors="pt",
                max_length=args.max_length,
                truncation=True,
            )
            output_beginning = tokenized.char_to_token(0, sequence_index=1)
            output_end = (
                tokenized.char_to_token(len(example["output"]) - 1, sequence_index=1)
                + 1
            )
            tokenized["input_ids"] = tokenized["input_ids"].squeeze()
            tokenized["attention_mask"] = tokenized["attention_mask"].squeeze()
            return tokenized, [output_beginning, output_end], forget

        def prepare_dataset(retain_set, forget_set):
            tokenized_retain = retain_set.apply(tokenize_function, axis=1, args=(0,))
            tokenized_forget = forget_set.apply(tokenize_function, axis=1, args=(1,))
            tokenized = pd.concat(
                [tokenized_retain, tokenized_forget], ignore_index=True
            )
            return tokenized

        return prepare_dataset(retain, forget)

    # returns a batched DataLoader
    def prepare_loader(
        data: pd.Series,
        tokenizer: AutoTokenizer,
        args: SimpleNamespace,
        shuffle: bool = False,
    ) -> DataLoader:
        def prepare_batch(data):
            inputs, ranges, tasks = zip(*data)
            inputs = tokenizer.pad(inputs, padding=True, return_tensors="pt")

            ranges = torch.tensor(np.asarray(ranges), dtype=torch.long)
            tasks = torch.tensor(np.asarray(tasks), dtype=torch.long)

            answer_mask = torch.clone(inputs.attention_mask)
            answer_mask[
                torch.arange(answer_mask.shape[1]).unsqueeze(0)
                < ranges[:, 0].unsqueeze(1)
            ] = 0

            return (inputs, answer_mask, ranges, tasks)

        return DataLoader(
            data,
            batch_size=args.batch_size,
            collate_fn=prepare_batch,
            shuffle=shuffle,
        )

    def prepare_args() -> SimpleNamespace:
        return SimpleNamespace(
            model="7B",
            logdir="logs",
            threads=1,
            seed=42,
            device=None,
            batch_size=1,
            epochs=15,
            learning_rate=1e-5,
            max_length=2048,
            evaluate_every=-1,
            save_model=True,
            save_logdir_name=False,
            beta=0.3,
            npo_mult=1.2,
            rt_mult=0.75,
            kl_mult=0.0,
        )

    def save_checkpoint(
        model: AutoModelForCausalLM, tokenizer: AutoTokenizer, path: str
    ):
        temporary_path = os.path.join(path, "tmp")
        os.makedirs(temporary_path, exist_ok=True)
        model.save_pretrained(temporary_path, max_shard_size="20GB")
        tokenizer.save_pretrained(temporary_path)

        os.makedirs(path, exist_ok=True)
        for name in os.listdir(temporary_path):
            src = os.path.join(temporary_path, name)
            dst = os.path.join(path, name)
            if os.path.isdir(src):
                if os.path.exists(dst):
                    shutil.rmtree(dst)
                shutil.move(src, dst)
            else:
                if os.path.isdir(dst):
                    shutil.rmtree(dst)
                elif os.path.exists(dst):
                    os.remove(dst)
                os.replace(src, dst)

        shutil.rmtree(temporary_path, ignore_errors=True)

    start_time = time.time()
    args = prepare_args()

    load_dtype = torch.float32
    if torch.cuda.is_available():
        is_bf16_supported = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
        load_dtype = torch.bfloat16 if is_bf16_supported else torch.float16

    reference_model = AutoModelForCausalLM.from_pretrained(
        input_path_to_unlearning_candidate_model,
        torch_dtype=load_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        input_path_to_unlearning_candidate_model,
        torch_dtype=load_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        input_path_to_unlearning_candidate_model,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    retain_train = pd.read_parquet(
        os.path.join(retain_set_path, "retain_sc.parquet"),
        engine="pyarrow",
    )
    forget_train = pd.read_parquet(
        os.path.join(forget_set_path, "forget_sc_1.parquet"),
        engine="pyarrow",
    )

    tokenized_train = prepare_data(tokenizer, retain_train, forget_train, args)
    train_loader = prepare_loader(tokenized_train, tokenizer, args, shuffle=True)

    print("Preparing model.")
    unlearn_model = UnlearningModel(
        model=model,
        reference_model=reference_model,
        tokenizer=tokenizer,
        args=args,
    )

    print("Unlearning.")
    unlearn_model.unlearn(
        tokenizer=tokenizer,
        train_data=train_loader,
        args=args,
        save_path=output_path_to_write_unlearned_model,
        start_time=start_time,
    )

    print("Saving final model.")
    extracted_model: AutoModelForCausalLM = unlearn_model.extract_model()
    os.makedirs(output_path_to_write_unlearned_model, exist_ok=True)
    save_checkpoint(
        model=extracted_model,
        tokenizer=tokenizer,
        path=output_path_to_write_unlearned_model,
    )


if __name__ == "__main__":
    LOAD_DIR = "allenai/OLMo-3-7B-Instruct"
    RETAIN_DIR = "./data"
    FORGET_DIR = "./data"
    OUTPUT_DIR = "./npo_rt_7B_full_sc"

    unlearn(LOAD_DIR, OUTPUT_DIR, FORGET_DIR, RETAIN_DIR)