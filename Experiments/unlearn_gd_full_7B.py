import os
import gc
import json
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


def _atomic_write_text(path: str, text: str) -> None:
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        handle.write(text)
    os.replace(tmp_path, path)


def _render_loss_svg(history: list[dict]) -> str:
    if not history:
        return (
            "<svg xmlns='http://www.w3.org/2000/svg' width='900' height='420' viewBox='0 0 900 420'>"
            "<rect width='100%' height='100%' fill='#0b1020'/>"
            "<text x='450' y='210' fill='#cbd5e1' font-family='sans-serif' font-size='20' text-anchor='middle'>"
            "No loss history available"
            "</text></svg>"
        )

    width = 900
    height = 420
    padding_left = 72
    padding_right = 24
    padding_top = 36
    padding_bottom = 64
    plot_width = width - padding_left - padding_right
    plot_height = height - padding_top - padding_bottom

    epochs = [int(row["epoch"]) for row in history]
    losses = [float(row["loss"]) for row in history]
    min_loss = min(losses)
    max_loss = max(losses)
    if max_loss == min_loss:
        max_loss += 1.0
        min_loss -= 1.0

    def x_for_index(index: int) -> float:
        if len(losses) == 1:
            return padding_left + plot_width / 2.0
        return padding_left + (index * plot_width / (len(losses) - 1))

    def y_for_loss(loss: float) -> float:
        return padding_top + (max_loss - loss) * plot_height / (max_loss - min_loss)

    x_points = [x_for_index(index) for index in range(len(losses))]
    y_points = [y_for_loss(loss) for loss in losses]
    polyline_points = " ".join(f"{x:.2f},{y:.2f}" for x, y in zip(x_points, y_points))

    grid_lines = []
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        loss_value = max_loss - fraction * (max_loss - min_loss)
        y = y_for_loss(loss_value)
        grid_lines.append(
            f"<line x1='{padding_left}' y1='{y:.2f}' x2='{width - padding_right}' y2='{y:.2f}' stroke='#1f2a44' stroke-width='1' stroke-dasharray='4 4' />"
        )
        grid_lines.append(
            f"<text x='{padding_left - 12}' y='{y + 4:.2f}' fill='#94a3b8' font-family='sans-serif' font-size='11' text-anchor='end'>{loss_value:.4f}</text>"
        )

    x_labels = []
    label_step = max(1, len(epochs) // 8)
    for index, epoch in enumerate(epochs):
        if index % label_step == 0 or index == len(epochs) - 1:
            x = x_for_index(index)
            x_labels.append(
                f"<text x='{x:.2f}' y='{height - 22}' fill='#94a3b8' font-family='sans-serif' font-size='11' text-anchor='middle'>{epoch}</text>"
            )
            x_labels.append(
                f"<line x1='{x:.2f}' y1='{padding_top}' x2='{x:.2f}' y2='{height - padding_bottom}' stroke='#1f2a44' stroke-width='1' stroke-dasharray='4 4' />"
            )

    point_nodes = [
        f"<circle cx='{x:.2f}' cy='{y:.2f}' r='3.5' fill='#38bdf8' stroke='#0f172a' stroke-width='1' />"
        for x, y in zip(x_points, y_points)
    ]

    title = (
        f"Loss curve over {len(history)} epoch(s); lower is better because the trainer maximizes the unlearning objective, so loss = -objective."
    )

    return f"""<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>
  <rect width='100%' height='100%' fill='#0b1020'/>
  <text x='{width / 2:.1f}' y='24' fill='#e2e8f0' font-family='sans-serif' font-size='16' text-anchor='middle'>{title}</text>
  <text x='18' y='{height / 2:.1f}' fill='#94a3b8' font-family='sans-serif' font-size='12' transform='rotate(-90 18 {height / 2:.1f})'>Loss = -objective</text>
  <text x='{width / 2:.1f}' y='{height - 8}' fill='#94a3b8' font-family='sans-serif' font-size='12' text-anchor='middle'>Epoch</text>
  <rect x='{padding_left}' y='{padding_top}' width='{plot_width}' height='{plot_height}' fill='#111827' stroke='#24314f' stroke-width='1.2' rx='8' />
  {''.join(grid_lines)}
  <line x1='{padding_left}' y1='{padding_top}' x2='{padding_left}' y2='{height - padding_bottom}' stroke='#94a3b8' stroke-width='1.2' />
  <line x1='{padding_left}' y1='{height - padding_bottom}' x2='{width - padding_right}' y2='{height - padding_bottom}' stroke='#94a3b8' stroke-width='1.2' />
  <polyline fill='none' stroke='#38bdf8' stroke-width='2.5' stroke-linecap='round' stroke-linejoin='round' points='{polyline_points}' />
  {''.join(point_nodes)}
  {''.join(x_labels)}
</svg>"""


def _render_series_svg(
    history: list[dict],
    series_specs: list[dict],
    title: str,
    y_label: str,
    no_data_message: str,
) -> str:
    valid_rows = []
    for row in history:
        epoch = row.get("epoch")
        if epoch is None:
            continue

        values = []
        missing_value = False
        for spec in series_specs:
            value = row.get(spec["key"])
            if value is None:
                missing_value = True
                break
            values.append(float(value))

        if not missing_value:
            valid_rows.append((int(epoch), values))

    if not valid_rows:
        return (
            "<svg xmlns='http://www.w3.org/2000/svg' width='900' height='420' viewBox='0 0 900 420'>"
            "<rect width='100%' height='100%' fill='#0b1020'/>"
            f"<text x='450' y='210' fill='#cbd5e1' font-family='sans-serif' font-size='20' text-anchor='middle'>{no_data_message}</text>"
            "</svg>"
        )

    width = 900
    height = 420
    padding_left = 72
    padding_right = 24
    padding_top = 40
    padding_bottom = 64
    plot_width = width - padding_left - padding_right
    plot_height = height - padding_top - padding_bottom

    epochs = [epoch for epoch, _ in valid_rows]
    min_epoch = min(epochs)
    max_epoch = max(epochs)

    series_values = {
        spec["key"]: [values[index] for _, values in valid_rows]
        for index, spec in enumerate(series_specs)
    }
    all_values = [value for values in series_values.values() for value in values]
    min_value = min(all_values)
    max_value = max(all_values)

    if max_value == min_value:
        padding = 1.0 if max_value == 0 else max(0.05, abs(max_value) * 0.1)
        min_value -= padding
        max_value += padding
    else:
        span = max_value - min_value
        min_value -= span * 0.08
        max_value += span * 0.08

    def x_for_epoch(epoch: int) -> float:
        if max_epoch == min_epoch:
            return padding_left + plot_width / 2.0
        return padding_left + ((epoch - min_epoch) * plot_width / (max_epoch - min_epoch))

    def y_for_value(value: float) -> float:
        return padding_top + (max_value - value) * plot_height / (max_value - min_value)

    grid_lines = []
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        value = max_value - fraction * (max_value - min_value)
        y = y_for_value(value)
        grid_lines.append(
            f"<line x1='{padding_left}' y1='{y:.2f}' x2='{width - padding_right}' y2='{y:.2f}' stroke='#1f2a44' stroke-width='1' stroke-dasharray='4 4' />"
        )
        grid_lines.append(
            f"<text x='{padding_left - 12}' y='{y + 4:.2f}' fill='#94a3b8' font-family='sans-serif' font-size='11' text-anchor='end'>{value:.4f}</text>"
        )

    x_labels = []
    label_step = max(1, len(epochs) // 8)
    for index, epoch in enumerate(epochs):
        if index % label_step == 0 or index == len(epochs) - 1:
            x = x_for_epoch(epoch)
            x_labels.append(
                f"<text x='{x:.2f}' y='{height - 22}' fill='#94a3b8' font-family='sans-serif' font-size='11' text-anchor='middle'>{epoch}</text>"
            )

    chart_elements = []
    for spec in series_specs:
        values = series_values[spec["key"]]
        points = " ".join(
            f"{x_for_epoch(epoch):.2f},{y_for_value(value):.2f}"
            for epoch, value in zip(epochs, values)
        )
        chart_elements.append(
            f"<polyline fill='none' stroke='{spec['color']}' stroke-width='2.5' stroke-linecap='round' stroke-linejoin='round' points='{points}' />"
        )
        chart_elements.extend(
            f"<circle cx='{x_for_epoch(epoch):.2f}' cy='{y_for_value(value):.2f}' r='3.5' fill='{spec['color']}' stroke='#0f172a' stroke-width='1' />"
            for epoch, value in zip(epochs, values)
        )

    legend_items = []
    legend_x = width - padding_right - 190
    legend_y = padding_top - 18
    for index, spec in enumerate(series_specs):
        y = legend_y + index * 20
        legend_items.append(
            f"<line x1='{legend_x}' y1='{y}' x2='{legend_x + 18}' y2='{y}' stroke='{spec['color']}' stroke-width='3' stroke-linecap='round' />"
        )
        legend_items.append(
            f"<text x='{legend_x + 24}' y='{y + 4}' fill='#e2e8f0' font-family='sans-serif' font-size='12'>{spec['label']}</text>"
        )

    return f"""<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>
  <rect width='100%' height='100%' fill='#0b1020'/>
  <text x='{width / 2:.1f}' y='24' fill='#e2e8f0' font-family='sans-serif' font-size='16' text-anchor='middle'>{title}</text>
  <text x='18' y='{height / 2:.1f}' fill='#94a3b8' font-family='sans-serif' font-size='12' transform='rotate(-90 18 {height / 2:.1f})'>{y_label}</text>
  <text x='{width / 2:.1f}' y='{height - 8}' fill='#94a3b8' font-family='sans-serif' font-size='12' text-anchor='middle'>Epoch</text>
  <rect x='{padding_left}' y='{padding_top}' width='{plot_width}' height='{plot_height}' fill='#111827' stroke='#24314f' stroke-width='1.2' rx='8' />
  {''.join(grid_lines)}
  <line x1='{padding_left}' y1='{padding_top}' x2='{padding_left}' y2='{height - padding_bottom}' stroke='#94a3b8' stroke-width='1.2' />
  <line x1='{padding_left}' y1='{height - padding_bottom}' x2='{width - padding_right}' y2='{height - padding_bottom}' stroke='#94a3b8' stroke-width='1.2' />
  {''.join(chart_elements)}
  {''.join(legend_items)}
  {''.join(x_labels)}
</svg>"""


def _normalize_history(history: list[dict]) -> list[dict]:
    defaults = {
        "forget_loss": None,
        "retain_loss": None,
        "forget_accuracy": None,
        "retain_accuracy": None,
        "forget_tokens": None,
        "retain_tokens": None,
        "forget_correct": None,
        "retain_correct": None,
    }
    normalized_by_epoch: dict[int, dict] = {}

    for row in history:
        epoch = row.get("epoch")
        if epoch is None:
            continue

        normalized_row = dict(row)
        for key, default in defaults.items():
            normalized_row.setdefault(key, default)

        normalized_by_epoch[int(epoch)] = normalized_row

    return [normalized_by_epoch[epoch] for epoch in sorted(normalized_by_epoch)]


def _save_training_artifacts(history: list[dict], path: str) -> None:
    os.makedirs(path, exist_ok=True)

    history = _normalize_history(history)

    history_json_path = os.path.join(path, "training_metrics.json")
    history_csv_path = os.path.join(path, "training_metrics.csv")
    history_svg_path = os.path.join(path, "training_loss.svg")
    history_task_loss_svg_path = os.path.join(path, "training_task_loss.svg")
    history_task_accuracy_svg_path = os.path.join(path, "training_task_accuracy.svg")

    history_json_tmp = f"{history_json_path}.tmp"
    with open(history_json_tmp, "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    os.replace(history_json_tmp, history_json_path)

    pd.DataFrame(history).to_csv(history_csv_path, index=False)
    _atomic_write_text(history_svg_path, _render_loss_svg(history))
    _atomic_write_text(
        history_task_loss_svg_path,
        _render_series_svg(
            history,
            [
                {"key": "forget_loss", "label": "Forget loss", "color": "#38bdf8"},
                {"key": "retain_loss", "label": "Retain loss", "color": "#f97316"},
            ],
            title="Forget vs Retain Loss",
            y_label="Token-level cross-entropy",
            no_data_message="No forget/retain loss history available",
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
            self._tokenizer = tokenizer

            if hasattr(self._llm, "config"):
                self._llm.config.use_cache = False
                if getattr(self._llm.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
                    self._llm.config.pad_token_id = tokenizer.pad_token_id

            if self._device.type == "cuda":
                if hasattr(self._llm, "gradient_checkpointing_enable"):
                    self._llm.gradient_checkpointing_enable()
                if hasattr(self._llm, "enable_input_require_grads"):
                    self._llm.enable_input_require_grads()

            self._llm = self._llm.to(dtype=self._dtype)
            for param in self._llm.parameters():
                param.requires_grad = True

            self.logdir, self._writers = args.logdir, {}
            self.to(self._device)

            self._optimizer = torch.optim.Adam(
                self._llm.parameters(),
                lr=args.learning_rate,
                maximize=True,
            )
            self._optimizer.zero_grad(set_to_none=True)

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

                running_objective = 0.0
                forget_loss_sum = 0.0
                retain_loss_sum = 0.0
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
                    running_objective += metrics["objective"]
                    forget_loss_sum += metrics["forget_loss"] * metrics["forget_tokens"]
                    retain_loss_sum += metrics["retain_loss"] * metrics["retain_tokens"]
                    forget_token_count += metrics["forget_tokens"]
                    retain_token_count += metrics["retain_tokens"]
                    forget_correct_count += metrics["forget_correct"]
                    retain_correct_count += metrics["retain_correct"]

                    running_forget_loss = forget_loss_sum / max(1, forget_token_count)
                    running_retain_loss = retain_loss_sum / max(1, retain_token_count)
                    running_forget_accuracy = forget_correct_count / max(1, forget_token_count)
                    running_retain_accuracy = retain_correct_count / max(1, retain_token_count)
                    running_epoch_objective = self._args.fg_w * running_forget_loss - self._args.rt_w * running_retain_loss

                    print(
                        f"obj={metrics['objective']:.4f} | "
                        f"forget_loss={metrics['forget_loss']:.4f} | "
                        f"retain_loss={metrics['retain_loss']:.4f} | "
                        f"forget_acc={metrics['forget_accuracy']:.3f} | "
                        f"retain_acc={metrics['retain_accuracy']:.3f} | "
                        f"forget_tokens={metrics['forget_tokens']} | "
                        f"retain_tokens={metrics['retain_tokens']}"
                    )

                    data_and_progress.set_postfix(
                        {
                            "loss": -running_epoch_objective,
                            "f_acc": running_forget_accuracy,
                            "r_acc": running_retain_accuracy,
                        }
                    )

                mean_forget_loss = forget_loss_sum / max(1, forget_token_count)
                mean_retain_loss = retain_loss_sum / max(1, retain_token_count)
                mean_forget_accuracy = forget_correct_count / max(1, forget_token_count)
                mean_retain_accuracy = retain_correct_count / max(1, retain_token_count)
                mean_objective = self._args.fg_w * mean_forget_loss - self._args.rt_w * mean_retain_loss
                mean_loss = -mean_objective
                history.append(
                    {
                        "epoch": epoch + 1,
                        "objective": mean_objective,
                        "loss": mean_loss,
                        "forget_loss": mean_forget_loss,
                        "retain_loss": mean_retain_loss,
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

            pass

        def train_step(self, inputs, answer_mask, tasks):
            """
            Gradient Difference Unlearning:
            objective = fg_w * CE_forget  -  rt_w * CE_retain
            with Adam(maximize=True):
            - CE_forget term goes UP   -> forget probs go DOWN (unlearn)
            - CE_retain term goes DOWN -> retain probs go UP (preserve ability)
            """

            outputs = self._llm(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
            )

            # token log-probs for next tokens: [B, T-1]
            next_token_logits = outputs.logits[:, :-1, :]
            logprob_all = F.log_softmax(outputs.logits[:, :-1, :], dim=-1).gather(
                2, inputs.input_ids[:, 1:].unsqueeze(-1)
            ).squeeze(-1)
            pred_ids = next_token_logits.argmax(dim=-1)

            # Build masks over the answer region
            token_mask = answer_mask[:, 1:] == 1
            f_mask_ex = (tasks == 1).unsqueeze(1).expand_as(token_mask)
            r_mask_ex = (tasks == 0).unsqueeze(1).expand_as(token_mask)
            f_mask = token_mask & f_mask_ex
            r_mask = token_mask & r_mask_ex

            device = logprob_all.device
            if f_mask.any():
                CE_forget = (-logprob_all[f_mask]).mean()
            else:
                CE_forget = torch.tensor(0.0, device=device)

            if r_mask.any():
                CE_retain = (-logprob_all[r_mask]).mean()
            else:
                CE_retain = torch.tensor(0.0, device=device)

            if f_mask.any():
                forget_correct = (pred_ids[f_mask] == inputs.input_ids[:, 1:][f_mask]).sum().item()
                forget_tokens = f_mask.sum().item()
                forget_accuracy = forget_correct / max(1, forget_tokens)
            else:
                forget_correct = 0
                forget_tokens = 0
                forget_accuracy = 0.0

            if r_mask.any():
                retain_correct = (pred_ids[r_mask] == inputs.input_ids[:, 1:][r_mask]).sum().item()
                retain_tokens = r_mask.sum().item()
                retain_accuracy = retain_correct / max(1, retain_tokens)
            else:
                retain_correct = 0
                retain_tokens = 0
                retain_accuracy = 0.0

            objective = self._args.fg_w * CE_forget - self._args.rt_w * CE_retain
            objective.backward()
            self._optimizer.step()
            self._optimizer.zero_grad(set_to_none=True)

            avg_p_forget = torch.exp(-CE_forget).item() if f_mask.any() else 0.0
            avg_p_retain = torch.exp(-CE_retain).item() if r_mask.any() else 0.0

            return {
                "objective": objective.item(),
                "CE_forget": CE_forget.item() if f_mask.any() else 0.0,
                "CE_retain": CE_retain.item() if r_mask.any() else 0.0,
                "forget_loss": CE_forget.item() if f_mask.any() else 0.0,
                "retain_loss": CE_retain.item() if r_mask.any() else 0.0,
                "forget_accuracy": forget_accuracy,
                "retain_accuracy": retain_accuracy,
                "forget_tokens": forget_tokens,
                "retain_tokens": retain_tokens,
                "forget_correct": forget_correct,
                "retain_correct": retain_correct,
                "p_forget": avg_p_forget,
                "p_retain": avg_p_retain,
                "forget_count": (tasks == 1).sum().item(),
                "retain_count": (tasks == 0).sum().item(),
            }

        def forward(self, x, **xs):
            return self._llm(x, **xs)

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
            epochs=30,
            learning_rate=1e-5,
            max_length=2048,
            evaluate_every=-1,
            save_model=True,
            save_logdir_name=False,
            fg_w=1.2,
            rt_w=1.5,
        )

    def save_checkpoint(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, path: str):
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
        os.path.join(forget_set_path, "forget_sc_2.parquet"),
        engine="pyarrow",
    )

    tokenized_train = prepare_data(tokenizer, retain_train, forget_train, args)
    train_loader = prepare_loader(tokenized_train, tokenizer, args, shuffle=True)

    print("Preparing model.")
    unlearn_model = UnlearningModel(
        model=model,
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
    OUTPUT_DIR = "./gd_7B_full_sc_plot_1"

    unlearn(LOAD_DIR, OUTPUT_DIR, FORGET_DIR, RETAIN_DIR)