import os
import shutil
import copy
import time
import re
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def unlearn(
    input_path_to_unlearning_candidate_model,
    output_path_to_write_unlearned_model,
    forget_set_path,
    retain_set_path=None,  # kept only for compatibility; unused in pure SimNPO
):
    """
    Pure SimNPO implementation:
      - reference-free
      - forget-set only
      - no retain loss
      - no KL loss
    """

    # -----------------------------
    # LoRA linear layer
    # -----------------------------
    class LoRALinear(torch.nn.Module):
        def __init__(self, original: torch.nn.Linear, rank: int):
            super().__init__()

            self.original: torch.nn.Linear = original
            self.A: torch.nn.Linear = torch.nn.Linear(
                original.in_features, rank, bias=False
            )
            self.B: torch.nn.Linear = torch.nn.Linear(
                rank, original.out_features, bias=False
            )

            torch.nn.init.xavier_normal_(self.A.weight)
            torch.nn.init.zeros_(self.B.weight)

        def forward(self, x):
            return self.original(x) + self.B(self.A(x))

        def merge(self):
            # Merge LoRA weights into original weights
            self.original.weight.data += self.B.weight @ self.A.weight
            torch.nn.init.xavier_normal_(self.A.weight)
            torch.nn.init.zeros_(self.B.weight)

    # -----------------------------
    # LoRA model wrapper
    # -----------------------------
    class LoRAModel(torch.nn.Module):
        def __init__(
            self,
            model: AutoModelForCausalLM,
            rank: int = 8,
            to_adapt=("q_proj", "v_proj"),
        ):
            super().__init__()
            self._llm: AutoModelForCausalLM = model
            self._loras: list[LoRALinear] = []

            # Freeze all backbone parameters
            for param in self._llm.parameters():
                param.requires_grad = False

            # Replace target linear layers with LoRA-wrapped versions
            modules = list(self._llm.named_modules())
            for name, module in modules:
                if isinstance(module, torch.nn.Linear) and any(
                    proj in name for proj in to_adapt
                ):
                    if "." not in name:
                        continue
                    parent_name, attr_name = name.rsplit(".", 1)
                    parent_module = self._llm.get_submodule(parent_name)

                    lora = LoRALinear(module, rank)
                    self._loras.append(lora)
                    lora.A.weight.requires_grad = True
                    lora.B.weight.requires_grad = True
                    setattr(parent_module, attr_name, lora)

            self._recovery = []

        def forward(self, input_ids, **kwargs):
            return self._llm(input_ids=input_ids, **kwargs)

        def merge_loras(self):
            for lora in self._loras:
                lora.merge()

        def lora_state_dict(self) -> dict[str, torch.Tensor]:
            state = {}
            for i, lora in enumerate(self._loras):
                state[f"{i}.A"] = lora.A.weight.detach().cpu()
                state[f"{i}.B"] = lora.B.weight.detach().cpu()
            return state

        def load_lora_state_dict(self, state: dict[str, torch.Tensor]) -> None:
            for i, lora in enumerate(self._loras):
                a_key = f"{i}.A"
                b_key = f"{i}.B"
                if a_key in state and b_key in state:
                    lora.A.weight.data.copy_(state[a_key].to(lora.A.weight.device))
                    lora.B.weight.data.copy_(state[b_key].to(lora.B.weight.device))

        def extract_model(self) -> AutoModelForCausalLM:
            """
            Merge LoRA weights into the base model and restore plain Linear layers.
            """
            self._recovery = []
            modules = list(self._llm.named_modules())

            for name, module in modules:
                if isinstance(module, LoRALinear):
                    if "." not in name:
                        continue
                    parent_name, attr_name = name.rsplit(".", 1)
                    parent_module = self._llm.get_submodule(parent_name)
                    self._recovery.append(
                        (parent_module, attr_name, copy.deepcopy(module))
                    )

            self.merge_loras()

            for name, module in modules:
                if isinstance(module, LoRALinear):
                    if "." not in name:
                        continue
                    parent_name, attr_name = name.rsplit(".", 1)
                    parent_module = self._llm.get_submodule(parent_name)
                    setattr(parent_module, attr_name, module.original)

            return self._llm

        def recover_loras(self):
            for parent_module, attr_name, module in self._recovery:
                setattr(parent_module, attr_name, module)

    # -----------------------------
    # Unlearning model wrapper
    # -----------------------------
    class UnlearningModel(torch.nn.Module):
        def __init__(
            self,
            model: AutoModelForCausalLM,
            tokenizer: AutoTokenizer,
            args: SimpleNamespace,
        ):
            super().__init__()
            self._args = args
            self._device = torch.device(
                args.device
                if args.device is not None
                else "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
            self._tokenizer = tokenizer
            self._llm = LoRAModel(model, rank=args.lora_rank)

            # Optimizer only on trainable LoRA params
            trainable_params = [p for p in self.parameters() if p.requires_grad]
            self._optimizer = torch.optim.Adam(trainable_params, lr=args.learning_rate)

            self.to(self._device)

        def forward(self, input_ids, **kwargs):
            return self._llm(input_ids, **kwargs)

        def extract_model(self) -> AutoModelForCausalLM:
            return self._llm.extract_model()

        def train_step(self, inputs, answer_mask):
            """
            Pure SimNPO loss:
                L = -(2/beta) * log_sigmoid( -beta * avg_logp(y|x) - gamma )
            where avg_logp(y|x) is the average log-probability over answer tokens.
            """

            input_ids = inputs.input_ids.to(self._device)
            attention_mask = inputs.attention_mask.to(self._device)
            answer_mask = answer_mask.to(self._device)

            outputs = self._llm(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            # token-level log-probabilities for next-token prediction
            # shape: [B, T-1]
            token_logprobs = F.log_softmax(outputs.logits[:, :-1, :], dim=-1).gather(
                2, input_ids[:, 1:].unsqueeze(-1)
            ).squeeze(-1)

            # mask only the answer tokens (exclude prompt tokens)
            answer_token_mask = answer_mask[:, 1:].bool()

            # number of answer tokens per example
            token_counts = answer_token_mask.sum(dim=1).clamp(min=1)

            # average log p(y|x) over answer tokens
            avg_answer_logprob = (
                (token_logprobs * answer_token_mask).sum(dim=1) / token_counts
            )  # [B]

            # SimNPO forget loss
            simnpo_loss = (
                -(2.0 / self._args.beta)
                * F.logsigmoid(
                    -self._args.beta * avg_answer_logprob - self._args.gamma
                ).mean()
            )
            simnpo_loss = simnpo_loss.nan_to_num()

            simnpo_loss.backward()
            self._optimizer.step()
            self._optimizer.zero_grad()

            return {
                "total_loss": simnpo_loss.item(),
                "simnpo_loss": simnpo_loss.item(),
                "forget_count": input_ids.size(0),
            }

        def unlearn(
            self,
            train_data: DataLoader,
            args: SimpleNamespace,
            save_path: str,
        ):
            lora_ckpt_dir = os.path.join(save_path, "lora_checkpoints")
            os.makedirs(lora_ckpt_dir, exist_ok=True)

            def _latest_lora_checkpoint(path: str):
                files = [f for f in os.listdir(path) if re.match(r"epoch_\d+\.pt$", f)]
                if not files:
                    return None
                files.sort(key=lambda x: int(re.search(r"\d+", x).group(0)))
                return os.path.join(path, files[-1])

            resume_epoch = 0
            latest_ckpt = _latest_lora_checkpoint(lora_ckpt_dir)
            if latest_ckpt is not None:
                ckpt = torch.load(latest_ckpt, map_location=self._device)
                self._llm.load_lora_state_dict(ckpt["lora_state"])
                if "optimizer_state" in ckpt:
                    self._optimizer.load_state_dict(ckpt["optimizer_state"])
                resume_epoch = int(ckpt.get("epoch", 0))
                print(f"Resumed from {latest_ckpt} (epoch {resume_epoch})")

            train_steps = 0

            for epoch in range(resume_epoch, args.epochs):
                print(f"\nEpoch {epoch + 1}/{args.epochs}")
                self.train()

                data_and_progress = tqdm(
                    train_data,
                    desc=f"Epoch {epoch + 1}/{args.epochs}",
                    unit="batch",
                    leave=False,
                )

                running_loss = 0.0
                running_examples = 0

                for inputs, answer_mask in data_and_progress:
                    losses = self.train_step(inputs, answer_mask)

                    train_steps += 1

                    if (
                        args.lora_merge_every > 0
                        and train_steps % args.lora_merge_every == 0
                    ):
                        self._llm.merge_loras()

                    running_loss += losses["total_loss"] * losses["forget_count"]
                    running_examples += losses["forget_count"]

                    avg_loss = running_loss / max(running_examples, 1)

                    print(
                        f"total_loss={losses['total_loss']:.6f} | "
                        f"simnpo_loss={losses['simnpo_loss']:.6f} | "
                        f"forget_count={losses['forget_count']}"
                    )

                    data_and_progress.set_postfix({"avg_loss": avg_loss})

                ckpt_path = os.path.join(lora_ckpt_dir, f"epoch_{epoch + 1}.pt")
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "lora_state": self._llm.lora_state_dict(),
                        "optimizer_state": self._optimizer.state_dict(),
                    },
                    ckpt_path,
                )
                print(f"Saved LoRA checkpoint: {ckpt_path}")

    # -----------------------------
    # Data preparation
    # -----------------------------
    def prepare_forget_data(
        tokenizer: AutoTokenizer,
        forget: pd.DataFrame,
    ) -> pd.Series:
        """
        Tokenize forget examples and record answer token span.
        Expects columns: 'input', 'output'
        """

        def tokenize_function(example):
            tokenized = tokenizer(
                example["input"],
                example["output"],
                return_tensors="pt",
                max_length=2048,
                truncation=True,
            )

            # locate the answer span in token space
            output_beginning = tokenized.char_to_token(0, sequence_index=1)
            output_end = tokenized.char_to_token(
                len(example["output"]) - 1, sequence_index=1
            )

            # fallback if tokenizer cannot map chars directly
            if output_beginning is None:
                output_beginning = 0
            if output_end is None:
                output_end = tokenized["input_ids"].shape[1] - 1

            output_end = output_end + 1

            tokenized["input_ids"] = tokenized["input_ids"].squeeze(0)
            tokenized["attention_mask"] = tokenized["attention_mask"].squeeze(0)

            return tokenized, [output_beginning, output_end]

        return forget.apply(tokenize_function, axis=1)

    def prepare_loader(
        data: pd.Series,
        tokenizer: AutoTokenizer,
        args: SimpleNamespace,
        shuffle: bool = False,
    ) -> DataLoader:
        def prepare_batch(batch):
            inputs, ranges = zip(*batch)

            # pad batch
            inputs = tokenizer.pad(inputs, padding=True, return_tensors="pt")
            ranges = torch.tensor(np.asarray(ranges), dtype=torch.long)

            # answer mask = 1 only for answer tokens
            answer_mask = torch.clone(inputs.attention_mask)

            # zero out everything before answer start
            answer_mask[
                torch.arange(answer_mask.shape[1]).unsqueeze(0)
                < ranges[:, 0].unsqueeze(1)
            ] = 0

            return inputs, answer_mask

        return DataLoader(
            data,
            batch_size=args.batch_size,
            collate_fn=prepare_batch,
            shuffle=shuffle,
        )

    # -----------------------------
    # Args
    # -----------------------------
    def prepare_args() -> SimpleNamespace:
        return SimpleNamespace(
            model="7B",
            logdir="logs",
            threads=1,
            seed=42,
            device=None,          # auto-select cuda if available
            batch_size=2,
            epochs=30,
            learning_rate=2e-5,
            beta=0.3,             # SimNPO hyperparameter
            gamma=0.0,            # SimNPO margin
            lora_rank=8,
            lora_merge_every=-1,  # -1 => never merge during training
            save_model=True,
            save_logdir_name=False,
        )

    # -----------------------------
    # Saving
    # -----------------------------
    def save_checkpoint(
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        path: str,
    ):
        os.makedirs(path, exist_ok=True)
        temporary_path = os.path.join(path, "tmp")
        os.makedirs(temporary_path, exist_ok=True)

        model.save_pretrained(temporary_path)
        tokenizer.save_pretrained(temporary_path)

        for f in os.listdir(temporary_path):
            shutil.move(os.path.join(temporary_path, f), os.path.join(path, f))

        shutil.rmtree(temporary_path, ignore_errors=True)

    # -----------------------------
    # Main body
    # -----------------------------
    start_time = time.time()
    _ = start_time  # kept to parallel your original structure

    args = prepare_args()

    # reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print("Loading model/tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(
        input_path_to_unlearning_candidate_model
    )
    tokenizer = AutoTokenizer.from_pretrained(
        input_path_to_unlearning_candidate_model
    )

    # Make padding work for decoder-only models if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading forget dataset...")
    forget_train = pd.read_parquet(
        os.path.join(forget_set_path, "forget_sc_1.parquet"),
        engine="pyarrow",
    )

    print("Tokenizing forget dataset...")
    tokenized_train = prepare_forget_data(tokenizer, forget_train)
    train_loader = prepare_loader(tokenized_train, tokenizer, args, shuffle=True)

    print("Preparing model...")
    unlearn_model = UnlearningModel(
        model=model,
        tokenizer=tokenizer,
        args=args,
    )

    print("Starting SimNPO unlearning...")
    unlearn_model.unlearn(
        train_data=train_loader,
        args=args,
        save_path=output_path_to_write_unlearned_model,
    )

    print("Merging LoRA weights and saving model...")
    extracted_model: AutoModelForCausalLM = unlearn_model.extract_model()
    os.makedirs(output_path_to_write_unlearned_model, exist_ok=True)

    save_checkpoint(
        model=extracted_model,
        tokenizer=tokenizer,
        path=output_path_to_write_unlearned_model,
    )

    print(f"Done. Saved unlearned model to: {output_path_to_write_unlearned_model}")


if __name__ == "__main__":
    # HARD-CODE YOUR PATHS HERE
    LOAD_DIR = "allenai/OLMo-3-7B-Instruct"          # path to base / candidate model
    FORGET_DIR = "./data"         # contains forget_set_q1.parquet
    OUTPUT_DIR = "./simnpo_7B_sc"    # output path

    unlearn(
        input_path_to_unlearning_candidate_model=LOAD_DIR,
        output_path_to_write_unlearned_model=OUTPUT_DIR,
        forget_set_path=FORGET_DIR,
        retain_set_path=None,     # unused in pure SimNPO
    )