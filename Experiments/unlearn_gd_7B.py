import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm
import os
import shutil
# import argparse  # removed
import pandas as pd
import numpy as np
import copy
import time
import re
from types import SimpleNamespace  # replacement for argparse.Namespace

def unlearn(
    input_path_to_unlearning_candidate_model,
    output_path_to_write_unlearned_model,
    forget_set_path,
    retain_set_path,
):
    # A LoRA adapted linear layer
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

            self._only_backbone = False

            torch.nn.init.xavier_normal_(self.A.weight)
            torch.nn.init.zeros_(self.B.weight)

        def forward(self, x):
            if self._only_backbone:
                return self.original(x)
            else:
                return self.original(x) + self.B(self.A(x))

        def merge(self):
            self.original.weight.data += self.B.weight @ self.A.weight
            torch.nn.init.xavier_normal_(self.A.weight)
            torch.nn.init.zeros_(self.B.weight)

    # A LoRA adapted LLM
    class LoRAModel(torch.nn.Module):
        def __init__(
            self,
            model: AutoModelForCausalLM,
            rank: int = 2,
            to_adapt: list[str] = ["q_proj", "v_proj"],
        ):
            super().__init__()
            self._llm: AutoModelForCausalLM = model
            self._loras: list[LoRALinear] = []

            # freeze all parameters of the llm
            for param in self._llm.parameters():
                param.requires_grad = False

            # change selected projections to LoRALinear
            modules = list(self._llm.named_modules())
            for name, module in modules:
                if isinstance(module, torch.nn.Linear) and any(
                    proj in name for proj in to_adapt
                ):
                    parent_name, attr_name = name.rsplit(".", 1)
                    parent_module = self._llm.get_submodule(parent_name)

                    lora = LoRALinear(module, rank)
                    self._loras.append(lora)
                    lora.A.weight.requires_grad = True
                    lora.B.weight.requires_grad = True
                    setattr(parent_module, attr_name, lora)

        def only_backbone(self, only_backbone: bool) -> None:
            for lora in self._loras:
                lora._only_backbone = only_backbone

        def forward(self, x, **xs):
            return self._llm(x, **xs)

        def merge_loras(self) -> None:
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

        # Merges all LoRA adapters into the base weights
        # converts all LoRA layers back to Linear layers
        # and returns the converted model
        def extract_model(self) -> AutoModelForCausalLM:
            self._recovery = []
            modules = list(self._llm.named_modules())

            for name, module in modules:
                if isinstance(module, LoRALinear):
                    parent_name, attr_name = name.rsplit(".", 1)
                    parent_module = self._llm.get_submodule(parent_name)
                    self._recovery.append((parent_module, attr_name, copy.deepcopy(module)))

            self.merge_loras()

            for name, module in modules:
                if isinstance(module, LoRALinear):
                    parent_name, attr_name = name.rsplit(".", 1)
                    parent_module = self._llm.get_submodule(parent_name)
                    setattr(parent_module, attr_name, module.original)

            return self._llm

        def recover_loras(self):
            for parent_module, attr_name, module in self._recovery:
                setattr(parent_module, attr_name, module)

    # unlearning wrapper for an LLM
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
            self._args: SimpleNamespace = args
            self._llm: LoRAModel = LoRAModel(model, args.lora_rank)
            self._tokenizer = tokenizer

            self.logdir, self._writers = args.logdir, {}

            self._optimizer = torch.optim.Adam(self.parameters(), lr=args.learning_rate, maximize=True,)
            self.to(self._device)

        def extract_model(self) -> AutoModelForCausalLM:
            return self._llm.extract_model()

        def unlearn(
            self,
            tokenizer: AutoTokenizer,
            train_data: DataLoader,
            args: SimpleNamespace,
            save_path: str,
            start_time: float,
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
                print("epoch: ", epoch)
                self.train()
                epoch_message = f"Epoch={epoch + 1}/{args.epochs}"
                data_and_progress = tqdm(train_data, epoch_message, unit="batch", leave=False)

                running_objective = 0.0
                forget_count = 0
                retain_count = 0

                for inputs, answer_mask, ranges, tasks in data_and_progress:
                    inputs.input_ids = inputs.input_ids.to(self._device)
                    inputs.attention_mask = inputs.attention_mask.to(self._device)
                    answer_mask = answer_mask.to(self._device)
                    ranges = ranges.to(self._device)
                    tasks = tasks.to(self._device)

                    # returns only total_loss, forget_count, retain_count
                    metrics = self.train_step(inputs, answer_mask, tasks)

                    train_steps += 1
                    if (args.lora_merge_every > 0) and (train_steps % args.lora_merge_every == 0):
                        self._llm.merge_loras()

                    running_objective += metrics["objective"]
                    forget_count += metrics["forget_count"]
                    retain_count += metrics["retain_count"]

                    print(   
                        f"obj={metrics['objective']:.4f} | "
                        f"CE_f={metrics['CE_forget']:.4f} (p≈{metrics['p_forget']:.3f}) | "
                        f"CE_r={metrics['CE_retain']:.4f} (p≈{metrics['p_retain']:.3f}) | "
                        f"forget_count={metrics['forget_count']} | "
                        f"retain_count={metrics['retain_count']}"
                    )

                    denom = max(1, (forget_count + retain_count))
                    data_and_progress.set_postfix({"loss": running_objective / denom})

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

            pass

        def train_step(self, inputs, answer_mask, tasks):
            """
            Gradient Difference Unlearning:
            objective = fg_w * CE_forget  -  rt_w * CE_retain
            with Adam(maximize=True):
            - CE_forget term goes UP   -> forget probs go DOWN (unlearn)
            - CE_retain term goes DOWN -> retain probs go UP (preserve ability)
            """

            # Forward with LoRA active
            self._llm.only_backbone(False)
            outputs = self._llm(
                torch.as_tensor(inputs.input_ids),
                attention_mask=torch.as_tensor(inputs.attention_mask),
            )

            # token log-probs for next tokens: [B, T-1]
            logprob_all = F.log_softmax(outputs.logits[:, :-1, :], dim=-1).gather(
                2, inputs.input_ids[:, 1:].unsqueeze(-1)
            ).squeeze(-1)

            # Build masks over the answer region
            token_mask = (answer_mask[:, 1:] == 1)                  # [B, T-1]
            f_mask_ex = (tasks == 1).unsqueeze(1).expand_as(token_mask)
            r_mask_ex = (tasks == 0).unsqueeze(1).expand_as(token_mask)
            f_mask = token_mask & f_mask_ex                          # forget tokens
            r_mask = token_mask & r_mask_ex                          # retain tokens

            # Cross-entropy = -log p (mean over selected tokens)
            device = logprob_all.device
            if f_mask.any():
                CE_forget = (-logprob_all[f_mask]).mean()
            else:
                CE_forget = torch.tensor(0.0, device=device)

            if r_mask.any():
                CE_retain = (-logprob_all[r_mask]).mean()
            else:
                CE_retain = torch.tensor(0.0, device=device)

            # Gradient Difference objective (ASCEND)
            objective = self._args.fg_w * CE_forget - self._args.rt_w * CE_retain
            objective.backward()
            self._optimizer.step()
            self._optimizer.zero_grad()

            # Friendly metrics:
            # avg probs = exp(-CE). For forget we want DOWN; for retain we want UP.
            avg_p_forget = torch.exp(-CE_forget).item() if f_mask.any() else 0.0
            avg_p_retain = torch.exp(-CE_retain).item() if r_mask.any() else 0.0

            return {
                "objective": objective.item(),            # should go UP if training works
                "CE_forget": CE_forget.item() if f_mask.any() else 0.0,
                "CE_retain": CE_retain.item() if r_mask.any() else 0.0,
                "p_forget": avg_p_forget,                 # should go DOWN
                "p_retain": avg_p_retain,                 # should go UP
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
            # tokenize string
            tokenized = tokenizer(
                example["input"],
                example["output"],
                return_tensors="pt",
                max_length=2048,
                truncation=True,
            )
            # get the range of output
            output_beginning = tokenized.char_to_token(0, sequence_index=1)
            output_end = (
                tokenized.char_to_token(len(example["output"]) - 1, sequence_index=1)
                + 1
            )
            # squeeze tensors
            tokenized["input_ids"] = tokenized["input_ids"].squeeze()
            tokenized["attention_mask"] = tokenized["attention_mask"].squeeze()
            return tokenized, [output_beginning, output_end], forget

        def prepare_dataset(retain_set, forget_set):
            # tokenize datasets
            tokenized_retain = retain_set.apply(tokenize_function, axis=1, args=(0,))
            tokenized_forget = forget_set.apply(tokenize_function, axis=1, args=(1,))
            # combine forget and retain sets into one
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
            # combined tokenized inputs into a single tensor
            inputs = tokenizer.pad(inputs, padding=True, return_tensors="pt")

            # create tensors for ranges and tasks
            ranges = torch.tensor(np.asarray(ranges), dtype=torch.long)
            tasks = torch.tensor(np.asarray(tasks), dtype=torch.long)

            answer_mask = torch.clone(inputs.attention_mask)
            answer_mask[
                torch.arange(answer_mask.shape[1]).unsqueeze(0)
                < ranges[:, 0].unsqueeze(1)
            ] = 0

            return (inputs, answer_mask, ranges, tasks)

        return DataLoader(
            data, batch_size=args.batch_size, collate_fn=prepare_batch, shuffle=shuffle
        )

    def prepare_args() -> SimpleNamespace:
        # replaces argparse with a static config
        return SimpleNamespace(
            model="7B",
            logdir="logs",
            threads=1,
            seed=42,
            device=None,            # auto-select cuda if available
            batch_size=4,
            epochs=10,
            learning_rate=1e-5,
            lora_rank=8,
            lora_merge_every=-1,    # -1 means never
            evaluate_every=-1,
            save_model=True,
            save_logdir_name=False,
            fg_w=1.2,               
            rt_w=1.5,
        )
        


    def save_checkpoint(
        model: AutoModelForCausalLM, tokenizer: AutoTokenizer, path: str
    ):
        # prepare tmp directory
        temporary_path = os.path.join(path, 'tmp')
        os.makedirs(temporary_path, exist_ok=True)
        # first save to a temporary directory to not break a checkpoint mid-save
        model.save_pretrained(temporary_path)
        tokenizer.save_pretrained(temporary_path)
        # finally move saved model from tmp to the actual directory
        for f in os.listdir(temporary_path):
            shutil.move(os.path.join(temporary_path, f), os.path.join(path, f))

    start_time = time.time()
    args = prepare_args()
    model = AutoModelForCausalLM.from_pretrained(
        "allenai/Olmo-3-7B-Instruct"
    )
    tokenizer = AutoTokenizer.from_pretrained("allenai/Olmo-3-7B-Instruct")

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
        tokenizer=tokenizer,
        args=args,
    )

    print("Unlearning.")
    unlearn_model.unlearn(
        tokenizer=tokenizer,
        train_data=train_loader,
        args=args,
        save_path=output_path_to_write_unlearned_model,
        start_time=start_time
    )

    print("Saving")
    extracted_model: AutoModelForCausalLM = unlearn_model.extract_model()
    os.makedirs(output_path_to_write_unlearned_model, exist_ok=True)
    save_checkpoint(
        model=extracted_model,
        tokenizer=tokenizer,
        path=output_path_to_write_unlearned_model,
    )
    pass


if __name__ == "__main__":
    # 🔁 HARD-CODE YOUR PATHS HERE
    LOAD_DIR   = "/models"         # e.g., "/models/zephyr-7b-beta"
    RETAIN_DIR = "./data"    # contains retain.parquet
    FORGET_DIR = "./data"    # contains forget.parquet
    OUTPUT_DIR = "./gd_7B_sc_test"    # where the adapted model will be saved

    unlearn(LOAD_DIR, OUTPUT_DIR, FORGET_DIR, RETAIN_DIR)