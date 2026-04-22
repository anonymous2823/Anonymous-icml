import os
import shutil
import datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm


def unlearn(
    input_path_to_unlearning_candidate_model,
    output_path_to_write_unlearned_model,
    forget_set_path,
    retain_set_path,
):
    """
    RMU implementation specialized for OLMo-3-7B-Instruct.

    Expected parquet files:
      - forget_set_path / "forget_set_q1.parquet"
      - retain_set_path / "retain_sc_val_1.parquet"

    Expected columns in both parquet files:
      - "input"
      - "output"
    """

    # ============================================================
    # Config
    # ============================================================
    def prepare_args() -> SimpleNamespace:
        return SimpleNamespace(
            # model / output
            model_name_or_path="allenai/Olmo-3-7B-Instruct",
            output_dir=output_path_to_write_unlearned_model,

            # data
            forget_file="forget_sc_1.parquet",
            retain_file="retain_sc.parquet",
            batch_size=1,
            max_num_batches=200,
            max_length=1024,

            # RMU hyperparameters
            lr=5e-5,
            steering_coeff=100.0,     # c in RMU
            alpha=3.0,             # retain weight

            # OLMo-3-7B-Instruct-specific layer selection
            module_str="{model_name}.model.layers[{layer_id}]",
            layer_id=31,             # hooked layer for RMU target/matching
            layer_ids=[29, 30, 31],  # train these layers

            # runtime
            seed=42,
            device=None,
            keep_frozen_on_cpu=True, # safer for VRAM, slower
            verbose=True,
        )

    # ============================================================
    # Utilities
    # ============================================================
    def set_seed(seed: int):
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    def get_model_device(model: torch.nn.Module) -> torch.device:
        return next(model.parameters()).device

    def move_inputs_to_device(inputs: dict, device: torch.device) -> dict:
        moved = {}
        for k, v in inputs.items():
            moved[k] = v.to(device) if torch.is_tensor(v) else v
        return moved

    # ============================================================
    # OLMo layer resolution
    # ============================================================
    def resolve_layer_module(model: torch.nn.Module, layer_id: int):
        """
        OLMo-3-7B-Instruct is supported in standard HF transformers and uses a
        Llama-style stack, so model.layers[layer_id] is the correct first path.
        """
        # Most likely path for OLMo-3 in HF
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return model.model.layers[layer_id]

        # Fallbacks if wrappers differ
        if hasattr(model, "model") and hasattr(model.model, "model") and hasattr(model.model.model, "layers"):
            return model.model.model.layers[layer_id]

        if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            return model.transformer.h[layer_id]

        raise ValueError(
            f"Could not resolve layer {layer_id}. "
            f"Please inspect the model structure with print(model)."
        )

    def get_params_for_layers(model: torch.nn.Module, layer_ids: list[int]):
        """
        Train all parameters inside the selected layers.
        This is more robust than relying on param_ids ordering.
        """
        params = []
        seen = set()

        for lid in layer_ids:
            layer_module = resolve_layer_module(model, lid)
            for p in layer_module.parameters():
                if id(p) not in seen:
                    params.append(p)
                    seen.add(id(p))

        return params

    # ============================================================
    # Forward hook activation cache
    # ============================================================
    def forward_with_cache(model, inputs, module, no_grad=True):
        """
        Runs a forward pass and returns the activation captured from `module`
        using a forward hook.
        """
        cache = {}

        def hook_fn(_module, _inputs, output):
            cache["activation"] = output

        handle = module.register_forward_hook(hook_fn)

        try:
            model_device = get_model_device(model)
            model_inputs = move_inputs_to_device(inputs, model_device)

            if no_grad:
                with torch.no_grad():
                    _ = model(
                        **model_inputs,
                        output_hidden_states=False,
                        return_dict=True,
                    )
            else:
                _ = model(
                    **model_inputs,
                    output_hidden_states=False,
                    return_dict=True,
                )
        finally:
            handle.remove()

        if "activation" not in cache:
            raise RuntimeError("Forward hook did not capture any activation.")

        activation = cache["activation"]
        if isinstance(activation, tuple):
            activation = activation[0]

        return activation

    # ============================================================
    # Text data
    # ============================================================
    class TextDataset(Dataset):
        def __init__(self, texts):
            self.texts = texts

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            return self.texts[idx]

    def prepare_texts(df: pd.DataFrame):
        texts = []
        for _, row in df.iterrows():
            inp = str(row["input"])
            out = str(row["output"])
            texts.append(inp + "\n" + out)
        return texts

    def prepare_loader(texts, batch_size, shuffle=True):
        dataset = TextDataset(texts)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    # ============================================================
    # Save model
    # ============================================================
    def save_checkpoint(model, tokenizer, path: str):
        tmp = os.path.join(path, "tmp")
        os.makedirs(tmp, exist_ok=True)

        model.save_pretrained(tmp)
        tokenizer.save_pretrained(tmp)

        os.makedirs(path, exist_ok=True)
        for f in os.listdir(tmp):
            shutil.move(os.path.join(tmp, f), os.path.join(path, f))

        shutil.rmtree(tmp, ignore_errors=True)

    # ============================================================
    # RMU Trainer
    # ============================================================
    class RMUTrainer:
        def __init__(self, updated_model, frozen_model, tokenizer, args: SimpleNamespace):
            self.updated_model = updated_model
            self.frozen_model = frozen_model
            self.tokenizer = tokenizer
            self.args = args

            self.updated_model.train()
            self.frozen_model.eval()

            # Freeze everything first
            for p in self.updated_model.parameters():
                p.requires_grad = False

            # Unfreeze only chosen layers
            self.params = get_params_for_layers(self.updated_model, args.layer_ids)
            for p in self.params:
                p.requires_grad = True

            self.optimizer = AdamW(self.params, lr=args.lr)

            self.updated_module = resolve_layer_module(self.updated_model, args.layer_id)
            self.frozen_module = resolve_layer_module(self.frozen_model, args.layer_id)

            self.control_vector = None

        def _init_control_vector(self, activation: torch.Tensor):
            """
            Fixed random RMU target vector c * u.
            Reused throughout training.
            """
            hidden_size = activation.shape[-1]
            device = activation.device
            dtype = activation.dtype

            random_vector = torch.rand(hidden_size, device=device, dtype=dtype)
            random_vector = random_vector / (torch.norm(random_vector) + 1e-12)
            random_vector = random_vector * self.args.steering_coeff

            if activation.dim() == 3:
                # [B, T, H] -> [1, 1, H]
                self.control_vector = random_vector.view(1, 1, hidden_size)
            elif activation.dim() == 2:
                # [B, H] -> [1, H]
                self.control_vector = random_vector.view(1, hidden_size)
            else:
                raise ValueError(f"Unexpected activation shape: {activation.shape}")

        def train(self, forget_loader, retain_loader):
            num_batches = min(
                self.args.max_num_batches,
                len(forget_loader),
                len(retain_loader),
            )

            old_truncation_side = self.tokenizer.truncation_side
            self.tokenizer.truncation_side = "right"

            progress = tqdm(
                zip(forget_loader, retain_loader),
                total=num_batches,
                desc="RMU unlearning",
                leave=False,
            )

            for step, (forget_batch, retain_batch) in enumerate(progress):
                if step >= num_batches:
                    break

                # -------------------------
                # Forget loss
                # -------------------------
                forget_inputs = self.tokenizer(
                    list(forget_batch),
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.args.max_length,
                )

                updated_forget_activations = forward_with_cache(
                    self.updated_model,
                    forget_inputs,
                    module=self.updated_module,
                    no_grad=False,
                )

                if self.control_vector is None:
                    self._init_control_vector(updated_forget_activations)

                forget_loss = F.mse_loss(
                    updated_forget_activations,
                    self.control_vector.expand_as(updated_forget_activations),
                )

                # -------------------------
                # Retain loss
                # -------------------------
                retain_inputs = self.tokenizer(
                    list(retain_batch),
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.args.max_length,
                )

                updated_retain_activations = forward_with_cache(
                    self.updated_model,
                    retain_inputs,
                    module=self.updated_module,
                    no_grad=False,
                )

                frozen_retain_activations = forward_with_cache(
                    self.frozen_model,
                    retain_inputs,
                    module=self.frozen_module,
                    no_grad=True,
                ).to(updated_retain_activations.device)

                retain_loss = F.mse_loss(
                    updated_retain_activations,
                    frozen_retain_activations,
                )
                retain_loss = retain_loss * self.args.alpha

                # -------------------------
                # Update
                # -------------------------
                loss = forget_loss + retain_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                grad_stat = 0.0
                if self.params[0].grad is not None:
                    grad_stat = self.params[0].grad.abs().mean().item()

                progress.set_postfix(
                    {
                        "loss": f"{loss.item():.4g}",
                        "forget": f"{forget_loss.item():.4g}",
                        "retain": f"{retain_loss.item():.4g}",
                    }
                )

                print(
                    f"step={step:04d} | "
                    f"loss={loss.item():.4g} | "
                    f"forget_loss={forget_loss.item():.4g} | "
                    f"retain_loss={retain_loss.item():.4g} | "
                    f"param_grad_mean={grad_stat:.4g}"
                )

                if self.args.verbose:
                    frozen_forget_activations = forward_with_cache(
                        self.frozen_model,
                        forget_inputs,
                        module=self.frozen_module,
                        no_grad=True,
                    ).to(updated_forget_activations.device)

                    unlearn_cosine = F.cosine_similarity(
                        updated_forget_activations,
                        frozen_forget_activations,
                        dim=-1,
                    ).mean()

                    retain_cosine = F.cosine_similarity(
                        updated_retain_activations,
                        frozen_retain_activations,
                        dim=-1,
                    ).mean()

                    print(f"unlearn_cosine_sim={unlearn_cosine.item():.6f}")
                    print(f"retain_cosine_sim={retain_cosine.item():.6f}")

                    def mean_norm(x):
                        if x.dim() == 3:
                            return x.norm(dim=-1).mean().item()
                        elif x.dim() == 2:
                            return x.norm(dim=-1).mean().item()
                        return x.float().norm().item()

                    print("updated_forget_activations.norm =", mean_norm(updated_forget_activations))
                    print("frozen_forget_activations.norm  =", mean_norm(frozen_forget_activations))
                    print("updated_retain_activations.norm =", mean_norm(updated_retain_activations))
                    print("frozen_retain_activations.norm  =", mean_norm(frozen_retain_activations))

            self.tokenizer.truncation_side = old_truncation_side

    # ============================================================
    # Main
    # ============================================================
    args = prepare_args()

    print("==== RMU Config (OLMo-3-7B-Instruct) ====")
    print("\n".join(f"{k}={v}" for k, v in vars(args).items()))
    print("=========================================")

    set_seed(args.seed)

    device = torch.device(
        args.device
        if args.device is not None
        else "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Using device:", device)

    # Load tokenizer + two model copies
    frozen_model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    updated_model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.keep_frozen_on_cpu:
        frozen_model.to("cpu")
    else:
        frozen_model.to(device)

    updated_model.to(device)

    for p in frozen_model.parameters():
        p.requires_grad = False

    # Load data
    retain_df = pd.read_parquet(
        os.path.join(retain_set_path, args.retain_file),
        engine="pyarrow",
    )
    forget_df = pd.read_parquet(
        os.path.join(forget_set_path, args.forget_file),
        engine="pyarrow",
    )

    retain_texts = prepare_texts(retain_df)
    forget_texts = prepare_texts(forget_df)

    retain_loader = prepare_loader(retain_texts, args.batch_size, shuffle=True)
    forget_loader = prepare_loader(forget_texts, args.batch_size, shuffle=True)

    # Train
    trainer = RMUTrainer(updated_model, frozen_model, tokenizer, args)
    trainer.train(forget_loader, retain_loader)

    # Save updated model
    save_path = args.output_dir
    if not save_path:
        date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        save_path = f"./rmu_olmo3_7b_{date}"

    print(f"Saving model to: {save_path}")
    os.makedirs(save_path, exist_ok=True)
    save_checkpoint(updated_model, tokenizer, save_path)
    print(f"Saved model to {save_path}")


if __name__ == "__main__":
    # HARD-CODE YOUR PATHS HERE
    LOAD_DIR = "/models"          # local HF model path for allenai/Olmo-3-7B-Instruct
    RETAIN_DIR = "./data"         # contains retain_sc_val_1.parquet
    FORGET_DIR = "./data"         # contains forget_set_q1.parquet
    OUTPUT_DIR = "./rmu_olmo3_7b_l22_24_sc"

    unlearn(
        input_path_to_unlearning_candidate_model=LOAD_DIR,
        output_path_to_write_unlearned_model=OUTPUT_DIR,
        forget_set_path=FORGET_DIR,
        retain_set_path=RETAIN_DIR,
    )