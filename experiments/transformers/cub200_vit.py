"""CUB-200 ViT optimizer comparison, configured through a Python CONFIG dictionary.

    uv run --extra vision -m experiments.transformers.cub200_vit
    uv run --extra vision -m experiments.transformers.cub200_vit --config path/to/run.py

Edit experiments/configs/transformers/cub200_vit.py to select AdamW, SOAP, Gnome, or
Muon. See docs/cub200_vit.md for configuration and parameter routing.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from functools import partial
import math
from pathlib import Path
import random
import runpy
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, get_worker_info
from torchvision import transforms as T
from torchvision.models.vision_transformer import VisionTransformer

from experiments.baselines import SOAP
from experiments.common import DIVERGED_EXIT, RunLogger, cosine_with_warmup, diverged, pick_device
from experiments.resnets.cub200 import CUB200, NUM_CLASSES, evaluate, prepare_data, seed_worker
from experiments.transformers.modded_nanogpt_model import Muon
from gnome import Gnome


EXPERIMENT = "cub200_vit"
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "transformers" / f"{EXPERIMENT}.py"

# The schema contains keys, not another set of experiment defaults. A run file
# must be complete, so its behavior never depends on silently merged settings.
ADAMW_KEYS = {"lr", "betas", "eps", "weight_decay"}
CURVATURE_KEYS = ADAMW_KEYS | {
    "shampoo_beta", "precondition_frequency", "max_precond_dim", "merge_dims", "precondition_1d",
}
SCHEMA = {
    "optimizer": None,
    "model": dict.fromkeys(("image_size", "patch_size", "num_layers", "num_heads", "hidden_dim",
                            "mlp_dim", "dropout", "attention_dropout", "layer_norm_eps", "head_init_std")),
    "data": {
        **dict.fromkeys(("data_dir", "download", "augment", "crop_scale", "crop_ratio",
                        "horizontal_flip_p", "eval_resize_ratio", "normalize_mean", "normalize_std")),
        "color_jitter": dict.fromkeys(("brightness", "contrast", "saturation", "hue")),
    },
    "training": dict.fromkeys(("seed", "device", "epochs", "batch_size", "eval_batch_size", "workers",
                               "max_steps", "warmup_steps", "min_lr_frac", "max_grad_norm")),
    "optimizers": {
        "adamw": dict.fromkeys(ADAMW_KEYS),
        "soap": dict.fromkeys(CURVATURE_KEYS | {"normalize_grads", "correct_bias"}),
        "gnome": dict.fromkeys(CURVATURE_KEYS | {"trust_radius", "norm_free", "loss", "aux_batch_size"}),
        "muon": {"lr": None, "weight_decay": None, "mu": None, "adamw": dict.fromkeys(ADAMW_KEYS)},
    },
    "output": dict.fromkeys(("runs_dir", "log_every", "eval_every_epochs", "save_checkpoint", "quiet")),
}


def validate_config(config):
    def keys(value, schema, path="config"):
        if not isinstance(value, dict):
            raise ValueError(f"{path} must be a dictionary")
        missing, unknown = schema.keys() - value.keys(), value.keys() - schema.keys()
        if missing or unknown:
            raise ValueError(f"{path}: missing keys {sorted(missing)}; unknown keys {sorted(unknown)}")
        for key, nested in schema.items():
            if nested is not None:
                keys(value[key], nested, f"{path}.{key}")

    def number(value, name, minimum=0, maximum=None, *, integer=False, positive=False):
        valid = type(value) is int if integer else type(value) in (int, float)
        if (not valid or not math.isfinite(value) or value < minimum
                or (positive and value <= 0) or (maximum is not None and value > maximum)):
            raise ValueError(f"Invalid {name}: {value!r}")

    def boolean(value, name):
        if type(value) is not bool:
            raise ValueError(f"{name} must be True or False")

    def sequence(value, name, length, minimum=0, maximum=None, positive=False):
        if not isinstance(value, list) or len(value) != length:
            raise ValueError(f"{name} must be an array of length {length}")
        for item in value:
            number(item, name, minimum, maximum, positive=positive)

    keys(config, SCHEMA)
    if config["optimizer"] not in ("adamw", "soap", "gnome", "muon"):
        raise ValueError("optimizer must be adamw, soap, gnome, or muon")
    model, data, training, output = (config[key] for key in ("model", "data", "training", "output"))
    for name in ("image_size", "patch_size", "num_layers", "num_heads", "hidden_dim", "mlp_dim"):
        number(model[name], f"model.{name}", integer=True, positive=True)
    if model["image_size"] % model["patch_size"] or model["hidden_dim"] % model["num_heads"]:
        raise ValueError("image_size must be divisible by patch_size; hidden_dim must be divisible by num_heads")
    for name in ("dropout", "attention_dropout"):
        number(model[name], f"model.{name}", maximum=1)
        if model[name] == 1:
            raise ValueError(f"model.{name} must be less than 1")
    number(model["layer_norm_eps"], "model.layer_norm_eps", positive=True)
    number(model["head_init_std"], "model.head_init_std")
    for name in ("download", "augment"):
        boolean(data[name], f"data.{name}")
    for name in ("crop_scale", "crop_ratio"):
        sequence(data[name], f"data.{name}", 2, positive=True,
                 maximum=1 if name == "crop_scale" else None)
        if data[name][0] > data[name][1]:
            raise ValueError(f"data.{name} must be in ascending order")
    number(data["horizontal_flip_p"], "data.horizontal_flip_p", maximum=1)
    number(data["eval_resize_ratio"], "data.eval_resize_ratio", minimum=1)
    sequence(data["normalize_mean"], "data.normalize_mean", 3, minimum=-math.inf)
    sequence(data["normalize_std"], "data.normalize_std", 3, positive=True)
    for name, value in data["color_jitter"].items():
        number(value, f"data.color_jitter.{name}", maximum=0.5 if name == "hue" else None)
    for name in ("epochs", "batch_size", "eval_batch_size"):
        number(training[name], f"training.{name}", integer=True, positive=True)
    for name in ("seed", "workers", "warmup_steps"):
        number(training[name], f"training.{name}", integer=True)
    number(training["seed"], "training.seed", maximum=2**32 - 1, integer=True)
    if training["max_steps"] is not None:
        number(training["max_steps"], "training.max_steps", integer=True, positive=True)
    number(training["min_lr_frac"], "training.min_lr_frac", maximum=1)
    if training["max_grad_norm"] is not None:
        number(training["max_grad_norm"], "training.max_grad_norm", positive=True)
    if training["device"] not in ("auto", "cpu", "mps", "cuda"):
        raise ValueError("training.device must be auto, cpu, mps, or cuda")
    for name in ("save_checkpoint", "quiet"):
        boolean(output[name], f"output.{name}")
    number(output["log_every"], "output.log_every", integer=True)
    number(output["eval_every_epochs"], "output.eval_every_epochs", integer=True, positive=True)
    for name, value in (("data.data_dir", data["data_dir"]), ("output.runs_dir", output["runs_dir"])):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a nonempty path string")
    for name, opt in config["optimizers"].items():
        for key in ("lr", "weight_decay"):
            number(opt[key], f"optimizers.{name}.{key}")
        if name == "muon":
            number(opt["mu"], "optimizers.muon.mu", maximum=1)
            if opt["mu"] == 1:
                raise ValueError("optimizers.muon.mu must be less than 1")
            opt = opt["adamw"]
            for key in ("lr", "weight_decay"):
                number(opt[key], f"optimizers.muon.adamw.{key}")
        sequence(opt["betas"], f"optimizers.{name}.betas", 2, maximum=1)
        if 1 in opt["betas"]:
            raise ValueError(f"optimizers.{name}.betas must be less than 1")
        number(opt["eps"], f"optimizers.{name}.eps", positive=True)
        if name in ("gnome", "soap"):
            number(opt["shampoo_beta"], f"optimizers.{name}.shampoo_beta", maximum=1)
            if opt["shampoo_beta"] == 1:
                raise ValueError(f"optimizers.{name}.shampoo_beta must be less than 1")
            for key in ("precondition_frequency", "max_precond_dim"):
                number(opt[key], f"optimizers.{name}.{key}", integer=True, positive=True)
            for key in ("merge_dims", "precondition_1d"):
                boolean(opt[key], f"optimizers.{name}.{key}")
    for key in ("normalize_grads", "correct_bias"):
        boolean(config["optimizers"]["soap"][key], f"optimizers.soap.{key}")
    gnome = config["optimizers"]["gnome"]
    boolean(gnome["norm_free"], "optimizers.gnome.norm_free")
    number(gnome["aux_batch_size"], "optimizers.gnome.aux_batch_size", integer=True, positive=True)
    if gnome["trust_radius"] is not None:
        number(gnome["trust_radius"], "optimizers.gnome.trust_radius", positive=True)
    if gnome["loss"] not in ("cce", "cce_hutchinson"):
        raise ValueError("optimizers.gnome.loss must be cce or cce_hutchinson")
    return config


def load_config(path=DEFAULT_CONFIG):
    """Execute a Python config file and validate its CONFIG dictionary."""
    namespace = runpy.run_path(str(Path(path).expanduser().resolve()))
    if "CONFIG" not in namespace:
        raise ValueError(f"{path} must define a CONFIG dictionary")
    return deepcopy(validate_config(namespace["CONFIG"]))


def build_model(config):
    options = dict(config["model"])
    eps, head_std = options.pop("layer_norm_eps"), options.pop("head_init_std")
    model = VisionTransformer(**options, num_classes=NUM_CLASSES,
                              norm_layer=partial(nn.LayerNorm, eps=eps))
    # torchvision initializes this head to zero. A configurable random head
    # allows backbone gradients from the first pass in scratch experiments.
    nn.init.normal_(model.heads.head.weight, std=head_std)
    nn.init.zeros_(model.heads.head.bias)
    return model


class IndependentAugmentation:
    """Keep CPU transform randomness separate from optimizer probes/dropout.

    Each loader worker gets its own seeded stream. This also handles workers=0,
    where torchvision transforms otherwise consume the model's global RNG.
    """

    def __init__(self, transform, seed):
        self.transform, self.seed, self.generator = transform, seed, None

    def __call__(self, image):
        if self.generator is None:
            seed = torch.initial_seed() if get_worker_info() is not None else self.seed
            self.generator = torch.Generator().manual_seed(seed)
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(self.generator.get_state())
            result = self.transform(image)
            self.generator.set_state(torch.get_rng_state())
        return result


def build_transforms(config):
    size, data = config["model"]["image_size"], config["data"]
    normalize = T.Normalize(data["normalize_mean"], data["normalize_std"])
    evaluation = T.Compose([
        T.Resize(round(size * data["eval_resize_ratio"])), T.CenterCrop(size),
        T.ToTensor(), normalize,
    ])
    training = T.Compose([
        T.RandomResizedCrop(size, scale=tuple(data["crop_scale"]), ratio=tuple(data["crop_ratio"])),
        T.RandomHorizontalFlip(data["horizontal_flip_p"]),
        T.ColorJitter(**data["color_jitter"]), T.ToTensor(), normalize,
    ]) if data["augment"] else evaluation
    return IndependentAugmentation(training, config["training"]["seed"]), evaluation


def build_optimizers(model, config):
    name = config["optimizer"]
    options = deepcopy(config["optimizers"][name])
    if name == "gnome":
        options.pop("aux_batch_size")
        return [Gnome(model.parameters(), **options, max_grad_norm=config["training"]["max_grad_norm"])]
    if name == "soap":
        return [SOAP(model.parameters(), **options)]
    if name == "adamw":
        return [torch.optim.AdamW(model.parameters(), **options)]
    # Include only attention/MLP matrices inside encoder blocks. In particular,
    # positional embeddings are excluded even if their shape changes to 2D.
    matrices = [p for p in model.encoder.layers.parameters() if p.ndim == 2]
    matrix_ids = {id(p) for p in matrices}
    auxiliary = [p for p in model.parameters() if id(p) not in matrix_ids]
    return [Muon(matrices, **{key: value for key, value in options.items() if key != "adamw"}),
            torch.optim.AdamW(auxiliary, **options["adamw"])]


def train_step(model, optimizers, images, labels, config, aux_rng):
    if config["optimizer"] == "gnome":
        k = min(len(labels), config["optimizers"]["gnome"]["aux_batch_size"])
        indices = torch.randperm(len(labels), generator=aux_rng)[:k].to(images.device)
        return optimizers[0].step(lambda: (model(images), labels),
                                  lambda: (model(images[indices]), labels[indices])).item()
    for optimizer in optimizers:
        optimizer.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(images), labels)
    loss.backward()
    if config["training"]["max_grad_norm"] is not None:
        nn.utils.clip_grad_norm_(model.parameters(), config["training"]["max_grad_norm"])
    for optimizer in optimizers:
        optimizer.step()
    return loss.detach().item()


def train(config, *, config_path=None):
    config = deepcopy(validate_config(config))
    training, output = config["training"], config["output"]
    seed = training["seed"]
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    device = pick_device() if training["device"] == "auto" else torch.device(training["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    try:
        root = prepare_data(config["data"]["data_dir"], config["data"]["download"])
    except FileNotFoundError as error:
        raise FileNotFoundError("CUB data is missing or incomplete. Set data.data_dir to the extracted "
                                "dataset or set data.download to true in the Python config.") from error
    train_transform, eval_transform = build_transforms(config)
    datasets = [CUB200(root, train=True, transform=train_transform),
                CUB200(root, train=False, transform=eval_transform)]
    loaders = [DataLoader(
        dataset, batch_size=training["batch_size"] if index == 0 else training["eval_batch_size"],
        shuffle=index == 0, num_workers=training["workers"], worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(seed + index),
        persistent_workers=training["workers"] > 0,
    ) for index, dataset in enumerate(datasets)]
    total_steps = training["epochs"] * len(loaders[0])
    if training["max_steps"] is not None:
        total_steps = min(total_steps, training["max_steps"])
    if training["warmup_steps"] >= total_steps:
        raise ValueError("training.warmup_steps must be less than total training steps")
    model = build_model(config).to(device=device, dtype=torch.float32)
    optimizers = build_optimizers(model, config)
    base_lrs = [[group["lr"] for group in opt.param_groups] for opt in optimizers]
    names = {id(p): name for name, p in model.named_parameters()}
    metadata = dict(
        config=config, config_path=str(Path(config_path).resolve()) if config_path else None,
        data_dir=str(root.resolve()), device=str(device), dtype="float32", num_classes=NUM_CLASSES,
        n_train=len(datasets[0]), n_test=len(datasets[1]), total_steps=total_steps,
        n_params=sum(p.numel() for p in model.parameters()), pretrained=False,
        evaluation_split="official_test", annotations="image_class_labels_only",
        aux_sampling="subset_of_main_without_replacement" if config["optimizer"] == "gnome" else None,
        optimizer_groups=[dict(optimizer=type(opt).__name__, groups=[
            {**{key: value for key, value in group.items() if key != "params"},
             "parameter_names": [names[id(p)] for p in group["params"]]}
            for group in opt.param_groups]) for opt in optimizers],
    )
    aux_rng = torch.Generator().manual_seed(seed + 2)
    step, training_seconds = 0, 0.0
    if not output["quiet"]:
        print(f"[{EXPERIMENT}] {config['optimizer']} | {device} | {metadata['n_params']:,} parameters | "
              f"{total_steps:,} steps", flush=True)
    with RunLogger(EXPERIMENT, config["optimizer"], seed, metadata, runs_dir=output["runs_dir"]) as run:
        for epoch in range(1, training["epochs"] + 1):
            model.train()
            epoch_loss, epoch_count = 0.0, 0
            started = time.perf_counter()
            for images, labels in loaders[0]:
                images, labels = images.to(device), labels.to(device)
                # Sample step 0..total_steps-1 so the final update reaches the
                # configured floor. Both Muon and its AdamW retain their LR ratio.
                multiplier = cosine_with_warmup(step, training["warmup_steps"],
                                                max(1, total_steps - 1), training["min_lr_frac"])
                for optimizer, lrs in zip(optimizers, base_lrs):
                    for group, lr in zip(optimizer.param_groups, lrs):
                        group["lr"] = lr * multiplier
                loss = train_step(model, optimizers, images, labels, config, aux_rng)
                step += 1
                if diverged(loss):
                    run.finish(completed=False, diverged=True, diverged_step=step)
                    raise SystemExit(DIVERGED_EXIT)
                epoch_loss += loss * labels.numel()
                epoch_count += labels.numel()
                lrs = {"lr": optimizers[0].param_groups[0]["lr"]}
                if len(optimizers) > 1:
                    lrs["adamw_lr"] = optimizers[1].param_groups[0]["lr"]
                run.log_train(step, epoch=epoch, loss=loss, **lrs)
                if not output["quiet"] and output["log_every"] and step % output["log_every"] == 0:
                    print(f"  step {step}/{total_steps} loss={epoch_loss / epoch_count:.5f} "
                          f"lr={lrs['lr']:.3e}", flush=True)
                if step == total_steps:
                    break
            training_seconds += time.perf_counter() - started
            if epoch % output["eval_every_epochs"] == 0 or step == total_steps:
                metrics = evaluate(model, loaders[1], device)
                if diverged(*metrics.values()):
                    run.finish(completed=False, diverged=True, diverged_step=step)
                    raise SystemExit(DIVERGED_EXIT)
                run.log_val(step, epoch=epoch, **metrics, **lrs,
                            train_loss=epoch_loss / epoch_count, training_seconds=training_seconds)
                if not output["quiet"]:
                    print(f"  epoch {epoch}: test loss={metrics['loss']:.5f} "
                          f"top1={metrics['top1']:.2%} top5={metrics['top5']:.2%}", flush=True)
            if step == total_steps:
                break
        if output["save_checkpoint"]:
            torch.save(dict(model=model.state_dict(), optimizers=[opt.state_dict() for opt in optimizers],
                            config=config, hyperparameters=metadata, step=step, epoch=epoch),
                       Path(run.path).with_suffix(".pt"))
        path = run.finish(completed=True, final_top1=metrics["top1"], final_top5=metrics["top5"],
                          final_val_loss=metrics["loss"], training_seconds=training_seconds)
    print(f"[{EXPERIMENT}] saved → {path}", flush=True)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Python file defining the CONFIG dictionary")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
    except (OSError, ValueError, SyntaxError) as error:
        parser.error(str(error))
    train(config, config_path=args.config)


if __name__ == "__main__":
    main()
