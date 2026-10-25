"""CUB-200-2011 fine-grained classification, from scratch or ImageNet weights.

Compare Gnome Fisher / Hutchinson, SOAP, and AdamW on the official 5,994
training / 5,794 test images. Only image-level species labels are used:
no bounding boxes, parts, or attributes. Test metrics
use the logger's `val` event type for compatibility with the run viewer;
the metadata identifies this as the official test split.

The shared GELU ResNet trunk gets a 7x7 stride-2 stem and max-pool for
224x224 inputs. GroupNorm is the default; BatchNorm is also supported for
every optimizer. resnet18_pretrained uses torchvision's ImageNet ResNet-18
with ReLU, BatchNorm by default, and a fresh 200-class head; all layers train.
All optimizers share CUB-specific
crop/flip/color-jitter augmentation, batches and a warmup/cosine
schedule. Gnome's auxiliary batch is a random subset of the main batch,
which remains intact. Optimizer configuration is owned by this experiment.

    # Edit experiments/configs/resnets/cub200.py: optimizer='gnome_hutchinson', download=True
    uv run --extra vision -m experiments.resnets.cub200

Download: https://data.caltech.edu/records/65de6-vp158 (about 1.2 GB).
Existing data: --data-dir may name CUB_200_2011 itself or its parent.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from experiments.common import (
    DIVERGED_EXIT, RunLogger, cosine_with_warmup, diverged, pick_device,
)
from experiments.common.resnet import MODEL_NAMES, build_model as build_shared_resnet
from experiments.baselines import SOAP
from gnome import Gnome


EXPERIMENT = "cub200"
NUM_CLASSES = 200
PRETRAINED_MODEL = "resnet18_pretrained"
PRETRAINED_WEIGHTS = "ResNet18_Weights.IMAGENET1K_V1"
DATASET_URL = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1"
DATASET_MD5 = "97eceeb196236b17998738112f37df78"
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
METADATA_FILES = ("images.txt", "image_class_labels.txt", "train_test_split.txt")
CROP_SCALE = (0.2, 1.0)
CROP_RATIO = (3 / 4, 4 / 3)
FLIP_PROBABILITY = 0.5
COLOR_JITTER = dict(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.0)
MERGE_DIMS_MODES = {
    "none": False,
    "greedy": True,
    "spatial": ((0,), (1,), (2, 3)),
    "patch": ((0,), (1, 2, 3)),
}


def prepare_data(data_dir: str, download: bool = False) -> Path:
    """Resolve extracted data, optionally downloading the checksummed archive."""
    root = Path(data_dir).expanduser()
    folder = root if root.name == "CUB_200_2011" else root / "CUB_200_2011"
    if not all((folder / name).is_file() for name in METADATA_FILES):
        if not download:
            raise FileNotFoundError(
                f"CUB metadata not found at {folder}. Pass --download or "
                "--data-dir pointing to the extracted CUB_200_2011 folder."
            )
        from torchvision.datasets.utils import download_and_extract_archive

        download_and_extract_archive(
            DATASET_URL, str(folder.parent), filename="CUB_200_2011.tgz",
            md5=DATASET_MD5,
        )
    if not all((folder / name).is_file() for name in METADATA_FILES):
        raise FileNotFoundError(f"Incomplete CUB metadata at {folder}")
    return folder


def _read_index(path: Path) -> dict[int, str]:
    records = {}
    for line in path.read_text().splitlines():
        image_id, value = line.split(maxsplit=1)
        image_id = int(image_id)
        if image_id in records:
            raise ValueError(f"Duplicate image ID {image_id} in {path}")
        records[image_id] = value
    return records


class CUB200(Dataset):
    """Join official annotation files by image ID, never by line order."""

    def __init__(self, root: Path, train: bool, transform=None):
        self.root = Path(root)
        self.transform = transform
        images, labels, splits = (
            _read_index(self.root / name) for name in METADATA_FILES
        )
        if not images or images.keys() != labels.keys() or images.keys() != splits.keys():
            raise ValueError("CUB metadata files must contain the same non-empty image IDs")
        self.samples = []
        self.image_ids = []
        for image_id in sorted(images):
            label, split = int(labels[image_id]), int(splits[image_id])
            if not 1 <= label <= NUM_CLASSES or split not in (0, 1):
                raise ValueError(f"Invalid CUB label/split for image ID {image_id}")
            if split != int(train):
                continue
            path = self.root / "images" / images[image_id]
            if not path.is_file():
                raise FileNotFoundError(path)
            self.samples.append((path, label - 1))
            self.image_ids.append(image_id)
        if not self.samples:
            raise ValueError(f"Empty CUB {'train' if train else 'test'} split")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        from PIL import Image

        path, label = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def build_transforms(image_size: int, augment: bool = True):
    """CUB's training policy; evaluation always uses deterministic center crops.

    Crops retain 20–100% of image area before resizing to image_size.
    Jitter runs on the PIL image. Disabling augmentation removes all three
    stochastic operations.
    """
    from torchvision import transforms as T

    evaluation = T.Compose([
        T.Resize(round(image_size * 256 / 224)), T.CenterCrop(image_size),
        T.ToTensor(), T.Normalize(MEAN, STD),
    ])
    training = T.Compose([
        T.RandomResizedCrop(image_size, scale=CROP_SCALE, ratio=CROP_RATIO),
        T.RandomHorizontalFlip(p=FLIP_PROBABILITY),
        T.ColorJitter(**COLOR_JITTER),
        T.ToTensor(), T.Normalize(MEAN, STD),
    ]) if augment else evaluation
    return training, evaluation


def default_norm(name: str) -> str:
    return "bn" if name == PRETRAINED_MODEL else "gn"


def _replace_batchnorm_with_groupnorm(module: nn.Module):
    """Retain pretrained affine parameters when explicitly opting into GN."""
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            replacement = nn.GroupNorm(32, child.num_features, eps=child.eps)
            with torch.no_grad():
                replacement.weight.copy_(child.weight)
                replacement.bias.copy_(child.bias)
            setattr(module, name, replacement)
        else:
            _replace_batchnorm_with_groupnorm(child)


def build_model(name: str, norm: str | None = None, *, load_pretrained: bool = True) -> nn.Module:
    """Build a CUB model; disable weight loading when restoring a checkpoint."""
    norm = norm or default_norm(name)
    if norm not in ("gn", "bn"):
        raise ValueError(f"unknown normalization: {norm}")
    if name == PRETRAINED_MODEL:
        from torchvision.models import ResNet18_Weights, resnet18

        weights = ResNet18_Weights.IMAGENET1K_V1 if load_pretrained else None
        model = resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
        if norm == "gn":
            _replace_batchnorm_with_groupnorm(model)
        return model
    layers = list(build_shared_resnet(name, num_outputs=NUM_CLASSES, norm=norm).children())
    layers[0] = nn.Conv2d(3, layers[0].out_channels, 7, stride=2, padding=3, bias=False)
    layers.insert(3, nn.MaxPool2d(3, stride=2, padding=1))
    return nn.Sequential(*layers)


def build_optimizer(
    name: str, params, lr: float, weight_decay: float,
    beta1: float = 0.9, beta2: float = 0.99, eps: float = 1e-6,
    trust_region: float = 1.0, aux_batch_size: int = 10,
    merge_dims: str = "none",
):
    """CUB optimizer settings, independent of other benchmark modules.

    Both Gnome variants differ only in their curvature surrogate. The
    training loop applies the same external LR schedule to every optimizer.
    """
    if name == "adamw":
        cfg = dict(lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999), eps=1e-8)
        return torch.optim.AdamW(params, **cfg), cfg
    if name not in ("gnome_fisher", "gnome_hutchinson", "soap"):
        raise ValueError(f"unknown optimizer: {name}")
    if merge_dims not in MERGE_DIMS_MODES:
        raise ValueError(f"unknown merge-dims mode {merge_dims!r}; expected one of {tuple(MERGE_DIMS_MODES)}")
    cfg = dict(
        lr=lr, weight_decay=weight_decay, betas=(beta1, beta2),
        shampoo_beta=beta2, precondition_frequency=10, precondition_1d=True,
        merge_dims=MERGE_DIMS_MODES[merge_dims],
    )
    if name == "soap":
        cfg["eps"] = 1e-8
        opt = SOAP(params, **cfg)
    else:
        cfg.update(
            eps=eps, trust_radius=trust_region if trust_region > 0 else None,
            norm_free=False,
            loss="cce" if name == "gnome_fisher" else "cce_hutchinson",
        )
        opt = Gnome(params, **cfg)
        cfg["aux_batch_size"] = aux_batch_size
    cfg["merge_dims_mode"] = merge_dims
    return opt, cfg


@contextmanager
def freeze_bn_running_stats(model: nn.Module):
    """Use training batch statistics without updating BN buffers or counters.

    Leave autograd and module train/eval modes intact. Restoring the tracking
    flag after the forward is safe: backward uses the saved batch statistics.
    """
    layers = [layer for layer in model.modules()
              if isinstance(layer, nn.BatchNorm2d)
              and layer.training and layer.track_running_stats]
    try:
        for layer in layers:
            layer.track_running_stats = False
        yield
    finally:
        for layer in layers:
            layer.track_running_stats = True


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def make_loader(dataset, args, shuffle=False, seed_offset=0):
    return DataLoader(
        dataset, batch_size=args.batch_size, shuffle=shuffle,
        num_workers=args.workers, worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(args.seed + seed_offset),
        persistent_workers=args.workers > 0,
    )


@torch.no_grad()
def evaluate(model, loader, device):
    """Sample-weighted CCE, top-1 and top-5 (fractions, not percentages)."""
    was_training = model.training
    model.eval()
    loss_sum, correct1, correct5, count = 0.0, 0, 0, 0
    try:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss_sum += F.cross_entropy(logits, labels, reduction="sum").item()
            top5 = logits.topk(5, dim=1).indices
            correct1 += (top5[:, 0] == labels).sum().item()
            correct5 += (top5 == labels[:, None]).any(dim=1).sum().item()
            count += labels.numel()
    finally:
        model.train(was_training)
    if count == 0:
        raise ValueError("Cannot evaluate an empty loader")
    return dict(loss=loss_sum / count, top1=correct1 / count, top5=correct5 / count)


def train(args) -> str:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device() if args.device == "auto" else torch.device(args.device)
    root = prepare_data(args.data_dir, args.download)
    train_transform, eval_transform = build_transforms(args.image_size, not args.no_augment)
    training = CUB200(root, train=True, transform=train_transform)
    train_eval = CUB200(root, train=True, transform=eval_transform)
    testing = CUB200(root, train=False, transform=eval_transform)
    loader = make_loader(training, args, shuffle=True)
    train_eval_loader = make_loader(train_eval, args, seed_offset=1)
    test_loader = make_loader(testing, args, seed_offset=2)
    model = build_model(args.model, args.norm).to(device)
    opt, opt_cfg = build_optimizer(
        args.optimizer, model.parameters(), args.lr, args.weight_decay,
        beta1=args.beta1, beta2=args.beta2, eps=args.eps,
        trust_region=args.trust_region, aux_batch_size=args.aux_batch_size,
        merge_dims=args.merge_dims,
    )
    total_steps = args.epochs * len(loader)
    if args.max_steps is not None:
        total_steps = min(total_steps, args.max_steps)
    is_gnome = args.optimizer.startswith("gnome")
    # Separate data-selection RNG from Gnome's randomized curvature probes.
    aux_rng = torch.Generator().manual_seed(args.seed + 3)
    hyperparameters = {
        **vars(args), "data_dir": str(root), "num_classes": NUM_CLASSES,
        "n_train": len(training), "n_test": len(testing),
        "evaluation_split": "official_test", "pretrained": args.model == PRETRAINED_MODEL,
        "pretrained_weights": PRETRAINED_WEIGHTS if args.model == PRETRAINED_MODEL else None,
        "activation": "relu" if args.model == PRETRAINED_MODEL else "gelu",
        "annotations": "image_class_labels_only", "stem": "7x7_s2_maxpool",
        "crop_scale": CROP_SCALE, "crop_ratio": CROP_RATIO,
        "normalize_mean": MEAN, "normalize_std": STD,
        "augmentation": {
            "enabled": not args.no_augment,
            "policy": "cub_crop_flip_jitter_v2",
            "crop_scale": CROP_SCALE, "crop_ratio": CROP_RATIO,
            "horizontal_flip_p": FLIP_PROBABILITY,
            "color_jitter": COLOR_JITTER,
        },
        "aux_sampling": "subset_of_main" if is_gnome else None,
        "aux_bn_running_stats": "frozen" if is_gnome else None,
        "total_steps": total_steps, "device": str(device),
        "n_params": sum(p.numel() for p in model.parameters()),
        **{f"opt.{key}": value for key, value in opt_cfg.items()},
    }
    if not args.quiet:
        print(f"[{EXPERIMENT}] {args.optimizer} | {args.model}/{args.norm} | "
              f"{args.image_size}px | device={device}\n"
              f"  train={len(training)} test={len(testing)} "
              f"params={hyperparameters['n_params']:,} steps={total_steps}\n"
              f"  augmentation={'off' if args.no_augment else 'crop + flip + color jitter'}",
              flush=True)

    with RunLogger(EXPERIMENT, args.optimizer, args.seed, hyperparameters,
                   runs_dir=args.runs_dir) as run:
        step, train_time = 0, 0.0
        for epoch in range(args.epochs):
            model.train()
            epoch_loss, epoch_count = 0.0, 0
            epoch_start = time.perf_counter()
            for images_cpu, labels_cpu in loader:
                images, labels = images_cpu.to(device), labels_cpu.to(device)
                lr = args.lr * cosine_with_warmup(
                    step, args.warmup_steps, total_steps, args.lr_min_frac
                )
                for group in opt.param_groups:
                    group["lr"] = lr
                if is_gnome:
                    k = min(args.aux_batch_size, labels.numel())
                    indices = torch.randperm(labels.numel(), generator=aux_rng)[:k].to(device)
                    aux_images, aux_labels = images[indices], labels[indices]

                    def main_closure():
                        return model(images), labels

                    def aux_closure():
                        with freeze_bn_running_stats(model):
                            return model(aux_images), aux_labels

                    loss = opt.step(main_closure, aux_closure)
                else:
                    opt.zero_grad(set_to_none=True)
                    loss = F.cross_entropy(model(images), labels)
                    loss.backward()
                    opt.step()
                loss_value = loss.detach().item()
                if diverged(loss_value):
                    run.finish(completed=False, diverged=True, diverged_step=step)
                    raise SystemExit(DIVERGED_EXIT)
                step += 1
                epoch_loss += loss_value * labels.numel()
                epoch_count += labels.numel()
                run.log_train(step, epoch=epoch + 1, loss=loss_value, lr=lr)
                if not args.quiet and args.log_every and step % args.log_every == 0:
                    print(f"  step {step}/{total_steps} | "
                          f"train_loss={epoch_loss / epoch_count:.4f} lr={lr:.2e}", flush=True)
                if step >= total_steps:
                    break
            # loss.item() above synchronizes the device; this includes data loading.
            train_time += time.perf_counter() - epoch_start
            train_metrics = evaluate(model, train_eval_loader, device)
            test_metrics = evaluate(model, test_loader, device)
            if any(not math.isfinite(v) for v in (*train_metrics.values(), *test_metrics.values())):
                run.finish(completed=False, diverged=True, diverged_step=step)
                raise SystemExit(DIVERGED_EXIT)
            run.log_val(
                step, epoch=epoch + 1, **test_metrics, lr=lr,
                train_loss=epoch_loss / epoch_count,
                **{f"train_eval_{key}": value for key, value in train_metrics.items()},
                training_seconds=train_time,
            )
            if not args.quiet:
                print(f"  epoch {epoch + 1}: train center-crop top1={train_metrics['top1']:.2%} | "
                      f"test loss={test_metrics['loss']:.4f} "
                      f"top1={test_metrics['top1']:.2%} top5={test_metrics['top5']:.2%}", flush=True)
            if step >= total_steps:
                break
        if args.save_checkpoint:
            torch.save({
                "model": model.state_dict(), "optimizer": opt.state_dict(),
                "hyperparameters": hyperparameters, "step": step, "epoch": epoch + 1,
            }, Path(run.path).with_suffix(".pt"))
        path = run.finish(
            completed=True, final_top1=test_metrics["top1"],
            final_top5=test_metrics["top5"], final_val_loss=test_metrics["loss"],
            final_train_top1=train_metrics["top1"], training_seconds=train_time,
        )
    print(f"[{EXPERIMENT}] saved → {path}", flush=True)
    return path


def parse_args(argv=None):
    from experiments.common.config import ConfigParser
    parser = ConfigParser(__file__)
    args = parser.parse_args(argv)
    args.norm = args.norm or default_norm(args.model)
    if min(args.epochs, args.batch_size, args.aux_batch_size) <= 0 or args.image_size < 32:
        parser.error("epochs and batch sizes must be positive; image-size must be at least 32")
    if min(args.workers, args.warmup_steps, args.log_every) < 0:
        parser.error("workers, warmup-steps, and log-every must be non-negative")
    if args.max_steps is not None and args.max_steps <= 0:
        parser.error("max-steps must be positive")
    if not 0 <= args.lr_min_frac <= 1:
        parser.error("lr-min-frac must be in [0, 1]")
    return args


def main():
    train(parse_args())


if __name__ == "__main__":
    main()
