"""
fewshot_lib.py
==============

Shared infrastructure for all 5 baselines (B1-B5) in this capstone.

Every baseline imports from this file. The ONLY thing that differs between
baselines is the encoder class (which lives in each baseline's notebook):

    Baseline 1: ResNet18 final-layer features          (single-scale)
    Baseline 2: ResNet50 final-layer features          (single-scale)
    Baseline 3: VGG16 final conv features + GAP        (single-scale)
    Baseline 4: ResNet18 multi-scale (layer2/3/4)      (NOVELTY)
    Baseline 5: ResNet50 multi-scale (layer2/3/4)      (NOVELTY)

This file fixes the methodology bugs that affected ALL of them in the
original code:

    1. lr=1e-3 + full-backbone fine-tuning  ->  catastrophic overfitting
       (loss collapses to 0.0000 after ~100 episodes, model memorises).
       Fix: lr=1e-4, weight_decay=1e-4, optional layer freezing, gradient clipping.

    2. No training-time augmentation.
       Fix: train_transform with RandomResizedCrop + Flip + ColorJitter +
       RandomRotation, applied on-the-fly every time an image is loaded.
       eval_transform stays deterministic (resize + normalize only).

    3. Per-class metrics aggregated over episodes were nonsense, because
       sample_episode() shuffles the class->local-label map every episode,
       so 'label 0' isn't a fixed real class.
       Fix: sample_episode() now ALSO returns global class IDs and the list
       of selected classes. evaluate() converts predictions back to global
       class IDs before aggregating, so per-class metrics & confusion
       matrices are interpretable.

    4. val split was unused; training ran for a fixed 2000 episodes and
       always reported the most-overfit final state.
       Fix: validation every val_every episodes, save best-val checkpoint,
       early stop on patience. Test is evaluated ONCE at the end with the
       best-val checkpoint.

    5. Tiny test set (~4 images for some classes) re-sampled 1000x with
       n_query=15 inflated the apparent stability.
       Fix: in addition to standard episodic test eval, also run a
       deterministic full_test_eval that classifies every test image
       once per support-seed, repeated over n_test_seeds (default 30).
       This gives a true per-class confusion matrix on the actual test set.

    6. Original sampler had a fallback `images * N` that silently mixed
       support and query images when a class was small (data leakage).
       Fix: support and query are kept disjoint; query may sample with
       replacement from `remaining` only when absolutely necessary.

    7. os.listdir included .DS_Store and label .txt files.
       Fix: filter to image extensions only.

USAGE FROM A NOTEBOOK
---------------------
    from fewshot_lib import (
        make_train_transform, make_eval_transform,
        FewShotDataset, run_few_shot,
        summarize_results, plot_confusion_matrices, plot_training_curves,
        predict_image, CLASS_NAMES,
    )

    # Define your encoder (this is the only baseline-specific piece):
    class MyEncoder(nn.Module):
        def __init__(self): ...
        def forward(self, x): ...

    results = run_few_shot(
        encoder_factory=lambda: MyEncoder(),
        baseline_name='Baseline X',
        data_root='clean_dataset',
        n_support_list=[1, 5, 10],
        n_train_episodes=400, val_every=20, patience=5,
        n_eval_episodes=200, n_test_seeds=30,
        learning_rate=1e-4, weight_decay=1e-4,
    )
    summarize_results(results, baseline_name='Baseline X')
    plot_confusion_matrices(results, baseline_name='Baseline X')
    plot_training_curves(results, baseline_name='Baseline X')
"""

from __future__ import annotations

import copy
import io
import os
import random
from typing import Callable, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

import matplotlib.pyplot as plt
import seaborn as sns


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

CLASS_NAMES = ["-K", "-N", "-P", "FN"]


# ─────────────────────────────────────────────────────────────────────────────
# Transforms
# ─────────────────────────────────────────────────────────────────────────────

def make_train_transform(image_size: int = 224) -> transforms.Compose:
    """Augmented transform used during TRAINING.

    Augmentation is applied on-the-fly every time an image is loaded, so even
    when the same source image gets reused across episodes (or within an
    episode for very small classes), each instance becomes a different
    tensor. This is the standard PyTorch pattern.
    """
    return transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def make_eval_transform(image_size: int = 224) -> transforms.Compose:
    """Deterministic transform used during VAL and TEST (no randomness)."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def make_strong_tta_transform(image_size: int = 224) -> transforms.Compose:
    """Strong test-time-augmentation (TTA) transform.

    Used by full_test_eval to evaluate the model under perturbed versions of
    each test image. Each call applies a different random perturbation (so
    feeding the same source image n times yields n different tensors).

    Why we want this:
        With only 4 test images per class and a strong pretrained backbone,
        per-class F1 saturates at 1.000 on visually-distinct classes (-K, -P).
        TTA produces realistic perturbations of the test images, breaking the
        perfect-recognition regime and giving a robustness-aware metric. This
        is a standard technique used in computer-vision robustness benchmarks
        (e.g., ImageNet-C, cifar-c).

    The perturbations applied (each randomly):
        - RandomResizedCrop scale 0.5-0.95 (cuts ~20-50% of pixels)
        - Horizontal flip
        - Rotation up to ±25°
        - ColorJitter (brightness/contrast/saturation 0.3, hue 0.05)
        - Gaussian blur (sigma up to 1.5)
    """
    return transforms.Compose([
        transforms.Resize((image_size + 48, image_size + 48)),
        transforms.RandomResizedCrop(image_size, scale=(0.5, 0.95)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=25),
        transforms.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05
        ),
        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 1.5)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

def _list_images(folder: str) -> list[str]:
    return sorted([
        f for f in os.listdir(folder)
        if not f.startswith(".") and os.path.splitext(f)[1].lower() in IMG_EXTS
    ])


class FewShotDataset:
    """
    Folder layout:  root_dir/<class_name>/<image_file>

    sample_episode() returns:
        sx          : (n_way*k_shot, C, H, W)  support images
        sy_local    : (n_way*k_shot,)          support labels in [0..n_way-1]
        qx          : (n_way*q_query, C, H, W) query images
        qy_local    : (n_way*q_query,)         query labels in [0..n_way-1]
        qy_global   : (n_way*q_query,)         query labels as GLOBAL class IDs
        selected    : list[str]                the n_way real class names

    The local/global distinction is the fix for per-class metric correctness:
    'label 2' is a different real class on every episode, but the global ID
    is fixed (-K=0, -N=1, -P=2, FN=3).
    """

    def __init__(self, root_dir: str, transform: transforms.Compose):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted([
            d for d in os.listdir(root_dir)
            if not d.startswith(".") and os.path.isdir(os.path.join(root_dir, d))
        ])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.class_to_images = {
            cls: _list_images(os.path.join(root_dir, cls)) for cls in self.classes
        }
        for cls, imgs in self.class_to_images.items():
            assert len(imgs) > 0, f"No images in {os.path.join(root_dir, cls)}"

    # ----- I/O helpers -----
    def _load(self, cls: str, name: str) -> torch.Tensor:
        img = Image.open(os.path.join(self.root_dir, cls, name)).convert("RGB")
        return self.transform(img)

    def load_all(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ALL images and their global class IDs (used by full_test_eval)."""
        xs, ys = [], []
        for cls in self.classes:
            for n in self.class_to_images[cls]:
                xs.append(self._load(cls, n))
                ys.append(self.class_to_idx[cls])
        return torch.stack(xs), torch.tensor(ys)

    # ----- Episodic sampler -----
    def sample_episode(
        self, n_way: int, k_shot: int, q_query: int, rng: random.Random | None = None
    ):
        rng = rng if rng is not None else random
        if n_way > len(self.classes):
            raise ValueError(
                f"n_way={n_way} > available classes={len(self.classes)} "
                f"in {self.root_dir}"
            )
        selected = rng.sample(self.classes, n_way)

        sx, sy_local = [], []
        qx, qy_local, qy_global = [], [], []

        for label_local, cls in enumerate(selected):
            label_global = self.class_to_idx[cls]
            images = self.class_to_images[cls]

            if len(images) >= k_shot + q_query:
                sampled = rng.sample(images, k_shot + q_query)
                support_imgs = sampled[:k_shot]
                query_imgs   = sampled[k_shot:]
            else:
                # Tight class: keep support and query DISJOINT to avoid silent
                # leakage. Only the query may resample with replacement, and
                # only from the remaining (non-support) pool.
                if len(images) <= k_shot:
                    support_imgs = list(images)
                    remaining = list(images)
                else:
                    support_imgs = rng.sample(images, k_shot)
                    remaining = [im for im in images if im not in support_imgs]

                if len(remaining) >= q_query:
                    query_imgs = rng.sample(remaining, q_query)
                else:
                    query_imgs = [rng.choice(remaining) for _ in range(q_query)]

            for n in support_imgs:
                sx.append(self._load(cls, n)); sy_local.append(label_local)
            for n in query_imgs:
                qx.append(self._load(cls, n))
                qy_local.append(label_local)
                qy_global.append(label_global)

        return (
            torch.stack(sx),
            torch.tensor(sy_local),
            torch.stack(qx),
            torch.tensor(qy_local),
            torch.tensor(qy_global),
            selected,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Misc utilities
# ─────────────────────────────────────────────────────────────────────────────

def pick_device(verbose: bool = True) -> torch.device:
    if torch.backends.mps.is_available():
        dev = torch.device("mps")
    elif torch.cuda.is_available():
        dev = torch.device("cuda")
    else:
        dev = torch.device("cpu")
    if verbose:
        print(f"Using device: {dev}")
    return dev


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def freeze_backbone(module: nn.Module, freeze: bool = True) -> None:
    for p in module.parameters():
        p.requires_grad = not freeze


def trainable_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def _embed_in_chunks(encoder: nn.Module, x: torch.Tensor, device: torch.device,
                     batch: int = 64) -> torch.Tensor:
    embs = []
    for i in range(0, x.size(0), batch):
        embs.append(encoder(x[i:i + batch].to(device)))
    return torch.cat(embs, dim=0)


# ─────────────────────────────────────────────────────────────────────────────
# Save / load checkpoints
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(
    path: str,
    encoder: nn.Module,
    extra_modules: Sequence[nn.Module] = (),
    *,
    metadata: dict | None = None,
) -> None:
    """Save encoder (and any extra trainable modules) to disk.

    The saved file is a torch.save dict with keys:
        - 'encoder'  : encoder.state_dict()
        - 'extras'   : list of state_dicts for extra modules (empty by default)
        - 'metadata' : free-form dict (best_val_acc, episode, hyperparams, etc.)
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "encoder": {k: v.detach().cpu() for k, v in encoder.state_dict().items()},
        "extras":  [
            {k: v.detach().cpu() for k, v in m.state_dict().items()}
            for m in extra_modules
        ],
        "metadata": metadata or {},
    }
    torch.save(payload, path)


def load_checkpoint(
    path: str,
    encoder: nn.Module,
    extra_modules: Sequence[nn.Module] = (),
    *,
    map_location: str | torch.device | None = None,
) -> dict:
    """Load a checkpoint produced by save_checkpoint into the given modules.

    Returns the metadata dict from the file.
    """
    payload = torch.load(path, map_location=map_location, weights_only=False)
    encoder.load_state_dict(payload["encoder"])
    for m, sd in zip(extra_modules, payload.get("extras", [])):
        m.load_state_dict(sd)
    return payload.get("metadata", {})


def evaluate_from_checkpoint(
    encoder_factory: Callable[[], nn.Module],
    checkpoint_path: str,
    *,
    data_root: str = "clean_dataset",
    n_way: int = 4,
    k_shot: int = 1,
    n_query: int = 10,
    n_eval_episodes: int = 200,
    n_test_seeds: int = 30,
    n_augs_per_image: int = 10,
    use_tta: bool = True,
    seed: int = 42,
    image_size: int = 224,
):
    """Load a saved encoder and run the full test evaluation, no training.

    Returns the same per-shot metric dict shape as run_few_shot's results,
    but with only the keys that are actually computable from a checkpoint
    (no `history`, no `best_val_*`).

    By default uses Test-Time Augmentation (TTA) so the metrics do not
    saturate at 1.000 on the small test set.
    """
    set_seed(seed)
    device = pick_device(verbose=True)

    eval_tx  = make_eval_transform(image_size)
    tta_tx   = make_strong_tta_transform(image_size) if use_tta else None
    train_data = FewShotDataset(os.path.join(data_root, "train"), eval_tx)
    test_data  = FewShotDataset(os.path.join(data_root, "test"),  eval_tx)
    n_classes  = len(test_data.classes)

    encoder = encoder_factory().to(device)
    md = load_checkpoint(checkpoint_path, encoder, map_location=device)
    print(f"Loaded checkpoint: {checkpoint_path}")
    if md:
        print(f"  metadata: {md}")
    encoder.eval()

    rng_test = random.Random(seed + 1234)
    acc, all_preds, all_labels, ep_accs = episodic_eval(
        encoder, test_data, n_way, k_shot, n_query,
        n_episodes=n_eval_episodes, device=device, rng=rng_test,
    )
    prec_macro = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    rec_macro  = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1_macro   = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    cm         = confusion_matrix(all_labels, all_preds, labels=list(range(n_classes)))
    stab_mean, stab_std = ep_accs.mean(), ep_accs.std()
    stab_ci  = 1.96 * stab_std / np.sqrt(len(ep_accs))

    seed_accs, full_preds, full_labels = full_test_eval(
        encoder, train_data, test_data, eval_tx,
        k_shot, n_seeds=n_test_seeds, device=device,
        tta_transform=tta_tx, n_augs_per_image=n_augs_per_image if use_tta else 1,
    )
    full_cm   = confusion_matrix(full_labels, full_preds, labels=list(range(n_classes)))
    full_prec = precision_score(full_labels, full_preds, average=None,
                                zero_division=0, labels=list(range(n_classes)))
    full_rec  = recall_score(full_labels, full_preds, average=None,
                             zero_division=0, labels=list(range(n_classes)))
    full_f1   = f1_score(full_labels, full_preds, average=None,
                         zero_division=0, labels=list(range(n_classes)))

    return {
        f"{k_shot}-shot": {
            "accuracy": acc, "precision": prec_macro, "recall": rec_macro,
            "f1_score": f1_macro,
            "confusion_matrix": cm,
            "stability_mean": stab_mean, "stability_std": stab_std,
            "stability_ci":   stab_ci,   "episode_accuracies": ep_accs,
            "full_seed_accs": seed_accs, "full_confusion_matrix": full_cm,
            "full_per_class_precision": full_prec,
            "full_per_class_recall":    full_rec,
            "full_per_class_f1":        full_f1,
            # No history/best_val_* — these only exist after a training run.
            "history": {"train_loss": [], "train_acc": [],
                        "val_acc": [], "val_episode": []},
            "best_val_acc": float("nan"),
            "best_val_episode": -1,
            "checkpoint_metadata": md,
        }
    }


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def episodic_eval(
    encoder: nn.Module,
    data: FewShotDataset,
    n_way: int,
    k_shot: int,
    q_query: int,
    n_episodes: int,
    device: torch.device,
    rng: random.Random,
):
    """Standard episodic evaluation (matches few-shot literature reporting).

    Returns:
        acc            : overall accuracy on aggregated query predictions
        all_preds      : np.ndarray of GLOBAL class IDs (predictions)
        all_labels     : np.ndarray of GLOBAL class IDs (true)
        ep_accs        : np.ndarray of per-episode accuracies (length=n_episodes)
    """
    encoder.eval()
    all_preds, all_labels, ep_accs = [], [], []
    with torch.no_grad():
        for _ in range(n_episodes):
            sx, sy, qx, qy_l, qy_g, sel = data.sample_episode(
                n_way, k_shot, q_query, rng=rng
            )
            sx, sy = sx.to(device), sy.to(device)
            qx = qx.to(device)
            s_emb = encoder(sx); q_emb = encoder(qx)
            protos = torch.stack([s_emb[sy == c].mean(0) for c in range(n_way)])
            preds_local = torch.cdist(q_emb, protos).argmin(1).cpu().numpy()
            local_to_global = np.array([data.class_to_idx[c] for c in sel])
            preds_global = local_to_global[preds_local]
            labels_global = qy_g.numpy()
            all_preds.append(preds_global); all_labels.append(labels_global)
            ep_accs.append((preds_global == labels_global).mean())
    all_preds  = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    return accuracy_score(all_labels, all_preds), all_preds, all_labels, np.array(ep_accs)


def full_test_eval(
    encoder: nn.Module,
    train_data: FewShotDataset,
    test_data: FewShotDataset,
    eval_tx: transforms.Compose,
    k_shot: int,
    n_seeds: int,
    device: torch.device,
    *,
    tta_transform: "transforms.Compose | None" = None,
    n_augs_per_image: int = 1,
):
    """Deterministic eval: classify every test image, repeat n_seeds times.

    For each seed:
        - draw k_shot random SUPPORT images per class (from train, no aug)
        - build prototypes
        - classify each test image n_augs_per_image times under
          `tta_transform` (or once with `eval_tx` if tta_transform is None)

    With tta_transform set (default in run_few_shot via make_strong_tta_transform),
    every test image is perturbed multiple times — random crop, color jitter,
    rotation, blur — and each perturbation is classified independently. This
    gives a robustness-aware metric that does NOT saturate at 1.000 on small
    test splits.

    Returns:
        seed_accs       : np.ndarray of shape (n_seeds,)
        all_run_preds   : np.ndarray of all predictions
        all_run_labels  : np.ndarray of all true labels
    """
    encoder.eval()
    n_classes = len(test_data.classes)
    use_tta   = tta_transform is not None and n_augs_per_image > 1

    # If no TTA, embed every test image ONCE and reuse.
    if not use_tta:
        test_x, test_y = test_data.load_all()
        with torch.no_grad():
            test_emb_static = _embed_in_chunks(encoder, test_x, device)
        labels_static = test_y.numpy()
    else:
        test_emb_static = None
        labels_static   = None

    seed_accs = []
    all_run_preds, all_run_labels = [], []
    for seed in range(n_seeds):
        rng = random.Random(1000 + seed)
        # Build support / prototypes (always with eval_tx — clean support).
        sx, sy = [], []
        for cls in train_data.classes:
            imgs = train_data.class_to_images[cls]
            picks = rng.sample(imgs, min(k_shot, len(imgs)))
            for n in picks:
                img = Image.open(
                    os.path.join(train_data.root_dir, cls, n)
                ).convert("RGB")
                sx.append(eval_tx(img))
                sy.append(train_data.class_to_idx[cls])
        sx = torch.stack(sx).to(device); sy = torch.tensor(sy)
        with torch.no_grad():
            s_emb = encoder(sx)
            protos = torch.stack(
                [s_emb[sy == c].mean(0) for c in range(n_classes)]
            )

        # Embed test images (TTA per-seed if requested, else use cached).
        if use_tta:
            torch.manual_seed(2000 + seed)
            aug_x, aug_y = [], []
            for cls in test_data.classes:
                cls_idx = test_data.class_to_idx[cls]
                for img_name in test_data.class_to_images[cls]:
                    img = Image.open(
                        os.path.join(test_data.root_dir, cls, img_name)
                    ).convert("RGB")
                    for _ in range(n_augs_per_image):
                        aug_x.append(tta_transform(img))
                        aug_y.append(cls_idx)
            aug_x = torch.stack(aug_x)
            with torch.no_grad():
                test_emb = _embed_in_chunks(encoder, aug_x, device)
            labels = np.array(aug_y)
        else:
            test_emb = test_emb_static
            labels   = labels_static

        with torch.no_grad():
            preds = torch.cdist(test_emb, protos).argmin(1).cpu().numpy()

        seed_accs.append((preds == labels).mean())
        all_run_preds.append(preds); all_run_labels.append(labels)

    return (
        np.array(seed_accs),
        np.concatenate(all_run_preds),
        np.concatenate(all_run_labels),
    )


def _pil_image_from_input(
    image: "str | os.PathLike | Image.Image | bytes | bytearray | memoryview",
) -> Image.Image:
    """Load a user-provided image as RGB PIL.Image."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, (bytes, bytearray, memoryview)):
        return Image.open(io.BytesIO(bytes(image))).convert("RGB")
    return Image.open(image).convert("RGB")


def predict_image(
    encoder_factory: Callable[[], nn.Module],
    checkpoint_path: str,
    image: "str | os.PathLike | Image.Image | bytes | bytearray | memoryview",
    *,
    data_root: str = "clean_dataset",
    k_shot: int = 1,
    support_draw_index: int = 0,
    n_tta: int = 1,
    image_size: int = 224,
    device: torch.device | None = None,
) -> dict[str, object]:
    """Classify one image with a saved encoder using Prototypical inference.

    Builds class prototypes from **training** images (same protocol as
    ``full_test_eval``): for each class, ``k_shot`` support images are drawn
    with ``random.Random(1000 + support_draw_index)``, embedded with the
    eval transform, and averaged. The query image is embedded and assigned
    to the nearest prototype (Euclidean distance in embedding space).

    ``n_tta > 1`` averages the query embedding over that many random
    strong augmentations (same family as test-time augmentation), which
    often stabilises confidence on a single out-of-split photo.

    Args:
        encoder_factory: Same lambda/class used when training this baseline.
        checkpoint_path: Path to ``.pt`` from ``run_few_shot`` /
            ``save_checkpoint``.
        image: File path, ``PIL.Image``, or raw bytes (e.g. from Jupyter
            ``FileUpload``).
        data_root: Dataset root containing ``train/`` and ``test/``.
        k_shot: Support images per class when building prototypes.
        support_draw_index: Integer ``i`` such that support is drawn with
            ``Random(1000 + i)``, matching the ``i``-th seed in
            ``full_test_eval``.
        n_tta: If ``> 1``, average query embedding over this many TTA views.
        image_size: Input resolution (must match training).
        device: Optional torch device; default auto (MPS / CUDA / CPU).

    Returns:
        Dict with ``predicted_class``, ``class_id``, ``class_names``,
        ``distances`` (per class), ``probabilities`` (softmax of negative
        squared distances), plus meta keys.
    """
    if device is None:
        device = pick_device(verbose=False)

    eval_tx = make_eval_transform(image_size)
    tta_tx = make_strong_tta_transform(image_size) if n_tta > 1 else None

    train_root = os.path.join(data_root, "train")
    train_data = FewShotDataset(train_root, eval_tx)
    n_classes = len(train_data.classes)

    encoder = encoder_factory().to(device)
    load_checkpoint(checkpoint_path, encoder, map_location=device)
    encoder.eval()

    rng = random.Random(1000 + support_draw_index)
    sx, sy = [], []
    for cls in train_data.classes:
        imgs = train_data.class_to_images[cls]
        picks = rng.sample(imgs, min(k_shot, len(imgs)))
        for n in picks:
            img = Image.open(os.path.join(train_data.root_dir, cls, n)).convert(
                "RGB"
            )
            sx.append(eval_tx(img))
            sy.append(train_data.class_to_idx[cls])
    sx = torch.stack(sx).to(device)
    sy = torch.tensor(sy)
    with torch.no_grad():
        s_emb = encoder(sx)
        protos = torch.stack([s_emb[sy == c].mean(0) for c in range(n_classes)])

    pil = _pil_image_from_input(image)
    with torch.no_grad():
        if n_tta <= 1:
            q = eval_tx(pil).unsqueeze(0).to(device)
            q_emb = encoder(q)
        else:
            assert tta_tx is not None
            torch.manual_seed(3000 + support_draw_index)
            aug = torch.stack([tta_tx(pil) for _ in range(n_tta)]).to(device)
            q_emb = encoder(aug).mean(0, keepdim=True)

        dists = torch.cdist(q_emb, protos).squeeze(0)
        logits = -(dists ** 2)
        probs = torch.softmax(logits, dim=0)
        pred_id = int(dists.argmin().item())

    names = list(train_data.classes)
    dists_np = dists.cpu().numpy()
    probs_np = probs.cpu().numpy()
    return {
        "predicted_class": names[pred_id],
        "class_id": pred_id,
        "class_names": names,
        "distances": {names[i]: float(dists_np[i]) for i in range(n_classes)},
        "probabilities": {names[i]: float(probs_np[i]) for i in range(n_classes)},
        "checkpoint_path": checkpoint_path,
        "k_shot": k_shot,
        "support_draw_index": support_draw_index,
        "n_tta": n_tta,
    }


# ─────────────────────────────────────────────────────────────────────────────
# The unified train+eval entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_few_shot(
    encoder_factory: Callable[[], nn.Module],
    *,
    baseline_name: str = "Baseline",
    data_root: str = "clean_dataset",
    n_support_list: Sequence[int] = (1, 5, 10),
    n_way: int = 4,
    n_query: int = 10,
    n_train_episodes: int = 400,
    n_eval_episodes: int = 200,
    val_every: int = 20,
    early_stop: bool = False,
    patience: int = 5,
    n_test_seeds: int = 30,
    n_augs_per_image: int = 10,
    use_tta: bool = True,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    grad_clip: float = 5.0,
    seed: int = 42,
    image_size: int = 224,
    extra_optimized_modules: Sequence[nn.Module] = (),
    save_dir: str | None = "checkpoints",
):
    """Train + evaluate a Prototypical Network for each shot count.

    Pipeline:
      1) Train on `train` split for ALL `n_train_episodes` (with augmentation).
      2) Validate on `val` split every `val_every` episodes (no aug); track
         training-episode accuracy on every step too.
      3) Always keep the BEST-VAL checkpoint in memory (and on disk if
         `save_dir` is given) — even when training continues past it.
         If `early_stop=True`, also halt after `patience` non-improvements.
      4) Load best-val state, then evaluate ONCE on `test`:
           a) Episodic test eval  (n_eval_episodes episodes).
           b) Full deterministic test eval (n_test_seeds support draws).

    Recommended setting (current default): `early_stop=False`. This makes
    every baseline run all `n_train_episodes` episodes so training-curve
    plots are directly comparable across baselines. The best-val checkpoint
    is still used for evaluation, so this does NOT cause overfitting in the
    reported metrics — it only changes which curves are visualised.

    Set `save_dir=None` to disable disk saving.

    Returns: dict keyed by f"{n_support}-shot" with all metrics + history.
    """
    set_seed(seed)
    device = pick_device(verbose=True)

    train_tx = make_train_transform(image_size)
    eval_tx  = make_eval_transform(image_size)
    tta_tx   = make_strong_tta_transform(image_size) if use_tta else None

    train_data = FewShotDataset(os.path.join(data_root, "train"), train_tx)
    val_data   = FewShotDataset(os.path.join(data_root, "val"),   eval_tx)
    test_data  = FewShotDataset(os.path.join(data_root, "test"),  eval_tx)

    if use_tta:
        print(f"  Test-Time Augmentation (TTA): ENABLED  "
              f"({n_augs_per_image} perturbations per test image)")
    else:
        print(f"  Test-Time Augmentation (TTA): disabled")

    # Sanity check: same class set in each split, in the same order.
    assert train_data.classes == val_data.classes == test_data.classes, (
        f"Class mismatch across splits: train={train_data.classes} "
        f"val={val_data.classes} test={test_data.classes}"
    )
    print(f"[{baseline_name}] Classes: {train_data.classes}")
    for split, d in [("train", train_data), ("val", val_data), ("test", test_data)]:
        print(f"  {split}: {{c: len(d.class_to_images[c]) for c in d.classes}}".replace(
            "{c: len(d.class_to_images[c]) for c in d.classes}",
            str({c: len(d.class_to_images[c]) for c in d.classes})
        ))

    n_classes = len(train_data.classes)
    if n_way > n_classes:
        raise ValueError(f"n_way={n_way} but only {n_classes} classes available.")

    results = {}

    for n_support in n_support_list:
        print(f"\n{'='*70}\n  [{baseline_name}] {n_support}-shot learning\n{'='*70}")

        encoder = encoder_factory().to(device)
        all_modules = [encoder] + list(extra_optimized_modules)
        params = []
        for m in all_modules:
            params += [p for p in m.parameters() if p.requires_grad]
        n_train_params = sum(p.numel() for p in params)
        print(f"  Trainable params: {n_train_params:,}")

        optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)

        # ---- Training ----
        rng_train = random.Random(seed)
        best_val_acc = -1.0
        best_state   = None
        best_episode = 0
        bad_checks   = 0
        history = {
            "train_loss":     [],
            "train_acc":      [],
            "train_episode":  [],
            "val_acc":        [],
            "val_episode":    [],
        }

        ckpt_path = None
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            safe_name = baseline_name.replace(" ", "_").replace("/", "_")
            ckpt_path = os.path.join(save_dir, f"{safe_name}_{n_support}shot.pt")

        for ep in range(1, n_train_episodes + 1):
            for m in all_modules:
                m.train()

            sx, sy, qx, qy_l, qy_g, _ = train_data.sample_episode(
                n_way, n_support, n_query, rng=rng_train
            )
            sx, sy, qx, qy_l = (
                sx.to(device), sy.to(device), qx.to(device), qy_l.to(device),
            )

            s_emb = encoder(sx); q_emb = encoder(qx)
            protos = torch.stack([s_emb[sy == c].mean(0) for c in range(n_way)])
            logits = -torch.cdist(q_emb, protos)
            loss = F.cross_entropy(logits, qy_l)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
            optimizer.step()

            # Per-episode training accuracy (on the query set of this episode).
            with torch.no_grad():
                train_acc = (logits.argmax(1) == qy_l).float().mean().item()
            history["train_loss"].append(loss.item())
            history["train_acc"].append(train_acc)
            history["train_episode"].append(ep)

            if ep % val_every == 0 or ep == n_train_episodes:
                val_rng = random.Random(seed + 2 + ep)
                val_acc, _, _, _ = episodic_eval(
                    encoder, val_data, n_way, n_support, n_query,
                    n_episodes=50, device=device, rng=val_rng,
                )
                history["val_acc"].append(val_acc)
                history["val_episode"].append(ep)
                msg = (f"  ep {ep:>4}/{n_train_episodes}  "
                       f"loss={loss.item():.4f}  train_acc={train_acc:.4f}  "
                       f"val_acc={val_acc:.4f}")
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_state   = {
                        "encoder": copy.deepcopy(encoder.state_dict()),
                        "extras":  [copy.deepcopy(m.state_dict())
                                    for m in extra_optimized_modules],
                    }
                    best_episode = ep
                    bad_checks   = 0
                    msg += "  ← new best"
                    if ckpt_path is not None:
                        save_checkpoint(
                            ckpt_path, encoder, extra_optimized_modules,
                            metadata={
                                "baseline": baseline_name,
                                "n_support": n_support,
                                "n_way": n_way,
                                "best_val_acc": best_val_acc,
                                "best_episode": best_episode,
                                "learning_rate": learning_rate,
                                "weight_decay": weight_decay,
                                "n_train_episodes_planned": n_train_episodes,
                            },
                        )
                        msg += f"  → saved {ckpt_path}"
                else:
                    bad_checks += 1
                    msg += f"  (no-improve {bad_checks}/{patience})"
                print(msg)
                if early_stop and bad_checks >= patience:
                    print(f"  Early stop at episode {ep}. "
                          f"Best val={best_val_acc:.4f} @ ep {best_episode}")
                    break

        if best_state is not None:
            encoder.load_state_dict(best_state["encoder"])
            for m, sd in zip(extra_optimized_modules, best_state["extras"]):
                m.load_state_dict(sd)
        print(f"  Loaded best-val checkpoint "
              f"(val_acc={best_val_acc:.4f}, ep={best_episode})")
        if ckpt_path is not None:
            print(f"  Best checkpoint persisted at: {ckpt_path}")

        # ---- Episodic test evaluation ----
        rng_test = random.Random(seed + 1234)
        acc, all_preds, all_labels, ep_accs = episodic_eval(
            encoder, test_data, n_way, n_support, n_query,
            n_episodes=n_eval_episodes, device=device, rng=rng_test,
        )
        prec_macro = precision_score(all_labels, all_preds, average="macro", zero_division=0)
        rec_macro  = recall_score(all_labels, all_preds, average="macro", zero_division=0)
        f1_macro   = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        cm         = confusion_matrix(
            all_labels, all_preds, labels=list(range(n_classes))
        )
        stab_mean, stab_std = ep_accs.mean(), ep_accs.std()
        stab_ci  = 1.96 * stab_std / np.sqrt(len(ep_accs))

        # ---- Full deterministic test eval ----
        seed_accs, full_preds, full_labels = full_test_eval(
            encoder, train_data, test_data, eval_tx,
            n_support, n_seeds=n_test_seeds, device=device,
            tta_transform=tta_tx, n_augs_per_image=n_augs_per_image if use_tta else 1,
        )
        full_cm = confusion_matrix(
            full_labels, full_preds, labels=list(range(n_classes))
        )
        full_prec = precision_score(full_labels, full_preds, average=None,
                                    zero_division=0, labels=list(range(n_classes)))
        full_rec  = recall_score(full_labels, full_preds, average=None,
                                 zero_division=0, labels=list(range(n_classes)))
        full_f1   = f1_score(full_labels, full_preds, average=None,
                             zero_division=0, labels=list(range(n_classes)))

        results[f"{n_support}-shot"] = {
            "accuracy": acc, "precision": prec_macro, "recall": rec_macro,
            "f1_score": f1_macro,
            "confusion_matrix": cm,
            "stability_mean": stab_mean, "stability_std": stab_std,
            "stability_ci":   stab_ci,   "episode_accuracies": ep_accs,
            "full_seed_accs": seed_accs, "full_confusion_matrix": full_cm,
            "full_per_class_precision": full_prec,
            "full_per_class_recall":    full_rec,
            "full_per_class_f1":        full_f1,
            "history": history,
            "best_val_acc": best_val_acc, "best_val_episode": best_episode,
            "checkpoint_path": ckpt_path,
        }

        print(f"\n  [Episodic test]  Acc={acc:.4f}  P={prec_macro:.4f}  "
              f"R={rec_macro:.4f}  F1={f1_macro:.4f}")
        print(f"                   Stability: {stab_mean:.4f} ± {stab_std:.4f}  "
              f"(95% CI ±{stab_ci:.4f})")
        print(f"  [Full-test eval] Mean acc per seed: "
              f"{seed_accs.mean():.4f} ± {seed_accs.std():.4f}  "
              f"(n_seeds={n_test_seeds})")
        print(f"                   Per-class F1 ({CLASS_NAMES}): "
              f"{[f'{x:.3f}' for x in full_f1]}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostics
# ─────────────────────────────────────────────────────────────────────────────

def diagnose_checkpoint(
    encoder_factory: Callable[[], nn.Module],
    checkpoint_path: str,
    *,
    data_root: str = "clean_dataset",
    k_shot: int = 1,
    n_seeds: int = 30,
    n_augs_per_image: int = 10,
    use_tta: bool = True,
    seed: int = 42,
    image_size: int = 224,
) -> dict:
    """Plain-English diagnosis of which classes a saved model is confusing.

    Loads the checkpoint, runs the deterministic full-test eval, and prints:
      - confusion matrix in tabular form with per-row recall, per-col precision
      - top error pairs (true → predicted) ranked by frequency
      - per-class breakdown in English: 'X of true class -N were predicted as Y'

    Returns the confusion matrix as a numpy array (n_classes, n_classes) so
    callers can do further analysis if they want.
    """
    if not os.path.exists(checkpoint_path):
        print(f"NOT FOUND: {checkpoint_path}")
        return np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=int)

    set_seed(seed)
    device = pick_device(verbose=False)
    eval_tx = make_eval_transform(image_size)
    tta_tx  = make_strong_tta_transform(image_size) if use_tta else None

    train_data = FewShotDataset(os.path.join(data_root, "train"), eval_tx)
    test_data  = FewShotDataset(os.path.join(data_root, "test"),  eval_tx)
    n_classes  = len(test_data.classes)

    encoder = encoder_factory().to(device)
    md = load_checkpoint(checkpoint_path, encoder, map_location=device)
    encoder.eval()
    print(f"Loaded: {checkpoint_path}")
    if md:
        print(f"  metadata: {md}\n")
    if use_tta:
        print(f"Test-Time Augmentation: ENABLED ({n_augs_per_image} perturbations per image)\n")

    seed_accs, all_preds, all_labels = full_test_eval(
        encoder, train_data, test_data, eval_tx,
        k_shot=k_shot, n_seeds=n_seeds, device=device,
        tta_transform=tta_tx, n_augs_per_image=n_augs_per_image if use_tta else 1,
    )

    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(all_labels, all_preds):
        cm[t, p] += 1

    total = cm.sum()
    diag  = sum(cm[i, i] for i in range(n_classes))
    err   = total - diag

    # Tabular CM
    print(f"Confusion matrix on {n_seeds} support draws × full test "
          f"({total} total predictions):\n")
    header_label = "TRUE/PRED"
    print(f"{header_label:>10} | "
          + " | ".join(f"{c:>7}" for c in CLASS_NAMES)
          + " | TOTAL  | RECALL")
    print("-" * 78)
    for i in range(n_classes):
        row_total = cm[i].sum()
        recall = cm[i, i] / row_total if row_total > 0 else 0
        cells  = " | ".join(f"{cm[i, j]:>7}" for j in range(n_classes))
        print(f"{CLASS_NAMES[i]:>10} | {cells} | {row_total:>5}  | {recall:.3f}")
    print("-" * 78)
    col_totals = cm.sum(axis=0)
    print(f"{'TOTAL':>10} | "
          + " | ".join(f"{x:>7}" for x in col_totals)
          + f" | {total}")
    precs = [cm[i, i] / col_totals[i] if col_totals[i] > 0 else 0
             for i in range(n_classes)]
    print(f"{'PRECISION':>10} | "
          + " | ".join(f"{p:>7.3f}" for p in precs))

    # Top error pairs
    print(f"\n{'─'*78}\nWhere does the error come from?\n{'─'*78}")
    print(f"  Total predictions: {total}")
    print(f"  Correct          : {diag} ({diag/total:.1%})")
    print(f"  Errors           : {err} ({err/total:.1%})")
    pairs = [
        (cm[i, j], CLASS_NAMES[i], CLASS_NAMES[j])
        for i in range(n_classes) for j in range(n_classes)
        if i != j and cm[i, j] > 0
    ]
    pairs.sort(reverse=True)
    print("\n  Top error pairs (true → predicted):")
    for count, t, p in pairs[:5]:
        print(f"    {t:>4}  →  {p:<4}  : {count:>4}  ({count/total:.1%} of all predictions)")

    # Per-class English explanation
    print(f"\n{'─'*78}\nPer-class breakdown\n{'─'*78}")
    for i in range(n_classes):
        c = CLASS_NAMES[i]
        row_total = cm[i].sum()
        col_total = col_totals[i]
        recall = cm[i, i] / row_total if row_total > 0 else 0
        prec   = cm[i, i] / col_total if col_total > 0 else 0
        print(f"\n  {c}:")
        print(f"    True {c} test predictions ({n_seeds} seeds): {row_total}")
        print(f"    Correctly predicted as {c}: {cm[i, i]} ({recall:.1%})")
        if recall < 1.0:
            misses = [(cm[i, j], CLASS_NAMES[j])
                      for j in range(n_classes) if j != i and cm[i, j] > 0]
            misses.sort(reverse=True)
            for cnt, mname in misses:
                print(f"      Missed → predicted as {mname}: {cnt}")
        if prec < 1.0:
            wrongs = [(cm[j, i], CLASS_NAMES[j])
                      for j in range(n_classes) if j != i and cm[j, i] > 0]
            wrongs.sort(reverse=True)
            print(f"    False alarms ({c} was predicted but truth was something else):")
            for cnt, oname in wrongs:
                print(f"      {cnt} predictions of {c} were actually {oname}")
        print(f"    Recall = {recall:.3f},  Precision = {prec:.3f}")

    return cm


# ─────────────────────────────────────────────────────────────────────────────
# Reporting helpers
# ─────────────────────────────────────────────────────────────────────────────

def summarize_results(results: dict, baseline_name: str = "Baseline") -> None:
    """Print the overall + per-class summary tables."""
    macro_rows = []
    for shot_type, m in results.items():
        macro_rows.append({
            "Shot":            shot_type,
            "Accuracy":        f"{m['accuracy']:.4f}",
            "Precision (M)":   f"{m['precision']:.4f}",
            "Recall (M)":      f"{m['recall']:.4f}",
            "F1 (M)":          f"{m['f1_score']:.4f}",
            "Episodic μ±σ":    f"{m['stability_mean']:.4f} ± {m['stability_std']:.4f}",
            "95% CI":          f"±{m['stability_ci']:.4f}",
            "Best val":        f"{m['best_val_acc']:.4f} @ ep{m['best_val_episode']}",
            "Full-test μ±σ":   f"{m['full_seed_accs'].mean():.4f} ± "
                               f"{m['full_seed_accs'].std():.4f}",
        })
    print("=" * 110)
    print(f"  {baseline_name} — overall")
    print("=" * 110)
    print(pd.DataFrame(macro_rows).to_string(index=False))
    print("=" * 110)

    print("\nPer-class metrics on TEST split (deterministic full-test eval):\n")
    rows = []
    for shot_type, m in results.items():
        for ci, cls in enumerate(CLASS_NAMES):
            rows.append({
                "Shot": shot_type, "Class": cls,
                "Precision": f"{m['full_per_class_precision'][ci]:.4f}",
                "Recall":    f"{m['full_per_class_recall'][ci]:.4f}",
                "F1":        f"{m['full_per_class_f1'][ci]:.4f}",
            })
    print(pd.DataFrame(rows).to_string(index=False))


def plot_confusion_matrices(results: dict, baseline_name: str = "Baseline") -> None:
    """Render two rows of confusion matrices (episodic + full-test)."""
    n = len(results)
    fig, axes = plt.subplots(2, n, figsize=(6 * n, 10))
    if n == 1:
        axes = axes.reshape(2, 1)
    for idx, (shot_type, m) in enumerate(results.items()):
        sns.heatmap(m["confusion_matrix"], annot=True, fmt="d", cmap="Blues",
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                    ax=axes[0, idx], cbar=False)
        axes[0, idx].set_title(f"{shot_type} — episodic", fontsize=12, fontweight="bold")
        axes[0, idx].set_xlabel("Predicted"); axes[0, idx].set_ylabel("True")

        sns.heatmap(m["full_confusion_matrix"], annot=True, fmt="d", cmap="Greens",
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                    ax=axes[1, idx], cbar=False)
        axes[1, idx].set_title(
            f"{shot_type} — full test (×{len(m['full_seed_accs'])} seeds)",
            fontsize=12, fontweight="bold",
        )
        axes[1, idx].set_xlabel("Predicted"); axes[1, idx].set_ylabel("True")

    plt.suptitle(f"{baseline_name} — Confusion Matrices",
                 fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.show()


def _smoothed(xs: list[float], window: int = 20) -> list[float]:
    """Simple moving-average smoothing for noisy per-episode metrics."""
    if len(xs) == 0:
        return xs
    arr = np.asarray(xs, dtype=float)
    if window <= 1 or len(arr) < window:
        return arr.tolist()
    kernel = np.ones(window) / window
    sm = np.convolve(arr, kernel, mode="same")
    # Trim edge artefacts by repeating the boundary mean
    half = window // 2
    sm[:half] = arr[:half].mean() if half > 0 else sm[:half]
    sm[-half:] = arr[-half:].mean() if half > 0 else sm[-half:]
    return sm.tolist()


def plot_training_loss_and_accuracy(
    results: dict, baseline_name: str = "Baseline", smooth_window: int = 20
) -> None:
    """Two separate figures: (1) training loss only, (2) train + val accuracy.

    Loss and accuracy are NOT overlaid on twin axes — each gets its own figure
    so they export cleanly for a thesis/paper.
    """
    n = len(results)

    # ── Figure 1: training loss only ─────────────────────────────────────
    fig1, axes1 = plt.subplots(1, n, figsize=(6 * n, 4.5))
    if n == 1:
        axes1 = [axes1]
    for ax, (shot_type, m) in zip(axes1, results.items()):
        hist = m["history"]
        ep, loss = hist["train_episode"], hist["train_loss"]
        ax.plot(ep, loss, color="steelblue", linewidth=0.6, alpha=0.35,
                label="Train loss (raw)")
        if smooth_window > 1:
            ax.plot(ep, _smoothed(loss, smooth_window),
                    color="steelblue", linewidth=2.0,
                    label=f"Train loss (avg-{smooth_window})")
        if m["best_val_episode"] >= 0:
            ax.axvline(m["best_val_episode"], color="red", linestyle="--",
                       alpha=0.7, label=f"Best-val @ ep{m['best_val_episode']}")
        ax.set_xlabel("Training episode")
        ax.set_ylabel("Loss")
        ax.set_yscale("symlog", linthresh=1e-3)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=7, framealpha=0.9)
        ax.set_title(f"{shot_type}", fontsize=12, fontweight="bold")
    fig1.suptitle(f"{baseline_name} — Training loss", fontsize=15, fontweight="bold")
    fig1.tight_layout()
    plt.show()

    # ── Figure 2: training accuracy + validation accuracy ─────────────────
    fig2, axes2 = plt.subplots(1, n, figsize=(6 * n, 4.5))
    if n == 1:
        axes2 = [axes2]
    for ax, (shot_type, m) in zip(axes2, results.items()):
        hist = m["history"]
        ep, tacc = hist["train_episode"], hist["train_acc"]
        ax.plot(ep, tacc, color="green", linewidth=0.6, alpha=0.35,
                label="Train acc (raw)")
        if smooth_window > 1:
            ax.plot(ep, _smoothed(tacc, smooth_window),
                    color="green", linewidth=2.0,
                    label=f"Train acc (avg-{smooth_window})")
        ax.plot(hist["val_episode"], hist["val_acc"],
                color="darkorange", marker="o", linewidth=2, label="Val acc")
        if m["best_val_episode"] >= 0:
            ax.axvline(m["best_val_episode"], color="red", linestyle="--",
                       alpha=0.7, label=f"Best-val @ ep{m['best_val_episode']}")
        ax.set_xlabel("Training episode")
        ax.set_ylabel("Accuracy")
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=7, framealpha=0.9)
        ax.set_title(f"{shot_type}", fontsize=12, fontweight="bold")
    fig2.suptitle(f"{baseline_name} — Training & validation accuracy",
                  fontsize=15, fontweight="bold")
    fig2.tight_layout()
    plt.show()


def plot_training_curves(results: dict, baseline_name: str = "Baseline") -> None:
    """3-row panel: training loss, training/val accuracy, episodic test stability.

    Loss and accuracy are on separate rows (no twin y-axis) for clearer figures.
    """
    n = len(results)
    fig, axes = plt.subplots(3, n, figsize=(6 * n, 12))
    if n == 1:
        axes = axes.reshape(3, 1)

    for idx, (shot_type, m) in enumerate(results.items()):
        hist = m["history"]
        ax_loss = axes[0, idx]
        ax_loss.plot(hist["train_episode"], hist["train_loss"],
                     color="steelblue", linewidth=0.6, alpha=0.4,
                     label="Train loss (raw)")
        ax_loss.plot(hist["train_episode"], _smoothed(hist["train_loss"], 20),
                     color="steelblue", linewidth=1.8, label="Train loss (avg-20)")
        if m["best_val_episode"] >= 0:
            ax_loss.axvline(m["best_val_episode"], color="red", linestyle="--",
                            alpha=0.7, label=f"Best @ ep{m['best_val_episode']}")
        ax_loss.set_xlabel("Training episode")
        ax_loss.set_ylabel("Loss")
        ax_loss.set_yscale("symlog", linthresh=1e-3)
        ax_loss.grid(True, alpha=0.3)
        ax_loss.legend(loc="upper right", fontsize=7, framealpha=0.85)
        ax_loss.set_title(f"{shot_type} — training loss",
                          fontsize=12, fontweight="bold")

        ax_acc = axes[1, idx]
        ax_acc.plot(hist["train_episode"], hist["train_acc"],
                    color="green", linewidth=0.6, alpha=0.35,
                    label="Train acc (raw)")
        ax_acc.plot(hist["train_episode"], _smoothed(hist["train_acc"], 20),
                    color="green", linewidth=1.5, alpha=0.9,
                    label="Train acc (avg-20)")
        ax_acc.plot(hist["val_episode"], hist["val_acc"], color="darkorange",
                    marker="o", linewidth=2, label="Val acc")
        if m["best_val_episode"] >= 0:
            ax_acc.axvline(m["best_val_episode"], color="red", linestyle="--",
                           alpha=0.7, label=f"Best @ ep{m['best_val_episode']}")
        ax_acc.set_xlabel("Training episode")
        ax_acc.set_ylabel("Accuracy")
        ax_acc.set_ylim([0, 1.05])
        ax_acc.grid(True, alpha=0.3)
        ax_acc.legend(loc="lower right", fontsize=7, framealpha=0.85)
        ax_acc.set_title(f"{shot_type} — training & val accuracy",
                         fontsize=12, fontweight="bold")

        ep_accs = m["episode_accuracies"]
        ax_bot = axes[2, idx]
        ax_bot.plot(ep_accs, alpha=0.5, linewidth=1, color="steelblue",
                    label="Episode acc")
        ax_bot.axhline(m["stability_mean"], color="red", linestyle="--", linewidth=2,
                       label=f"Mean: {m['stability_mean']:.4f}")
        ax_bot.fill_between(
            range(len(ep_accs)),
            m["stability_mean"] - m["stability_std"],
            m["stability_mean"] + m["stability_std"],
            alpha=0.2, color="red", label=f"±Std: {m['stability_std']:.4f}",
        )
        ax_bot.set_xlabel("Test episode"); ax_bot.set_ylabel("Accuracy")
        ax_bot.set_ylim([0, 1.05]); ax_bot.grid(True, alpha=0.3)
        ax_bot.legend(loc="lower right", fontsize=8)
        ax_bot.set_title(f"{shot_type} — episodic test stability",
                         fontsize=12, fontweight="bold")

    plt.suptitle(f"{baseline_name} — Training & Stability",
                 fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.show()
