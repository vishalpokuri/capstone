"""
Rebuilds the 5 baseline notebooks from a uniform template.

Each notebook has identical methodology (imported from fewshot_lib.py) and
differs ONLY in the encoder definition cell.

Run once after editing this file:
    python3 _rebuild_baselines.py

This file is a build-tool, not part of the experiment. Safe to delete.
"""

from __future__ import annotations

import json
import textwrap
import uuid
from pathlib import Path

NB_DIR = Path(__file__).parent

# ─── Per-baseline configuration ─────────────────────────────────────────────

BASELINES = [
    {
        "filename": "Baseline1_ResNet18_ProtoNet_FewShot.ipynb",
        "title": "Baseline 1: ResNet18 + Prototypical Network (Few-Shot Learning)",
        "summary": (
            "**Backbone:** ResNet18 (ImageNet-pretrained)\n"
            "**Features:** final-layer global-avg-pool features  →  512-D embedding\n"
            "**Classifier head:** Prototypical Network (Euclidean distance to class prototypes)\n\n"
            "This is a *single-scale* baseline. It uses only the deepest layer of "
            "ResNet18, which is the standard ProtoNet recipe."
        ),
        "encoder_name": "ResNet18Encoder",
        "encoder_code": '''\
class ResNet18Encoder(nn.Module):
    """ResNet18 trunk used as a 512-D feature extractor.

    `freeze_until` regularises by freezing early conv blocks:
        'none'   : fine-tune everything (overfits on small data)
        'layer3' : freeze conv1/bn1/layer1/layer2 (recommended default)
        'layer4' : freeze everything except the last residual block
        'all'    : pure feature extractor, no fine-tuning
    """

    def __init__(self, freeze_until: str = "layer3"):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])  # drop FC
        self._apply_freeze(freeze_until)

    def _apply_freeze(self, mode: str):
        if mode == "none":
            return
        for p in self.encoder.parameters():
            p.requires_grad = False
        if mode == "all":
            return
        unfreeze_from = {"layer3": 6, "layer4": 7}[mode]
        for module in list(self.encoder.children())[unfreeze_from:]:
            for p in module.parameters():
                p.requires_grad = True

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)
''',
        "encoder_factory_call": "lambda: ResNet18Encoder(freeze_until='layer3')",
    },
    {
        "filename": "Baseline2_ResNet50_ProtoNet_FewShot.ipynb",
        "title": "Baseline 2: ResNet50 + Prototypical Network (Few-Shot Learning)",
        "summary": (
            "**Backbone:** ResNet50 (ImageNet-pretrained)\n"
            "**Features:** final-layer global-avg-pool features  →  2048-D embedding\n"
            "**Classifier head:** Prototypical Network\n\n"
            "Same recipe as Baseline 1 but with a deeper backbone — tests whether "
            "a larger trunk on its own improves performance."
        ),
        "encoder_name": "ResNet50Encoder",
        "encoder_code": '''\
class ResNet50Encoder(nn.Module):
    """ResNet50 trunk → 2048-D embedding.

    `freeze_until`: 'none' | 'layer3' | 'layer4' | 'all'  (see ResNet18Encoder).
    """

    def __init__(self, freeze_until: str = "layer3"):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])  # drop FC
        self._apply_freeze(freeze_until)

    def _apply_freeze(self, mode: str):
        if mode == "none":
            return
        for p in self.encoder.parameters():
            p.requires_grad = False
        if mode == "all":
            return
        unfreeze_from = {"layer3": 6, "layer4": 7}[mode]
        for module in list(self.encoder.children())[unfreeze_from:]:
            for p in module.parameters():
                p.requires_grad = True

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)
''',
        "encoder_factory_call": "lambda: ResNet50Encoder(freeze_until='layer3')",
    },
    {
        "filename": "Baseline3_VGG16_ProtoNet_FewShot_.ipynb",
        "title": "Baseline 3: VGG16 + Prototypical Network (Few-Shot Learning)",
        "summary": (
            "**Backbone:** VGG16 (ImageNet-pretrained)\n"
            "**Features:** final convolutional feature map → Global Average Pooling → 512-D embedding\n"
            "**Classifier head:** Prototypical Network\n\n"
            "VGG16 is a deeper, older-style architecture without residual connections. "
            "Compares a non-residual backbone against the ResNet baselines."
        ),
        "encoder_name": "VGG16Encoder",
        "encoder_code": '''\
class VGG16Encoder(nn.Module):
    """VGG16 features → GAP → 512-D embedding.

    Because VGG has no logical 'residual blocks', we expose two granularities
    of freezing:
        'none'  : fine-tune all conv layers
        'last3' : freeze all but the last 3 conv layers (recommended)
        'all'   : pure feature extractor
    """

    def __init__(self, freeze_until: str = "last3"):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        self.features = vgg.features        # nn.Sequential
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self._apply_freeze(freeze_until)

    def _apply_freeze(self, mode: str):
        if mode == "none":
            return
        for p in self.features.parameters():
            p.requires_grad = False
        if mode == "all":
            return
        if mode == "last3":
            # Unfreeze the last 3 Conv2d layers (and their following ReLU/BN).
            conv_idxs = [i for i, m in enumerate(self.features)
                         if isinstance(m, nn.Conv2d)]
            cutoff = conv_idxs[-3]
            for m in list(self.features)[cutoff:]:
                for p in m.parameters():
                    p.requires_grad = True
        else:
            raise ValueError(f"Unknown freeze_until={mode!r}")

    def forward(self, x):
        x = self.features(x)         # (B, 512, 7, 7)
        x = self.gap(x)              # (B, 512, 1, 1)
        return x.view(x.size(0), -1) # (B, 512)
''',
        "encoder_factory_call": "lambda: VGG16Encoder(freeze_until='last3')",
    },
    {
        "filename": "Baseline4_MS_ProtoNet_ResNet18.ipynb",
        "title": "Baseline 4 (Proposed): MS-ProtoNet-18 — Multi-Scale ResNet18 + ProtoNet",
        "summary": (
            "## Novelty\n"
            "Standard ProtoNet uses **only the final layer** of the backbone. "
            "This baseline introduces a *multi-scale* feature extractor that combines "
            "**three different layers** of ResNet18:\n\n"
            "- `layer2` — low/mid-level features (texture, local patterns)  (128 ch)\n"
            "- `layer3` — mid/high-level features (parts, structures)        (256 ch)\n"
            "- `layer4` — high-level semantic features                       (512 ch)\n\n"
            "Each level is global-average-pooled, projected through a learnable Linear "
            "to a common 128-D space, and the three 128-D vectors are concatenated into a "
            "**384-D multi-scale embedding** that the Prototypical Network operates on.\n\n"
            "**Hypothesis:** richer, multi-scale features should improve classification of "
            "fine-grained nutrient-deficiency patterns on lettuce leaves, especially in the "
            "1-shot setting where every bit of feature richness matters."
        ),
        "encoder_name": "ResNet18MultiScale",
        "encoder_code": '''\
class ResNet18MultiScale(nn.Module):
    """Multi-scale feature extractor on ResNet18.

    Forward pass:
        image (224x224)
          ── conv1/bn1/relu/maxpool/layer1
          ── layer2 ─→ GAP ─→ fc2 (128)  ┐
          ── layer3 ─→ GAP ─→ fc3 (128)  ├─→ concat → 384-D embedding
          ── layer4 ─→ GAP ─→ fc4 (128)  ┘

    `freeze_until`: how much of the trunk is fine-tunable.
        'none'   : fine-tune everything (overfits on small data)
        'layer3' : trunk frozen up to & including layer2 (default)
        'layer4' : freeze everything except layer4
        'all'    : pure feature extractor (only the fc projections train)
    """

    def __init__(self, freeze_until: str = "layer3"):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.conv1   = base.conv1
        self.bn1     = base.bn1
        self.relu    = base.relu
        self.maxpool = base.maxpool
        self.layer1  = base.layer1
        self.layer2  = base.layer2  # 128 ch
        self.layer3  = base.layer3  # 256 ch
        self.layer4  = base.layer4  # 512 ch

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # Projections to common 128-D space (always trainable: this IS the novelty).
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(512, 128)

        self._apply_freeze(freeze_until)

    def _apply_freeze(self, mode: str):
        trunk = [self.conv1, self.bn1, self.layer1, self.layer2,
                 self.layer3, self.layer4]
        if mode == "none":
            return
        # Freeze the whole trunk first; selectively unfreeze.
        for m in trunk:
            for p in m.parameters():
                p.requires_grad = False
        if mode == "all":
            return
        # Unfreeze layer3 + layer4 (and everything after layer3 in the list).
        unfreeze = {"layer3": [self.layer3, self.layer4],
                    "layer4": [self.layer4]}[mode]
        for m in unfreeze:
            for p in m.parameters():
                p.requires_grad = True

    def forward(self, x):
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x); x = self.maxpool(x)
        x  = self.layer1(x)
        f2 = self.layer2(x)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)

        v2 = self.gap(f2).view(f2.size(0), -1)
        v3 = self.gap(f3).view(f3.size(0), -1)
        v4 = self.gap(f4).view(f4.size(0), -1)

        z2 = self.fc2(v2)
        z3 = self.fc3(v3)
        z4 = self.fc4(v4)

        return torch.cat([z2, z3, z4], dim=1)  # 384-D
''',
        "encoder_factory_call": "lambda: ResNet18MultiScale(freeze_until='layer3')",
    },
    {
        "filename": "Baseline5_MS_ProtoNet_ResNet50.ipynb",
        "title": "Baseline 5 (Proposed): MS-ProtoNet-50 — Multi-Scale ResNet50 + ProtoNet",
        "summary": (
            "## Novelty\n"
            "Same multi-scale recipe as Baseline 4, but using a **ResNet50** trunk.\n\n"
            "- `layer2` —  512 channels  →  Linear(512 → 128)\n"
            "- `layer3` — 1024 channels  →  Linear(1024 → 128)\n"
            "- `layer4` — 2048 channels  →  Linear(2048 → 128)\n\n"
            "Final embedding: concat → **384-D**.\n\n"
            "**Hypothesis:** the deeper trunk should give richer mid-level features at "
            "layer2/layer3, amplifying the benefit of multi-scale fusion compared to "
            "Baseline 4."
        ),
        "encoder_name": "ResNet50MultiScale",
        "encoder_code": '''\
class ResNet50MultiScale(nn.Module):
    """Multi-scale ResNet50 → 384-D embedding (concat of three 128-D scales).

    `freeze_until`: 'none' | 'layer3' | 'layer4' | 'all'.
    """

    def __init__(self, freeze_until: str = "layer3"):
        super().__init__()
        base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.conv1   = base.conv1
        self.bn1     = base.bn1
        self.relu    = base.relu
        self.maxpool = base.maxpool
        self.layer1  = base.layer1
        self.layer2  = base.layer2  #  512 ch
        self.layer3  = base.layer3  # 1024 ch
        self.layer4  = base.layer4  # 2048 ch

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(1024, 128)
        self.fc4 = nn.Linear(2048, 128)

        self._apply_freeze(freeze_until)

    def _apply_freeze(self, mode: str):
        trunk = [self.conv1, self.bn1, self.layer1, self.layer2,
                 self.layer3, self.layer4]
        if mode == "none":
            return
        for m in trunk:
            for p in m.parameters():
                p.requires_grad = False
        if mode == "all":
            return
        unfreeze = {"layer3": [self.layer3, self.layer4],
                    "layer4": [self.layer4]}[mode]
        for m in unfreeze:
            for p in m.parameters():
                p.requires_grad = True

    def forward(self, x):
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x); x = self.maxpool(x)
        x  = self.layer1(x)
        f2 = self.layer2(x)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)

        v2 = self.gap(f2).view(f2.size(0), -1)
        v3 = self.gap(f3).view(f3.size(0), -1)
        v4 = self.gap(f4).view(f4.size(0), -1)

        z2 = self.fc2(v2)
        z3 = self.fc3(v3)
        z4 = self.fc4(v4)

        return torch.cat([z2, z3, z4], dim=1)
''',
        "encoder_factory_call": "lambda: ResNet50MultiScale(freeze_until='layer3')",
    },
]


# ─── Cell helpers ────────────────────────────────────────────────────────────

def md_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }


def code_cell(text: str) -> dict:
    return {
        "cell_type": "code",
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


# ─── Build a single notebook ─────────────────────────────────────────────────

def build_notebook(cfg: dict) -> dict:
    cells = []

    # 0. Title / overview
    cells.append(md_cell(f"# {cfg['title']}\n\n{cfg['summary']}\n"))

    # 1. Library imports
    cells.append(md_cell("## 1. Imports & shared methodology\n\n"
                         "Everything except the encoder lives in `fewshot_lib.py` so "
                         "all 5 baselines use the **identical training & evaluation pipeline**.\n"))
    cells.append(code_cell(
        "import torch\n"
        "import torch.nn as nn\n"
        "from torchvision import models\n\n"
        "from fewshot_lib import (\n"
        "    run_few_shot,\n"
        "    summarize_results,\n"
        "    plot_confusion_matrices,\n"
        "    plot_training_curves,\n"
        "    plot_training_loss_and_accuracy,\n"
        "    evaluate_from_checkpoint,\n"
        "    diagnose_checkpoint,\n"
        "    predict_image,\n"
        ")\n"
    ))

    # 2. Encoder
    cells.append(md_cell(f"## 2. Encoder definition (the only baseline-specific piece)\n"))
    cells.append(code_cell(cfg["encoder_code"]))

    # 3. Run
    cells.append(md_cell(
        "## 3. Train + evaluate (1-shot, 5-shot, 10-shot)\n\n"
        "Hyperparameters used for every baseline so cross-baseline comparisons are fair:\n\n"
        "| Param | Value | Note |\n"
        "|---|---:|---|\n"
        "| `n_train_episodes` | 400 | Every baseline runs all 400 episodes for comparable curves. |\n"
        "| `early_stop` | False | We run all episodes and **only use the best-val checkpoint** for evaluation. |\n"
        "| `val_every` | 20 | Validation accuracy checked every 20 training episodes. |\n"
        "| `n_eval_episodes` | 200 | Episodic test evaluation. |\n"
        "| `n_test_seeds` | 30 | Deterministic full-test evaluation. |\n"
        "| `n_query` | 10 | Some val classes only have 8 images; 10 is the safe upper bound. |\n"
        "| `learning_rate` | 1e-4 | Was 1e-3 originally — that's what made loss collapse to 0. |\n"
        "| `weight_decay` | 1e-4 | Mild L2. |\n"
        "| Augmentation | RandomResizedCrop + Flip + ColorJitter + Rotation | Training only; eval is deterministic. |\n"
        "| Backbone freezing | per encoder default | See the encoder docstring. |\n\n"
        "The best-val checkpoint per shot count is automatically saved to `checkpoints/`.\n"
    ))
    cells.append(code_cell(
        f"results = run_few_shot(\n"
        f"    encoder_factory={cfg['encoder_factory_call']},\n"
        f"    baseline_name={cfg['title'].split(':')[0]!r},\n"
        f"    data_root='clean_dataset',\n"
        f"    n_support_list=[1, 5, 10],\n"
        f"    n_way=4,\n"
        f"    n_query=10,\n"
        f"    n_train_episodes=400,\n"
        f"    val_every=20,\n"
        f"    early_stop=False,    # run all 400 episodes for comparable curves\n"
        f"    n_eval_episodes=200,\n"
        f"    n_test_seeds=30,\n"
        f"    learning_rate=1e-4,\n"
        f"    weight_decay=1e-4,\n"
        f"    seed=42,\n"
        f"    save_dir='checkpoints',\n"
        f")\n"
    ))

    # 4. Summary table
    cells.append(md_cell("## 4. Results summary (overall + per-class)\n"))
    cells.append(code_cell(
        f"summarize_results(results, baseline_name={cfg['title'].split(':')[0]!r})\n"
    ))

    # 5. Training loss + training accuracy (focused, per request)
    cells.append(md_cell(
        "## 5. Training loss & training accuracy\n\n"
        "How to read this plot:\n\n"
        "- **Blue (left axis, log-scaled)** — training loss per episode (raw + smoothed average).\n"
        "  - Healthy: loss decreases gradually, plateaus around 1e-2 to 1e-1.\n"
        "  - Loss → 0 means the encoder has memorised the training set. **This is OK as long as the val accuracy is also high** — and the red dashed line marks where we stopped using the model for evaluation.\n"
        "- **Green (right axis)** — training accuracy on the query of each training episode (smoothed).\n"
        "- **Orange markers (right axis)** — validation accuracy at each `val_every` checkpoint.\n"
        "- **Red dashed line** — the episode where we got the BEST validation accuracy. The model state at that episode is what was saved to `checkpoints/` and what produced the test results above.\n\n"
        "If train_acc → 1.00 while val_acc plateaus or drops, that's textbook overfitting — and the early-stopping checkpoint protects you from it.\n"
    ))
    cells.append(code_cell(
        f"plot_training_loss_and_accuracy(results, baseline_name={cfg['title'].split(':')[0]!r})\n"
    ))

    # 6. Confusion matrices
    cells.append(md_cell(
        "## 6. Confusion matrices\n\n"
        "Two views per shot count:\n"
        "- **Episodic (top, blue):** aggregated over evaluation episodes (standard few-shot reporting).\n"
        "- **Full-test (bottom, green):** every test image classified once, repeated over 30 random "
        "support draws from train. This is the most honest per-class view for the small test set.\n"
    ))
    cells.append(code_cell(
        f"plot_confusion_matrices(results, baseline_name={cfg['title'].split(':')[0]!r})\n"
    ))

    # 7. Combined training curves + stability
    cells.append(md_cell(
        "## 7. Combined view — training trace + test episodic stability\n\n"
        "Three-row panel for each shot count:\n"
        "- **Top:** training loss (raw + smoothed) and best-val marker.\n"
        "- **Middle:** training accuracy + validation accuracy (no twin y-axis).\n"
        "- **Bottom:** per-episode accuracy on the test set during episodic evaluation, with mean ± 1σ.\n"
    ))
    cells.append(code_cell(
        f"plot_training_curves(results, baseline_name={cfg['title'].split(':')[0]!r})\n"
    ))

    # 8. Re-evaluate from saved checkpoint (no training)
    cells.append(md_cell(
        "## 8. Re-evaluate from saved checkpoint (no training)\n\n"
        "Once training has been run once, the best-val encoder for each shot count is saved to "
        "`checkpoints/`. The cell below shows how to **reload** any of those checkpoints and "
        "regenerate test metrics + confusion matrices **without training again** — useful when "
        "you just want to re-render the reports, or share a trained model with a collaborator.\n\n"
        "It demonstrates reload for 1-shot. To reload another shot count, change `k_shot` and the file path.\n"
    ))
    cells.append(code_cell(
        f"# Replace `k_shot` with 1, 5, or 10 to reload that checkpoint.\n"
        f"k_shot = 1\n"
        f"safe_name = {cfg['title'].split(':')[0]!r}.replace(' ', '_')\n"
        f"ckpt_path = f'checkpoints/{{safe_name}}_{{k_shot}}shot.pt'\n"
        f"\n"
        f"reloaded = evaluate_from_checkpoint(\n"
        f"    encoder_factory={cfg['encoder_factory_call']},\n"
        f"    checkpoint_path=ckpt_path,\n"
        f"    data_root='clean_dataset',\n"
        f"    n_way=4,\n"
        f"    k_shot=k_shot,\n"
        f"    n_query=10,\n"
        f"    n_eval_episodes=200,\n"
        f"    n_test_seeds=30,\n"
        f"    seed=42,\n"
        f")\n"
        f"\n"
        f"summarize_results(reloaded, baseline_name=f'{cfg['title'].split(':')[0]} (reloaded {{k_shot}}-shot)')\n"
        f"plot_confusion_matrices(reloaded, baseline_name=f'{cfg['title'].split(':')[0]} (reloaded {{k_shot}}-shot)')\n"
    ))

    # 9. Diagnose: which classes are confused (plain English breakdown)
    cells.append(md_cell(
        "## 9. Diagnose: which classes is the model actually confusing?\n\n"
        "The macro F1 score hides asymmetric errors. This cell loads a saved checkpoint, "
        "runs the deterministic full-test evaluation, and **prints in plain English** which "
        "classes are being confused with which others — including the actual confusion matrix "
        "in tabular form, the top error pairs, and a per-class breakdown.\n\n"
        "Use this to defend your results in the viva. If you see, for example, "
        "*'136 FN images were predicted as -N'*, that's a real biological signal — early-stage "
        "nitrogen deficiency in lettuce visually resembles healthy leaf variation.\n"
    ))
    cells.append(code_cell(
        f"# Diagnose the 1-shot checkpoint — change k_shot to 5 or 10 for the others.\n"
        f"k_shot = 1\n"
        f"safe_name = {cfg['title'].split(':')[0]!r}.replace(' ', '_')\n"
        f"ckpt_path = f'checkpoints/{{safe_name}}_{{k_shot}}shot.pt'\n"
        f"\n"
        f"_ = diagnose_checkpoint(\n"
        f"    encoder_factory={cfg['encoder_factory_call']},\n"
        f"    checkpoint_path=ckpt_path,\n"
        f"    data_root='clean_dataset',\n"
        f"    k_shot=k_shot,\n"
        f"    n_seeds=30,\n"
        f"    seed=42,\n"
        f")\n"
    ))

    # 10. Upload + predict (interactive)
    cells.append(md_cell(
        "## 10. Upload a new image and predict\n\n"
        "Uses the **same Prototypical inference** as evaluation: class prototypes are built from "
        "`k_shot` random **training** images per class (RNG matches `full_test_eval` when "
        "`support_draw_index` matches that loop index). Your photo is embedded and classified "
        "by nearest prototype. **`n_tta=5`** averages several strong augmentations of the query "
        "for stabler probabilities.\n\n"
        "Requires **`ipywidgets`** (`pip install ipywidgets`). In VS Code / JupyterLab, pick a file "
        "with **FileUpload**, then click **Predict**. To use a file path instead, call "
        "`predict_image(..., image=\"path/to.jpg\")` in code.\n"
    ))
    cells.append(code_cell(
        f"from IPython.display import display\n"
        f"import ipywidgets as widgets\n"
        f"\n"
        f"# predict_image is also in the import cell above\n"
        f"\n"
        f"k_shot = 1\n"
        f"support_draw_index = 0  # same RNG stream as full_test_eval's first seed (1000+0)\n"
        f"safe_name = {cfg['title'].split(':')[0]!r}.replace(' ', '_')\n"
        f"ckpt_path = f'checkpoints/{{safe_name}}_{{k_shot}}shot.pt'\n"
        f"\n"
        f"uploader = widgets.FileUpload(accept='image/*', multiple=False)\n"
        f"go = widgets.Button(description='Predict', button_style='primary')\n"
        f"out = widgets.Output()\n"
        f"\n"
        f"\n"
        f"def _extract_upload_bytes(u):\n"
        f"    v = u.value\n"
        f"    if not v:\n"
        f"        return None, None\n"
        f"    if isinstance(v, tuple) and len(v) > 0:\n"
        f"        e = v[0]\n"
        f"        if isinstance(e, dict):\n"
        f"            return e.get('content'), e.get('name', 'upload')\n"
        f"        c = getattr(e, 'content', None)\n"
        f"        if c is not None:\n"
        f"            return bytes(c), getattr(e, 'name', 'upload')\n"
        f"    if isinstance(v, dict) and v:\n"
        f"        name, info = next(iter(v.items()))\n"
        f"        return info.get('content'), name\n"
        f"    return None, None\n"
        f"\n"
        f"\n"
        f"def _on_predict(_):\n"
        f"    with out:\n"
        f"        out.clear_output()\n"
        f"        raw, fname = _extract_upload_bytes(uploader)\n"
        f"        if not raw:\n"
        f"            print('Upload an image, then click Predict.')\n"
        f"            return\n"
        f"        pred = predict_image(\n"
        f"            encoder_factory={cfg['encoder_factory_call']},\n"
        f"            checkpoint_path=ckpt_path,\n"
        f"            image=raw,\n"
        f"            data_root='clean_dataset',\n"
        f"            k_shot=k_shot,\n"
        f"            support_draw_index=support_draw_index,\n"
        f"            n_tta=5,\n"
        f"        )\n"
        f"        print(f'File: {{fname}}')\n"
        f"        print(\n"
        f"            f\"Predicted: {{pred['predicted_class']}}  \"\n"
        f"            f\"(class id {{pred['class_id']}})\"\n"
        f"        )\n"
        f"        top = sorted(pred['probabilities'].items(), key=lambda kv: -kv[1])\n"
        f"        print('Probabilities (softmax over negative squared distances):')\n"
        f"        for c, p in top:\n"
        f"            print(f'  {{c}}: {{p:.3f}}')\n"
        f"\n"
        f"\n"
        f"go.on_click(_on_predict)\n"
        f"display(widgets.VBox([\n"
        f"    widgets.HTML(\n"
        f"        '<b>Upload an image</b> — expected classes: -K, -N, -P, FN '\n"
        f"        '(hydroponic lettuce deficiency signs).'\n"
        f"    ),\n"
        f"    uploader,\n"
        f"    go,\n"
        f"    out,\n"
        f"]))\n"
    ))

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return nb


# ─── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    for cfg in BASELINES:
        path = NB_DIR / cfg["filename"]
        nb = build_notebook(cfg)
        with open(path, "w") as f:
            json.dump(nb, f, indent=1)
        print(f"  wrote {cfg['filename']}  ({len(nb['cells'])} cells)")


if __name__ == "__main__":
    main()
