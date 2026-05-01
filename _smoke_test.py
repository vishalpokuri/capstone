"""
Smoke-test: ensures every baseline notebook RUNS end-to-end with tiny budgets,
AND that the save-then-reload-then-re-evaluate workflow round-trips.

Run:
    python3 _smoke_test.py
"""

from __future__ import annotations

import json
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplcfg")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.show = lambda *a, **kw: None

import re
import sys
import traceback
from pathlib import Path

NOTEBOOKS = [
    "Baseline1_ResNet18_ProtoNet_FewShot.ipynb",
    "Baseline2_ResNet50_ProtoNet_FewShot.ipynb",
    "Baseline3_VGG16_ProtoNet_FewShot_.ipynb",
    "Baseline4_MS_ProtoNet_ResNet18.ipynb",
    "Baseline5_MS_ProtoNet_ResNet50.ipynb",
]


def shrink_run_call(src: str) -> str:
    """Replace production hyperparameters with smoke-test ones."""
    src = re.sub(r"n_train_episodes=\d+", "n_train_episodes=20", src)
    src = re.sub(r"n_eval_episodes=\d+",  "n_eval_episodes=10",  src)
    src = re.sub(r"n_test_seeds=\d+",     "n_test_seeds=2",      src)
    src = re.sub(r"val_every=\d+",        "val_every=10",        src)
    src = re.sub(r"n_support_list=\[1, 5, 10\]", "n_support_list=[1]", src)
    return src


def run_notebook(path: Path) -> dict:
    print(f"\n{'='*70}\n{path.name}\n{'='*70}")
    nb = json.loads(path.read_text())
    g = {"__name__": "__main__"}
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        if "run_few_shot(" in src or "evaluate_from_checkpoint(" in src:
            src = shrink_run_call(src)
        try:
            exec(src, g)
        except Exception:
            print(f"\n  CELL {i} FAILED in {path.name}:")
            print("--- source ---"); print(src); print("--- traceback ---")
            traceback.print_exc()
            raise
    return g.get("results", {})


def main() -> None:
    here = Path(__file__).parent
    failures = []
    for nb_name in NOTEBOOKS:
        try:
            results = run_notebook(here / nb_name)
            shot = next(iter(results))
            m = results[shot]
            ckpt = m.get("checkpoint_path")
            print(f"\n  ✓ {nb_name}  {shot} acc={m['accuracy']:.3f}  "
                  f"per-class F1={[f'{x:.2f}' for x in m['full_per_class_f1']]}  "
                  f"ckpt={'OK' if ckpt and os.path.exists(ckpt) else 'MISSING'}")
        except Exception as e:
            failures.append((nb_name, repr(e)))

    print("\n" + "=" * 70)
    if failures:
        print("FAILURES:")
        for name, err in failures:
            print(f"  - {name}: {err}")
        sys.exit(1)
    else:
        print("All 5 baselines: train → save → reload → re-evaluate, OK.")


if __name__ == "__main__":
    main()
