"""common.smoke — smoke-test utilities.

  * smoke_common()            — self-test of the common layer (data loading,
                                metrics on both binary & multiclass dummy data,
                                confusion-matrix rendering with dpi=600 + TNR).
  * pick_smoke_dataset(name)  — DatasetDescriptor for a small KEEL set.
  * run_smoke(build_fn, key)  — run one model's build() on a single fold of a
                                small dataset; the per-module `__main__` driver.
"""
import os
import numpy as np

from . import paths
from .dataio import DatasetDescriptor, load_folds
from .metrics import compute_metrics, save_confusion_matrix, METRIC_KEYS
from .base import run_model_on_dataset


def pick_smoke_dataset(name="ecoli"):
    return DatasetDescriptor(name=name, source="keel", keel_name=name)


def smoke_common():
    """Self-test the common layer. Raises AssertionError on any failure."""
    paths.ensure_dirs()
    print("[smoke_common] loading KEEL ecoli (5-fold)...")
    d1 = pick_smoke_dataset("ecoli")
    folds1, classes1 = load_folds(d1)
    assert len(folds1) == 5, f"expected 5 folds, got {len(folds1)}"
    assert folds1[0].X_train.ndim == 2 and folds1[0].X_test.ndim == 2
    print(f"  ecoli: 5 folds, classes={classes1.tolist()}, "
          f"fold1 train={folds1[0].X_train.shape} test={folds1[0].X_test.shape}")

    # --- metrics on dummy predictions: binary + multiclass ---
    rng = np.random.RandomState(0)
    for tag, n in (("multi", len(classes1)),):
        y = rng.randint(0, n, size=50)
        yp = rng.randint(0, n, size=50)
        proba = rng.rand(50, n); proba /= proba.sum(1, keepdims=True)
        m = compute_metrics(y, yp, proba, n, list(range(n)))
        assert set(m.keys()) == set(METRIC_KEYS), f"{tag}: metric keys mismatch"
        assert all(np.isfinite(v) for v in m.values()), f"{tag}: non-finite metric"
        print(f"  metrics[{tag},C={n}]: " + ", ".join(f"{k}={v:.3f}" for k, v in m.items()))

    # --- confusion matrix render: assert dpi=600 + TNR font active ---
    import matplotlib.pyplot as plt
    out_png = os.path.join(paths.FIGURES_DIR, "_smoke_common_cm.png")
    yt = [np.array([0, 1, 2, 0, 1, 2, 0, 1]), np.array([2, 0, 1, 0, 2, 1, 0, 2])]
    yp = [np.array([0, 1, 1, 0, 1, 2, 0, 1]), np.array([2, 0, 0, 0, 2, 1, 1, 2])]
    save_confusion_matrix(yt, yp, [0, 1, 2], out_png, title="smoke_common", dpi=600)
    from PIL import Image
    im = Image.open(out_png)
    dpi_x = im.info.get("dpi", (0, 0))[0]
    assert abs(dpi_x - 600) < 1, f"dpi!=600, got {im.info.get('dpi')}"
    fam = plt.rcParams["font.serif"][0]
    assert fam == "Times New Roman", f"font not TNR, got '{fam}'"
    print(f"  CM png: {out_png} dpi={im.info['dpi'][0]} font={fam}")

    print("[smoke_common] PASS")
    return True


def run_smoke(build_fn, model_key, dataset="ecoli", n_folds=5, save_cm=True,
              scale=True, base_seed=42):
    """Run `build_fn(random_state=seed, smoke=True)` on `n_folds` folds of a
    small dataset. Used by each model module's `__main__` smoke driver."""
    paths.ensure_dirs()
    desc = pick_smoke_dataset(dataset)
    folds, classes = load_folds(desc)
    folds = folds[:n_folds]
    print(f"\n[SMOKE] {model_key} on {dataset} "
          f"(classes={classes.tolist()}, folds={len(folds)})")

    def factory(seed):
        return build_fn(random_state=seed, smoke=True)

    res = run_model_on_dataset(factory, desc, folds, classes,
                               fig_dir=paths.FIGURES_DIR, model_key=model_key,
                               scale=scale, base_seed=base_seed, save_cm=save_cm)
    print(f"[SMOKE] {model_key} ->")
    for k in METRIC_KEYS:
        print(f"    {k:10s}: {res[f'{k}_mean']:.4f} +/- {res[f'{k}_std']:.4f}")
    print(f"    {'Runtime':10s}: {res['Runtime_mean']:.3f} +/- {res['Runtime_std']:.3f} s")
    return res


if __name__ == "__main__":
    smoke_common()
