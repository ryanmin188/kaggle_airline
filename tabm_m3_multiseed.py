# -*- coding: utf-8 -*-
"""
TabM M3 multi-seed ensemble experiment for the airline satisfaction project.

What it does
------------
1. Reuses the existing M3 OOF/test predictions (the original TabM seed).
2. Trains two additional M3 models with different neural-network seeds while
   keeping the same 5-fold split, features, and 20-epoch protocol.
3. Averages the three seeds' OOF and test probabilities.
4. Compares the single-seed TabM and multi-seed TabM inside the fixed
   80% existing champion + 20% TabM rank blend.

Expected project files in the same directory as this script:
    tabm_m3_full_oof.py
    tabm_m3_fold2_confirm.py

Data and prediction paths follow the current project setup under D:\\kaggle\\airline.
This script does not overwrite the current champion predictions or submission.
"""

from __future__ import annotations

import gc
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


# =============================================================================
# Configuration
# =============================================================================

PROJECT_DIR = Path(__file__).resolve().parent
SOURCE_SCRIPT = PROJECT_DIR / "tabm_m3_full_oof.py"

BASELINE_DIR = Path(r"D:\kaggle\airline\tabm_m3_full_oof")
OUTPUT_DIR = Path(r"D:\kaggle\airline\tabm_m3_multiseed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Two extra random seeds; the original M3 run is reused as the first seed.
EXTRA_MODEL_SEEDS = [3022, 4022]

N_FOLDS = 5
SPLIT_SEED = 42

TABM_WEIGHT = 0.20
CHAMPION_WEIGHT = 0.80

ID_COL = "id"
TARGET = "satisfaction"


# =============================================================================
# Utilities
# =============================================================================

def check_file(path: Path | str) -> Path:
    """Raise a clear error if a required file does not exist."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Required file was not found: {path}"
        )
    return path


def load_prediction_file(
    path: Path | str,
    expected_ids: np.ndarray,
    label: str,
    y: np.ndarray | None = None,
) -> np.ndarray:
    """
    Load predictions and align them to the expected ID order.

    OOF files must also contain the target column so target alignment
    can be verified before comparing predictions.
    """
    path = check_file(path)
    df = pd.read_csv(path)

    required = {ID_COL, "prediction"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"{label}: expected columns {sorted(required)}, "
            f"found {df.columns.tolist()}"
        )

    df[ID_COL] = pd.to_numeric(
        df[ID_COL],
        errors="raise",
    )

    if df[ID_COL].duplicated().any():
        raise ValueError(
            f"{label}: duplicate IDs detected in {path}"
        )

    if set(df[ID_COL].to_numpy()) != set(expected_ids):
        raise ValueError(
            f"{label}: IDs do not match the current train/test file"
        )

    df = (
        df.set_index(ID_COL)
        .loc[expected_ids]
    )

    if y is not None:
        if TARGET not in df.columns:
            raise ValueError(
                f"{label}: missing target column '{TARGET}'"
            )

        aligned_y = df[TARGET].to_numpy(dtype=np.int8)

        if not np.array_equal(
            aligned_y,
            y.astype(np.int8),
        ):
            raise ValueError(
                f"{label}: target values are misaligned with train.csv"
            )

    pred = df["prediction"].to_numpy(dtype=np.float64)

    if not np.isfinite(pred).all():
        raise ValueError(
            f"{label}: NaN or infinite predictions detected"
        )

    return pred


def run_existing_m3_source(model_seed: int) -> None:
    """
    Execute the existing M3 pipeline after changing only:
      1. Output directory.
      2. Neural-network training seed.

    The original source file is read but never modified or overwritten.
    """
    check_file(SOURCE_SCRIPT)

    src = SOURCE_SCRIPT.read_text(encoding="utf-8-sig")

    # Redirect every run to its own folder.
    output_pattern = (
        r'OUTPUT_DIR\s*=\s*\(\s*'
        r'r"D:\\kaggle\\airline\\tabm_m3_full_oof"\s*\)'
    )

    replacement = (
        'OUTPUT_DIR = os.path.join('
        'r"D:\\kaggle\\airline\\tabm_m3_multiseed", '
        'f"seed_{MODEL_SEED}")'
    )

    src, n_output = re.subn(
        output_pattern,
        lambda _match: replacement,
        src,
        count=1,
    )

    if n_output != 1:
        raise RuntimeError(
            "Could not safely patch OUTPUT_DIR in "
            "tabm_m3_full_oof.py. The source script may have changed. "
            "Inspect its configuration block before continuing."
        )

    # The original script uses 2042 + fold_num for each fold.
    # We replace this with a distinct base seed for each new run.
    src, n_seed = re.subn(
        r"fold_seed\s*=\s*SEED\s*\+\s*2000\s*\+\s*fold_num",
        "fold_seed = MODEL_SEED + fold_num",
        src,
        count=1,
    )

    if n_seed != 1:
        raise RuntimeError(
            "Could not safely patch fold_seed in "
            "tabm_m3_full_oof.py. The source script may have changed. "
            "Inspect its training block before continuing."
        )

    namespace = {
        "__name__": "__main__",
        "__file__": str(SOURCE_SCRIPT),
        "MODEL_SEED": int(model_seed),
    }

    print("\n" + "=" * 100)
    print(f"RUNNING EXTRA TABM M3 SEED: {model_seed}")
    print(
        "Same fold split / features / 20 epochs; "
        "neural-network initialization seed differs."
    )
    print("=" * 100)

    exec(
        compile(src, str(SOURCE_SCRIPT), "exec"),
        namespace,
    )


def normalized_rank(values: np.ndarray) -> np.ndarray:
    """Convert predictions into normalized ranks for rank blending."""
    from scipy.stats import rankdata

    values = np.asarray(values, dtype=np.float64)

    if not np.isfinite(values).all():
        raise ValueError(
            "Cannot rank non-finite predictions"
        )

    return rankdata(
        values,
        method="average",
    ) / len(values)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    if str(PROJECT_DIR) not in sys.path:
        sys.path.insert(0, str(PROJECT_DIR))

    check_file(SOURCE_SCRIPT)

    # Import the existing helper module and its data paths.
    import tabm_m3_fold2_confirm as base

    train_path = check_file(base.TRAIN_PATH)
    test_path = check_file(
        Path(base.DATA_DIR) / "test.csv"
    )

    baseline_oof_path = check_file(
        BASELINE_DIR / "tabm_m3_oof.csv"
    )
    baseline_test_path = check_file(
        BASELINE_DIR / "tabm_m3_test_predictions.csv"
    )

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    ids = pd.to_numeric(
        train[ID_COL],
        errors="raise",
    ).to_numpy()

    test_ids = pd.to_numeric(
        test[ID_COL],
        errors="raise",
    ).to_numpy()

    y = train[TARGET].astype(np.int8).to_numpy()

    print("=" * 100)
    print("TABM M3 MULTI-SEED ENSEMBLE")
    print("Project:", PROJECT_DIR)
    print("Train:", train.shape, " Test:", test.shape)
    print("Device:", base.DEVICE)
    print("Extra seeds:", EXTRA_MODEL_SEEDS)
    print("Total seeds in ensemble:", 1 + len(EXTRA_MODEL_SEEDS))
    print("=" * 100)

    # -------------------------------------------------------------------------
    # 1. Load the original M3 OOF/test predictions.
    # -------------------------------------------------------------------------

    oof_by_seed: dict[str, np.ndarray] = {
        "original_seed": load_prediction_file(
            baseline_oof_path,
            ids,
            "original TabM M3 OOF",
            y=y,
        )
    }

    test_by_seed: dict[str, np.ndarray] = {
        "original_seed": load_prediction_file(
            baseline_test_path,
            test_ids,
            "original TabM M3 test predictions",
        )
    }

    # -------------------------------------------------------------------------
    # 2. Train two additional seeds.
    #
    # Each seed runs the existing fixed 5-fold, 20-epoch M3 protocol.
    # Ten new fold-models are trained in total.
    # -------------------------------------------------------------------------

    for model_seed in EXTRA_MODEL_SEEDS:
        run_existing_m3_source(model_seed)

        seed_dir = OUTPUT_DIR / f"seed_{model_seed}"

        oof_path = (
            seed_dir / "tabm_m3_oof.csv"
        )

        test_pred_path = (
            seed_dir / "tabm_m3_test_predictions.csv"
        )

        key = f"seed_{model_seed}"

        oof_by_seed[key] = load_prediction_file(
            oof_path,
            ids,
            f"{key} OOF",
            y=y,
        )

        test_by_seed[key] = load_prediction_file(
            test_pred_path,
            test_ids,
            f"{key} test predictions",
        )

        gc.collect()

    # -------------------------------------------------------------------------
    # 3. Average predictions across random seeds.
    #
    # This is neural-network seed averaging, not blend-weight optimization.
    # All OOF predictions use the same outer-fold split.
    # -------------------------------------------------------------------------

    seed_oof_matrix = np.column_stack(
        list(oof_by_seed.values())
    )

    seed_test_matrix = np.column_stack(
        list(test_by_seed.values())
    )

    multi_oof = seed_oof_matrix.mean(axis=1)
    multi_test = seed_test_matrix.mean(axis=1)

    single_oof = oof_by_seed["original_seed"]

    # -------------------------------------------------------------------------
    # 4. Load and align the fixed LightGBM / RealMLP OOF predictions.
    # -------------------------------------------------------------------------

    lgb_oof = base.load_oof(
        base.LGBM_OOF_PATH,
        ids,
        y,
        "Conditional-TE LGBM",
    )

    mlp_oof = base.load_oof(
        base.REALMLP_OOF_PATH,
        ids,
        y,
        "Lean RealMLP V1",
    )

    champion_rank = (
        0.40 * normalized_rank(lgb_oof)
        + 0.60 * normalized_rank(mlp_oof)
    )

    single_rank = normalized_rank(single_oof)
    multi_rank = normalized_rank(multi_oof)

    champion_auc = roc_auc_score(
        y,
        champion_rank,
    )

    single_tabm_auc = roc_auc_score(
        y,
        single_oof,
    )

    multi_tabm_auc = roc_auc_score(
        y,
        multi_oof,
    )

    single_blend = (
        CHAMPION_WEIGHT * champion_rank
        + TABM_WEIGHT * single_rank
    )

    multi_blend = (
        CHAMPION_WEIGHT * champion_rank
        + TABM_WEIGHT * multi_rank
    )

    single_blend_auc = roc_auc_score(
        y,
        single_blend,
    )

    multi_blend_auc = roc_auc_score(
        y,
        multi_blend,
    )

    # -------------------------------------------------------------------------
    # 5. Fold-level comparison.
    #
    # Calculate rank blends within each held-out fold, just as in the
    # existing experiment. The split and blend weights stay fixed.
    # -------------------------------------------------------------------------

    splitter = StratifiedKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=SPLIT_SEED,
    )

    fold_rows = []

    for fold_num, (_, val_idx) in enumerate(
        splitter.split(np.zeros(len(y)), y),
        start=1,
    ):
        y_val = y[val_idx]

        champ_fold = (
            0.40 * normalized_rank(lgb_oof[val_idx])
            + 0.60 * normalized_rank(mlp_oof[val_idx])
        )

        single_fold = (
            CHAMPION_WEIGHT * champ_fold
            + TABM_WEIGHT
            * normalized_rank(single_oof[val_idx])
        )

        multi_fold = (
            CHAMPION_WEIGHT * champ_fold
            + TABM_WEIGHT
            * normalized_rank(multi_oof[val_idx])
        )

        champ_auc_fold = roc_auc_score(
            y_val,
            champ_fold,
        )

        single_auc_fold = roc_auc_score(
            y_val,
            single_fold,
        )

        multi_auc_fold = roc_auc_score(
            y_val,
            multi_fold,
        )

        fold_rows.append({
            "fold": fold_num,
            "champion_auc": champ_auc_fold,
            "single_seed_blend_auc": single_auc_fold,
            "multi_seed_blend_auc": multi_auc_fold,
            "single_seed_gain": (
                single_auc_fold - champ_auc_fold
            ),
            "multi_seed_gain": (
                multi_auc_fold - champ_auc_fold
            ),
            "multi_vs_single_gain": (
                multi_auc_fold - single_auc_fold
            ),
        })

    fold_df = pd.DataFrame(fold_rows)

    # -------------------------------------------------------------------------
    # 6. Summaries for each seed and the mean ensemble.
    # -------------------------------------------------------------------------

    summary_rows = []

    for name, pred in oof_by_seed.items():
        pred_rank = normalized_rank(pred)

        summary_rows.append({
            "model": name,
            "tabm_oof_auc": roc_auc_score(y, pred),
            "blended_oof_auc": roc_auc_score(
                y,
                CHAMPION_WEIGHT * champion_rank
                + TABM_WEIGHT * pred_rank,
            ),
        })

    summary_rows.append({
        "model": f"mean_of_{len(oof_by_seed)}_seeds",
        "tabm_oof_auc": multi_tabm_auc,
        "blended_oof_auc": multi_blend_auc,
    })

    summary_df = pd.DataFrame(summary_rows)

    # -------------------------------------------------------------------------
    # 7. Save outputs.
    # -------------------------------------------------------------------------

    summary_df.to_csv(
        OUTPUT_DIR / "multiseed_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    fold_df.to_csv(
        OUTPUT_DIR / "multiseed_fold_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame({
        ID_COL: ids,
        TARGET: y,
        "tabm_original_seed_prediction": single_oof,
        "tabm_multiseed_mean_prediction": multi_oof,
        "champion_rank": champion_rank,
        "blend_original_seed_rank": single_blend,
        "blend_multiseed_rank": multi_blend,
    }).to_csv(
        OUTPUT_DIR / "multiseed_oof_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame({
        ID_COL: test_ids,
        "prediction": multi_test,
    }).to_csv(
        OUTPUT_DIR / "tabm_m3_multiseed_test_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -------------------------------------------------------------------------
    # 8. Report final comparison.
    # -------------------------------------------------------------------------

    print("\n" + "=" * 100)
    print("MULTI-SEED SUMMARY")
    print("=" * 100)

    print(
        f"Current champion OOF AUC          : {champion_auc:.9f}"
    )

    print(
        f"Single-seed TabM OOF AUC          : {single_tabm_auc:.9f}"
    )

    print(
        f"Multi-seed TabM OOF AUC           : {multi_tabm_auc:.9f}"
    )

    print(
        f"Single-seed blended OOF AUC       : {single_blend_auc:.9f}"
    )

    print(
        f"Multi-seed blended OOF AUC        : {multi_blend_auc:.9f}"
    )

    print(
        "Blend change vs champion, single : "
        f"{single_blend_auc - champion_auc:+.9f}"
    )

    print(
        "Blend change vs champion, multi  : "
        f"{multi_blend_auc - champion_auc:+.9f}"
    )

    print(
        "Multi-seed vs single-seed change : "
        f"{multi_blend_auc - single_blend_auc:+.9f}"
    )

    print("\nPer-seed / ensemble results:")

    print(
        summary_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.9f}",
        )
    )

    print("\nFold-level blend comparison:")

    print(
        fold_df.to_string(
            index=False,
            float_format=lambda x: f"{x:+.9f}",
        )
    )

    positive_folds = int(
        (fold_df["multi_seed_gain"] > 0).sum()
    )

    print(
        "\nPositive folds for multi-seed gain:",
        f"{positive_folds}/{N_FOLDS}",
    )

    print("\nSaved outputs to:", OUTPUT_DIR)
    print(
        "No current champion predictions were overwritten, "
        "and no submission was submitted automatically."
    )


if __name__ == "__main__":
    main()