# -*- coding: utf-8 -*-
"""
TabM M3 - Full 5-fold OOF confirmation.

Reuses feature/model helpers from:
    tabm_m3_fold2_confirm.py

Protocol:
    5-fold StratifiedKFold, shuffle=True, random_state=42
    M3 feature set only
    20 fixed epochs per fold
    80% Champion + 20% TabM rank blend
    No per-fold epoch selection
    No blend-weight search

Outputs:
    D:\\kaggle\\airline\\tabm_m3_full_oof\\
        tabm_m3_oof.csv
        tabm_m3_test_predictions.csv
        tabm_m3_fold_metrics.csv
        tabm_m3_full_summary.csv
"""

import os
import gc
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import tabm_m3_fold2_confirm as base


# =============================================================================
# 1. CONFIGURATION
# =============================================================================

TRAIN_PATH = base.TRAIN_PATH
TEST_PATH = os.path.join(
    base.DATA_DIR,
    "test.csv",
)

LGBM_OOF_PATH = base.LGBM_OOF_PATH
REALMLP_OOF_PATH = base.REALMLP_OOF_PATH

OUTPUT_DIR = (
    r"D:\kaggle\airline\tabm_m3_full_oof"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_FOLDS = 5
FIXED_EPOCHS = 20
SEED = 42

TABM_WEIGHT = 0.20
CHAMPION_WEIGHT = 0.80

ID_COL = "id"
TARGET = "satisfaction"


# =============================================================================
# 2. MAIN
# =============================================================================

def main():
    base.set_seed(SEED)

    print("=" * 100)
    print("TABM M3 FULL 5-FOLD OOF CONFIRMATION")
    print("Device:", base.DEVICE)

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    base.require_file(TRAIN_PATH)
    base.require_file(TEST_PATH)

    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    ids = pd.to_numeric(
        train[ID_COL],
        errors="raise",
    ).to_numpy()

    test_ids = pd.to_numeric(
        test[ID_COL],
        errors="raise",
    ).to_numpy()

    y = train[TARGET].astype(
        np.int8
    ).to_numpy()

    if len(np.unique(ids)) != len(ids):
        raise ValueError("Duplicate train IDs.")

    if len(np.unique(test_ids)) != len(test_ids):
        raise ValueError("Duplicate test IDs.")

    print("Train shape:", train.shape)
    print("Test shape:", test.shape)

    # -------------------------------------------------------------------------
    # Load existing OOF predictions.
    #
    # IMPORTANT:
    # These predictions must have been generated using the same five-fold
    # split: StratifiedKFold(5, shuffle=True, random_state=42).
    # -------------------------------------------------------------------------

    lgb_oof = base.load_oof(
        LGBM_OOF_PATH,
        ids,
        y,
        "Conditional-TE LGBM",
    )

    mlp_oof = base.load_oof(
        REALMLP_OOF_PATH,
        ids,
        y,
        "Lean RealMLP V1",
    )

    # Full-data ranks for the final OOF ensemble calculation.
    lgb_rank_full = base.normalized_rank(lgb_oof)
    mlp_rank_full = base.normalized_rank(mlp_oof)

    champion_rank_full = (
        0.40 * lgb_rank_full
        + 0.60 * mlp_rank_full
    )

    champion_full_auc = roc_auc_score(
        y,
        champion_rank_full,
    )

    print(
        f"\nChampion full OOF AUC = "
        f"{champion_full_auc:.9f}"
    )

    # -------------------------------------------------------------------------
    # Create identical folds for TabM and existing OOF.
    # -------------------------------------------------------------------------

    splitter = StratifiedKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=SEED,
    )

    folds = list(
        splitter.split(
            np.zeros(len(y)),
            y,
        )
    )

    tabm_oof = np.full(
        len(train),
        np.nan,
        dtype=np.float64,
    )

    tabm_test_sum = np.zeros(
        len(test),
        dtype=np.float64,
    )

    fold_metrics = []

    oof_path = os.path.join(
        OUTPUT_DIR,
        "tabm_m3_oof.csv",
    )

    test_pred_path = os.path.join(
        OUTPUT_DIR,
        "tabm_m3_test_predictions.csv",
    )

    fold_metrics_path = os.path.join(
        OUTPUT_DIR,
        "tabm_m3_fold_metrics.csv",
    )

    # -------------------------------------------------------------------------
    # Run the five outer folds.
    # -------------------------------------------------------------------------

    for fold_num, (fit_idx, val_idx) in enumerate(
        folds,
        start=1,
    ):
        print("\n" + "#" * 100)
        print(f"FOLD {fold_num}/{N_FOLDS}")
        print("#" * 100)

        y_fit = y[fit_idx]
        y_val = y[val_idx]

        # ---------------------------------------------------------------------
        # A. Build raw feature views.
        # ---------------------------------------------------------------------

        row_fit = base.build_row_features(
            train.iloc[fit_idx]
        )

        row_val = base.build_row_features(
            train.iloc[val_idx]
        )

        row_test = base.build_row_features(
            test
        )

        raw_fit = row_fit["raw"]
        raw_val = row_val["raw"]
        raw_test = row_test["raw"]

        num_fit = row_fit["numeric_base"]
        num_val = row_val["numeric_base"]
        num_test = row_test["numeric_base"]

        eng_fit = row_fit["numeric_engineered"]
        eng_val = row_val["numeric_engineered"]
        eng_test = row_test["numeric_engineered"]

        cat_fit = pd.concat(
            [
                row_fit["categorical_base"],
                row_fit["categorical_engineered"],
            ],
            axis=1,
        )

        cat_val = pd.concat(
            [
                row_val["categorical_base"],
                row_val["categorical_engineered"],
            ],
            axis=1,
        )

        cat_test = pd.concat(
            [
                row_test["categorical_base"],
                row_test["categorical_engineered"],
            ],
            axis=1,
        )

        # ---------------------------------------------------------------------
        # B. Distance profiles.
        #
        # Training rows use cross-fitted profiles.
        # Validation/test rows use the outer fitting partition only.
        # ---------------------------------------------------------------------

        print("Building cross-fitted distance profiles...")

        profile_fit = base.crossfit_distance_profiles(
            raw_fit,
            y_fit,
        )

        profile_val = base.make_distance_profile(
            raw_fit,
            raw_val,
        )

        profile_test = base.make_distance_profile(
            raw_fit,
            raw_test,
        )

        # ---------------------------------------------------------------------
        # C. Target encoding.
        # ---------------------------------------------------------------------

        print("Building target encodings...")

        te_encoder = base.TargetEncoder(
            target_type="binary",
            smooth="auto",
            cv=5,
            shuffle=True,
            random_state=SEED + 900,
        )

        te_fit_values = te_encoder.fit_transform(
            row_fit["te_keys"],
            y_fit,
        ).astype(np.float32)

        te_val_values = te_encoder.transform(
            row_val["te_keys"]
        ).astype(np.float32)

        te_test_values = te_encoder.transform(
            row_test["te_keys"]
        ).astype(np.float32)

        te_columns = [
            f"te__{col}"
            for col in row_fit["te_keys"].columns
        ]

        te_fit = pd.DataFrame(
            te_fit_values,
            columns=te_columns,
        )

        te_val = pd.DataFrame(
            te_val_values,
            columns=te_columns,
        )

        te_test = pd.DataFrame(
            te_test_values,
            columns=te_columns,
        )

        del te_encoder
        del te_fit_values, te_val_values, te_test_values

        # ---------------------------------------------------------------------
        # D. Frequency features.
        # ---------------------------------------------------------------------

        print("Building frequency features...")

        freq_fit, freq_val = base.fit_frequency_features(
            row_fit["te_keys"],
            row_val["te_keys"],
        )

        _, freq_test = base.fit_frequency_features(
            row_fit["te_keys"],
            row_test["te_keys"],
        )

        # ---------------------------------------------------------------------
        # E. Create numeric matrices.
        # ---------------------------------------------------------------------

        numeric_fit_frames = [
            num_fit,
            eng_fit,
            profile_fit,
            te_fit,
            freq_fit,
        ]

        numeric_val_frames = [
            num_val,
            eng_val,
            profile_val,
            te_val,
            freq_val,
        ]

        numeric_test_frames = [
            num_test,
            eng_test,
            profile_test,
            te_test,
            freq_test,
        ]

        X_fit_num, X_val_num, numeric_names = (
            base.encode_numeric(
                numeric_fit_frames,
                numeric_val_frames,
            )
        )

        X_fit_num_test, X_test_num, numeric_names_test = (
            base.encode_numeric(
                numeric_fit_frames,
                numeric_test_frames,
            )
        )

        if numeric_names != numeric_names_test:
            raise ValueError(
                "Numeric feature mismatch between validation and test."
            )

        if not np.allclose(
            X_fit_num,
            X_fit_num_test,
            rtol=0,
            atol=0,
        ):
            raise ValueError(
                "Training numeric matrix differs between validation/test "
                "preprocessing."
            )

        # ---------------------------------------------------------------------
        # F. Encode categories.
        # ---------------------------------------------------------------------

        X_fit_cat, X_val_cat, cat_cardinalities = (
            base.encode_categorical(
                cat_fit,
                cat_val,
            )
        )

        X_fit_cat_test, X_test_cat, cat_cards_test = (
            base.encode_categorical(
                cat_fit,
                cat_test,
            )
        )

        if cat_cardinalities != cat_cards_test:
            raise ValueError(
                "Categorical cardinalities differ."
            )

        if not np.array_equal(
            X_fit_cat,
            X_fit_cat_test,
        ):
            raise ValueError(
                "Training categorical matrix differs between validation/test "
                "preprocessing."
            )

        print(
            f"Numeric features: {X_fit_num.shape[1]}"
        )

        print(
            f"Categorical features: {X_fit_cat.shape[1]}"
        )

        # Release pandas feature objects before training.
        del row_fit, row_val, row_test
        del raw_fit, raw_val, raw_test
        del num_fit, num_val, num_test
        del eng_fit, eng_val, eng_test
        del cat_fit, cat_val, cat_test
        del profile_fit, profile_val, profile_test
        del te_fit, te_val, te_test
        del freq_fit, freq_val, freq_test
        del numeric_fit_frames, numeric_val_frames
        del numeric_test_frames
        del X_fit_num_test, X_fit_cat_test

        gc.collect()

        # ---------------------------------------------------------------------
        # G. Train for exactly 20 epochs.
        #
        # No validation AUC is consulted during training.
        # ---------------------------------------------------------------------

        fold_seed = SEED + 2000 + fold_num
        base.set_seed(fold_seed)

        model = base.create_model(
            X_fit_num,
            cat_cardinalities,
        )

        optimizer = base.AdamW(
            model.parameters(),
            lr=base.LEARNING_RATE,
            weight_decay=base.WEIGHT_DECAY,
        )

        for epoch in range(
            1,
            FIXED_EPOCHS + 1,
        ):
            loss = base.train_one_epoch(
                model,
                optimizer,
                X_fit_num,
                X_fit_cat,
                y_fit,
            )

            if (
                epoch == 1
                or epoch % 5 == 0
                or epoch == FIXED_EPOCHS
            ):
                print(
                    f"Fold {fold_num} | "
                    f"Epoch {epoch:02d}/{FIXED_EPOCHS} | "
                    f"loss={loss:.6f}"
                )

        # ---------------------------------------------------------------------
        # H. Predict validation and test rows.
        # ---------------------------------------------------------------------

        val_pred = base.predict(
            model,
            X_val_num,
            X_val_cat,
        )

        test_pred = base.predict(
            model,
            X_test_num,
            X_test_cat,
        )

        tabm_oof[val_idx] = val_pred
        tabm_test_sum += test_pred / N_FOLDS

        del model, optimizer
        del X_fit_num, X_val_num, X_fit_cat, X_val_cat
        del X_test_num, X_test_cat
        del val_pred, test_pred

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ---------------------------------------------------------------------
        # I. Fold-wise diagnostics using fixed 20% TabM weight.
        # ---------------------------------------------------------------------

        tabm_fold_pred = tabm_oof[val_idx]

        lgb_rank_fold = base.normalized_rank(
            lgb_oof[val_idx]
        )

        mlp_rank_fold = base.normalized_rank(
            mlp_oof[val_idx]
        )

        tabm_rank_fold = base.normalized_rank(
            tabm_fold_pred
        )

        champion_rank_fold = (
            0.40 * lgb_rank_fold
            + 0.60 * mlp_rank_fold
        )

        blend_rank_fold = (
            CHAMPION_WEIGHT * champion_rank_fold
            + TABM_WEIGHT * tabm_rank_fold
        )

        champion_fold_auc = roc_auc_score(
            y_val,
            champion_rank_fold,
        )

        tabm_fold_auc = roc_auc_score(
            y_val,
            tabm_fold_pred,
        )

        blend_fold_auc = roc_auc_score(
            y_val,
            blend_rank_fold,
        )

        fold_gain = (
            blend_fold_auc
            - champion_fold_auc
        )

        fold_metrics.append({
            "fold": fold_num,
            "fit_rows": len(fit_idx),
            "validation_rows": len(val_idx),
            "fixed_epochs": FIXED_EPOCHS,
            "tabm_auc": tabm_fold_auc,
            "champion_auc": champion_fold_auc,
            "blend_auc": blend_fold_auc,
            "gain_vs_champion": fold_gain,
        })

        print("\nFold result:")
        print(
            f"TabM AUC       = {tabm_fold_auc:.9f}"
        )
        print(
            f"Champion AUC   = {champion_fold_auc:.9f}"
        )
        print(
            f"Fixed blend    = {blend_fold_auc:.9f}"
        )
        print(
            f"Gain           = {fold_gain:+.9f}"
        )

        # Save checkpoints after every fold.
        pd.DataFrame({
            ID_COL: ids,
            TARGET: y,
            "prediction": tabm_oof,
        }).to_csv(
            oof_path,
            index=False,
            encoding="utf-8-sig",
        )

        pd.DataFrame({
            ID_COL: test_ids,
            "prediction": tabm_test_sum,
        }).to_csv(
            test_pred_path,
            index=False,
            encoding="utf-8-sig",
        )

        pd.DataFrame(
            fold_metrics
        ).to_csv(
            fold_metrics_path,
            index=False,
            encoding="utf-8-sig",
        )

        print(
            f"Checkpoint saved after fold {fold_num}."
        )

    # =============================================================================
    # 3. FULL OOF EVALUATION
    # =============================================================================

    if np.isnan(tabm_oof).any():
        raise ValueError(
            "Some OOF predictions are missing."
        )

    if not np.isfinite(tabm_oof).all():
        raise ValueError(
            "Non-finite OOF predictions detected."
        )

    tabm_rank_full = base.normalized_rank(
        tabm_oof
    )

    full_blend_rank = (
        CHAMPION_WEIGHT * champion_rank_full
        + TABM_WEIGHT * tabm_rank_full
    )

    tabm_full_auc = roc_auc_score(
        y,
        tabm_oof,
    )

    champion_full_auc = roc_auc_score(
        y,
        champion_rank_full,
    )

    blend_full_auc = roc_auc_score(
        y,
        full_blend_rank,
    )

    full_gain = (
        blend_full_auc
        - champion_full_auc
    )

    metrics_df = pd.DataFrame(
        fold_metrics
    )

    mean_fold_gain = metrics_df[
        "gain_vs_champion"
    ].mean()

    positive_folds = int(
        (
            metrics_df["gain_vs_champion"] > 0
        ).sum()
    )

    summary = pd.DataFrame([{
        "tabm_full_oof_auc": tabm_full_auc,
        "champion_full_oof_auc": champion_full_auc,
        "fixed_80_20_blend_full_oof_auc": blend_full_auc,
        "full_oof_gain_vs_champion": full_gain,
        "mean_fold_gain": mean_fold_gain,
        "positive_fold_count": positive_folds,
        "n_folds": N_FOLDS,
        "fixed_epochs": FIXED_EPOCHS,
        "tabm_weight": TABM_WEIGHT,
        "champion_weight": CHAMPION_WEIGHT,
    }])

    summary_path = os.path.join(
        OUTPUT_DIR,
        "tabm_m3_full_summary.csv",
    )

    summary.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 100)
    print("FULL OOF SUMMARY")
    print("=" * 100)

    print(
        f"TabM full OOF AUC       = {tabm_full_auc:.9f}"
    )

    print(
        f"Champion full OOF AUC   = {champion_full_auc:.9f}"
    )

    print(
        f"Fixed 80/20 blend AUC   = {blend_full_auc:.9f}"
    )

    print(
        f"Full OOF gain           = {full_gain:+.9f}"
    )

    print(
        f"Mean fold gain          = {mean_fold_gain:+.9f}"
    )

    print(
        f"Positive folds          = "
        f"{positive_folds}/{N_FOLDS}"
    )

    print("\nFold-by-fold diagnostics:")
    print(
        metrics_df.to_string(
            index=False,
            formatters={
                "tabm_auc": "{:.9f}".format,
                "champion_auc": "{:.9f}".format,
                "blend_auc": "{:.9f}".format,
                "gain_vs_champion": "{:+.9f}".format,
            },
        )
    )

    print(f"\nSaved full summary: {summary_path}")
    print(f"Saved OOF predictions: {oof_path}")
    print(f"Saved test predictions: {test_pred_path}")
    print(f"Saved fold metrics: {fold_metrics_path}")


if __name__ == "__main__":
    main()