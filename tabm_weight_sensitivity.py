
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


# ============================================================
# 1. 路径与实验配置
# ============================================================

BASE_DIR = Path(r"D:\kaggle\airline")
DATA_DIR = BASE_DIR / "data"

TRAIN_PATH = DATA_DIR / "train.csv"

LGBM_OOF_PATH = (
    BASE_DIR
    / "conditional_te_experiment"
    / "conditional_te_lgbm_oof.csv"
)

REALMLP_OOF_PATH = (
    BASE_DIR
    / "lean_realmlp_blend"
    / "lean_realmlp_oof.csv"
)

TABM_OOF_PATH = (
    BASE_DIR
    / "tabm_m3_full_oof"
    / "tabm_m3_oof.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "tabm_m3_full_oof"
    / "weight_sensitivity"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ID_COL = "id"
TARGET_COL = "satisfaction"

N_SPLITS = 5
FOLD_RANDOM_STATE = 42

# 从 0% 到 30%，每次增加 2.5 个百分点
TABM_WEIGHTS = np.round(
    np.arange(0.0, 0.300001, 0.025),
    3,
)

LGBM_WEIGHT_IN_CHAMPION = 0.40
REALMLP_WEIGHT_IN_CHAMPION = 0.60


# ============================================================
# 2. 通用工具
# ============================================================

def normalize_name(name):
    """忽略大小写、下划线和其他标点。"""
    return "".join(
        char.lower()
        for char in str(name)
        if char.isalnum()
    )


def find_id_column(df):
    """识别 ID 列。"""
    preferred = {
        "id",
        "passengerid",
        "customerid",
        "recordid",
        "rowid",
    }

    for col in df.columns:
        if normalize_name(col) in preferred:
            return col

    return None


def canonicalize_ids(values):
    """
    将 ID 标准化，兼容 123、123.0 和 '123'。
    """
    values = pd.Series(values).reset_index(drop=True)
    numeric = pd.to_numeric(values, errors="coerce")

    if numeric.notna().all():
        arr = numeric.to_numpy(dtype=float)

        if np.all(np.isclose(arr, np.round(arr))):
            return pd.Series(
                np.round(arr).astype(np.int64).astype(str)
            )

        return pd.Series(
            np.round(arr, 10).astype(str)
        )

    return values.astype(str).str.strip()


def find_prediction_column(df, path):
    """
    自动识别预测列，避免误把 ID、标签或 fold 当成预测。
    优先寻找含义明确的预测列名。
    """
    normalized_columns = {
        normalize_name(col): col
        for col in df.columns
    }

    preferred = [
        "prediction",
        "predictions",
        "oofprediction",
        "oofpred",
        "tabmprediction",
        "tabmpred",
        "tabmm3prediction",
        "tabmm3pred",
        "lgbmprediction",
        "lgbmpred",
        "realmlpprediction",
        "realmlppred",
        "pred",
        "preds",
        "probability",
        "prob",
        "proba",
        "score",
    ]

    for name in preferred:
        if name in normalized_columns:
            col = normalized_columns[name]
            values = pd.to_numeric(df[col], errors="coerce")

            if values.notna().all():
                return col

    # 兜底：筛选数值列，排除 ID、标签、fold 和排名列
    excluded = {
        "id",
        "passengerid",
        "customerid",
        "recordid",
        "rowid",
        "satisfaction",
        "target",
        "label",
        "y",
        "ytrue",
        "true",
        "fold",
        "foldid",
        "index",
        "rowindex",
    }

    candidates = []

    for col in df.columns:
        name = normalize_name(col)

        if (
            name in excluded
            or name.startswith("unnamed")
            or "rank" in name
        ):
            continue

        values = pd.to_numeric(df[col], errors="coerce")

        if values.notna().all():
            candidates.append(col)

    if len(candidates) == 1:
        return candidates[0]

    raise ValueError(
        f"\n无法唯一识别预测列：{path}\n"
        f"文件列名：{list(df.columns)}\n"
        f"候选数值列：{candidates}\n"
        "请检查该文件，并在 find_prediction_column() 中指定列名。"
    )


def load_predictions(path, expected_ids):
    """
    读取 OOF 预测，并根据 ID 与 train.csv 对齐。
    若无 ID，则只有在行数相同时才按行顺序对齐。
    """
    if not path.exists():
        raise FileNotFoundError(f"找不到文件：{path}")

    df = pd.read_csv(path)

    print(f"\nLoading: {path}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    pred_col = find_prediction_column(df, path)

    predictions = pd.to_numeric(
        df[pred_col],
        errors="raise",
    ).to_numpy(dtype=float)

    if not np.isfinite(predictions).all():
        raise ValueError(f"{path} 的预测列含 NaN 或无穷值。")

    if len(predictions) != len(expected_ids):
        raise ValueError(
            f"{path} 行数为 {len(predictions)}，"
            f"但训练集行数为 {len(expected_ids)}。"
        )

    file_id_col = find_id_column(df)

    if file_id_col is None:
        print(
            "WARNING: 文件没有可识别的 ID 列，"
            "将按行顺序对齐。请确认原始行序完全一致。"
        )
        return predictions

    file_ids = canonicalize_ids(df[file_id_col])
    train_ids = canonicalize_ids(expected_ids)

    if file_ids.duplicated().any():
        raise ValueError(f"{path} 的 ID 存在重复值。")

    if train_ids.duplicated().any():
        raise ValueError("train.csv 的 ID 存在重复值。")

    pred_map = dict(zip(file_ids, predictions))

    if set(pred_map.keys()) != set(train_ids):
        missing = set(train_ids) - set(pred_map.keys())
        extra = set(pred_map.keys()) - set(train_ids)

        raise ValueError(
            f"{path} 的 ID 与训练集不一致。\n"
            f"缺少 ID 数量：{len(missing)}\n"
            f"多余 ID 数量：{len(extra)}"
        )

    aligned = np.array(
        [pred_map[key] for key in train_ids],
        dtype=float,
    )

    print(f"Prediction column: {pred_col}")
    print("ID alignment: verified")

    return aligned


def encode_target(series):
    """
    将比赛标签转换成 0/1。
    本比赛的正类应是 satisfied。
    """
    if series.isna().any():
        raise ValueError("目标列包含缺失值。")

    numeric = pd.to_numeric(series, errors="coerce")

    if numeric.notna().all():
        unique_values = set(numeric.unique())

        if unique_values.issubset({0, 1}):
            return numeric.to_numpy(dtype=int)

    normalized = (
        series.astype(str)
        .str.strip()
        .str.lower()
    )

    unique_values = set(normalized.unique())

    positive_names = {
        "satisfied",
        "1",
        "true",
        "yes",
        "positive",
    }

    if len(unique_values) == 2 and unique_values.intersection(
        positive_names
    ):
        y = normalized.isin(positive_names).astype(int).to_numpy()

        if set(np.unique(y)) == {0, 1}:
            print("Target encoding: satisfied = 1, other class = 0")
            return y

    # 兜底：二分类标签排序后编码，并明确打印映射
    if len(unique_values) != 2:
        raise ValueError(
            f"期望二分类标签，实际标签为：{unique_values}"
        )

    sorted_labels = sorted(unique_values)
    mapping = {
        sorted_labels[0]: 0,
        sorted_labels[1]: 1,
    }

    print(f"WARNING: 使用备用标签映射：{mapping}")

    return normalized.map(mapping).to_numpy(dtype=int)


def normalized_rank(values):
    """
    转为 [0, 1] 之间的归一化排名。
    分数更高的样本排名更高。
    """
    values = np.asarray(values, dtype=float)

    if not np.isfinite(values).all():
        raise ValueError("预测数据包含 NaN 或无穷值。")

    if len(values) <= 1:
        return np.zeros(len(values), dtype=float)

    ranks = rankdata(values, method="average")

    return (ranks - 1.0) / (len(values) - 1.0)


# ============================================================
# 3. 主实验
# ============================================================

def main():
    print("=" * 88)
    print("TABM WEIGHT SENSITIVITY ANALYSIS")
    print("=" * 88)

    # --------------------------------------------------------
    # 读取标签和 ID
    # --------------------------------------------------------
    train = pd.read_csv(TRAIN_PATH)

    if ID_COL not in train.columns:
        raise ValueError(
            f"训练集没有 {ID_COL} 列。"
            f"当前列名：{list(train.columns)}"
        )

    if TARGET_COL not in train.columns:
        raise ValueError(
            f"训练集没有 {TARGET_COL} 列。"
            f"当前列名：{list(train.columns)}"
        )

    train_ids = train[ID_COL]

    if train_ids.isna().any() or train_ids.duplicated().any():
        raise ValueError("训练集 ID 缺失或重复。")

    y = encode_target(train[TARGET_COL])

    print(f"\nTrain shape: {train.shape}")
    print(f"Positive rate: {y.mean():.6f}")
    print(f"Rows: {len(y):,}")

    # --------------------------------------------------------
    # 读取三组 OOF 预测
    # --------------------------------------------------------
    lgbm_pred = load_predictions(
        LGBM_OOF_PATH,
        train_ids,
    )

    realmlp_pred = load_predictions(
        REALMLP_OOF_PATH,
        train_ids,
    )

    tabm_pred = load_predictions(
        TABM_OOF_PATH,
        train_ids,
    )

    # --------------------------------------------------------
    # 全量 OOF 排名
    # --------------------------------------------------------
    lgbm_rank_global = normalized_rank(lgbm_pred)
    realmlp_rank_global = normalized_rank(realmlp_pred)
    tabm_rank_global = normalized_rank(tabm_pred)

    champion_rank_global = (
        LGBM_WEIGHT_IN_CHAMPION * lgbm_rank_global
        + REALMLP_WEIGHT_IN_CHAMPION * realmlp_rank_global
    )

    champion_full_auc = roc_auc_score(
        y,
        champion_rank_global,
    )

    print("\n" + "=" * 88)
    print("BASELINE")
    print("=" * 88)
    print(f"Champion full OOF AUC = {champion_full_auc:.9f}")

    # --------------------------------------------------------
    # 复现相同的五折划分
    # --------------------------------------------------------
    splitter = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=FOLD_RANDOM_STATE,
    )

    fold_indices = []

    for fold_num, (_, val_idx) in enumerate(
        splitter.split(np.zeros(len(y)), y),
        start=1,
    ):
        fold_indices.append(val_idx)

    # 预先计算每个验证折中的排名及冠军基准
    fold_components = []
    fold_baseline_aucs = []

    for fold_num, val_idx in enumerate(
        fold_indices,
        start=1,
    ):
        fold_lgbm_rank = normalized_rank(lgbm_pred[val_idx])
        fold_realmlp_rank = normalized_rank(realmlp_pred[val_idx])
        fold_tabm_rank = normalized_rank(tabm_pred[val_idx])

        fold_champion_rank = (
            LGBM_WEIGHT_IN_CHAMPION * fold_lgbm_rank
            + REALMLP_WEIGHT_IN_CHAMPION * fold_realmlp_rank
        )

        champion_auc = roc_auc_score(
            y[val_idx],
            fold_champion_rank,
        )

        fold_components.append({
            "fold": fold_num,
            "indices": val_idx,
            "tabm_rank": fold_tabm_rank,
            "champion_rank": fold_champion_rank,
            "champion_auc": champion_auc,
        })

        fold_baseline_aucs.append(champion_auc)

    # --------------------------------------------------------
    # 遍历融合权重
    # --------------------------------------------------------
    summary_rows = []
    detail_rows = []

    for tabm_weight in TABM_WEIGHTS:
        champion_weight = 1.0 - tabm_weight

        # 全量 OOF：先对各模型全量预测做排名，再融合
        global_blend = (
            champion_weight * champion_rank_global
            + tabm_weight * tabm_rank_global
        )

        global_auc = roc_auc_score(y, global_blend)
        global_gain = global_auc - champion_full_auc

        fold_aucs = []
        fold_gains = []

        # 分别计算五个验证折的 AUC
        for component in fold_components:
            fold_num = component["fold"]
            val_idx = component["indices"]

            champion_rank = component["champion_rank"]
            tabm_rank = component["tabm_rank"]

            fold_blend = (
                champion_weight * champion_rank
                + tabm_weight * tabm_rank
            )

            fold_auc = roc_auc_score(
                y[val_idx],
                fold_blend,
            )

            fold_gain = fold_auc - component["champion_auc"]

            fold_aucs.append(fold_auc)
            fold_gains.append(fold_gain)

            detail_rows.append({
                "tabm_weight": tabm_weight,
                "champion_weight": champion_weight,
                "fold": fold_num,
                "champion_auc": component["champion_auc"],
                "blend_auc": fold_auc,
                "gain_vs_champion": fold_gain,
            })

        summary_row = {
            "tabm_weight": tabm_weight,
            "champion_weight": champion_weight,
            "full_oof_auc": global_auc,
            "full_oof_gain": global_gain,
            "mean_fold_auc": float(np.mean(fold_aucs)),
            "mean_fold_gain": float(np.mean(fold_gains)),
            "min_fold_gain": float(np.min(fold_gains)),
            "max_fold_gain": float(np.max(fold_gains)),
            "positive_gain_folds": int(
                np.sum(np.asarray(fold_gains) > 0)
            ),
        }

        for i, auc in enumerate(fold_aucs, start=1):
            summary_row[f"fold_{i}_auc"] = auc

        for i, gain in enumerate(fold_gains, start=1):
            summary_row[f"fold_{i}_gain"] = gain

        summary_rows.append(summary_row)

    summary = pd.DataFrame(summary_rows)
    details = pd.DataFrame(detail_rows)

    # --------------------------------------------------------
    # 保存结果
    # --------------------------------------------------------
    summary_path = OUTPUT_DIR / "tabm_weight_sensitivity_summary.csv"
    details_path = OUTPUT_DIR / "tabm_weight_sensitivity_by_fold.csv"

    summary.to_csv(summary_path, index=False)
    details.to_csv(details_path, index=False)

    # --------------------------------------------------------
    # 控制台结果
    # --------------------------------------------------------
    display_cols = [
        "tabm_weight",
        "full_oof_auc",
        "full_oof_gain",
        "mean_fold_gain",
        "min_fold_gain",
        "positive_gain_folds",
    ]

    print("\n" + "=" * 110)
    print("WEIGHT SENSITIVITY RESULTS")
    print("=" * 110)

    with pd.option_context(
        "display.max_rows", None,
        "display.max_columns", None,
        "display.width", 200,
        "display.float_format", lambda x: f"{x:.9f}",
    ):
        print(summary[display_cols].to_string(index=False))

    # --------------------------------------------------------
    # 对比 20% 已知基准
    # --------------------------------------------------------
    target_row = summary.loc[
        np.isclose(summary["tabm_weight"], 0.20)
    ]

    if not target_row.empty:
        row = target_row.iloc[0]

        print("\n" + "=" * 88)
        print("20% TABM REPRODUCTION CHECK")
        print("=" * 88)
        print(f"Full OOF AUC : {row['full_oof_auc']:.9f}")
        print(f"Full OOF gain: {row['full_oof_gain']:+.9f}")
        print(f"Mean fold gain: {row['mean_fold_gain']:+.9f}")
        print(
            f"Positive folds: "
            f"{int(row['positive_gain_folds'])}/{N_SPLITS}"
        )

        print("\nPreviously observed reference:")
        print("Full OOF AUC around 0.961322378")
        print("Mean fold gain around +0.000052306")
        print("Positive folds: 5/5")

        if (
            abs(row["full_oof_auc"] - 0.961322378) > 0.00005
            or int(row["positive_gain_folds"]) != 5
        ):
            print(
                "\nWARNING: The result differs materially from the "
                "previous reference. Check prediction files, target "
                "encoding, and fold assignments before interpreting "
                "the weight comparison."
            )

    # --------------------------------------------------------
    # 仅作诊断：列出全量 OOF AUC 最高的权重
    # --------------------------------------------------------
    top_rows = summary.sort_values(
        "full_oof_auc",
        ascending=False,
    ).head(5)

    print("\n" + "=" * 88)
    print("TOP 5 WEIGHTS BY FULL OOF AUC")
    print("=" * 88)

    with pd.option_context(
        "display.max_columns", None,
        "display.width", 180,
        "display.float_format", lambda x: f"{x:.9f}",
    ):
        print(top_rows[display_cols].to_string(index=False))

    print("\nSaved files:")
    print(summary_path)
    print(details_path)

    print("\nNo models were trained in this script.")
    print(
        "Do not automatically select the highest single OOF score; "
        "also inspect fold stability and avoid overfitting the weight."
    )


if __name__ == "__main__":
    main()