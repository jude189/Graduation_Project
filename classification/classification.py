
import warnings
warnings.filterwarnings("ignore")
import os
import pickle
import json
os.environ["PYTHONWARNINGS"] = "ignore"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    precision_score, recall_score,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc, classification_report,
)
import xgboost as xgb
import lightgbm as lgb

# ── Paths ─────────────────────────────────────────────────────────────────────
FEAT_CSV   = "/kaggle/input/datasets/englojainhamdan/featuress/features_claude_v3.csv"
ISIC_ROOT  = "/kaggle/input/datasets/mahmudulhasantasin/isic-2017-original-dataset/isic 2017"
OUTPUT_DIR = Path("/kaggle/working")

GT_PATHS = {
    "train": f"{ISIC_ROOT}/ISIC-2017_Training_Part3_GroundTruth.csv",
    "val":   f"{ISIC_ROOT}/ISIC-2017_Validation_Part3_GroundTruth.csv",
    "test":  f"{ISIC_ROOT}/ISIC-2017_Test_v2_Part3_GroundTruth.csv",
}

SEED          = 42
N_FOLDS       = 5
K_BEST_LINEAR = 120     # for LR / MLP only — trees use full feature space
ES_ROUNDS     = 50      # early-stopping rounds for XGB / LGBM


# =============================================================================
# 1)  DATA LOADING  (unchanged from v9)
# =============================================================================

def load_gt(split):
    df = pd.read_csv(GT_PATHS[split])
    if "image_id" not in df.columns:
        df = df.rename(columns={df.columns[0]: "image_id"})
    df["image_id"] = df["image_id"].astype(str).str.strip()
    def _label(row):
        if row.get("melanoma", 0) == 1:             return "melanoma"
        if row.get("seborrheic_keratosis", 0) == 1: return "seborrheic_keratosis"
        return "nevus"
    df["label"] = df.apply(_label, axis=1)
    return df[["image_id", "label"]]


def build_dataset():
    print("── Loading features …")
    feat = pd.read_csv(FEAT_CSV, low_memory=False)
    feat["image_id"] = feat["image_id"].astype(str).str.strip()
    feat["image_id_bare"] = (feat["image_id"]
                             .str.replace(r"^isic17__", "", regex=True)
                             .str.strip())
    frames = []
    for split in tqdm(["train", "val", "test"], desc="Loading splits", unit="split"):
        gt   = load_gt(split)
        sub  = feat[feat["split"] == split].copy()
        sub  = sub.merge(gt, left_on="image_id_bare", right_on="image_id",
                         how="inner", suffixes=("", "_gt"))
        tqdm.write(f"   {split:5s}: {len(sub)} rows  "
                   f"labels: {sub['label'].value_counts().to_dict()}")
        frames.append(sub)
    df = pd.concat(frames, ignore_index=True)
    print(f"   Total: {len(df)}")
    return df


def prepare_Xy(df):
    drop_cols = {
        "image_id", "image_id_bare", "image_id_gt",
        "split", "label",
        "melanoma", "seborrheic_keratosis",
    }
    feat_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feat_cols].apply(pd.to_numeric, errors="coerce")
    X = X.dropna(axis=1, how="all")
    le = LabelEncoder()
    y  = le.fit_transform(df["label"])
    print(f"   Classes : {le.classes_}")
    print(f"   Features: {X.shape[1]}")
    return X, y, le, X.columns.tolist()


# =============================================================================
# 2)  PREPROCESSING  +  MODEL FACTORIES
# =============================================================================

def base_prep(k=None):
    """Imputer + scaler, optionally with SelectKBest."""
    steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ]
    if k is not None:
        steps.append(("selector", SelectKBest(f_classif, k=k)))
    return steps


def _sample_weights(y):
    """Balanced sample weights (used for XGBoost multiclass)."""
    classes = np.unique(y)
    cw = compute_class_weight("balanced", classes=classes, y=y)
    cw_map = dict(zip(classes, cw))
    return np.array([cw_map[v] for v in y])


# ── XGB wrapper with early-stopping built in ──────────────────────────────────
from sklearn.base import BaseEstimator, ClassifierMixin

class XGBEarlyStop(ClassifierMixin, BaseEstimator):
    """
    XGBClassifier wrapper that:
      • auto-computes balanced sample weights for multiclass.
      • carves off a small internal validation set (10%) for early stopping.
      • uses up to 800 trees but stops when val mlogloss plateaus.
    Fits ~2-3x faster than fixed-500 trees and usually yields better AUC.
    """
    _estimator_type = "classifier"

    def __init__(self, n_estimators=800, learning_rate=0.05, max_depth=5,
                 subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                 gamma=0.1, reg_alpha=0.1, reg_lambda=1.5,
                 es_rounds=ES_ROUNDS, val_frac=0.10, random_state=SEED):
        self.n_estimators     = n_estimators
        self.learning_rate    = learning_rate
        self.max_depth        = max_depth
        self.subsample        = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.gamma            = gamma
        self.reg_alpha        = reg_alpha
        self.reg_lambda       = reg_lambda
        self.es_rounds        = es_rounds
        self.val_frac         = val_frac
        self.random_state     = random_state

    def fit(self, X, y, sample_weight=None):
        self.classes_ = np.unique(y)
        if sample_weight is None:
            sample_weight = _sample_weights(y)
        # Stratified internal split for early stopping
        X_tr, X_val, y_tr, y_val, w_tr, _ = train_test_split(
            X, y, sample_weight, test_size=self.val_frac,
            stratify=y, random_state=self.random_state,
        )
        self._estimator = xgb.XGBClassifier(
            n_estimators     = self.n_estimators,
            learning_rate    = self.learning_rate,
            max_depth        = self.max_depth,
            subsample        = self.subsample,
            colsample_bytree = self.colsample_bytree,
            min_child_weight = self.min_child_weight,
            gamma            = self.gamma,
            reg_alpha        = self.reg_alpha,
            reg_lambda       = self.reg_lambda,
            eval_metric      = "mlogloss",
            early_stopping_rounds = self.es_rounds,
            n_jobs           = -1,
            random_state     = self.random_state,
            verbosity        = 0,
        )
        self._estimator.fit(
            X_tr, y_tr, sample_weight=w_tr,
            eval_set=[(X_val, y_val)], verbose=False,
        )
        return self

    def predict(self, X):       return self._estimator.predict(X)
    def predict_proba(self, X): return self._estimator.predict_proba(X)


class LGBMEarlyStop(ClassifierMixin, BaseEstimator):
    """LightGBM wrapper with class_weight='balanced' + early stopping."""
    _estimator_type = "classifier"

    def __init__(self, n_estimators=800, learning_rate=0.05, num_leaves=40,
                 max_depth=6, subsample=0.8, colsample_bytree=0.8,
                 min_child_samples=10, reg_alpha=0.1, reg_lambda=1.5,
                 es_rounds=ES_ROUNDS, val_frac=0.10, random_state=SEED):
        self.n_estimators      = n_estimators
        self.learning_rate     = learning_rate
        self.num_leaves        = num_leaves
        self.max_depth         = max_depth
        self.subsample         = subsample
        self.colsample_bytree  = colsample_bytree
        self.min_child_samples = min_child_samples
        self.reg_alpha         = reg_alpha
        self.reg_lambda        = reg_lambda
        self.es_rounds         = es_rounds
        self.val_frac          = val_frac
        self.random_state      = random_state

    def fit(self, X, y, sample_weight=None):
        self.classes_ = np.unique(y)
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=self.val_frac, stratify=y,
            random_state=self.random_state,
        )
        self._estimator = lgb.LGBMClassifier(
            n_estimators      = self.n_estimators,
            learning_rate     = self.learning_rate,
            num_leaves        = self.num_leaves,
            max_depth         = self.max_depth,
            subsample         = self.subsample,
            colsample_bytree  = self.colsample_bytree,
            min_child_samples = self.min_child_samples,
            reg_alpha         = self.reg_alpha,
            reg_lambda        = self.reg_lambda,
            class_weight      = "balanced",
            n_jobs            = -1,
            random_state      = self.random_state,
            verbose           = -1,
        )
        self._estimator.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(self.es_rounds, verbose=False)],
        )
        return self

    def predict(self, X):       return self._estimator.predict(X)
    def predict_proba(self, X): return self._estimator.predict_proba(X)

    @property
    def feature_importances_(self):
        return self._estimator.feature_importances_


def build_models():
    """
    Six base models. Trees are NOT wrapped in CalibratedClassifierCV
    (calibration discretises probabilities and hurts AUC). Final ensemble
    is a soft-voting blend of the three strongest base learners.
    """
    # ── Linear / NN models — class_weight='balanced', SelectKBest=120
    lr_pipe = Pipeline(base_prep(k=K_BEST_LINEAR) + [
        ("clf", LogisticRegression(
            C=0.5, max_iter=1000, solver="lbfgs",
            class_weight="balanced", random_state=SEED)),
    ])

    mlp_pipe = Pipeline(base_prep(k=K_BEST_LINEAR) + [
        ("clf", MLPClassifier(
            hidden_layer_sizes=(256, 128, 64), activation="relu",
            alpha=0.001, max_iter=400, early_stopping=True,
            validation_fraction=0.1, random_state=SEED)),
    ])

    # ── Tree models — full feature space, balanced weighting, no calibration
    rf_pipe = Pipeline(base_prep() + [
        ("clf", RandomForestClassifier(
            n_estimators=600, max_depth=None, min_samples_leaf=1,
            max_features="sqrt", class_weight="balanced",
            n_jobs=-1, random_state=SEED)),
    ])

    # HistGradientBoosting replaces the slow SVM — class_weight='balanced'
    # is supported natively, fits in seconds, AUC competitive with XGB.
    hgb_pipe = Pipeline(base_prep() + [
        ("clf", HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.05, max_depth=6,
            l2_regularization=1.0, class_weight="balanced",
            early_stopping=True, validation_fraction=0.10,
            n_iter_no_change=ES_ROUNDS, random_state=SEED)),
    ])

    xgb_pipe = Pipeline(base_prep() + [("clf", XGBEarlyStop())])
    lgbm_pipe = Pipeline(base_prep() + [("clf", LGBMEarlyStop())])

    return {
        "LogisticRegression": lr_pipe,
        "HistGradBoost":      hgb_pipe,
        "MLP":                mlp_pipe,
        "RandomForest":       rf_pipe,
        "XGBoost":            xgb_pipe,
        "LightGBM":           lgbm_pipe,
    }


def build_ensemble(trained_models):
    """
    Soft-voting ensemble of the three strongest learners:
        LightGBM + XGBoost + LogisticRegression
    Wraps already-fitted pipelines in a prefit VotingClassifier so we don't
    refit. This typically lifts macro-AUC by 0.005–0.02 over the best
    individual model at zero extra training cost.
    """
    members = [
        ("lgbm", trained_models["LightGBM"]),
        ("xgb",  trained_models["XGBoost"]),
        ("lr",   trained_models["LogisticRegression"]),
    ]
    # Manual soft-voting (sklearn's VotingClassifier(prefit=...) is brittle
    # across versions). We define a tiny wrapper instead.
    return _PrefitSoftVote(members)


class _PrefitSoftVote:
    """Average predict_proba across already-fitted estimators."""
    def __init__(self, members):
        self.members = members  # list of (name, fitted_pipeline)
        self.classes_ = members[0][1].classes_

    def predict_proba(self, X):
        probs = np.mean([m.predict_proba(X) for _, m in self.members], axis=0)
        return probs

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def fit(self, X, y):
        # Members are already fit; no-op to satisfy interface.
        return self


# =============================================================================
# 3)  METRICS
# =============================================================================

def compute_metrics(y_true, y_pred, y_prob, le):
    n  = len(le.classes_)
    yb = label_binarize(y_true, classes=list(range(n)))
    return {
        "AUC"      : round(roc_auc_score(yb, y_prob,
                           multi_class="ovr", average="macro"), 4),
        "Accuracy" : round(accuracy_score(y_true, y_pred), 4),
        "F1"       : round(f1_score(y_true, y_pred,
                           average="macro", zero_division=0), 4),
        "Precision": round(precision_score(y_true, y_pred,
                           average="macro", zero_division=0), 4),
        "Recall"   : round(recall_score(y_true, y_pred,
                           average="macro", zero_division=0), 4),
    }


# =============================================================================
# 4)  CROSS-VALIDATION  (also collects OOF probs — single-pass, no rerun)
# =============================================================================

def run_cv_with_oof(name, pipe, X, y, le):
    """Run 5-fold CV and return both summary stats AND OOF probabilities.
    Saves us from recomputing OOF probs later for the operating-point report.
    """
    print(f"\n   CV  [{name}] …")
    skf  = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    aucs, accs, f1s, precs, recs = [], [], [], [], []
    n_cls   = len(le.classes_)
    oof_prob = np.zeros((len(X), n_cls))

    fold_bar = tqdm(enumerate(skf.split(X, y), 1), total=N_FOLDS,
                    desc=f"   {name} folds", unit="fold", leave=False)
    for fold, (tr_idx, va_idx) in fold_bar:
        Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
        ytr, yva = y[tr_idx],      y[va_idx]
        # Clone the pipeline each fold (fresh fit)
        from sklearn.base import clone
        fold_pipe = clone(pipe)
        fold_pipe.fit(Xtr, ytr)
        y_prob = fold_pipe.predict_proba(Xva)
        y_pred = fold_pipe.predict(Xva)
        oof_prob[va_idx] = y_prob
        m = compute_metrics(yva, y_pred, y_prob, le)
        aucs.append(m["AUC"]);  accs.append(m["Accuracy"])
        f1s.append(m["F1"]);    precs.append(m["Precision"])
        recs.append(m["Recall"])
        fold_bar.set_postfix(auc=f"{m['AUC']:.4f}", acc=f"{m['Accuracy']:.4f}")

    def _f(v): return f"{np.mean(v):.4f} ± {np.std(v):.4f}"
    res = {
        "AUC":       _f(aucs),  "AUC_mean":  np.mean(aucs),
        "Accuracy":  _f(accs),  "Acc_mean":  np.mean(accs),
        "F1":        _f(f1s),
        "Precision": _f(precs),
        "Recall":    _f(recs),
    }
    print(f"      AUC={res['AUC']}  Acc={res['Accuracy']}  F1={res['F1']}")
    return res, oof_prob


# =============================================================================
# 5)  PLOTS  (unchanged from v9)
# =============================================================================

def plot_roc(model_name, y_true, y_prob, le, out_path):
    n  = len(le.classes_)
    yb = label_binarize(y_true, classes=list(range(n)))
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, (cls, col) in enumerate(
            zip(le.classes_, ["#e41a1c", "#377eb8", "#4daf4a"])):
        fpr, tpr, _ = roc_curve(yb[:, i], y_prob[:, i])
        ax.plot(fpr, tpr, lw=2, color=col,
                label=f"{cls} (AUC={auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set(xlabel="FPR", ylabel="TPR", title=f"ROC — {model_name}")
    ax.legend(loc="lower right"); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()


def plot_confusion(model_name, y_true, y_pred, le, out_path):
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(
        confusion_matrix(y_true, y_pred),
        display_labels=le.classes_
    ).plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {model_name}")
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()


def plot_feature_importance(imp_df, out_path, top_n=30):
    df  = imp_df.head(top_n)
    fig, ax = plt.subplots(figsize=(10, top_n * 0.35 + 1))
    colors  = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(df)))[::-1]
    ax.barh(df["feature"][::-1], df["importance"][::-1], color=colors[::-1])
    ax.set(xlabel="Mean Importance Score",
           title=f"Top {top_n} Features (RF + XGB + LGBM averaged)")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_summary(cv_results, te_results, out_path):
    names   = list(te_results.keys())
    cv_aucs = [cv_results.get(n, {}).get("AUC_mean", 0) for n in names]
    te_aucs = [te_results[n]["AUC"]      for n in names]
    te_accs = [te_results[n]["Accuracy"] for n in names]
    x   = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(x - 0.2, cv_aucs, 0.38, label="CV AUC",   color="#377eb8", alpha=0.85)
    axes[0].bar(x + 0.2, te_aucs, 0.38, label="Test AUC", color="#e41a1c", alpha=0.85)
    axes[0].axhline(0.85, color="black", ls="--", lw=1, label="Target 0.85")
    axes[0].set_xticks(x); axes[0].set_xticklabels(names, rotation=25, ha="right")
    axes[0].set(ylabel="AUC", title="CV vs Test AUC")
    axes[0].legend(); axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(x, te_accs, color="#4daf4a", alpha=0.85)
    axes[1].axhline(0.80, color="black", ls="--", lw=1, label="Target 0.80")
    axes[1].set_xticks(x); axes[1].set_xticklabels(names, rotation=25, ha="right")
    axes[1].set(ylabel="Accuracy", title="Test Accuracy by Model")
    axes[1].legend(); axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()


# =============================================================================
# 6)  FEATURE IMPORTANCE  (no calibration wrappers anymore — simpler)
# =============================================================================

def get_feature_importance(trained_models, feat_names):
    """
    Average feature_importances_ across RF / XGB / LGBM. Trees use the full
    feature space (no SelectKBest), so importances align 1:1 with feat_names.
    """
    scores = pd.DataFrame({"feature": feat_names})
    for name in ["RandomForest", "XGBoost", "LightGBM"]:
        if name not in trained_models:
            continue
        pipe = trained_models[name]
        try:
            inner = pipe.named_steps["clf"]
            # Unwrap our XGB / LGBM wrappers if needed
            if hasattr(inner, "_estimator"):
                imp = inner._estimator.feature_importances_
            else:
                imp = inner.feature_importances_
            # Trees have no SelectKBest in v10, so imp aligns with feat_names.
            # Defensive: if a future change adds a selector, map back.
            selector = pipe.named_steps.get("selector")
            if selector is not None:
                full_imp = np.zeros(len(feat_names))
                support  = selector.get_support()
                full_imp[support] = imp
                imp = full_imp
            imp = imp / (imp.sum() + 1e-12)
            scores[name] = imp
        except Exception as e:
            print(f"   [warn] could not extract importances for {name}: {e}")
            continue
    imp_cols = [c for c in scores.columns if c != "feature"]
    scores["importance"] = scores[imp_cols].mean(axis=1) if imp_cols else 0.0
    return scores.sort_values("importance", ascending=False).reset_index(drop=True)


# =============================================================================
# 7)  FEATURE-GROUP BREAKDOWN  (unchanged from v9)
# =============================================================================

def feature_group_breakdown(feat_cols, out_csv, out_png):
    rx_cols          = [c for c in feat_cols if c.startswith("rx_")]
    asym_centre      = ([c for c in feat_cols if c.startswith("abcde_asym_")
                         and "_pa_" not in c]
                        + [c for c in feat_cols if c.startswith("abcde_color_asym_")])
    asym_pa          = [c for c in feat_cols if c.startswith("abcde_asym_pa_")]
    border_contour   = [c for c in feat_cols if any(
                            c.startswith(f"abcde_{x}") for x in
                            ["compactness", "border_roughness", "convexity",
                             "solidity", "elongation", "radial_dist_std",
                             "border_notch_cnt", "fourier_coef"])]
    border_octants   = [c for c in feat_cols if c.startswith("abcde_border_b8")
                        or c.startswith("abcde_border_oct")
                        or c.startswith("abcde_border_mean")
                        or c.startswith("abcde_border_gradient")
                        or c.startswith("abcde_border_sharpness")]
    color_stats      = [c for c in feat_cols if c.startswith("abcde_col_")
                        and "_het_" not in c]
    color_6          = [c for c in feat_cols if c.startswith("col6_")]
    color_het        = [c for c in feat_cols if c.startswith("abcde_col_het_")]
    radial           = [c for c in feat_cols if c.startswith("abcde_radial_")]
    diameter         = [c for c in feat_cols if c.startswith("abcde_area_")
                        or c.startswith("abcde_diam_")]
    rx_firstorder = [c for c in rx_cols if "firstorder" in c.lower()]
    rx_shape2d    = [c for c in rx_cols if "shape2d" in c.lower()]
    rx_glcm       = [c for c in rx_cols if "glcm" in c.lower()]
    rx_glrlm      = [c for c in rx_cols if "glrlm" in c.lower()]
    rx_glszm      = [c for c in rx_cols if "glszm" in c.lower()]
    rx_ngtdm      = [c for c in rx_cols if "ngtdm" in c.lower()]

    groups = {
        "PyRadiomics — firstorder":                          rx_firstorder,
        "PyRadiomics — shape2D":                             rx_shape2d,
        "PyRadiomics — GLCM":                                rx_glcm,
        "PyRadiomics — GLRLM":                               rx_glrlm,
        "PyRadiomics — GLSZM":                               rx_glszm,
        "PyRadiomics — NGTDM":                               rx_ngtdm,
        "A — Asymmetry (image-centre fold)":                 asym_centre,
        "A — Asymmetry (principal-axis, Stolz 1993)":        asym_pa,
        "B — Border (contour: roughness/Fourier/notch)":     border_contour,
        "B — Border (8-octant sharpness, Stolz 1994)":       border_octants,
        "C — Color stats (RGB/HSV/LAB + BWV + regression)":  color_stats,
        "C — 6-color palette (Argenziano 2003)":             color_6,
        "C — Color heterogeneity (IQR/MAD/CV)":              color_het,
        "C — Radial color distribution":                     radial,
        "D — Diameter / area":                               diameter,
    }
    rows = [{"group": g, "n_features": len(cols)} for g, cols in groups.items()]
    rows.append({"group": "── TOTAL", "n_features": sum(r["n_features"] for r in rows)})
    df_grp = pd.DataFrame(rows)
    df_grp.to_csv(out_csv, index=False)

    plot_df = df_grp[df_grp["group"] != "── TOTAL"].sort_values("n_features")
    fig, ax = plt.subplots(figsize=(10, 0.55 * len(plot_df) + 1))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(plot_df)))
    ax.barh(plot_df["group"], plot_df["n_features"], color=colors)
    for i, v in enumerate(plot_df["n_features"]):
        ax.text(v + 0.5, i, str(int(v)), va="center", fontsize=9)
    total = sum(r["n_features"] for r in rows[:-1])
    ax.set(xlabel="Number of features",
           title=f"Feature-group composition ({total} total)")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    return df_grp


# =============================================================================
# 8)  DIM-REDUCTION  (unchanged from v9)
# =============================================================================

def dim_reduction_analysis(X_cv, y_cv, le, out_dir):
    print("\n── Dim-reduction analysis (PCA + t-SNE) …")
    Xc = SimpleImputer(strategy="median").fit_transform(X_cv)
    Xs = StandardScaler().fit_transform(Xc)

    pca2  = PCA(n_components=2, random_state=SEED).fit(Xs)
    X_pca = pca2.transform(Xs)

    print("   running t-SNE (~1–3 min) …")
    X_tsne = TSNE(n_components=2, random_state=SEED, perplexity=30,
                  init="pca", learning_rate="auto").fit_transform(Xs)

    pca_full = PCA(random_state=SEED).fit(Xs)
    cumvar   = np.cumsum(pca_full.explained_variance_ratio_)
    n95 = int(np.argmax(cumvar >= 0.95) + 1)
    n99 = int(np.argmax(cumvar >= 0.99) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = ["#e41a1c", "#377eb8", "#4daf4a"]
    for i, cls in enumerate(le.classes_):
        m = y_cv == i
        axes[0].scatter(X_pca[m, 0],  X_pca[m, 1],  c=colors[i], label=cls, alpha=0.55, s=18)
        axes[1].scatter(X_tsne[m, 0], X_tsne[m, 1], c=colors[i], label=cls, alpha=0.55, s=18)
    pc1, pc2 = pca2.explained_variance_ratio_
    axes[0].set(xlabel="PC1", ylabel="PC2",
                title=f"PCA  (PC1+PC2 = {(pc1+pc2)*100:.1f}% variance)")
    axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].set(xlabel="t-SNE 1", ylabel="t-SNE 2", title="t-SNE projection")
    axes[1].legend(); axes[1].grid(alpha=0.3)
    axes[2].plot(range(1, len(cumvar) + 1), cumvar, lw=2, color="#333333")
    axes[2].axhline(0.95, ls="--", color="red",    label="95% var")
    axes[2].axhline(0.99, ls="--", color="orange", label="99% var")
    axes[2].axvline(n95,  ls=":",  color="red",    alpha=0.5)
    axes[2].axvline(n99,  ls=":",  color="orange", alpha=0.5)
    axes[2].set(xlabel="Number of components", ylabel="Cumulative variance",
                title=f"PCA scree  (95% → {n95},  99% → {n99})")
    axes[2].legend(); axes[2].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "dim_reduction.png", dpi=150, bbox_inches="tight")
    plt.close()

    summary = pd.DataFrame([{
        "n_features_total"       : int(X_cv.shape[1]),
        "n_components_95pct_var" : n95,
        "n_components_99pct_var" : n99,
        "compression_ratio_95"   : round(n95 / X_cv.shape[1], 4),
        "pc1_explained_var"      : round(float(pc1), 6),
        "pc2_explained_var"      : round(float(pc2), 6),
        "pc1_pc2_combined"       : round(float(pc1 + pc2), 6),
    }])
    summary.to_csv(out_dir / "dim_reduction_summary.csv", index=False)
    print(f"   95% variance captured by {n95}/{X_cv.shape[1]} components "
          f"({n95 / X_cv.shape[1] * 100:.1f}% of original dim).")
    return summary


# =============================================================================
# 9)  PER-CLASS METRICS
# =============================================================================

def per_class_and_cm(trained_models, X_te, y_te, le, out_dir):
    """
    All predictions here are MODEL ARGMAX (the "final verdict"). No threshold
    is applied — this is the canonical evaluation.
    """
    print("\n── Per-class metrics + numeric confusion matrices (argmax) …")
    rows = []
    for name, pipe in tqdm(trained_models.items(), desc="models", leave=False):
        y_pred = pipe.predict(X_te)
        cm = confusion_matrix(y_te, y_pred)
        pd.DataFrame(cm, index=le.classes_, columns=le.classes_)\
          .to_csv(out_dir / f"cm_{name}.csv")
        rep = classification_report(
            y_te, y_pred, target_names=le.classes_,
            digits=4, output_dict=True, zero_division=0)
        pd.DataFrame(rep).T.round(4)\
          .to_csv(out_dir / f"per_class_{name}.csv")
        for cls in le.classes_:
            rows.append({
                "model"    : name, "class": cls,
                "precision": round(rep[cls]["precision"], 4),
                "recall"   : round(rep[cls]["recall"],    4),
                "f1"       : round(rep[cls]["f1-score"],  4),
                "support"  : int(rep[cls]["support"]),
            })
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "per_class_summary.csv", index=False)
    return summary


def plot_melanoma_sensitivity(per_class_summary, out_path):
    mel = (per_class_summary[per_class_summary["class"] == "melanoma"]
           .sort_values("recall", ascending=True))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.85, len(mel)))
    ax.barh(mel["model"], mel["recall"], color=colors)
    for i, v in enumerate(mel["recall"]):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=10)
    ax.set(xlabel="Recall (sensitivity) on melanoma class — model argmax",
           title="Melanoma sensitivity by model — primary clinical metric",
           xlim=(0, 1))
    ax.axvline(0.80, ls="--", color="black", lw=1, label="0.80 target")
    ax.legend(); ax.grid(axis="x", alpha=0.3)
    plt.tight_layout(); plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_per_class_grouped_bars(per_class_summary, le, out_path):
    metrics = ["precision", "recall", "f1"]
    classes = list(le.classes_)
    models  = per_class_summary["model"].unique().tolist()
    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 5),
                             sharey=True)
    width = 0.8 / len(models)
    x     = np.arange(len(classes))
    cmap  = plt.cm.tab10(np.linspace(0, 1, len(models)))
    for ax, met in zip(axes, metrics):
        for j, mdl in enumerate(models):
            sub = (per_class_summary[per_class_summary["model"] == mdl]
                   .set_index("class").loc[classes, met].values)
            ax.bar(x + j * width - 0.4 + width / 2, sub,
                   width, label=mdl, color=cmap[j])
        ax.set_xticks(x); ax.set_xticklabels(classes, rotation=15)
        ax.set_ylim(0, 1.0); ax.set_title(met.capitalize())
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Score")
    axes[-1].legend(loc="upper right", bbox_to_anchor=(1.35, 1.0))
    plt.suptitle("Per-class precision / recall / F1 by model (argmax)", y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# =============================================================================
# 10)  CLINICAL METRICS  +  THRESHOLD CALIBRATION  (informational only)
# =============================================================================
# IMPORTANT v10 FRAMING:
#   The headline test results above use MODEL ARGMAX as the final verdict.
#   The operating-point table below is INFORMATIONAL — it shows what the
#   sensitivity/specificity tradeoff would look like IF a deployment chose
#   to apply a probability threshold instead of argmax. Useful for clinical
#   stakeholders but not the primary evaluation.

def _melanoma_idx(le):
    return int(np.where(le.classes_ == "melanoma")[0][0])


def compute_clinical_metrics(y_true, y_pred, le):
    mi = _melanoma_idx(le)
    tp = int(np.sum((y_true == mi) & (y_pred == mi)))
    fn = int(np.sum((y_true == mi) & (y_pred != mi)))
    fp = int(np.sum((y_true != mi) & (y_pred == mi)))
    tn = int(np.sum((y_true != mi) & (y_pred != mi)))
    sens = tp / (tp + fn + 1e-12)
    spec = tn / (tn + fp + 1e-12)
    ppv  = tp / (tp + fp + 1e-12)
    npv  = tn / (tn + fn + 1e-12)
    f2   = (5 * ppv * sens) / (4 * ppv + sens + 1e-12)
    return {
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "sensitivity": round(sens, 4), "specificity": round(spec, 4),
        "PPV":         round(ppv,  4), "NPV":         round(npv,  4),
        "F2":          round(f2,   4),
    }


def apply_melanoma_threshold(y_prob, le, threshold):
    mi   = _melanoma_idx(le)
    n_cl = len(le.classes_)
    y_pred = np.full(len(y_prob), -1, dtype=int)
    for i, probs in enumerate(y_prob):
        if probs[mi] >= threshold:
            y_pred[i] = mi
        else:
            others = [(j, probs[j]) for j in range(n_cl) if j != mi]
            y_pred[i] = max(others, key=lambda x: x[1])[0]
    return y_pred


def threshold_sweep(y_prob, y_true, le, t_lo=0.05, t_hi=0.50, n_steps=181):
    thresholds = np.linspace(t_lo, t_hi, n_steps)
    rows = []
    for t in thresholds:
        yp = apply_melanoma_threshold(y_prob, le, t)
        cm = compute_clinical_metrics(y_true, yp, le)
        rows.append({"threshold": round(float(t), 4), **cm})
    return pd.DataFrame(rows)


def find_operating_points(sweep):
    points = {}
    scr = sweep[sweep["sensitivity"] >= 0.85]
    r = (scr.loc[scr["F2"].idxmax()] if len(scr) > 0
         else sweep.loc[sweep["sensitivity"].idxmax()])
    points["screening"] = (float(r["threshold"]), r)

    r = sweep.loc[sweep["F2"].idxmax()]
    points["balanced"] = (float(r["threshold"]), r)

    spc = sweep[sweep["specificity"] >= 0.90]
    r = (spc.loc[spc["F2"].idxmax()] if len(spc) > 0
         else sweep.loc[sweep["specificity"].idxmax()])
    points["specialist"] = (float(r["threshold"]), r)
    return points


def plot_threshold_sweep(sweep, op_points, model_name, out_path):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(sweep["threshold"], sweep["sensitivity"], lw=2,
            color="#e41a1c", label="Sensitivity")
    ax.plot(sweep["threshold"], sweep["specificity"], lw=2,
            color="#377eb8", label="Specificity")
    ax.plot(sweep["threshold"], sweep["PPV"], lw=2, ls="--",
            color="#4daf4a", label="PPV")
    ax.plot(sweep["threshold"], sweep["F2"], lw=2, ls="-.",
            color="#984ea3", label="F2")
    style = {"screening": ("#e41a1c", ":"),
             "balanced":  ("#000000", ":"),
             "specialist":("#377eb8", ":")}
    for nm, (t, row) in op_points.items():
        c, ls = style[nm]
        ax.axvline(t, color=c, ls=ls, lw=1.2,
                   label=f"{nm}: t={t:.2f} (Sens={row['sensitivity']:.2f}, "
                         f"Spec={row['specificity']:.2f})")
    ax.set(xlabel="Melanoma probability threshold", ylabel="Score",
           title=f"Threshold sweep + operating points (informational) — {model_name}",
           xlim=(sweep["threshold"].min(), sweep["threshold"].max()),
           ylim=(0, 1.05))
    ax.legend(loc="center right", fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()


def print_clinical_summary(name, cm_dict):
    print(f"   {name:<28}  "
          f"Sens={cm_dict['sensitivity']:.3f}  "
          f"Spec={cm_dict['specificity']:.3f}  "
          f"PPV={cm_dict['PPV']:.3f}  "
          f"NPV={cm_dict['NPV']:.3f}  "
          f"F2={cm_dict['F2']:.3f}  "
          f"(TP={cm_dict['TP']} FN={cm_dict['FN']} FP={cm_dict['FP']})")


# =============================================================================
# 11)  MAIN
# =============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("  ISIC-2017  ·  ML Classification  v10 — argmax verdict, faster, higher AUC")
    print("=" * 70)

    df = build_dataset()

    print("\n── Preparing feature matrix …")
    X, y, le, feat_names = prepare_Xy(df)

    split_series = df["split"].values
    X_tr, y_tr = X[split_series == "train"], y[split_series == "train"]
    X_va, y_va = X[split_series == "val"],   y[split_series == "val"]
    X_te, y_te = X[split_series == "test"],  y[split_series == "test"]
    X_cv = pd.concat([X_tr, X_va], ignore_index=True)
    y_cv = np.concatenate([y_tr, y_va])
    print(f"   Train:{len(X_tr)}  Val:{len(X_va)}  Test:{len(X_te)}  CV:{len(X_cv)}")

    print("\n── Feature-group breakdown …")
    grp_df = feature_group_breakdown(
        feat_names,
        OUTPUT_DIR / "feature_groups.csv",
        OUTPUT_DIR / "feature_groups.png",
    )
    print(grp_df.to_string(index=False))

    models = build_models()

    # ── Cross-validation (collects OOF probs in same pass) ────────────────
    cv_results = {}
    oof_probs  = {}
    print("\n" + "=" * 70)
    print("  CROSS-VALIDATION  (5-fold stratified, with OOF probabilities)")
    print("=" * 70)
    model_bar = tqdm(models.items(), desc="Models (CV)", unit="model")
    for name, pipe in model_bar:
        model_bar.set_description(f"CV: {name}")
        cv_results[name], oof_probs[name] = run_cv_with_oof(name, pipe, X_cv, y_cv, le)

    print("\n" + "=" * 70)
    print("  CV SUMMARY")
    print("=" * 70)
    cv_df = pd.DataFrame({
        n: {k: v for k, v in r.items() if "mean" not in k}
        for n, r in cv_results.items()
    }).T
    print(cv_df.to_string())
    cv_df.to_csv(OUTPUT_DIR / "cv_results.csv")

    # ── Final fit on full CV data + test evaluation (ARGMAX) ──────────────
    print("\n" + "=" * 70)
    print("  FINAL EVALUATION ON TEST SET  (model argmax — the final verdict)")
    print("=" * 70)
    te_results = {}
    trained    = {}
    fit_bar = tqdm(models.items(), desc="Training final models", unit="model")
    for name, pipe in fit_bar:
        fit_bar.set_description(f"Fitting: {name}")
        pipe.fit(X_cv, y_cv)
        trained[name] = pipe
        y_pred = pipe.predict(X_te)              # ← ARGMAX = final verdict
        y_prob = pipe.predict_proba(X_te)
        m      = compute_metrics(y_te, y_pred, y_prob, le)
        te_results[name] = m
        ta = "✓" if m["AUC"]      >= 0.85 else "✗"
        tb = "✓" if m["Accuracy"] >= 0.80 else "✗"
        tqdm.write(f"   [{name}]  AUC={m['AUC']} {ta}  Acc={m['Accuracy']} {tb}  "
                   f"F1={m['F1']}  Prec={m['Precision']}  Rec={m['Recall']}")
        plot_roc(name, y_te, y_prob, le, OUTPUT_DIR / f"roc_{name}.png")
        plot_confusion(name, y_te, y_pred, le, OUTPUT_DIR / f"cm_{name}.png")

    # ── Soft-voting ensemble (free AUC lift) ──────────────────────────────
    print("\n── Building soft-voting ensemble (LightGBM + XGBoost + LogReg) …")
    ensemble = build_ensemble(trained)
    trained["Ensemble"] = ensemble
    y_pred_ens = ensemble.predict(X_te)          # ← ARGMAX of mean probs
    y_prob_ens = ensemble.predict_proba(X_te)
    te_results["Ensemble"] = compute_metrics(y_te, y_pred_ens, y_prob_ens, le)
    m = te_results["Ensemble"]
    print(f"   [Ensemble]  AUC={m['AUC']}  Acc={m['Accuracy']}  F1={m['F1']}  "
          f"Prec={m['Precision']}  Rec={m['Recall']}")
    plot_roc("Ensemble", y_te, y_prob_ens, le, OUTPUT_DIR / "roc_Ensemble.png")
    plot_confusion("Ensemble", y_te, y_pred_ens, le, OUTPUT_DIR / "cm_Ensemble.png")

    pd.DataFrame(te_results).T.to_csv(OUTPUT_DIR / "test_results.csv")
    plot_summary(cv_results, te_results, OUTPUT_DIR / "summary_plot.png")

    # ── Save deployment artefacts ─────────────────────────────────────────
    print("\n── Saving deployment artefacts …")
    with open(OUTPUT_DIR / "pipeline_XGBoost.pkl", "wb") as f:
        pickle.dump(trained["XGBoost"], f, protocol=4)
    with open(OUTPUT_DIR / "pipeline_LightGBM.pkl", "wb") as f:
        pickle.dump(trained["LightGBM"], f, protocol=4)
    with open(OUTPUT_DIR / "label_encoder.pkl", "wb") as f:
        pickle.dump(le, f, protocol=4)
    pd.Series(feat_names).to_csv(OUTPUT_DIR / "feature_columns_v2.csv",
                                  index=False, header=["feature"])
    print(f"   ✓ pipelines + encoder + feature_columns_v2.csv  ({len(feat_names)} cols)")

    # ── Feature importance ────────────────────────────────────────────────
    print("\n── Computing feature importance …")
    imp_df = get_feature_importance(trained, feat_names)
    imp_df.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)
    plot_feature_importance(imp_df, OUTPUT_DIR / "feature_importance.png", top_n=30)

    # ── Per-class + clinical (argmax) ─────────────────────────────────────
    print("\n" + "=" * 70)
    print("  SUPPLEMENTARY ANALYSIS  (all argmax-based)")
    print("=" * 70)

    per_class_summary = per_class_and_cm(trained, X_te, y_te, le, OUTPUT_DIR)
    plot_per_class_grouped_bars(per_class_summary, le,
                                OUTPUT_DIR / "per_class_grouped_bars.png")
    plot_melanoma_sensitivity(per_class_summary,
                              OUTPUT_DIR / "melanoma_sensitivity.png")

    print("\n   Melanoma recall (sensitivity) by model — argmax:")
    mel = (per_class_summary[per_class_summary["class"] == "melanoma"]
           .sort_values("recall", ascending=False))
    print(mel[["model", "recall", "precision", "f1"]].to_string(index=False))

    print("\n── Clinical metrics — melanoma vs rest, ARGMAX (the final verdict) …")
    clinical_rows = []
    for name, pipe in trained.items():
        y_pred = pipe.predict(X_te)
        cm_dict = compute_clinical_metrics(y_te, y_pred, le)
        cm_dict["model"] = name
        clinical_rows.append(cm_dict)
        print_clinical_summary(name, cm_dict)
    clinical_df = pd.DataFrame(clinical_rows).set_index("model")
    clinical_df.to_csv(OUTPUT_DIR / "clinical_metrics.csv")

    # ── Operating-point report (INFORMATIONAL — for clinical reference) ──
    # Reuses OOF probs already collected during CV, so no extra training cost.
    print("\n" + "=" * 70)
    print("  OPERATING-POINT REPORT  (informational — final verdict is argmax)")
    print("=" * 70)

    op_table_rows = []
    thresholds_json = {}
    op_models = [m for m in ["XGBoost", "LightGBM", "Ensemble"] if m in trained]

    for model_name in op_models:
        print(f"\n── {model_name}: threshold sweep on OOF probabilities …")
        if model_name == "Ensemble":
            # Average OOF across the ensemble members
            members_oof = [oof_probs[n] for n in ["LightGBM", "XGBoost",
                                                  "LogisticRegression"]
                           if n in oof_probs]
            oof = np.mean(members_oof, axis=0)
        else:
            oof = oof_probs[model_name]

        sweep_oof = threshold_sweep(oof, y_cv, le)
        sweep_oof.to_csv(OUTPUT_DIR / f"threshold_sweep_OOF_{model_name}.csv",
                         index=False)
        op_points = find_operating_points(sweep_oof)
        thresholds_json[model_name] = {nm: t for nm, (t, _) in op_points.items()}

        y_prob_test = trained[model_name].predict_proba(X_te)
        sweep_test  = threshold_sweep(y_prob_test, y_te, le)
        sweep_test.to_csv(OUTPUT_DIR / f"threshold_sweep_TEST_{model_name}.csv",
                          index=False)
        plot_threshold_sweep(sweep_test, op_points, model_name,
                             OUTPUT_DIR / f"threshold_sweep_{model_name}.png")

        print(f"\n   {model_name} — three operating points evaluated on TEST set:")
        for nm, (t, row_oof) in op_points.items():
            y_pred = apply_melanoma_threshold(y_prob_test, le, t)
            cm     = compute_clinical_metrics(y_te, y_pred, le)
            acc    = round(accuracy_score(y_te, y_pred), 4)
            f1m    = round(f1_score(y_te, y_pred, average="macro",
                                    zero_division=0), 4)
            print_clinical_summary(f"{model_name}/{nm} t={t:.2f}", cm)
            op_table_rows.append({
                "model": model_name, "operating_point": nm,
                "threshold": t, "accuracy": acc, "macro_f1": f1m,
                **cm,
            })

    op_table = pd.DataFrame(op_table_rows)
    op_table.to_csv(OUTPUT_DIR / "operating_points_test.csv", index=False)
    with open(OUTPUT_DIR / "melanoma_thresholds.json", "w") as f:
        json.dump(thresholds_json, f, indent=2)
    print(f"\n   ✓ melanoma_thresholds.json (informational only)  {thresholds_json}")

    # ── Dim-reduction ─────────────────────────────────────────────────────
    dim_summary = dim_reduction_analysis(X_cv, y_cv, le, OUTPUT_DIR)
    print(f"\n   {dim_summary.to_string(index=False)}")

    # ── Final summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  FINAL SUMMARY — TEST SET (model argmax = final verdict)")
    print("=" * 70)
    print(f"\n  {'Model':<22} {'AUC':>7} {'Acc':>7} {'F1':>7} {'Prec':>7} {'Rec':>7}")
    print(f"  {'-' * 57}")
    for name, m in sorted(te_results.items(), key=lambda x: -x[1]["AUC"]):
        ta = "✓" if m["AUC"]      >= 0.85 else " "
        tb = "✓" if m["Accuracy"] >= 0.80 else " "
        print(f"  {name:<22} {m['AUC']:>6}{ta}  {m['Accuracy']:>5}{tb}  "
              f"{m['F1']:>6}  {m['Precision']:>6}  {m['Recall']:>6}")

    print("\n── Clinical metrics summary (argmax, melanoma vs rest):")
    print(clinical_df[["sensitivity", "specificity", "PPV", "NPV", "F2"]].to_string())

    print("\n── Operating-point table (informational, applied to test set):")
    print(op_table[["model","operating_point","threshold","sensitivity",
                    "specificity","PPV","NPV","F2","accuracy"]].to_string(index=False))

    print("\n  Top 10 most important features:")
    print(imp_df[["feature", "importance"]].head(10).to_string(index=False))

    print("\n  Outputs in /kaggle/working/:")
    for f in sorted(OUTPUT_DIR.glob("*")):
        if f.is_file() and f.suffix in {".csv", ".png", ".pkl", ".json"}:
            print(f"    {f.name}")
    print("\n  Done.")


if __name__ == "__main__":
    main()
