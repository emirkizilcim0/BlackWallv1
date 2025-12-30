import os
from typing import Any, Dict

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)

MODELS_DIR = "models"
DATA_PATH = "data/CIC_IDS_2017/Friday-WorkingHours-Morning.pcap_ISCX.csv"
LABEL_COL = "Label"


def load_data(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise SystemExit(f"Dataset not found at {path}")
    df = pd.read_csv(path)
    # Normalize column names (strip whitespace)
    df.columns = [c.strip() for c in df.columns]
    if LABEL_COL not in df.columns:
        raise SystemExit(f"Label column '{LABEL_COL}' not found in {path}")

    y_raw = df[LABEL_COL]
    y = y_raw.apply(lambda x: 0 if str(x).strip().upper() == "BENIGN" else 1).astype(int)
    X_raw = df.drop(columns=[LABEL_COL])
    return {"X_raw": X_raw, "y": y}


def prepare_features(preprocessor, X_raw: pd.DataFrame) -> np.ndarray:
    """
    Use saved preprocessor (preprocessor.joblib) to align/scale features.
    Falls back to numeric-only median fill if preprocessor missing.
    """
    if preprocessor is None:
        numeric = X_raw.select_dtypes(include=[np.number]).fillna(X_raw.median())
        return numeric.values
    # The preprocessor expects a DataFrame; it will align to training columns
    return preprocessor.prepare_real_time_features(X_raw.copy())


def plot_confusion_matrix(cm: np.ndarray, title: str) -> None:
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix - {title}")
    plt.tight_layout()
    plt.show()


def plot_roc(y_true: np.ndarray, scores: np.ndarray, title: str) -> None:
    fpr, tpr, _ = roc_curve(y_true, scores)
    auc_val = roc_auc_score(y_true, scores) if len(np.unique(y_true)) > 1 else float("nan")
    plt.figure(figsize=(4, 3))
    plt.plot(fpr, tpr, label=f"AUC = {auc_val:.3f}")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.6)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {title}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()


def evaluate_classifier(name: str, model, X: np.ndarray, y: np.ndarray) -> None:
    y_pred = model.predict(X)
    # Some models may not have predict_proba
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X)
    else:
        scores = y_pred

    cm = confusion_matrix(y, y_pred)
    f1 = f1_score(y, y_pred)
    auc = roc_auc_score(y, scores) if len(np.unique(y)) > 1 else float("nan")

    print(f"\n===== {name} =====")
    print(classification_report(y, y_pred, digits=3))
    print(f"F1 score: {f1:.3f}")
    print(f"AUC-ROC : {auc:.3f}")

    plot_confusion_matrix(cm, name)
    plot_roc(y, scores, name)


def evaluate_isolation_forest(model, X: np.ndarray, y: np.ndarray) -> None:
    y_pred = model.predict(X)
    y_pred = np.where(y_pred == -1, 1, 0)  # -1 -> anomaly -> attack
    scores = -model.decision_function(X)   # higher means more anomalous

    cm = confusion_matrix(y, y_pred)
    f1 = f1_score(y, y_pred)
    auc = roc_auc_score(y, scores) if len(np.unique(y)) > 1 else float("nan")

    print(f"\n===== Isolation Forest =====")
    print(classification_report(y, y_pred, digits=3))
    print(f"F1 score: {f1:.3f}")
    print(f"AUC-ROC : {auc:.3f}")

    plot_confusion_matrix(cm, "Isolation Forest")
    plot_roc(y, scores, "Isolation Forest")


def main() -> None:
    data = load_data(DATA_PATH)
    X_raw, y = data["X_raw"], data["y"]

    # Load preprocessor if available
    preprocessor_path = os.path.join(MODELS_DIR, "preprocessor.joblib")
    preprocessor = joblib.load(preprocessor_path) if os.path.exists(preprocessor_path) else None

    X = prepare_features(preprocessor, X_raw)

    # Load models
    log_reg = joblib.load(os.path.join(MODELS_DIR, "logistic_regression.joblib"))
    gnb = joblib.load(os.path.join(MODELS_DIR, "gaussian_nb.joblib"))
    iso = joblib.load(os.path.join(MODELS_DIR, "isolation_forest.joblib"))

    # Evaluate sequentially
    evaluate_classifier("Logistic Regression", log_reg, X, y)
    evaluate_classifier("GaussianNB", gnb, X, y)
    evaluate_isolation_forest(iso, X, y)


if __name__ == "__main__":
    main()
