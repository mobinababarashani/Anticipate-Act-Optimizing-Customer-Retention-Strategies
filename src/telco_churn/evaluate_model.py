from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

from .utils import FIGURES_DIR, MODELS_DIR, REPORTS_DIR, ensure_project_dirs, load_model, load_processed_data


def evaluate(threshold: float = 0.5) -> dict[str, float]:
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1.")

    ensure_project_dirs()
    _, X_test, _, y_test = load_processed_data()
    model = load_model(MODELS_DIR / "final_model.pkl")

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    roc_auc = roc_auc_score(y_test, probabilities)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        predictions,
        average="binary",
        zero_division=0,
    )
    cm = confusion_matrix(y_test, predictions)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "roc_auc": float(roc_auc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "true_positives": float(tp),
        "false_positives": float(fp),
        "false_negatives": float(fn),
        "true_negatives": float(tn),
    }

    pd.DataFrame([{"threshold": threshold, **metrics}]).to_csv(
        REPORTS_DIR / "evaluation_metrics.csv",
        index=False,
    )
    report = classification_report(
        y_test,
        predictions,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(REPORTS_DIR / "classification_report.csv")
    pd.DataFrame(
        cm,
        index=["Actual Non-Churn", "Actual Churn"],
        columns=["Predicted Non-Churn", "Predicted Churn"],
    ).to_csv(REPORTS_DIR / "confusion_matrix.csv")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    RocCurveDisplay.from_predictions(y_test, probabilities)
    plt.title("ROC Curve")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "roc_curve.png", dpi=150)
    plt.close()

    PrecisionRecallDisplay.from_predictions(y_test, probabilities)
    plt.title("Precision-Recall Curve")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "precision_recall_curve.png", dpi=150)
    plt.close()

    ConfusionMatrixDisplay.from_predictions(y_test, predictions)
    plt.title(f"Confusion Matrix - Threshold {threshold:.2f}")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=150)
    plt.close()

    if hasattr(model, "feature_importances_"):
        importance = (
            pd.DataFrame({"feature": X_test.columns, "importance": model.feature_importances_})
            .sort_values("importance", ascending=False)
        )
        importance.to_csv(REPORTS_DIR / "feature_importance.csv", index=False)
        ax = importance.head(15).sort_values("importance").plot.barh(
            x="feature",
            y="importance",
            legend=False,
            figsize=(8, 6),
        )
        ax.set_title("Top Feature Importances")
        ax.set_xlabel("Importance")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "feature_importance.png", dpi=150)
        plt.close()

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the final churn model.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for churn.")
    args = parser.parse_args()

    metrics = evaluate(threshold=args.threshold)
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    print("Saved evaluation CSV files and PNG figures under reports/.")


if __name__ == "__main__":
    main()
