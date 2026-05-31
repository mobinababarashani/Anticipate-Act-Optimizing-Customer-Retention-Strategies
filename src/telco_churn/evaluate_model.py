from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
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


def _save_pdf_report(report: str, metrics: dict[str, float], threshold: float) -> None:
    figure_paths = [
        FIGURES_DIR / "roc_curve.png",
        FIGURES_DIR / "precision_recall_curve.png",
        FIGURES_DIR / "confusion_matrix.png",
        FIGURES_DIR / "feature_importance.png",
    ]

    with PdfPages(REPORTS_DIR / "churn_report.pdf") as pdf:
        fig = plt.figure(figsize=(8.5, 11))
        fig.text(0.08, 0.94, "Customer Retention Intelligence", fontsize=18, weight="bold")
        fig.text(0.08, 0.90, "A churn model packaged as a retention prioritization workflow.", fontsize=11)
        fig.text(0.08, 0.86, f"Decision threshold: {threshold:.2f}", fontsize=11)

        metric_lines = [
            f"ROC-AUC: {metrics['roc_auc']:.4f}",
            f"Precision: {metrics['precision']:.4f}",
            f"Recall: {metrics['recall']:.4f}",
            f"F1-score: {metrics['f1']:.4f}",
            f"True positives: {metrics['true_positives']:.0f}",
            f"False positives: {metrics['false_positives']:.0f}",
            f"False negatives: {metrics['false_negatives']:.0f}",
            f"True negatives: {metrics['true_negatives']:.0f}",
        ]
        fig.text(0.08, 0.76, "\n".join(metric_lines), fontsize=11, linespacing=1.6)
        fig.text(0.08, 0.50, "Classification Report", fontsize=13, weight="bold")
        fig.text(0.08, 0.25, report, fontsize=9, family="monospace", linespacing=1.2)
        fig.text(
            0.08,
            0.10,
            "Business reading: use probability ranking to prioritize retention outreach under campaign budget constraints.",
            fontsize=10,
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        for path in figure_paths:
            if not path.exists():
                continue
            image = plt.imread(path)
            fig, ax = plt.subplots(figsize=(8.5, 6))
            ax.imshow(image)
            ax.axis("off")
            ax.set_title(path.stem.replace("_", " ").title())
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def evaluate(threshold: float = 0.5) -> dict[str, float]:
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1.")

    ensure_project_dirs()
    X_train, X_test, _, y_test = load_processed_data()
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
            pd.DataFrame({"feature": X_train.columns, "importance": model.feature_importances_})
            .sort_values("importance", ascending=False)
            .head(15)
        )
        importance.to_csv(REPORTS_DIR / "feature_importance.csv", index=False)
        ax = importance.sort_values("importance").plot.barh(
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

    report = classification_report(y_test, predictions, digits=3, zero_division=0)
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
    report_text = f"""# Telco Churn Model Evaluation

## Executive Summary

This project predicts customer churn and translates the predictions into retention priorities. The final model is useful for ranking customers by churn risk before a campaign.

The most important business reading is not "the model is correct or wrong." The useful reading is: "which customers deserve attention first, and what type of intervention should they receive?"

## Model Snapshot

| Metric | Value |
| --- | ---: |
| ROC-AUC | {roc_auc:.4f} |
| Precision | {precision:.4f} |
| Recall | {recall:.4f} |
| F1-score | {f1:.4f} |
| True positives | {tp} |
| False positives | {fp} |
| False negatives | {fn} |
| True negatives | {tn} |

## Threshold

Current decision threshold: `{threshold:.2f}`

This threshold is a starting point. In a real retention workflow, I would tune it based on campaign budget, expected saved revenue, customer lifetime value, and the cost of each retention offer.

## Classification Report

```text
{report}
```

## Retention Playbook

| Segment | Churn probability | Action |
| --- | ---: | --- |
| High risk | `>= 0.70` | Immediate outreach, contract incentive, support bundle |
| Medium risk | `0.40 - 0.69` | Nurture campaign, service review, controlled offer test |
| Low risk | `< 0.40` | Monitor; avoid unnecessary campaign spend |

## Why This Matters

False negatives are expensive because they are customers the company fails to save. False positives are also costly because the business may spend money on customers who were not going to churn. The practical solution is to use probability ranking and campaign capacity, not a hard model label alone.

## Interview-Ready Takeaway

This project shows the full data science loop: framing a business problem, building a reproducible ML pipeline, evaluating trade-offs, creating customer-level scores, and packaging the result in a demo app that a non-technical stakeholder can use.
"""
    (REPORTS_DIR / "churn_report.md").write_text(report_text, encoding="utf-8")
    _save_pdf_report(report=report, metrics=metrics, threshold=threshold)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the final churn model.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for churn.")
    args = parser.parse_args()

    metrics = evaluate(threshold=args.threshold)
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    print("Saved report and figures under reports/.")


if __name__ == "__main__":
    main()
