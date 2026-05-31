from __future__ import annotations

import argparse

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from .utils import MODELS_DIR, ensure_project_dirs, load_processed_data, save_json, save_model


def train_candidates(X_train: pd.DataFrame, y_train: pd.Series) -> dict[str, object]:
    return {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            random_state=42,
        ),
        "xgboost": XGBClassifier(
            n_estimators=200,
            learning_rate=0.03,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="logloss",
        ),
    }


def train_and_select_model() -> tuple[str, object, dict[str, float]]:
    ensure_project_dirs()
    X_train, X_test, y_train, y_test = load_processed_data()

    scores: dict[str, float] = {}
    models = train_candidates(X_train, y_train)

    for name, model in models.items():
        model.fit(X_train, y_train)
        probabilities = model.predict_proba(X_test)[:, 1]
        scores[name] = float(roc_auc_score(y_test, probabilities))

    best_name = max(scores, key=scores.get)
    best_model = models[best_name]

    save_model(best_model, MODELS_DIR / "final_model.pkl")
    save_json(scores, MODELS_DIR / "model_scores.json")
    save_json(list(X_train.columns), MODELS_DIR / "feature_columns.json")

    return best_name, best_model, scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Train churn prediction models.")
    parser.parse_args()

    best_name, _, scores = train_and_select_model()
    print("Model comparison by ROC-AUC:")
    for name, score in scores.items():
        print(f"- {name}: {score:.4f}")
    print(f"Saved final model: {best_name}")


if __name__ == "__main__":
    main()
