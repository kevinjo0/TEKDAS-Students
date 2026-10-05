"""Latih dan evaluasi model status akademik mahasiswa.

Jalankan setelah data_prep.py:
    python ml_model.py
"""
from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from data_prep import CATEGORICAL_COLUMNS, FEATURE_COLUMNS, ID_COLUMN, NUMERIC_COLUMNS, TARGET

BASE_DIR = Path(__file__).resolve().parent


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), NUMERIC_COLUMNS),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), CATEGORICAL_COLUMNS),
    ])
    classifier = RandomForestClassifier(
        n_estimators=350,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline([("preprocess", preprocessor), ("model", classifier)])


def main() -> None:
    train = pd.read_csv(BASE_DIR / "student_train.csv")
    test = pd.read_csv(BASE_DIR / "student_test.csv")
    pipeline = build_pipeline()
    print("Melatih Random Forest untuk status akademik...")
    pipeline.fit(train[FEATURE_COLUMNS], train[TARGET])

    y_pred = pipeline.predict(test[FEATURE_COLUMNS])
    labels = list(pipeline.classes_)
    metrics = {
        "accuracy": float(accuracy_score(test[TARGET], y_pred)),
        "precision_macro": float(precision_score(test[TARGET], y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(test[TARGET], y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(test[TARGET], y_pred, average="macro", zero_division=0)),
        "classes": labels,
        "confusion_matrix": confusion_matrix(test[TARGET], y_pred, labels=labels).tolist(),
    }
    print("\nLaporan klasifikasi:\n")
    print(classification_report(test[TARGET], y_pred, digits=3, zero_division=0))

    joblib.dump(pipeline, BASE_DIR / "student_outcome_pipeline.pkl")
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    importance = pipeline.named_steps["model"].feature_importances_
    pd.DataFrame({"feature": names, "importance": importance}).sort_values(
        "importance", ascending=False
    ).to_csv(BASE_DIR / "feature_importance.csv", index=False)

    predictions = test[[ID_COLUMN, TARGET]].copy()
    predictions["predicted_status"] = y_pred
    for i, label in enumerate(labels):
        predictions[f"probability_{label}"] = pipeline.predict_proba(test[FEATURE_COLUMNS])[:, i]
    predictions.to_csv(BASE_DIR / "student_predictions.csv", index=False)
    (BASE_DIR / "model_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("Disimpan: student_outcome_pipeline.pkl, feature_importance.csv, student_predictions.csv, model_metrics.json")


if __name__ == "__main__":
    main()
