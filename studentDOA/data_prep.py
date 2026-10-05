"""Siapkan dataset Student DOA untuk klasifikasi status akademik.

Jalankan:
    python data_prep.py
"""
from pathlib import Path
import json

import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
SOURCE_FILE = BASE_DIR / "dataset_studentDOA.csv"
TARGET = "Target"
ID_COLUMN = "student_id"
RANDOM_STATE = 42

# Kolom kode yang mewakili kategori, bukan urutan/ukuran numerik.
CATEGORICAL_COLUMNS = [
    "Marital status", "Application mode", "Application order", "Course",
    "Daytime/evening attendance", "Previous qualification", "Nacionality",
    "Mother's qualification", "Father's qualification", "Mother's occupation",
    "Father's occupation", "Displaced", "Educational special needs", "Debtor",
    "Tuition fees up to date", "Gender", "Scholarship holder", "International",
]
NUMERIC_COLUMNS = [
    "Age at enrollment", "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (enrolled)", "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)", "Curricular units 1st sem (grade)",
    "Curricular units 1st sem (without evaluations)",
    "Curricular units 2nd sem (credited)",
    "Curricular units 2nd sem (enrolled)", "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (approved)", "Curricular units 2nd sem (grade)",
    "Curricular units 2nd sem (without evaluations)", "Unemployment rate",
    "Inflation rate", "GDP",
]
FEATURE_COLUMNS = CATEGORICAL_COLUMNS + NUMERIC_COLUMNS


def load_and_validate() -> pd.DataFrame:
    """Muat data sumber dan pastikan kontrak kolomnya benar."""
    if not SOURCE_FILE.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {SOURCE_FILE}")
    df = pd.read_csv(SOURCE_FILE)
    required = set(FEATURE_COLUMNS + [TARGET])
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Kolom wajib tidak ditemukan: {missing}")
    if df.empty:
        raise ValueError("Dataset kosong.")
    if df[TARGET].nunique(dropna=True) < 2:
        raise ValueError("Target harus memiliki minimal dua kelas.")
    return df


def prepare_student_features(df: pd.DataFrame) -> pd.DataFrame:
    """Pilih fitur, buang duplikat, dan buat ID baris yang stabil untuk dashboard."""
    prepared = df[FEATURE_COLUMNS + [TARGET]].drop_duplicates().dropna(subset=[TARGET]).copy()
    # ID hanya untuk menelusuri baris di dashboard; tidak menjadi fitur model.
    prepared.insert(0, ID_COLUMN, [f"STU-{i:05d}" for i in range(1, len(prepared) + 1)])
    return prepared


def main() -> None:
    source = load_and_validate()
    dataset = prepare_student_features(source)
    train, test = train_test_split(
        dataset,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=dataset[TARGET],
    )

    dataset.to_csv(BASE_DIR / "student_ml_dataset.csv", index=False)
    train.to_csv(BASE_DIR / "student_train.csv", index=False)
    test.to_csv(BASE_DIR / "student_test.csv", index=False)

    report = {
        "source_rows": int(len(source)),
        "model_rows": int(len(dataset)),
        "duplicate_rows_removed": int(len(source) - len(dataset)),
        "feature_count_before_one_hot": len(FEATURE_COLUMNS),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "target_distribution": dataset[TARGET].value_counts().to_dict(),
        "missing_by_column": dataset[FEATURE_COLUMNS].isna().sum().to_dict(),
    }
    (BASE_DIR / "data_quality_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"Selesai: {len(train):,} train dan {len(test):,} test")
    print("Dibuat: student_ml_dataset.csv, student_train.csv, student_test.csv")


if __name__ == "__main__":
    main()
