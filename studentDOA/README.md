# Student DOA: BI + ML + LLM

Dashboard Streamlit untuk mempelajari prediksi status akademik mahasiswa: `Dropout`, `Enrolled`, dan `Graduate`.

## Menjalankan

```powershell
cd studentDOA
pip install -r requirements.txt
python data_prep.py
python ml_model.py
streamlit run app.py
```

Tab LLM bersifat opsional dan membutuhkan Ollama lokal:

```powershell
ollama serve
ollama pull llama3.2
```

Jika Ollama belum tersedia, ketiga tab lain tetap dapat digunakan.

## File keluaran

- `student_ml_dataset.csv`, `student_train.csv`, `student_test.csv`: data siap model dan pembagian train/test;
- `student_outcome_pipeline.pkl`: pipeline preprocessing dan Random Forest;
- `feature_importance.csv`, `student_predictions.csv`, `model_metrics.json`: keluaran evaluasi model.
