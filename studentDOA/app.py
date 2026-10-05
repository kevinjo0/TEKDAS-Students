"""Dashboard Streamlit: data mahasiswa -> BI -> ML -> LLM.

Jalankan dari folder studentDOA:
    streamlit run app.py
"""
from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st

from data_prep import FEATURE_COLUMNS, TARGET
from vai_analyst import DEFAULT_MODEL, build_prompt, generate_student_strategy

BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(page_title="Student DOA: BI + ML + LLM", layout="wide")
st.title("Student DOA — dari Data ke ML dan LLM")
st.caption("Dashboard pembelajaran untuk menganalisis status akademik: Dropout, Enrolled, atau Graduate.")


@st.cache_data
def load_data() -> pd.DataFrame:
    path = BASE_DIR / "student_ml_dataset.csv"
    if not path.exists():
        raise FileNotFoundError("Jalankan `python data_prep.py` terlebih dahulu.")
    return pd.read_csv(path)


@st.cache_resource
def load_model():
    path = BASE_DIR / "student_outcome_pipeline.pkl"
    return joblib.load(path) if path.exists() else None


@st.cache_data
def load_optional_outputs():
    fi_path = BASE_DIR / "feature_importance.csv"
    metric_path = BASE_DIR / "model_metrics.json"
    importance = pd.read_csv(fi_path) if fi_path.exists() else pd.DataFrame()
    metrics = json.loads(metric_path.read_text(encoding="utf-8")) if metric_path.exists() else {}
    return importance, metrics


try:
    students = load_data()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

model = load_model()
importance, metrics = load_optional_outputs()

total = len(students)
dropout_rate = students[TARGET].eq("Dropout").mean()
graduate_rate = students[TARGET].eq("Graduate").mean()
average_age = students["Age at enrollment"].mean()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Mahasiswa", f"{total:,}")
c2.metric("Dropout pada data", f"{dropout_rate:.1%}")
c3.metric("Graduate pada data", f"{graduate_rate:.1%}")
c4.metric("Rata-rata usia masuk", f"{average_age:.1f}")

labels = students["student_id"].tolist()
selected_id = st.sidebar.selectbox("Pilih ID mahasiswa", labels)
selected = students.loc[students["student_id"].eq(selected_id)].iloc[0]

tab_data, tab_bi, tab_ml, tab_llm = st.tabs([
    "1 · Dataset", "2 · Business Intelligence", "3 · Machine Learning", "4 · LLM Analyst"
])

with tab_data:
    st.subheader("Dataset Student DOA")
    st.markdown(
        "Satu baris mewakili satu mahasiswa. Target `Target` terdiri dari `Dropout`, "
        "`Enrolled`, dan `Graduate`. Kode pada sebagian besar kolom kategorikal dipertahankan "
        "apa adanya dan diubah menjadi one-hot encoding hanya di dalam pipeline model."
    )
    st.write(f"Ukuran: {students.shape[0]:,} baris × {students.shape[1]} kolom")
    st.dataframe(students.head(100), width="stretch")
    st.subheader("Profil mahasiswa terpilih")
    st.dataframe(pd.DataFrame([selected.to_dict()]), width="stretch")

with tab_bi:
    st.subheader("Descriptive BI sebelum predictive ML")
    distribution = students[TARGET].value_counts().rename_axis("Status").to_frame("Jumlah")
    st.markdown("**Distribusi status akademik**")
    st.bar_chart(distribution)

    course_status = pd.crosstab(students["Course"].astype(str), students[TARGET], normalize="index")
    if "Dropout" in course_status:
        st.markdown("**Proporsi dropout menurut kode program studi**")
        st.bar_chart(course_status[["Dropout"]].sort_values("Dropout", ascending=False))

    grade_by_status = students.groupby(TARGET)[[
        "Curricular units 1st sem (grade)", "Curricular units 2nd sem (grade)"
    ]].mean()
    st.markdown("**Rata-rata nilai per semester menurut status**")
    st.bar_chart(grade_by_status)
    st.caption("Catatan: nilai dan kode program studi bersifat deskriptif; grafik tidak menunjukkan hubungan sebab-akibat.")

with tab_ml:
    st.subheader("Prediksi status akademik")
    st.dataframe(pd.DataFrame([selected.to_dict()]), width="stretch")
    if model is None:
        st.error("Model belum tersedia. Jalankan `python data_prep.py` lalu `python ml_model.py`.")
    else:
        probability_values = model.predict_proba(pd.DataFrame([selected[FEATURE_COLUMNS]]))[0]
        probabilities = dict(zip(model.classes_, probability_values))
        predicted = max(probabilities, key=probabilities.get)
        a, b, c = st.columns(3)
        a.metric("Prediksi model", predicted)
        b.metric("Probabilitas tertinggi", f"{probabilities[predicted]:.1%}")
        c.metric("Label pada dataset", selected[TARGET])
        st.bar_chart(pd.DataFrame.from_dict(probabilities, orient="index", columns=["Probabilitas"]))
        st.info("Prediksi menunjukkan pola yang dipelajari dari data historis, bukan kepastian maupun penilaian atas kemampuan mahasiswa.")

        if metrics:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
            m2.metric("Precision macro", f"{metrics['precision_macro']:.3f}")
            m3.metric("Recall macro", f"{metrics['recall_macro']:.3f}")
            m4.metric("F1 macro", f"{metrics['f1_macro']:.3f}")
        if not importance.empty:
            st.markdown("**15 fitur global terpenting menurut model**")
            st.bar_chart(importance.head(15).set_index("feature")["importance"])

with tab_llm:
    st.subheader("LLM sebagai analyst — bukan pengganti model")
    st.caption(f"Ollama model: {DEFAULT_MODEL} (ubah melalui environment variable OLLAMA_MODEL)")
    if model is None:
        st.error("Latih model terlebih dahulu agar LLM menerima probabilitas ML yang nyata.")
    else:
        probability_values = model.predict_proba(pd.DataFrame([selected[FEATURE_COLUMNS]]))[0]
        probabilities = dict(zip(model.classes_, probability_values))
        model_factors = importance.head(8).to_dict("records") if not importance.empty else []
        prompt = build_prompt(selected.to_dict(), probabilities, model_factors)
        with st.expander("Lihat prompt yang dikirim ke LLM"):
            st.code(prompt, language="text")
        if st.button("Buat rekomendasi dukungan berbasis bukti"):
            with st.spinner("LLM menyusun interpretasi dari profil dan prediksi model..."):
                answer = generate_student_strategy(selected.to_dict(), probabilities, model_factors)
            st.markdown(answer)

st.markdown("---")
st.caption("Prinsip pembelajaran: data menggambarkan pola; ML memprediksi target; LLM mengomunikasikan bukti dan batasannya.")
