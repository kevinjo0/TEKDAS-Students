"""Lapisan LLM yang menjelaskan hasil model secara terbatas pada bukti data."""
import os
from typing import Any

import ollama

DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")


def build_prompt(student_profile: dict[str, Any], probabilities: dict[str, float], model_factors: list[dict]) -> str:
    profile_keys = [
        "student_id", "Age at enrollment", "Course", "Gender", "Debtor",
        "Tuition fees up to date", "Scholarship holder",
        "Curricular units 1st sem (approved)", "Curricular units 1st sem (grade)",
        "Curricular units 2nd sem (approved)", "Curricular units 2nd sem (grade)",
    ]
    profile = {key: student_profile.get(key) for key in profile_keys if key in student_profile}
    formatted_probability = {label: f"{value:.1%}" for label, value in probabilities.items()}
    return f"""
Anda adalah asisten analisis pendidikan. Gunakan HANYA bukti data berikut.
Nilai kategori berupa kode numerik; jangan menebak arti spesifik dari kode tersebut.
Probabilitas model adalah prediksi, bukan kepastian atau penyebab.
Jangan mengklaim kondisi pribadi, kemampuan, motivasi, atau masa depan mahasiswa.

PROFIL MAHASISWA
{profile}

PROBABILITAS MODEL
{formatted_probability}

FAKTOR GLOBAL MODEL
{model_factors}

TUGAS
Tulis dalam bahasa Indonesia untuk staf akademik:
1. Interpretasi singkat status dengan probabilitas tertinggi.
2. Dua observasi berdasarkan nilai profil yang tersedia.
3. Dua tindakan dukungan akademik yang proporsional dan tidak menghakimi.
4. Satu batasan singkat tentang prediksi ini.
Gunakan butir-butir ringkas.
""".strip()


def generate_student_strategy(student_profile: dict[str, Any], probabilities: dict[str, float], model_factors: list[dict], model: str = DEFAULT_MODEL) -> str:
    prompt = build_prompt(student_profile, probabilities, model_factors)
    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.2},
        )
        return response["message"]["content"]
    except Exception as exc:
        return (
            f"Tidak dapat menghubungi Ollama model '{model}'. Pastikan `ollama serve` aktif "
            f"dan model sudah diunduh. Detail: {exc}"
        )
