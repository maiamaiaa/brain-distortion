import streamlit as st
import torch
import numpy as np
import pickle
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="Deteksi Distorsi Kognitif",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Aplikasi Deteksi Distorsi Kognitif")
st.markdown(
    "Aplikasi ini menggunakan model **IndoBERT** untuk mendeteksi distorsi kognitif pada teks."
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_resources():

    model_path = "maiamaiaa/brain-distortion-indobert"

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path
    )

    model.to(device)
    model.eval()

    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    return tokenizer, model, label_encoder


try:
    tokenizer, model, label_encoder = load_resources()
    st.success("✅ Model berhasil dimuat!")
except Exception as e:
    st.error(f"❌ Gagal memuat model: {e}")
    st.stop()


# =========================
# PREDICTION FUNCTION
# =========================
def predict_mental_health_clean(text):

    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        probs = torch.nn.functional.softmax(
            outputs.logits,
            dim=1
        ).cpu().numpy()[0]

        pred_idx = np.argmax(probs)

    predicted_label = label_encoder.inverse_transform(
        [pred_idx]
    )[0]

    confidence = probs[pred_idx]

    return predicted_label, confidence


# =========================
# UI
# =========================
user_input = st.text_area(
    "Masukkan teks/kalimat pikiran negatif di sini:",
    height=150
)

if st.button("Deteksi Distorsi Kognitif", type="primary"):

    if not user_input.strip():
        st.warning("Silakan masukkan teks terlebih dahulu!")

    else:

        with st.spinner("Menganalisis..."):

            label, confidence = predict_mental_health_clean(
                user_input
            )

        st.markdown("### Hasil Analisis")

        st.error(
            f"**Terdeteksi Distorsi Kognitif:** {label}"
        )

        st.info(
            f"**Confidence Score:** {confidence * 100:.2f}%"
        )
