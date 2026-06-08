import streamlit as st
import torch
import numpy as np
import pickle
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

st.set_page_config(page_title="Deteksi Distorsi Kognitif", page_icon="🧠", layout="centered")
st.title("🧠 Aplikasi Deteksi Distorsi Kognitif")
st.markdown("Aplikasi ini menggunakan model **IndoBERT** untuk mendeteksi distorsi kognitif pada teks.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_resources():
    base_path = "saved_model"
    model_path = base_path
    
    # PERBAIKAN OTOMATIS: Jika di dalam folder saved_model ada folder lagi, kode ini akan otomatis masuk ke dalamnya
    if os.path.exists(base_path) and os.path.isdir(base_path):
        subdirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
        if subdirs and not os.path.exists(os.path.join(base_path, "config.json")):
            model_path = os.path.join(base_path, subdirs[0])
            
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    
    # Memuat label encoder
    try:
        with open('label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
    except:
        label_encoder = None
        
    return tokenizer, model, label_encoder

try:
    tokenizer, model, label_encoder = load_resources()
    st.success("Model, Tokenizer, dan Label Encoder berhasil dimuat!")
except Exception as e:
    st.error(f"Gagal memuat model. Pastikan folder 'saved_model' sudah di-upload ke GitHub. Error: {e}")

def predict_mental_health_clean(text, label_encoder):
    model.eval()
    encoding = tokenizer(
        text, add_special_tokens=True, max_length=128, padding='max_length',
        truncation=True, return_attention_mask=True, return_tensors='pt',
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1).cpu().numpy()[0]
        pred_label_idx = np.argmax(probs)
    
    if label_encoder is not None:
        predicted_label = label_encoder.inverse_transform([pred_label_idx])[0]
    else:
        predicted_label = f"Class Index: {pred_label_idx}"
    return predicted_label, probs[pred_label_idx]

user_input = st.text_area("Masukkan teks/kalimat pikiran negatif di sini:", height=150)

if st.button("Deteksi Distorsi Kognitif", type="primary"):
    if user_input.strip() == "":
        st.warning("Silakan masukkan teks terlebih dahulu!")
    else:
        with st.spinner("Menganalisis..."):
            label, confidence = predict_mental_health_clean(user_input, label_encoder)
        st.markdown("### **Hasil Analisis:**")
        st.error(f"**Terdeteksi Distorsi Kognitif:** {label}")
        st.info(f"**Confidence Score:** {confidence * 100:.2f}%")
