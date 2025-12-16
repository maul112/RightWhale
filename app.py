import streamlit as st
import soundfile as sf
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import librosa
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --- 1. FUNGSI TRANSFORMASI & PREPROCESSING (JANGAN DIUBAH) ---
def transform_dataframe_to_fft(df, fs=2000):
    # Pisahkan Fitur dan Label (jika ada)
    if 'class' in df.columns:
        X = df.drop('class', axis=1).values
        has_label = True
    else:
        X = df.values
        has_label = False

    X = X.astype(float)
    N = X.shape[1] 

    # FFT Vectorized
    fft_complex = np.fft.rfft(X, axis=1)
    fft_magnitude = 2.0 / N * np.abs(fft_complex)

    freqs = np.fft.rfftfreq(N, d=1/fs)
    new_columns = [f"freq_{f:.2f}Hz" for f in freqs]

    df_fft = pd.DataFrame(fft_magnitude, columns=new_columns)

    if has_label:
        df_fft['class'] = df['class'].values
        
    return df_fft

def batch_eliminate_noise(df_fft, percentile=70):
    feature_cols = [c for c in df_fft.columns if str(c).startswith('freq_')]
    X = df_fft[feature_cols].values
    
    thresholds = np.percentile(X, percentile, axis=1, keepdims=True)
    X_clean = X.copy()
    X_clean[X_clean < thresholds] = 0
    
    df_clean = pd.DataFrame(X_clean, columns=feature_cols)
    
    if 'class' in df_fft.columns:
        df_clean['class'] = df_fft['class'].values
        
    return df_clean

def batch_clean_islands_1d(df_clean):
    feature_cols = [c for c in df_clean.columns if str(c).startswith('freq_')]
    X = df_clean[feature_cols].values
    
    mask = (X > 0).astype(int)
    left_neighbor = np.pad(mask[:, :-1], ((0, 0), (1, 0)), mode='constant')
    right_neighbor = np.pad(mask[:, 1:], ((0, 0), (0, 1)), mode='constant')
    neighbor_count = left_neighbor + right_neighbor
    
    islands = (mask == 1) & (neighbor_count == 0)
    
    X_final = X.copy()
    X_final[islands] = 0
    
    df_final = pd.DataFrame(X_final, columns=feature_cols)
    
    if 'class' in df_clean.columns:
        df_final['class'] = df_clean['class'].values
        
    return df_final

# --- 2. FUNGSI BARU: PROSES WAV MENTAH ---
def process_wav_file(uploaded_file):
    """
    Membaca file audio, resample ke 2000Hz, dan potong/pad jadi 4000 titik.
    """
    # Load menggunakan librosa (otomatis resample ke 2000 Hz)
    # sr=2000 SANGAT PENTING agar cocok dengan training data
    y, sr = librosa.load(uploaded_file, sr=2000)
    
    target_length = 4000 # 2 Detik @ 2000Hz
    
    # Fix Duration
    if len(y) > target_length:
        y = y[:target_length]
    else:
        padding = target_length - len(y)
        y = np.pad(y, (0, padding), mode='constant')
        
    # Ubah ke DataFrame (1 Baris)
    return pd.DataFrame([y])

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="NARW Detection System",
    page_icon="🐋",
    layout="wide"
)

st.title("🐋 Deteksi Suara Paus Sikat (NARW)")
st.markdown("""
Aplikasi ini menerima input file audio **.WAV**, memprosesnya melalui FFT dan pembersihan noise, 
lalu menggunakan **Random Forest** untuk mendeteksi keberadaan *North Atlantic Right Whale*.
""")
st.write("Link Dataset dan Laporan (akses menggunakan akun kampus):")
st.write("https://drive.google.com/drive/folders/1RxI7vsFrz34ioO1mdWbubDDio6Ym7RxU?usp=sharing")

# --- SIDEBAR ---
st.sidebar.header("Panel Kontrol")
st.sidebar.info("Silakan upload file audio (.wav). Sistem akan otomatis melakukan resampling ke 2000Hz.")

# PERBAIKAN: Ubah type jadi wav
uploaded_file = st.sidebar.file_uploader("Upload Audio (WAV)", type=["wav"])

# --- LOAD MODEL ---
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('rf_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except FileNotFoundError:
        st.error("File model.pkl atau scaler.pkl tidak ditemukan!")
        return None, None

model, scaler = load_assets()

# --- LOGIKA UTAMA ---
if uploaded_file is not None and model is not None:
    
    # 1. BACA & PRE-PROCESS WAV KE RAW DATAFRAME
    try:
        with st.spinner('Membaca dan menstandarisasi audio (2000Hz)...'):
            # Langkah ini mengubah WAV -> Array 4000 kolom
            df_raw = process_wav_file(uploaded_file)
            
        st.write(f"📂 Audio berhasil dimuat: **{uploaded_file.name}**")
        
        audio_2k = df_raw.iloc[0].values
        audio_44k = librosa.resample(audio_2k, orig_sr=2000, target_sr=44100)
        audio_44k = np.clip(audio_44k, -1.0, 1.0) 
        virtual_file = io.BytesIO()
        sf.write(virtual_file, audio_44k, 44100, format='WAV', subtype='PCM_16')
        virtual_file.seek(0)
        st.caption("🔊 Audio Input (Resampled for Playback):")
        st.audio(virtual_file, format='audio/wav')

    except Exception as e:
        st.error(f"Gagal memproses file audio: {e}")
        st.stop()

    # 2. PROSES PREDIKSI (PIPELINE LENGKAP)
    if st.button("Jalankan Analisis & Prediksi", type="primary"):
        with st.spinner('Sedang melakukan Transformasi FFT & Denoising...'):
            
            # --- PIPELINE DATA (URUTAN SANGAT PENTING) ---
            
            # A. Transformasi FFT
            # df_raw (Time Domain) -> df_fft (Frequency Domain)
            df_fft = transform_dataframe_to_fft(df_raw, fs=2000)
            
            # B. Eliminasi Noise (Thresholding)
            df_clean = batch_eliminate_noise(df_fft, percentile=70) # Sesuaikan percentile training
            
            # C. Hapus Island (Speckle Noise)
            df_final_features = batch_clean_islands_1d(df_clean)
            
            # D. Scaling (StandardScaler)
            # Pastikan nama kolom df_final_features sesuai dengan scaler
            try:
                X_scaled = scaler.transform(df_final_features)
            except Exception as e:
                st.error(f"Error Scaling: {e}. Kemungkinan jumlah fitur FFT beda.")
                st.stop()

            # E. Prediksi Model
            y_pred = model.predict(X_scaled)
            y_prob = model.predict_proba(X_scaled)[:, 1] # Confidence Paus

            # 3. TAMPILKAN HASIL
            hasil_text = '🐋 PAUS TERDETEKSI' if y_pred[0] == 1 else '🔊 HANYA NOISE'
            confidence = y_prob[0] * 100

            st.divider()
            
            # Tampilan Besar
            col1, col2 = st.columns([2, 1])
            
            with col1:
                if y_pred[0] == 1:
                    st.success(f"### Hasil: {hasil_text}")
                else:
                    st.warning(f"### Hasil: {hasil_text}")
            
            with col2:
                st.metric("Tingkat Kepercayaan (Confidence)", f"{confidence:.2f}%")

            # --- VISUALISASI SPEKTRUM (OPSIONAL TAPI BAGUS) ---
            st.subheader("🔍 Visualisasi Spektrum Sinyal (Setelah Denoising)")
            
            # Ambil data untuk plot
            freqs_plot = [float(c.replace('freq_', '').replace('Hz', '')) for c in df_final_features.columns]
            magnitudes = df_final_features.iloc[0].values
            
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(freqs_plot, magnitudes, color='blue', linewidth=0.8)
            ax.set_title("Spektrum Frekuensi Audio Input")
            ax.set_xlabel("Frekuensi (Hz)")
            ax.set_ylabel("Magnitudo")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

else:
    # Tampilan awal jika belum upload
    st.info("👋 Menunggu upload file .WAV...")