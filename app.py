import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def transform_dataframe_to_fft(df, fs=2000):
    print("Mulai transformasi FFT...")
    
    # 1. Pisahkan Fitur dan Label
    if 'class' in df.columns:
        X = df.drop('class', axis=1).values
        y = df['class'].values
        has_label = True
    else:
        X = df.values
        has_label = False

    # Pastikan tipe data float
    X = X.astype(float)
    N = X.shape[1] # Harusnya 4000

    # 2. Lakukan FFT Vectorized (Sekaligus untuk semua baris)
    fft_complex = np.fft.rfft(X, axis=1)

    # 3. Hitung Magnitudo (Amplitudo Termodulasi)
    # Normalisasi dengan 2/N agar skala sesuai fisik
    fft_magnitude = 2.0 / N * np.abs(fft_complex)

    # 4. Buat Nama Kolom Baru (Label Frekuensi)
    # Hitung frekuensi untuk setiap bin
    freqs = np.fft.rfftfreq(N, d=1/fs)
    new_columns = [f"freq_{f:.2f}Hz" for f in freqs]

    # 5. Gabungkan kembali ke DataFrame
    df_fft = pd.DataFrame(fft_magnitude, columns=new_columns)

    if has_label:
        df_fft['class'] = y
        
    print(f"Selesai. Dimensi awal: {df.shape} -> Dimensi FFT: {df_fft.shape}")
    return df_fft

def batch_eliminate_noise(df_fft, percentile=70):
    # 1. Pisahkan Fitur dan Label
    feature_cols = [c for c in df_fft.columns if str(c).startswith('freq_')]
    
    # Ambil matriks data (numpy array) untuk kecepatan tinggi
    X = df_fft[feature_cols].values
    
    # 2. Hitung Threshold untuk SETIAP BARIS (axis=1)
    # keepdims=True agar dimensi tetap (N_samples, 1) untuk broadcasting
    thresholds = np.percentile(X, percentile, axis=1, keepdims=True)
    
    # 3. Terapkan Masking (Set nilai < threshold menjadi 0)
    X_clean = X.copy()
    X_clean[X_clean < thresholds] = 0
    
    # 4. Gabungkan kembali ke DataFrame
    df_clean = pd.DataFrame(X_clean, columns=feature_cols)
    
    # Tambahkan kembali kolom 'class' jika ada
    if 'class' in df_fft.columns:
        df_clean['class'] = df_fft['class'].values
        
    # Cek statistik sederhana untuk memastikan nol sudah masuk
    n_zeros = np.count_nonzero(X_clean==0)
    n_total = X_clean.size
    return df_clean

def batch_clean_islands_1d(df_clean):
    # 1. Pisahkan Fitur dan Label
    feature_cols = [c for c in df_clean.columns if str(c).startswith('freq_')]
    X = df_clean[feature_cols].values
    
    # 2. Buat Masker Biner (1 jika ada data, 0 jika kosong)
    # Ini untuk memudahkan pengecekan "apakah tetangga aktif?"
    mask = (X > 0).astype(int)
    
    # 3. Hitung Tetangga (Kiri dan Kanan)
    # Kita geser array ke kiri dan kanan untuk cek tetangga
    # Pad dengan 0 di ujung-ujung
    left_neighbor = np.pad(mask[:, :-1], ((0, 0), (1, 0)), mode='constant')
    right_neighbor = np.pad(mask[:, 1:], ((0, 0), (0, 1)), mode='constant')
    
    # Hitung jumlah tetangga aktif (Maksimal 2: kiri dan kanan)
    neighbor_count = left_neighbor + right_neighbor
    
    # 4. Identifikasi Pulau
    # Definisi Pulau di 1D: Titik Aktif (mask==1) TAPI Tetangga Aktif = 0
    # (Artinya dia berdiri sendiri, kiri-kanan nol)
    islands = (mask == 1) & (neighbor_count == 0)
    
    # 5. Hapus Pulau
    X_final = X.copy()
    X_final[islands] = 0
    
    # 6. Kembalikan ke DataFrame
    df_final = pd.DataFrame(X_final, columns=feature_cols)
    
    if 'class' in df_clean.columns:
        df_final['class'] = df_clean['class'].values
        
    # Statistik
    n_islands = np.count_nonzero(islands)
    
    return df_final

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="NARW Detection System",
    page_icon="🐋",
    layout="wide"
)

# --- JUDUL & HEADER ---
st.title("🐋 Deteksi Suara Paus Sikat (NARW)")
st.markdown("""
Aplikasi ini menggunakan **Random Forest Classifier** dengan input **Data Spektral (FFT)** untuk mendeteksi keberadaan *North Atlantic Right Whale*.
""")

# --- SIDEBAR (UPLOAD & INFO) ---
st.sidebar.header("Panel Kontrol")
st.sidebar.info("Silakan upload file CSV berisi fitur ekstraksi/FFT.")

uploaded_file = st.sidebar.file_uploader("Upload Data Uji (CSV)", type=["csv"])

# --- FUNGSI LOAD MODEL ---
@st.cache_resource
def load_assets():
    try:
        # Ganti nama file sesuai yang Anda save
        model = joblib.load('rf_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except FileNotFoundError:
        st.error("File model.pkl atau scaler.pkl tidak ditemukan!")
        return None, None

model, scaler = load_assets()

# --- LOGIKA UTAMA ---
if uploaded_file is not None and model is not None:
    # 1. BACA DATA
    try:
        df = pd.read_csv(uploaded_file)
        df.loc[df['class'] == 'NoWhale', 'class'] = 0
        df.loc[df['class'] == 'RightWhale', 'class'] = 1
        df['class'] = df['class'].astype(int)
        st.write(f"📂 File berhasil dimuat: **{uploaded_file.name}** ({df.shape[0]} baris)")
        
        # Tampilkan cuplikan data
        with st.expander("Lihat Cuplikan Data Mentah"):
            st.dataframe(df.head())

    except Exception as e:
        st.error(f"Gagal membaca file: {e}")
        st.stop()

    # 2. PROSES PREDIKSI
    if st.button("Jalankan Prediksi", type="primary"):
        with st.spinner('Sedang melakukan analisis...'):
            
            # A. Cek Ground Truth (Apakah ada kolom 'class'?)
            target_col = 'class'
            has_label = target_col in df.columns
            st.info(f"Deteksi Label: {'Ada' if has_label else 'Tidak Ada'}")
            
            # B. Pisahkan Fitur dan Label
            if has_label:
                X_input = df.drop(target_col, axis=1)
                y_true = df[target_col]

                df_fft = transform_dataframe_to_fft(X_input)
                df_fft_clean = batch_eliminate_noise(df_fft, percentile=65)
                X_input = batch_clean_islands_1d(df_fft_clean)
            else:
                X_input = df.copy()
                y_true = None

            # C. Scaling (Wajib!)
            try:
                X_scaled = scaler.transform(X_input)
            except Exception as e:
                st.error(f"Error pada Scaling: {e}. Pastikan jumlah kolom sama dengan data latih!")
                st.stop()

            # D. Prediksi
            y_pred = model.predict(X_scaled)
            y_prob = model.predict_proba(X_scaled)[:, 1] # Ambil probabilitas kelas Paus

            # 3. TAMPILKAN HASIL
            
            # Buat DataFrame Hasil
            res_df = df.copy()
            res_df['Prediksi_Label'] = y_pred
            res_df['Confidence_Paus'] = y_prob
            res_df['Hasil'] = res_df['Prediksi_Label'].map({1: '🐋 PAUS', 0: '🔊 NOISE'})

            # --- METRIK RINGKASAN ---
            col1, col2, col3 = st.columns(3)
            total_paus = np.sum(y_pred == 1)
            total_noise = np.sum(y_pred == 0)
            
            col1.metric("Total Data Diuji", len(df))
            col2.metric("Terdeteksi Paus", f"{total_paus}", delta_color="normal")
            col3.metric("Terdeteksi Noise", f"{total_noise}", delta_color="off")

            st.divider()

            # --- TAMPILAN JIKA ADA KUNCI JAWABAN (EVALUASI) ---
            if has_label:
                st.subheader("📊 Evaluasi Kinerja (Berdasarkan Label Asli)")
                
                acc = accuracy_score(y_true, y_pred)
                st.success(f"Akurasi pada data ini: **{acc*100:.2f}%**")
                
                col_graph1, col_graph2 = st.columns(2)
                
                with col_graph1:
                    st.write("**Confusion Matrix**")
                    cm = confusion_matrix(y_true, y_pred)
                    fig, ax = plt.subplots()
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                    ax.set_xlabel('Prediksi')
                    ax.set_ylabel('Aktual')
                    st.pyplot(fig)

                with col_graph2:
                    st.write("**Classification Report**")
                    report_dict = classification_report(y_true, y_pred, output_dict=True)
                    st.dataframe(pd.DataFrame(report_dict).transpose())

            # --- TABEL HASIL DETAIL ---
            st.subheader("📝 Detail Hasil Prediksi")
            
            # Highlight baris yang Paus
            def highlight_paus(row):
                return ['background-color: #d1ffd6' if row['Hasil'] == '🐋 PAUS' else '' for _ in row]

            st.dataframe(
                res_df[['Hasil', 'Confidence_Paus', 'Prediksi_Label']].style.apply(highlight_paus, axis=1),
                use_container_width=True
            )

            # --- DOWNLOAD BUTTON ---
            csv_result = res_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Hasil Prediksi (CSV)",
                data=csv_result,
                file_name='hasil_prediksi_narw.csv',
                mime='text/csv',
            )

else:
    # Tampilan awal jika belum upload
    st.info("👋 Selamat Datang! Silakan upload file CSV di sidebar sebelah kiri untuk memulai.")
    
    # Ilustrasi Dummy
    st.markdown("---")
    st.markdown("#### Contoh Format Data yang Diharapkan:")
    st.markdown("File CSV harus memiliki kolom fitur frekuensi (FFT) yang sama dengan data training.")