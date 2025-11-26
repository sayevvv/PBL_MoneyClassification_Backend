# -*- coding: utf-8 -*-
"""
Backend Flask untuk Model Deteksi Uang (Versi 3)
--- SERVER GABUNGAN (SVM & XGBOOST) ---
--- MAMPU MENDETEKSI "BUKAN UANG/NEGATIVE" ---

API Endpoint:
- URL: /predict
- Method: POST
- Input JSON: {
    "image": "...",
    "model": "svm"  (atau "xgboost")
  }
- Output JSON: { "prediction": "50000", "model_used": "svm" }
"""

import os
import cv2
import numpy as np
import joblib
import base64
import io
from flask import Flask, request, jsonify, render_template
from skimage.feature import local_binary_pattern # Untuk LBP
from PIL import Image
# (Tidak perlu 'import xgboost', karena joblib menanganinya)

# -----------------------------------------------------
# 1. INISIALISASI APLIKASI
# -----------------------------------------------------
app = Flask(__name__)


# -----------------------------------------------------
# 2. KONFIGURASI DAN MUAT MODEL (DIPERBARUI)
# -----------------------------------------------------
# Variabel ini HARUS SAMA PERSIS dengan skrip preprocessing V2
IMG_SIZE = (250, 250)
H_BINS = 180
S_BINS = 256
LBP_POINTS = 24
LBP_RADIUS = 8
LBP_BINS = int(LBP_POINTS + 2)

# Path model yang telah diperbarui sesuai struktur folder Anda
MODEL_DIR = 'models'

# --- Path Model SVM ---
MODEL_SVM_PATH = os.path.join(MODEL_DIR, 'svm', 'svm_model_v3.joblib')
SCALER_SVM_PATH = os.path.join(MODEL_DIR, 'svm', 'svm_scaler_v3.joblib')

# --- Path Model XGBoost ---
MODEL_XGB_PATH = os.path.join(MODEL_DIR, 'xgboost', 'xgb_model_v3.joblib')

# --- Path Encoder (Dipakai Bersama) ---
LE_PATH = os.path.join(MODEL_DIR, 'label_encoder_v3.joblib')

# Muat SEMUA model, scaler, dan label encoder saat startup
try:
    model_svm = joblib.load(MODEL_SVM_PATH)
    scaler_svm = joblib.load(SCALER_SVM_PATH)
    print("Model SVM dan Scaler berhasil dimuat.")

    model_xgb = joblib.load(MODEL_XGB_PATH)
    print("Model XGBoost berhasil dimuat.")

    le = joblib.load(LE_PATH)
    print("Label Encoder berhasil dimuat.")

    print(f"\nKelas yang diketahui: {list(le.classes_)}")

except FileNotFoundError as e:
    print(f"Error: File model tidak ditemukan. {e}")
    print("Pastikan file .joblib ada di dalam folder 'models/svm' dan 'models/xgboost'")
    # Inisialisasi semua sebagai None jika ada kegagalan
    model_svm, scaler_svm, model_xgb, le = None, None, None, None

# -----------------------------------------------------
# 3. FUNGSI HELPER
# -----------------------------------------------------
def get_color_name(hue_degree):
    """Helper untuk mengkonversi hue degree ke nama warna"""
    if 0 <= hue_degree < 15 or 345 <= hue_degree <= 360:
        return "Merah"
    elif 15 <= hue_degree < 45:
        return "Oranye"
    elif 45 <= hue_degree < 75:
        return "Kuning"
    elif 75 <= hue_degree < 150:
        return "Hijau"
    elif 150 <= hue_degree < 210:
        return "Biru"
    elif 210 <= hue_degree < 270:
        return "Ungu"
    else:
        return "Merah Muda"

# -----------------------------------------------------
# 4. FUNGSI EKSTRAKSI FITUR V2 (TIDAK BERUBAH)
# -----------------------------------------------------
def preprocess_and_extract_features_v2(image_array):
    """
    Fungsi ini mereplika 'extract_features_v2' dari Colab,
    tapi menggunakan image array (bukan path) sebagai input.
    """
    try:
        # 'image_array' diharapkan dalam format BGR dari OpenCV

        # --- PREPROCESSING GAMBAR ---
        image = cv2.resize(image_array, IMG_SIZE)
        image = cv2.GaussianBlur(image, (5, 5), 0)

        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # --- FITUR 1: WARNA (HSV HISTOGRAM) ---
        h_hist = cv2.calcHist([hsv_image], [0], None, [H_BINS], [0, 180])
        s_hist = cv2.calcHist([hsv_image], [1], None, [S_BINS], [0, 256])

        h_hist = cv2.normalize(h_hist, None, 0, 1, cv2.NORM_MINMAX)
        s_hist = cv2.normalize(s_hist, None, 0, 1, cv2.NORM_MINMAX)

        color_features = np.concatenate((h_hist, s_hist)).flatten()

        # --- FITUR 2: TEKSTUR (LBP) ---
        lbp = local_binary_pattern(gray_image, LBP_POINTS, LBP_RADIUS, method='uniform')
        (texture_features, _) = np.histogram(lbp.ravel(),
                                             bins=LBP_BINS,
                                             range=(0, LBP_BINS))

        texture_features = cv2.normalize(texture_features, None, 0, 1, cv2.NORM_MINMAX)
        texture_features = texture_features.flatten()

        # --- FITUR 3: BENTUK (HU MOMENTS) ---
        _, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        moments = cv2.moments(thresh)
        hu_moments = cv2.HuMoments(moments)

        shape_features = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-7)
        shape_features = shape_features.flatten()

        # --- GABUNGKAN SEMUA FITUR ---
        final_feature_vector = np.concatenate((color_features, texture_features, shape_features))

        return final_feature_vector

    except Exception as e:
        print(f"Error saat ekstraksi fitur: {e}")
        return None

# -----------------------------------------------------
# 5. ROUTE API (DIPERBARUI)
# -----------------------------------------------------
@app.route("/")
def index():
    # Halaman HTML sederhana untuk info
    return "<h1>Backend Model Gabungan (SVM & XGBoost) Aktif!</h1><p>Gunakan endpoint /predict untuk mengirim gambar.</p>"

@app.route("/predict", methods=["POST"])
def predict():
    # Cek apakah semua model sudah di-load
    if not all([model_svm, scaler_svm, model_xgb, le]):
        return jsonify({"error": "Satu atau lebih file model (SVM/XGB/Scaler/Encoder) tidak berhasil dimuat."}), 500

    try:
        # 1. Dapatkan data JSON dari request
        data = request.get_json()
        if 'image' not in data:
            return jsonify({"error": "Request JSON tidak memiliki key 'image'"}), 400

        # (BARU) Dapatkan model yang dipilih, default ke 'svm' jika tidak ada
        model_choice = data.get('model', 'svm').lower()

        # 2. Decode gambar Base64
        # Hapus header jika ada (misal: "data:image/jpeg;base64,")
        image_b64_string = data['image'].split(',')[-1]
        image_data = base64.b64decode(image_b64_string)

        # Gunakan PIL untuk membuka gambar dari string bytes
        pil_image = Image.open(io.BytesIO(image_data))

        # Konversi ke format OpenCV (BGR)
        # PIL (RGB) -> NumPy Array (RGB) -> OpenCV (BGR)
        cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        # 3. Ekstraksi Fitur V2 (Fungsi yang sama untuk kedua model)
        features = preprocess_and_extract_features_v2(cv_image)

        if features is None:
            return jsonify({"error": "Gagal memproses gambar."}), 500

        # Fitur harus dalam 2D array (1 baris) untuk scaler/model
        features_2d = features.reshape(1, -1)

        # --- ANALISIS KARAKTERISTIK GAMBAR INI ---
        color_size = H_BINS + S_BINS
        texture_size = LBP_BINS

        color_features = features[:color_size]
        texture_features = features[color_size:color_size + texture_size]
        shape_features = features[color_size + texture_size:]

        # Cari karakteristik DOMINAN dari gambar ini
        dominant_hue_idx = np.argmax(color_features[:H_BINS])
        dominant_saturation_idx = np.argmax(color_features[H_BINS:])
        dominant_lbp_pattern = np.argmax(texture_features)

        hue_degree = int(dominant_hue_idx * 2)
        saturation_strength = float(color_features[H_BINS + dominant_saturation_idx])
        texture_std = float(np.std(texture_features))
        texture_max = float(np.max(texture_features))

        image_characteristics = {
            "dominant_color": {
                "hue_degree": hue_degree,
                "color_name": get_color_name(hue_degree),
                "saturation_level": int(dominant_saturation_idx),
                "color_strength": "kuat" if saturation_strength > 0.5 else "lemah",
                "saturation_value": round(saturation_strength, 4)
            },
            "texture_pattern": {
                "dominant_lbp_pattern": int(dominant_lbp_pattern),
                "texture_complexity": "tinggi" if texture_std > 0.1 else "rendah",
                "uniformity": round(texture_max, 4),
                "variation": round(texture_std, 4)
            },
            "shape_info": {
                "hu_moments": [round(float(val), 6) for val in shape_features],
                "symmetry": "simetris" if abs(shape_features[0]) < 0.5 else "asimetris",
                "elongation": round(float(shape_features[1]), 6)
            }
        }

        # --- (BARU) LOGIKA PEMILIHAN MODEL ---

        if model_choice == 'svm':
            # 4. Scaling Fitur (WAJIB untuk SVM)
            features_scaled = scaler_svm.transform(features_2d)

            # 5. Lakukan Prediksi SVM
            prediction_int = model_svm.predict(features_scaled)
            model_used = "SVM"

        elif model_choice == 'xgboost':
            # 4. (Scaling TIDAK DIPERLUKAN untuk XGBoost)

            # 5. Lakukan Prediksi XGBoost
            prediction_int = model_xgb.predict(features_2d)
            model_used = "XGBoost"

        else:
            return jsonify({"error": f"Model '{model_choice}' tidak dikenali. Pilih 'svm' atau 'xgboost'."}), 400

        # 6. Ubah prediksi (angka) kembali ke label (string)
        prediction_label = le.inverse_transform(prediction_int)

        # 7. Buat penjelasan berdasarkan karakteristik gambar
        explanation = f"Gambar ini memiliki warna dominan {image_characteristics['dominant_color']['color_name']} ({hue_degree}°) dengan intensitas {image_characteristics['dominant_color']['color_strength']}, tekstur dengan kompleksitas {image_characteristics['texture_pattern']['texture_complexity']}, dan bentuk yang {image_characteristics['shape_info']['symmetry']}."

        # 8. Kirim Respons dengan karakteristik gambar
        return jsonify({
            "prediction": prediction_label[0],
            "model_version": "v3 (Warna+Tekstur+Bentuk + Deteksi Negative)",
            "model_used": model_used,
            "image_characteristics": image_characteristics,
            "explanation": explanation,
            "total_features_extracted": int(len(features))
        })

    except Exception as e:
        print(f"[ERROR] di endpoint /predict: {e}")
        return jsonify({"error": f"Terjadi kesalahan internal: {str(e)}"}), 500

# -----------------------------------------------------
# 6. JALANKAN APLIKASI
# -----------------------------------------------------
if __name__ == "__main__":
    # host='0.0.0.0' membuat server bisa diakses dari jaringan lokal (HP Anda)
    # debug=False untuk testing (gunakan True saat development)
    app.run(host='0.0.0.0', port=5000, debug=False)