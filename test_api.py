# -*- coding: utf-8 -*-
"""
Skrip Klien untuk Menguji Backend Flask GABUNGAN

Cara Penggunaan:
1. Pastikan server Flask (app.py) Anda SEDANG BERJALAN di terminal lain.
2. Buat folder 'gambar_uji' dan letakkan gambar-gambar uji di dalamnya.
3. (OPSIONAL) Ubah `MODEL_TO_TEST` ke 'svm' atau 'xgboost'.
4. Jalankan skrip ini dari terminal (venv) terpisah:
   python test_api.py
5. Skrip akan otomatis menguji SEMUA gambar di folder 'gambar_uji'
"""

import requests
import base64
import os
from pathlib import Path
import time

# --- UBAH INI ---
# Folder yang berisi gambar-gambar uji
TEST_IMAGES_FOLDER = "gambar_uji"

# --- (BARU) UBAH INI UNTUK MEMILIH MODEL ---
MODEL_TO_TEST = "xgboost"  # Ganti ini menjadi "svm" untuk menguji model lainnya

# Alamat server Flask Anda (pastikan app.py sedang berjalan)
API_URL = "http://127.0.0.1:5000/predict"

# Ekstensi file gambar yang didukung
SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}

def encode_image_to_base64(image_path):
    """Membaca file gambar dan meng-encode-nya ke string Base64."""
    try:
        with open(image_path, "rb") as image_file:
            # Baca data biner
            binary_data = image_file.read()
            # Encode ke Base64 dan ubah ke string utf-8
            base64_string = base64.b64encode(binary_data).decode('utf-8')
            return base64_string
    except FileNotFoundError:
        print(f"Error: File gambar tidak ditemukan di {image_path}")
        print("Pastikan Anda sudah meletakkan file gambar dan nama filenya benar.")
        return None
    except Exception as e:
        print(f"Error saat encode gambar: {e}")
        return None

def get_image_files(folder_path):
    """Mendapatkan semua file gambar dari folder."""
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' tidak ditemukan!")
        print(f"Silakan buat folder '{folder_path}' dan letakkan gambar-gambar uji di dalamnya.")
        return []
    
    image_files = []
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if os.path.isfile(file_path):
            ext = Path(file).suffix.lower()
            if ext in SUPPORTED_EXTENSIONS:
                image_files.append(file_path)
    
    return sorted(image_files)

def test_single_image(image_path, model_name):
    """Menguji satu gambar dengan model yang dipilih."""
    print(f"\n{'='*70}")
    print(f"Menguji: {os.path.basename(image_path)}")
    print(f"{'='*70}")
    
    # 1. Encode gambar
    b64_string = encode_image_to_base64(image_path)
    
    if b64_string is None:
        return {"status": "error", "message": "Gagal encode gambar"}

    # 2. Siapkan data JSON (payload)
    payload = {
        "image": b64_string,
        "model": model_name
    }

    try:
        # 3. Kirim request POST
        start_time = time.time()
        response = requests.post(API_URL, json=payload)
        elapsed_time = time.time() - start_time
        
        # 4. Parse hasil
        print(f"Status Code: {response.status_code}")
        print(f"Waktu Respons: {elapsed_time:.2f} detik")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Prediksi: Rp {result.get('prediction', 'N/A')}")
            print(f"  Model: {result.get('model_used', 'N/A')}")
            print(f"  Versi: {result.get('model_version', 'N/A')}")
            
            # Tampilkan Penjelasan dan Karakteristik
            explanation = result.get('explanation', '')
            if explanation:
                print(f"  Penjelasan: {explanation}")
            
            characteristics = result.get('image_characteristics', {})
            if characteristics:
                print("  Karakteristik:")
                dom_color = characteristics.get('dominant_color', {})
                print(f"    - Warna Dominan: {dom_color.get('color_name', 'N/A')} (Hue: {dom_color.get('hue_degree', 'N/A')}°)")
                
                tex_pattern = characteristics.get('texture_pattern', {})
                print(f"    - Tekstur: Kompleksitas {tex_pattern.get('texture_complexity', 'N/A')}")
                
                shape_info = characteristics.get('shape_info', {})
                print(f"    - Bentuk: {shape_info.get('symmetry', 'N/A')}")

            return {"status": "success", "result": result, "time": elapsed_time}
        else:
            try:
                error_data = response.json()
                print(f"✗ Error: {error_data.get('error', 'Unknown error')}")
            except:
                print(f"✗ Error: {response.text}")
            return {"status": "error", "message": response.text}

    except requests.exceptions.ConnectionError:
        print("✗ [GAGAL KONEK]")
        print("  Error: Tidak bisa terhubung ke server.")
        print("  Pastikan server Flask (app.py) sedang berjalan!")
        return {"status": "error", "message": "Connection error"}
    except Exception as e:
        print(f"✗ Terjadi error: {e}")
        return {"status": "error", "message": str(e)}

def test_all_images():
    """Menguji semua gambar di folder."""
    print("="*70)
    print("BACKEND FLASK - TEST API UNTUK SEMUA GAMBAR")
    print("="*70)
    print(f"Folder gambar: {TEST_IMAGES_FOLDER}")
    print(f"Model yang digunakan: {MODEL_TO_TEST}")
    print(f"API URL: {API_URL}")
    
    # Dapatkan semua file gambar
    image_files = get_image_files(TEST_IMAGES_FOLDER)
    
    if not image_files:
        print("\n✗ Tidak ada gambar ditemukan!")
        print(f"  Pastikan ada file gambar (.jpg, .png, dll) di folder '{TEST_IMAGES_FOLDER}'")
        return
    
    print(f"\nDitemukan {len(image_files)} gambar untuk diuji.")
    print("-"*70)
    
    # Test setiap gambar
    results = []
    for image_path in image_files:
        result = test_single_image(image_path, MODEL_TO_TEST)
        results.append({
            "image": os.path.basename(image_path),
            "result": result
        })
    
    # Tampilkan ringkasan
    print("\n" + "="*70)
    print("RINGKASAN HASIL TEST")
    print("="*70)
    
    success_count = sum(1 for r in results if r["result"]["status"] == "success")
    error_count = len(results) - success_count
    
    print(f"\nTotal gambar diuji: {len(results)}")
    print(f"✓ Berhasil: {success_count}")
    print(f"✗ Gagal: {error_count}")
    
    if success_count > 0:
        print("\n--- Detail Hasil Prediksi ---")
        for item in results:
            if item["result"]["status"] == "success":
                prediction = item["result"]["result"].get("prediction", "N/A")
                time_taken = item["result"].get("time", 0)
                print(f"  {item['image']:<30} → Rp {prediction:<10} ({time_taken:.2f}s)")
    
    if error_count > 0:
        print("\n--- Gambar yang Gagal ---")
        for item in results:
            if item["result"]["status"] == "error":
                print(f"  {item['image']}: {item['result'].get('message', 'Unknown error')}")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    test_all_images()