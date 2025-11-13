# 🏦 Backend API Deteksi Uang Kertas Indonesia

Backend Flask untuk sistem deteksi dan klasifikasi uang kertas Indonesia menggunakan Machine Learning (SVM & XGBoost).

## 📋 Daftar Isi

- [Fitur Utama](#-fitur-utama)
- [Teknologi yang Digunakan](#-teknologi-yang-digunakan)
- [Struktur Project](#-struktur-project)
- [Instalasi](#-instalasi)
- [Cara Menjalankan](#-cara-menjalankan)
- [API Documentation](#-api-documentation)
- [Testing](#-testing)
- [Model Machine Learning](#-model-machine-learning)
- [Troubleshooting](#-troubleshooting)

## ✨ Fitur Utama

- **Dual Model Support**: Mendukung prediksi menggunakan SVM atau XGBoost
- **Ekstraksi Fitur Advanced**: Kombinasi fitur warna (HSV), tekstur (LBP), dan bentuk (Hu Moments)
- **REST API**: Endpoint API yang mudah digunakan
- **Base64 Image Processing**: Input gambar dalam format Base64
- **Real-time Prediction**: Prediksi cepat dan akurat

## 🛠 Teknologi yang Digunakan

### Backend Framework
- **Flask 3.1.2** - Web framework Python

### Machine Learning & Computer Vision
- **scikit-learn 1.7.2** - Model SVM dan preprocessing
- **XGBoost 3.1.1** - Model XGBoost
- **OpenCV 4.12.0** - Image processing
- **scikit-image 0.25.2** - Feature extraction (LBP)
- **NumPy 2.2.6** - Numerical computing
- **joblib 1.5.2** - Model serialization

### Image Processing
- **Pillow 12.0.0** - Image handling

### Python Version
- **Python 3.13.5**

## 📁 Struktur Project

```
PBL-Backend/
│
├── app.py                          # Aplikasi Flask utama
├── test_api.py                     # Script untuk testing API
├── requirements.txt                # Dependencies Python
├── README.md                       # Dokumentasi (file ini)
│
├── models/                         # Folder model ML
│   ├── label_encoder_v2.joblib    # Label encoder (shared)
│   │
│   ├── svm/                       # Model SVM
│   │   ├── svm_model_v2.joblib   
│   │   └── svm_scaler_v2.joblib  
│   │
│   └── xgboost/                   # Model XGBoost
│       └── xgb_model_v2.joblib   
│
└── venv/                          # Virtual environment (tidak di-commit)
```

## 🚀 Instalasi

### Prasyarat
- Python 3.13 atau lebih tinggi
- pip (Python package manager)
- Git (opsional)

### Langkah-langkah Instalasi

1. **Clone atau Download Repository**
   ```powershell
   git clone <repository-url>
   cd PBL-Backend
   ```

2. **Buat Virtual Environment**
   ```powershell
   python -m venv venv
   ```

3. **Aktifkan Virtual Environment**
   ```powershell
   # Windows PowerShell
   .\venv\Scripts\Activate.ps1
   
   # Windows CMD
   venv\Scripts\activate.bat
   ```

4. **Install Dependencies**
   ```powershell
   pip install -r requirements.txt
   ```

5. **Verifikasi Instalasi**
   ```powershell
   python -c "import flask, cv2, sklearn, xgboost; print('✓ Semua modul berhasil diinstall!')"
   ```

## 🎯 Cara Menjalankan

### 1. Menjalankan Server

```powershell
# Pastikan virtual environment aktif
python app.py
```

Server akan berjalan di:
- **Local**: `http://127.0.0.1:5000`
- **Network**: `http://0.0.0.0:5000` (dapat diakses dari perangkat lain di jaringan yang sama)

### 2. Akses Homepage

Buka browser dan kunjungi:
```
http://127.0.0.1:5000
```

Anda akan melihat status model yang tersedia.

### 3. Testing API

Gunakan script `test_api.py` yang disediakan:

```powershell
# Edit test_api.py terlebih dahulu:
# - Ubah TEST_IMAGE_PATH ke path gambar Anda
# - Ubah MODEL_TO_TEST ke "svm" atau "xgboost"

python test_api.py
```

## 📡 API Documentation

### Base URL
```
http://127.0.0.1:5000
```

### Endpoints

#### 1. GET `/` - Homepage
Menampilkan informasi status server dan model yang tersedia.

**Response:**
```html
<h1>Backend Model Gabungan (SVM & XGBoost) Aktif!</h1>
<p>Gunakan endpoint /predict untuk mengirim gambar.</p>
```

---

#### 2. POST `/predict` - Prediksi Uang

Melakukan prediksi denominasi uang berdasarkan gambar yang dikirim.

**Request Headers:**
```
Content-Type: application/json
```

**Request Body:**
```json
{
  "image": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
  "model": "svm"
}
```

**Parameters:**
| Parameter | Type   | Required | Description                           | Default |
|-----------|--------|----------|---------------------------------------|---------|
| `image`   | string | Yes      | Base64 encoded image string           | -       |
| `model`   | string | No       | Model to use: "svm" or "xgboost"      | "svm"   |

**Response (Success - 200 OK):**
```json
{
  "prediction": "50000",
  "model_version": "v2 (Warna+Tekstur+Bentuk)",
  "model_used": "SVM"
}
```

**Response Fields:**
| Field           | Type   | Description                                    |
|-----------------|--------|------------------------------------------------|
| `prediction`    | string | Denominasi uang yang diprediksi               |
| `model_version` | string | Versi model dan fitur yang digunakan          |
| `model_used`    | string | Model yang digunakan untuk prediksi           |

**Response (Error - 400 Bad Request):**
```json
{
  "error": "Request JSON tidak memiliki key 'image'"
}
```

**Response (Error - 500 Internal Server Error):**
```json
{
  "error": "Satu atau lebih file model (SVM/XGB/Scaler/Encoder) tidak berhasil dimuat."
}
```

### Contoh Penggunaan dengan cURL

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "image": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
    "model": "svm"
  }'
```

### Contoh Penggunaan dengan Python

```python
import requests
import base64

# Encode gambar
with open("gambar_uji.jpg", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode('utf-8')

# Kirim request
response = requests.post(
    "http://127.0.0.1:5000/predict",
    json={
        "image": f"data:image/jpeg;base64,{image_b64}",
        "model": "svm"  # atau "xgboost"
    }
)

# Tampilkan hasil
print(response.json())
```

## 🧪 Testing

### Testing Manual dengan Script

1. **Siapkan Gambar Test**
   - Letakkan gambar uang di folder `PBL-Backend`
   - Format: JPG, PNG, atau format gambar lainnya

2. **Edit `test_api.py`**
   ```python
   TEST_IMAGE_PATH = "gambar_uji.jpg"  # Nama file gambar Anda
   MODEL_TO_TEST = "svm"                # atau "xgboost"
   ```

3. **Jalankan Test**
   ```powershell
   python test_api.py
   ```

4. **Hasil Test**
   ```
   Menguji server di http://127.0.0.1:5000/predict...
   Menggunakan gambar: gambar_uji.jpg
   Meminta model: svm

   ✓ Prediksi berhasil!
   Hasil Prediksi: 50000
   Model yang Digunakan: SVM
   Versi Model: v2 (Warna+Tekstur+Bentuk)
   ```

### Testing dengan Postman

1. Buka Postman
2. Buat request baru:
   - Method: `POST`
   - URL: `http://127.0.0.1:5000/predict`
   - Headers: `Content-Type: application/json`
   - Body (raw JSON):
     ```json
     {
       "image": "data:image/jpeg;base64,YOUR_BASE64_STRING",
       "model": "svm"
     }
     ```
3. Klik **Send**

## 🤖 Model Machine Learning

### Arsitektur Model

#### 1. **SVM (Support Vector Machine)**
- **Algoritma**: Support Vector Classification
- **Scaling**: StandardScaler (wajib)
- **Feature Engineering**: Warna + Tekstur + Bentuk
- **Akurasi**: ~95% (tergantung dataset training)

#### 2. **XGBoost**
- **Algoritma**: Extreme Gradient Boosting
- **Scaling**: Tidak diperlukan
- **Feature Engineering**: Warna + Tekstur + Bentuk
- **Akurasi**: ~97% (tergantung dataset training)

### Ekstraksi Fitur

Model menggunakan 3 jenis fitur yang dikombinasikan:

#### 1. **Fitur Warna (Color Features)**
- **Metode**: HSV Histogram
- **Detail**:
  - Hue Histogram (180 bins)
  - Saturation Histogram (256 bins)
  - Total: 436 fitur

#### 2. **Fitur Tekstur (Texture Features)**
- **Metode**: Local Binary Pattern (LBP)
- **Detail**:
  - Points: 24
  - Radius: 8
  - Method: uniform
  - Total: 26 fitur (bins)

#### 3. **Fitur Bentuk (Shape Features)**
- **Metode**: Hu Moments
- **Detail**:
  - 7 Hu Moments invariants
  - Log-transformed
  - Total: 7 fitur

**Total Fitur**: 469 fitur per gambar

### Preprocessing Pipeline

1. **Resize**: 250x250 pixels
2. **Gaussian Blur**: (5,5) kernel untuk noise reduction
3. **Color Space Conversion**: BGR → HSV, BGR → Grayscale
4. **Normalization**: MINMAX normalization untuk setiap jenis fitur
5. **Scaling** (SVM only): StandardScaler untuk normalisasi fitur

### Kelas yang Didukung

Model dapat memprediksi denominasi uang kertas Indonesia:
- 1000
- 2000
- 5000
- 10000
- 20000
- 50000
- 100000

## 🔧 Troubleshooting

### Error: "Module not found"

**Solusi:**
```powershell
# Pastikan virtual environment aktif
.\venv\Scripts\Activate.ps1

# Install ulang dependencies
pip install -r requirements.txt
```

### Error: "Model file tidak ditemukan"

**Solusi:**
1. Pastikan struktur folder `models/` sudah benar
2. Pastikan file `.joblib` ada di lokasi yang tepat:
   ```
   models/
   ├── label_encoder_v2.joblib
   ├── svm/
   │   ├── svm_model_v2.joblib
   │   └── svm_scaler_v2.joblib
   └── xgboost/
       └── xgb_model_v2.joblib
   ```

### Error: "Port 5000 already in use"

**Solusi 1 - Ubah Port:**
Edit `app.py`:
```python
app.run(host='0.0.0.0', port=5001, debug=True)
```

**Solusi 2 - Kill Process:**
```powershell
# Cari process yang menggunakan port 5000
netstat -ano | findstr :5000

# Kill process (ganti PID dengan ID yang ditemukan)
taskkill /PID <PID> /F
```

### Error: "Connection Refused" saat Testing

**Solusi:**
1. Pastikan server Flask sedang berjalan
2. Cek apakah port benar (default: 5000)
3. Cek firewall Windows

### Gambar Tidak Bisa Diprediksi

**Solusi:**
1. Pastikan gambar dalam format yang didukung (JPG, PNG)
2. Pastikan encoding Base64 benar
3. Cek ukuran file tidak terlalu besar (maks ~10MB)
4. Pastikan gambar tidak corrupt

### XGBoost "Module not found"

**Solusi:**
```powershell
# Install XGBoost secara manual
pip install xgboost

# Atau install ulang semua dependencies
pip install -r requirements.txt
```

## � Integrasi dengan Flutter App

### Persiapan Backend untuk Flutter

#### 1. Pastikan Server Dapat Diakses dari Jaringan

Server Flask sudah dikonfigurasi untuk menerima koneksi dari jaringan lokal:
```python
app.run(host='0.0.0.0', port=5000, debug=True)
```

#### 2. Temukan IP Address Komputer Anda

**Windows PowerShell:**
```powershell
ipconfig
```

Cari bagian **IPv4 Address**, contoh: `192.168.1.100`

**Linux/Mac:**
```bash
ifconfig
# atau
ip addr show
```

#### 3. Test Koneksi dari Perangkat Mobile

Pastikan laptop/PC dan HP berada di **jaringan WiFi yang sama**, lalu buka browser di HP:
```
http://192.168.1.100:5000
```

Jika berhasil, Anda akan melihat homepage backend.

### Implementasi di Flutter

#### Setup Dependencies (pubspec.yaml)

```yaml
dependencies:
  flutter:
    sdk: flutter
  http: ^1.1.0          # untuk HTTP requests
  image_picker: ^1.0.4   # untuk ambil gambar dari kamera/galeri
  image: ^4.1.3         # untuk image processing
```

#### Konfigurasi API Service

Buat file `lib/services/api_service.dart`:

```dart
import 'dart:convert';
import 'dart:io';
import 'package:http/http.dart' as http;

class ApiService {
  // GANTI dengan IP address komputer Anda
  static const String baseUrl = 'http://192.168.1.100:5000';
  
  // Timeout untuk request
  static const Duration timeout = Duration(seconds: 30);
  
  /// Kirim gambar untuk prediksi
  /// 
  /// [imagePath] - Path file gambar dari device
  /// [model] - Model yang digunakan: 'svm' atau 'xgboost'
  /// 
  /// Returns Map dengan hasil prediksi
  static Future<Map<String, dynamic>> predictImage({
    required String imagePath,
    String model = 'svm',
  }) async {
    try {
      // 1. Baca file gambar
      File imageFile = File(imagePath);
      List<int> imageBytes = await imageFile.readAsBytes();
      
      // 2. Encode ke Base64
      String base64Image = base64Encode(imageBytes);
      String imageBase64 = 'data:image/jpeg;base64,$base64Image';
      
      // 3. Siapkan request body
      Map<String, dynamic> requestBody = {
        'image': imageBase64,
        'model': model,
      };
      
      // 4. Kirim POST request
      final response = await http
          .post(
            Uri.parse('$baseUrl/predict'),
            headers: {
              'Content-Type': 'application/json',
            },
            body: jsonEncode(requestBody),
          )
          .timeout(timeout);
      
      // 5. Handle response
      if (response.statusCode == 200) {
        return jsonDecode(response.body);
      } else {
        throw Exception('Error ${response.statusCode}: ${response.body}');
      }
    } catch (e) {
      throw Exception('Gagal menghubungi server: $e');
    }
  }
  
  /// Test koneksi ke server
  static Future<bool> testConnection() async {
    try {
      final response = await http
          .get(Uri.parse(baseUrl))
          .timeout(Duration(seconds: 5));
      return response.statusCode == 200;
    } catch (e) {
      return false;
    }
  }
}
```

#### Contoh Penggunaan di Widget

```dart
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'dart:io';
import 'services/api_service.dart';

class PredictionScreen extends StatefulWidget {
  @override
  _PredictionScreenState createState() => _PredictionScreenState();
}

class _PredictionScreenState extends State<PredictionScreen> {
  File? _image;
  String? _prediction;
  String? _modelUsed;
  bool _isLoading = false;
  String _selectedModel = 'svm';

  final ImagePicker _picker = ImagePicker();

  // Ambil gambar dari kamera
  Future<void> _pickImageFromCamera() async {
    final XFile? photo = await _picker.pickImage(
      source: ImageSource.camera,
      imageQuality: 85,
    );
    
    if (photo != null) {
      setState(() {
        _image = File(photo.path);
        _prediction = null;
      });
    }
  }

  // Ambil gambar dari galeri
  Future<void> _pickImageFromGallery() async {
    final XFile? photo = await _picker.pickImage(
      source: ImageSource.gallery,
      imageQuality: 85,
    );
    
    if (photo != null) {
      setState(() {
        _image = File(photo.path);
        _prediction = null;
      });
    }
  }

  // Kirim gambar untuk prediksi
  Future<void> _predictImage() async {
    if (_image == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Pilih gambar terlebih dahulu!')),
      );
      return;
    }

    setState(() {
      _isLoading = true;
      _prediction = null;
    });

    try {
      // Panggil API
      final result = await ApiService.predictImage(
        imagePath: _image!.path,
        model: _selectedModel,
      );

      setState(() {
        _prediction = result['prediction'];
        _modelUsed = result['model_used'];
        _isLoading = false;
      });

      // Tampilkan hasil
      _showResultDialog();
    } catch (e) {
      setState(() {
        _isLoading = false;
      });

      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('Error: $e'),
          backgroundColor: Colors.red,
        ),
      );
    }
  }

  void _showResultDialog() {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: Text('Hasil Prediksi'),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('Denominasi: Rp ${_prediction}',
                style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold)),
            SizedBox(height: 10),
            Text('Model: $_modelUsed'),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: Text('OK'),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Deteksi Uang'),
      ),
      body: SingleChildScrollView(
        padding: EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // Pilih Model
            Text('Pilih Model:', style: TextStyle(fontSize: 16)),
            Row(
              children: [
                Expanded(
                  child: RadioListTile<String>(
                    title: Text('SVM'),
                    value: 'svm',
                    groupValue: _selectedModel,
                    onChanged: (value) {
                      setState(() => _selectedModel = value!);
                    },
                  ),
                ),
                Expanded(
                  child: RadioListTile<String>(
                    title: Text('XGBoost'),
                    value: 'xgboost',
                    groupValue: _selectedModel,
                    onChanged: (value) {
                      setState(() => _selectedModel = value!);
                    },
                  ),
                ),
              ],
            ),
            SizedBox(height: 20),

            // Preview Gambar
            if (_image != null)
              Container(
                height: 300,
                decoration: BoxDecoration(
                  border: Border.all(color: Colors.grey),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: ClipRRect(
                  borderRadius: BorderRadius.circular(8),
                  child: Image.file(_image!, fit: BoxFit.cover),
                ),
              )
            else
              Container(
                height: 300,
                decoration: BoxDecoration(
                  border: Border.all(color: Colors.grey),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Center(
                  child: Text('Belum ada gambar'),
                ),
              ),
            SizedBox(height: 20),

            // Tombol Ambil Gambar
            Row(
              children: [
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: _pickImageFromCamera,
                    icon: Icon(Icons.camera_alt),
                    label: Text('Kamera'),
                  ),
                ),
                SizedBox(width: 10),
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: _pickImageFromGallery,
                    icon: Icon(Icons.photo_library),
                    label: Text('Galeri'),
                  ),
                ),
              ],
            ),
            SizedBox(height: 10),

            // Tombol Prediksi
            ElevatedButton(
              onPressed: _isLoading ? null : _predictImage,
              style: ElevatedButton.styleFrom(
                padding: EdgeInsets.symmetric(vertical: 16),
              ),
              child: _isLoading
                  ? CircularProgressIndicator(color: Colors.white)
                  : Text('PREDIKSI', style: TextStyle(fontSize: 18)),
            ),

            // Hasil Prediksi
            if (_prediction != null) ...[
              SizedBox(height: 20),
              Card(
                color: Colors.green[50],
                child: Padding(
                  padding: EdgeInsets.all(16),
                  child: Column(
                    children: [
                      Text('Hasil Prediksi:',
                          style: TextStyle(fontSize: 16)),
                      SizedBox(height: 8),
                      Text('Rp $_prediction',
                          style: TextStyle(
                            fontSize: 32,
                            fontWeight: FontWeight.bold,
                            color: Colors.green[700],
                          )),
                      Text('Model: $_modelUsed',
                          style: TextStyle(color: Colors.grey[600])),
                    ],
                  ),
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }
}
```

### Troubleshooting Koneksi Flutter

#### ❌ Error: "Connection Refused" / "Network Unreachable"

**Solusi:**

1. **Cek IP Address**
   ```powershell
   ipconfig
   ```
   Pastikan IP di Flutter sama dengan IP komputer

2. **Cek Firewall Windows**
   - Buka **Windows Defender Firewall**
   - Pilih **Allow an app through firewall**
   - Pastikan Python/Flask diizinkan di **Private networks**

3. **Cek Server Berjalan**
   ```powershell
   # Di komputer, pastikan server running
   python app.py
   ```

4. **Test dari Browser HP**
   ```
   http://IP_KOMPUTER:5000
   ```

#### ❌ Error: "TimeoutException"

**Solusi:**
- Tingkatkan timeout di Flutter:
  ```dart
  static const Duration timeout = Duration(seconds: 60);
  ```
- Kompres gambar sebelum dikirim:
  ```dart
  final XFile? photo = await _picker.pickImage(
    source: ImageSource.camera,
    imageQuality: 50,  // Kurangi kualitas
    maxWidth: 800,     // Batasi ukuran
  );
  ```

#### ❌ Error: "Certificate Verification Failed" (HTTPS)

**Solusi:**
Backend menggunakan HTTP (bukan HTTPS), pastikan di Flutter:
- Gunakan `http://` bukan `https://`
- Tambahkan di `android/app/src/main/AndroidManifest.xml`:
  ```xml
  <application
      android:usesCleartextTraffic="true"
      ...>
  ```

### Tips Optimasi

1. **Kompres Gambar Sebelum Kirim**
   ```dart
   import 'package:image/image.dart' as img;
   
   Future<File> compressImage(File file) async {
     final bytes = await file.readAsBytes();
     final image = img.decodeImage(bytes);
     final compressed = img.encodeJpg(image!, quality: 70);
     
     final compressedFile = File('${file.path}_compressed.jpg');
     await compressedFile.writeAsBytes(compressed);
     return compressedFile;
   }
   ```

2. **Tambahkan Loading Indicator**
   - Prediksi bisa memakan waktu 2-5 detik
   - Gunakan `CircularProgressIndicator` atau animasi loading

3. **Cache Hasil Prediksi**
   - Simpan hasil prediksi terakhir
   - Gunakan `SharedPreferences` atau database lokal

4. **Retry Mechanism**
   ```dart
   Future<Map<String, dynamic>> predictWithRetry({
     required String imagePath,
     String model = 'svm',
     int maxRetries = 3,
   }) async {
     for (int i = 0; i < maxRetries; i++) {
       try {
         return await ApiService.predictImage(
           imagePath: imagePath,
           model: model,
         );
       } catch (e) {
         if (i == maxRetries - 1) rethrow;
         await Future.delayed(Duration(seconds: 2));
       }
     }
     throw Exception('Max retries reached');
   }
   ```

### Keamanan

⚠️ **Catatan Penting untuk Production:**

1. **Gunakan HTTPS** (bukan HTTP) di production
2. **Tambahkan API Key/Token** untuk autentikasi
3. **Rate Limiting** untuk mencegah abuse
4. **Input Validation** untuk file size dan format
5. **Deploy ke Cloud** (AWS, Google Cloud, Heroku, dll)

Contoh dengan API Key:
```dart
static Future<Map<String, dynamic>> predictImage({
  required String imagePath,
  String model = 'svm',
}) async {
  final response = await http.post(
    Uri.parse('$baseUrl/predict'),
    headers: {
      'Content-Type': 'application/json',
      'X-API-Key': 'YOUR_SECRET_API_KEY',  // Tambahkan API key
    },
    body: jsonEncode(requestBody),
  );
  // ...
}
```

## �📝 Catatan Pengembangan

### Menambah Model Baru

1. Train model baru di Colab/Jupyter
2. Export model sebagai `.joblib`:
   ```python
   import joblib
   joblib.dump(model, 'new_model.joblib')
   joblib.dump(scaler, 'new_scaler.joblib')
   ```
3. Upload ke folder `models/`
4. Update `app.py` untuk menambah logika model baru

### Mengupdate Dependencies

```powershell
# Update semua packages
pip install --upgrade -r requirements.txt

# Generate requirements.txt baru
pip freeze > requirements.txt
```

### Debug Mode

Server berjalan dengan `debug=True` secara default, yang artinya:
- Auto-reload saat code berubah
- Detailed error messages
- **JANGAN** gunakan di production

Untuk production, ubah di `app.py`:
```python
app.run(host='0.0.0.0', port=5000, debug=False)
```

## 📄 License

Project ini dibuat untuk keperluan pembelajaran dan penelitian.

## 👥 Kontributor

- **Developer**: [Nama Anda]
- **Contact**: [Email/GitHub]

## 🙏 Acknowledgments

- Dataset uang kertas Indonesia
- scikit-learn documentation
- XGBoost documentation
- Flask documentation

---

**Last Updated**: November 13, 2025

**Version**: 2.0.0

Untuk pertanyaan atau issue, silakan buat issue di repository ini.
