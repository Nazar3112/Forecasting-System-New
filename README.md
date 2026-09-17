# Sistem Prediksi Tren Penjualan Retail (MySQL Native Edition)
### PT. Indomarco Prismatama — Skripsi Alfiyan Nazar (220511053)
**Model Hibrida: Facebook Prophet + LightGBM Residual Regression**

---

## 🌟 Keunggulan Versi Ini (MySQL Native — Tanpa Docker)

1. **Tanpa Docker**: Cukup menggunakan **XAMPP** atau layanan MySQL standar yang umum dipakai di lingkungan akademik.
2. **Pure Python Driver (`pymysql`)**: Tidak memerlukan kompilasi C++ (*Visual C++ Build Tools*) saat `pip install`.
3. **Auto-Create Database & Tables**: Skrip otomatis mengeksekusi `CREATE DATABASE IF NOT EXISTS db_forecasting` dan membuat semua tabel ORM — tidak perlu membuka phpMyAdmin secara manual.
4. **Performa Tinggi**:
   - Ingest data besar (~686 MB) menggunakan *chunking* 50.000 (read) dan 25.000 (insert) untuk mencegah error *MySQL server has gone away*.
   - Kueri peramalan instan (<0.1 detik) membaca deret waktu teragregasi (~160 baris) dari tabel `sales_aggregations`.
5. **Alur Pembaruan Inkremental**: Data bulanan baru (~15–25 MB) diunggah langsung lewat dashboard dan otomatis di-UPSERT via MySQL `ON DUPLICATE KEY UPDATE` tanpa memproses ulang dataset 3 tahun.

---

## 📁 Struktur Direktori

```
Forecasting System New Ver/
├── .env.example              ← Template konfigurasi MySQL (XAMPP default)
├── .gitignore                ← Proteksi data rahasia & berkas besar
├── requirements.txt          ← Dependensi Python murni (termasuk pymysql)
├── README.md                 ← Dokumentasi ini
├── app.py                    ← Dasbor operasional Streamlit multi-tab
│
├── database/                 ← Lapisan Basis Data MySQL
│   ├── __init__.py
│   ├── connection.py         ← Engine SQLAlchemy, auto-create DB, connection pool
│   ├── models.py             ← ORM SQLAlchemy: raw_transactions & sales_aggregations
│   └── repository.py         ← Kueri bulk insert, ON DUPLICATE KEY UPDATE, fetch series
│
├── core/                     ← Modul Peramalan Hibrida
│   ├── __init__.py
│   ├── config.py             ← Parameter peramalan & konstanta tema
│   ├── preprocessor.py       ← DeltaPreprocessor untuk validasi CSV delta
│   ├── feature_engine.py     ← Rekayasa fitur lag & kalender
│   ├── prophet_engine.py     ← Tahap 1: Prophet Baseline Decomposition
│   ├── lgbm_engine.py        ← Tahap 2: LightGBM Residual Error Regression
│   ├── engine.py             ← HybridForecastingEngine Orchestrator
│   ├── evaluator.py          ← Metrik akurasi: MAPE, RMSE, MAE
│   └── visualizer.py         ← Grafik Plotly & ekspor CSV
│
├── scripts/                  ← Utilitas Terminal / CLI
│   ├── __init__.py
│   └── seed_historical.py    ← Skrip CLI pemuatan master dataset (~686 MB)
│
└── tests/                    ← Pengujian Otomatis (Unit Tests)
    ├── __init__.py
    ├── test_database.py      ← Pengujian koneksi & model ORM
    └── test_hybrid.py        ← Pengujian model peramalan hibrida
```

---

## 🚀 Panduan Menjalankan Sistem

### 1. Nyalakan MySQL di XAMPP
- Buka aplikasi **XAMPP Control Panel**.
- Klik tombol **Start** pada baris **MySQL** (pastikan statusnya berwarna hijau pada port `3306`).

### 2. Pasang Dependensi Python
Buka terminal pada direktori `Forecasting System New Ver`:
```bash
pip install -r requirements.txt
```

### 3. Konfigurasi Lingkungan (`.env`)
Salin berkas template konfigurasi:
```bash
# Windows PowerShell:
Copy-Item .env.example .env

# Atau CMD / Git Bash:
cp .env.example .env
```
Isi default `.env` sudah disesuaikan untuk XAMPP standar:
```ini
DB_HOST=localhost
DB_PORT=3306
DB_USER=root
DB_PASS=
DB_NAME=db_forecasting
```

### 4. Pemuatan Dataset Master Historis (Hanya Sekali)
Muat berkas CSV master 3 tahun (~686 MB) ke MySQL:
```bash
python -m scripts.seed_historical --file "D:/Kuli ahh__/TUGASSSS/SKRIPSI/merged_MTRAN_with_desc.csv"
```
*(Sesuaikan path berkas CSV dengan lokasi berkas di laptop Anda).*

### 5. Jalankan Aplikasi Web Dasbor
```bash
streamlit run app.py
```
Aplikasi akan terbuka otomatis di peramban web pada alamat `http://localhost:8501`.

---

## 🧪 Menjalankan Pengujian (Testing)

Untuk memvalidasi bahwa seluruh pipeline matematika Prophet + LightGBM dan skema database berjalan normal:
```bash
python -m pytest tests/test_hybrid.py tests/test_database.py -v
```

---

## 📐 Formulasi Ilmiah Model Hibrida

1. **Tahap 1 — Facebook Prophet (Taylor & Letham, 2018)**:
   $$\hat{y}_{\text{Prophet}}(t) = g(t) + s(t) + h(t)$$
   Mendekomposisi tren makro $g(t)$ dan fluktuasi musiman $s(t)$.

2. **Tahap 2 — LightGBM Residual Error Regression**:
   $$\hat{e}_{\text{LightGBM}}(t) \approx y_{\text{aktual}}(t) - \hat{y}_{\text{Prophet}}(t)$$
   Menangkap pola non-linier dan autoregresif dari sisa galat (*residual*) menggunakan fitur *lag* dan kalender.

3. **Rekonsiliasi Akhir Non-Negatif**:
   $$\hat{y}_{\text{hybrid}}(t) = \max\left(0, \hat{y}_{\text{Prophet}}(t) + \hat{e}_{\text{LightGBM}}(t)\right)$$
