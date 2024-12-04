# Analisis Big Data untuk Optimalisasi Produksi dan Distribusi Telur

Proyek ini mengimplementasikan analisis Big Data untuk mengoptimalkan produksi dan distribusi telur di Indonesia menggunakan tiga metode utama: **Clustering**, **Forecasting**, dan **Klasifikasi Gambar**. Semua analisis ini disajikan dalam bentuk aplikasi web berbasis **Flask**.

## Features

1. **Clustering Produksi Telur**  
   - Memetakan jumlah produksi telur di tiap provinsi di Indonesia pada periode 2018-2023.
   - Data diklasifikasikan ke dalam 3 cluster: **Rendah**, **Sedang**, dan **Tinggi**.  
   - Visualisasi interaktif peta Indonesia dengan warna berbeda untuk setiap cluster (hijau untuk rendah, kuning untuk sedang, dan merah untuk tinggi).

2. **Forecasting Harga Telur**  
   - Meramalkan harga telur pada tahun 2023-2024 untuk setiap provinsi.
   - Grafik harga telur berdasarkan data historis yang disajikan dalam format interaktif.

3. **Klasifikasi Gambar Telur**  
   - Menggunakan model pembelajaran mesin untuk mengklasifikasikan apakah telur pecah atau tidak.
   - Fitur kamera real-time memungkinkan pengguna untuk mengecek apakah telur yang ditampilkan pecah atau tidak.

## Technology Used

- **Python 3.x**
- **Flask** (Web framework)
- **TensorFlow/Keras** (Model klasifikasi gambar)
- **OpenCV** (Pemrosesan gambar)
- **Folium** (Peta interaktif)
- **Matplotlib dan Seaborn** (Visualisasi data)
- **Pandas dan NumPy** (pemrosesan data)

## Installation Guide

### Prasyarat
Pastikan Anda sudah menginstal Python 3.x di sistem Anda. Anda juga memerlukan beberapa pustaka Python yang digunakan dalam proyek ini.

### Langkah-langkah Instalasi

1. **Clone repository ini**:
    ```bash
    git clone https://github.com/Glendy1208/eggABD.git
    cd eggABD
    ```

2. **Buat dan aktifkan virtual environment** (Opsional, namun disarankan untuk menghindari konflik dengan paket lain):
    - Pengguna Windows
        ```bash
        python -m venv venv
        source venv/bin/activate
        ```

3. **Install dependencies**:
    Instal semua pustaka yang dibutuhkan dengan menggunakan pip:
    ```bash
    pip install -r requirements.txt
    ```

4. **Menjalankan aplikasi**:
    Setelah semua langkah di atas selesai, jalankan aplikasi Flask:
    ```bash
    python app.py
    ```

    Aplikasi akan berjalan di `http://127.0.0.1:5000/`.

## Penggunaan

1. **Clustering**:
   - Pilih tahun (2018-2023) dan lihat pemetaan produksi telur berdasarkan cluster (Rendah, Sedang, Tinggi).
   - Peta interaktif menunjukkan distribusi produksi telur di Indonesia dengan warna yang berbeda.
   - ketika di hover atau klik pada suatu titik provinsi tertentu maka akan menampilkan jumlah produksi dalam 1 tahun dengan satuan ton dan hasil dari klaster-nya

2. **Forecasting**:
   - Pilih provinsi yang diinginkan (default semua provinsi/se-Indonesia) untuk melihat grafik ramalan harga telur per bulan.
   - pada pojok kanan tengah/bawah terdapat tombol "<<" untuk menampilkan grafik harga telur.
   - Grafik ini menunjukkan harga rata-rata telur berdasarkan data historis.

3. **Klasifikasi Gambar**:
   - Tekan tombol kamera pojok kanan atas untuk memulai kamera dan aplikasi akan menampilkan hasil klasifikasi secara real-time.
   - Gambar akan dianalisis untuk menentukan apakah telur tersebut pecah (Damaged) atau tidak (NotDamaged).