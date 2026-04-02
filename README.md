# 🍜 KostFoodApp-TOPSIS

Aplikasi berbasis web rekomendasi makanan untuk anak kost berdasarkan harga, jarak, rating, dan jenis. Tugas Case Based Learning Mata Kuliah AI.

---

## 📌 Tentang Aplikasi

KostApp-TOPSIS membantu penghuni kost memilih makanan terbaik di sekitar mereka dengan mempertimbangkan **4 kriteria sekaligus** menggunakan metode TOPSIS — sebuah algoritma pengambilan keputusan multi-kriteria (MCDM).

Pengguna cukup mengatur bobot preferensi masing-masing kriteria, lalu sistem secara otomatis menghitung dan menampilkan ranking rekomendasi makanan terbaik.

---

## 🚀 Cara Menjalankan

```bash
pip install streamlit numpy pandas
streamlit run kostfood_simple.py
```

---

## 🍽️ Data & Kriteria

Terdapat **20 alternatif makanan** yang tersedia, masing-masing memiliki atribut berikut:

| Kriteria   | Tipe    | Keterangan                          |
|------------|---------|-------------------------------------|
| Harga      | Cost    | Lebih murah → lebih baik            |
| Jarak      | Cost    | Lebih dekat → lebih baik            |
| Rating     | Benefit | Lebih tinggi → lebih baik           |
| Jenis      | Benefit | Skor kesesuaian jenis makanan       |

Kolom **Jenis** dikonversi ke nilai numerik sebelum dihitung:

```python
JENIS_SCORE = {
    "berat":      1.0,   # Makanan mengenyangkan
    "vegetarian": 0.8,   # Pilihan sehat
    "cepat saji": 0.7,   # Praktis
    "ringan":     0.5,   # Cemilan / snack
}
```

---

## 🧮 Metode TOPSIS — Alur Perhitungan

TOPSIS *(Technique for Order Preference by Similarity to Ideal Solution)* bekerja melalui **6 langkah utama**:

```
Data Mentah (X)
     ↓  normalisasi vektor
Matriks R
     ↓  × bobot pengguna
Matriks Terbobot V
     ↓  ambil max / min sesuai tipe
Solusi Ideal A+ dan A−
     ↓  jarak Euclidean
D+ dan D−
     ↓  CC = D− / (D+ + D−)
Ranking → Rekomendasi
```

---

### Langkah 1 — Matriks Keputusan (X)

Setiap baris adalah satu alternatif makanan, setiap kolom adalah satu kriteria.

```python
X = np.array([
    [f["harga"], f["jarak"], f["rating"], JENIS_SCORE[f["jenis"]]]
    for f in data
], dtype=float)
```

Contoh (5 baris pertama):

| Makanan             | Harga  | Jarak | Rating | Jenis |
|---------------------|--------|-------|--------|-------|
| Nasi Goreng Spesial | 12000  | 0.3   | 4.8    | 1.0   |
| Mie Ayam Bakso      | 14000  | 0.5   | 4.6    | 1.0   |
| Burger Mini         | 18000  | 0.8   | 4.2    | 0.7   |
| Ayam Geprek         | 16000  | 0.4   | 4.7    | 1.0   |
| Roti Bakar Coklat   | 8000   | 0.2   | 4.3    | 0.5   |

---

### Langkah 2 — Normalisasi Matriks (R)

Karena setiap kriteria memiliki satuan berbeda (Rupiah, km, bintang), semua nilai diseragamkan menggunakan **Vector Normalization**:

$$r_{ij} = \frac{x_{ij}}{\sqrt{\sum_{i=1}^{n} x_{ij}^2}}$$

```python
norm = np.sqrt((X ** 2).sum(axis=0))
norm[norm == 0] = 1          # hindari pembagian nol
R = X / norm
```

Setiap kolom dibagi dengan panjang vektornya sehingga semua nilai berada di rentang 0–1 dan skala antar kriteria menjadi setara.

---

### Langkah 3 — Matriks Terbobot (V)

Nilai ternormalisasi dikalikan dengan bobot preferensi pengguna:

$$v_{ij} = w_j \times r_{ij}$$

```python
weights = np.array([w_harga, w_jarak, w_rating, w_jenis])
# Bobot sudah dinormalisasi: total = 1
V = R * weights
```

Contoh jika pengguna mengatur bobot `Harga=30, Jarak=25, Rating=30, Jenis=15`:

| Kriteria | Input | Bobot Ternormalisasi |
|----------|-------|----------------------|
| Harga    | 30    | 0.30                 |
| Jarak    | 25    | 0.25                 |
| Rating   | 30    | 0.30                 |
| Jenis    | 15    | 0.15                 |

---

### Langkah 4 — Solusi Ideal A⁺ dan A⁻

Ditentukan berdasarkan **tipe** setiap kriteria:

$$A^+_j = \begin{cases} \min(V_{:,j}) & \text{jika Cost} \\ \max(V_{:,j}) & \text{jika Benefit} \end{cases}
\qquad
A^-_j = \begin{cases} \max(V_{:,j}) & \text{jika Cost} \\ \min(V_{:,j}) & \text{jika Benefit} \end{cases}$$

```python
is_benefit = np.array([False, False, True, True])
#                       harga  jarak  rating jenis

A_pos = np.where(is_benefit, V.max(axis=0), V.min(axis=0))
A_neg = np.where(is_benefit, V.min(axis=0), V.max(axis=0))
```

| Kriteria | Tipe    | A⁺ (Terbaik)              | A⁻ (Terburuk)              |
|----------|---------|---------------------------|----------------------------|
| Harga    | Cost    | V terkecil (termurah)     | V terbesar (termahal)      |
| Jarak    | Cost    | V terkecil (terdekat)     | V terbesar (terjauh)       |
| Rating   | Benefit | V terbesar (rating tinggi)| V terkecil (rating rendah) |
| Jenis    | Benefit | V terbesar (paling cocok) | V terkecil (kurang cocok)  |

---

### Langkah 5 — Jarak Euclidean (D⁺ dan D⁻)

Setiap alternatif dihitung jaraknya ke solusi ideal positif dan negatif:

$$D^+_i = \sqrt{\sum_{j=1}^{m} (v_{ij} - A^+_j)^2}$$

$$D^-_i = \sqrt{\sum_{j=1}^{m} (v_{ij} - A^-_j)^2}$$

```python
D_pos = np.sqrt(((V - A_pos) ** 2).sum(axis=1))
D_neg = np.sqrt(((V - A_neg) ** 2).sum(axis=1))
```

- **D⁺ kecil** → makanan dekat ke kondisi ideal terbaik ✅
- **D⁻ besar** → makanan jauh dari kondisi terburuk ✅

---

### Langkah 6 — Closeness Coefficient (CC) & Ranking

Skor akhir yang menggabungkan D⁺ dan D⁻ menjadi satu nilai antara 0 dan 1:

$$CC_i = \frac{D^-_i}{D^+_i + D^-_i}$$

```python
CC = D_neg / (D_pos + D_neg + 1e-10)
# +1e-10 untuk mencegah pembagian dengan nol

results.sort(key=lambda x: x["CC"], reverse=True)
```

| Nilai CC       | Interpretasi                        |
|----------------|-------------------------------------|
| Mendekati 1.0  | Sangat dekat ke solusi ideal terbaik — **direkomendasikan** |
| Sekitar 0.5    | Posisi menengah                     |
| Mendekati 0.0  | Sangat dekat ke solusi terburuk     |

---

## 🖥️ Tampilan Aplikasi

Aplikasi dijalankan via **Streamlit** dengan layout:

- **Sidebar** — filter jenis makanan, filter harga maksimum, slider bobot kriteria, tombol "Cari Rekomendasi"
- **Main area** — rekomendasi terbaik beserta skor CC, tabel ranking semua alternatif, expander detail perhitungan tiap langkah TOPSIS

---

## 📦 Teknologi

| Library    | Kegunaan                              |
|------------|---------------------------------------|
| Streamlit  | Framework web app interaktif          |
| NumPy      | Komputasi matriks TOPSIS              |
| Pandas     | Tampilan tabel hasil ranking          |

---

## 👨‍💻 Tim Pengembang

**Kelompok 9** — Tugas Case Based Learning Mata Kuliah Kecerdasan Buatan
