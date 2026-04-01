import streamlit as st
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="KostFood — Rekomendasi TOPSIS",
    page_icon="🍜",
    layout="wide",
)

# ─────────────────────────────────────────────
#  DATA MAKANAN
# ─────────────────────────────────────────────
FOODS = [
    {"id": 1,  "nama": "Nasi Goreng Spesial",  "jenis": "berat",      "harga": 12000, "jarak": 0.3, "rating": 4.8},
    {"id": 2,  "nama": "Mie Ayam Bakso",        "jenis": "berat",      "harga": 14000, "jarak": 0.5, "rating": 4.6},
    {"id": 3,  "nama": "Burger Mini",           "jenis": "cepat saji", "harga": 18000, "jarak": 0.8, "rating": 4.2},
    {"id": 4,  "nama": "Ayam Geprek",           "jenis": "berat",      "harga": 16000, "jarak": 0.4, "rating": 4.7},
    {"id": 5,  "nama": "Roti Bakar Coklat",     "jenis": "ringan",     "harga": 8000,  "jarak": 0.2, "rating": 4.3},
    {"id": 6,  "nama": "Soto Ayam",             "jenis": "berat",      "harga": 13000, "jarak": 0.6, "rating": 4.5},
    {"id": 7,  "nama": "Salad Sayur",           "jenis": "vegetarian", "harga": 15000, "jarak": 1.2, "rating": 4.1},
    {"id": 8,  "nama": "Indomie Goreng Jumbo",  "jenis": "berat",     "harga": 10000, "jarak": 0.1, "rating": 3.9},
    {"id": 9,  "nama": "Gado-Gado",             "jenis": "vegetarian", "harga": 12000, "jarak": 0.7, "rating": 4.4},
    {"id": 10, "nama": "Nasi Padang",           "jenis": "berat",      "harga": 22000, "jarak": 1.0, "rating": 4.9},
    {"id": 11, "nama": "Martabak Mini",         "jenis": "ringan",     "harga": 7000,  "jarak": 0.3, "rating": 4.0},
    {"id": 12, "nama": "KFC (1 potong)",        "jenis": "cepat saji", "harga": 28000, "jarak": 1.5, "rating": 4.3},
    {"id": 13, "nama": "Bakso Malang",          "jenis": "berat",      "harga": 14000, "jarak": 0.4, "rating": 4.6},
    {"id": 14, "nama": "Pisang Goreng",         "jenis": "ringan",     "harga": 5000,  "jarak": 0.1, "rating": 3.8},
    {"id": 15, "nama": "Nasi Kuning",           "jenis": "berat",      "harga": 11000, "jarak": 0.5, "rating": 4.5},
    {"id": 16, "nama": "Smoothie Bowl",         "jenis": "vegetarian", "harga": 30000, "jarak": 1.8, "rating": 4.7},
    {"id": 17, "nama": "Gorengan Mix",          "jenis": "ringan",     "harga": 6000,  "jarak": 0.2, "rating": 3.7},
    {"id": 18, "nama": "McDonald's Burger",     "jenis": "cepat saji", "harga": 35000, "jarak": 2.0, "rating": 4.4},
    {"id": 19, "nama": "Tempe Orek",            "jenis": "vegetarian", "harga": 9000,  "jarak": 0.3, "rating": 4.2},
    {"id": 20, "nama": "Ayam Bakar Madu",       "jenis": "berat",      "harga": 20000, "jarak": 0.9, "rating": 4.8},
]

# Skor numerik untuk kriteria Jenis (benefit: semakin tinggi semakin baik)
JENIS_SCORE = {"cepat saji": 0.7, "berat": 1.0, "ringan": 0.5, "vegetarian": 0.8}


# ─────────────────────────────────────────────
#  FUNGSI TOPSIS
# ─────────────────────────────────────────────
def run_topsis(data, w_harga, w_jarak, w_rating, w_jenis):
    """
    TOPSIS — 6 langkah utama:
    1. Buat matriks keputusan (X)
    2. Normalisasi matriks → R
    3. Bobot matriks ternormalisasi → V
    4. Tentukan solusi ideal positif (A+) dan negatif (A-)
    5. Hitung jarak Euclidean D+ dan D-
    6. Hitung Closeness Coefficient (CC), lalu ranking
    """
    weights = np.array([w_harga, w_jarak, w_rating, w_jenis])

    # === LANGKAH 1: Matriks keputusan ===
    X = np.array([
        [f["harga"], f["jarak"], f["rating"], JENIS_SCORE[f["jenis"]]]
        for f in data
    ], dtype=float)

    # === LANGKAH 2: Normalisasi (Vector Normalization) ===
    norm = np.sqrt((X ** 2).sum(axis=0))
    norm[norm == 0] = 1          # hindari pembagian nol
    R = X / norm

    # === LANGKAH 3: Matriks terbobot ===
    V = R * weights

    # === LANGKAH 4: Solusi ideal ===
    # Benefit  = semakin besar semakin baik  → rating, jenis
    # Cost     = semakin kecil semakin baik  → harga, jarak
    is_benefit = np.array([False, False, True, True])   # [harga, jarak, rating, jenis]

    A_pos = np.where(is_benefit, V.max(axis=0), V.min(axis=0))
    A_neg = np.where(is_benefit, V.min(axis=0), V.max(axis=0))

    # === LANGKAH 5: Jarak Euclidean ===
    D_pos = np.sqrt(((V - A_pos) ** 2).sum(axis=1))
    D_neg = np.sqrt(((V - A_neg) ** 2).sum(axis=1))

    # === LANGKAH 6: Closeness Coefficient & ranking ===
    CC = D_neg / (D_pos + D_neg + 1e-10)

    results = []
    for i, f in enumerate(data):
        results.append({
            **f,
            "CC":      round(CC[i], 4),
            "D_pos":   round(D_pos[i], 4),
            "D_neg":   round(D_neg[i], 4),
            "R_harga":  round(R[i][0], 4),
            "R_jarak":  round(R[i][1], 4),
            "R_rating": round(R[i][2], 4),
            "R_jenis":  round(R[i][3], 4),
        })

    results.sort(key=lambda x: x["CC"], reverse=True)
    for rank, r in enumerate(results, start=1):
        r["rank"] = rank

    return results


# ─────────────────────────────────────────────
#  HEADER APLIKASI
# ─────────────────────────────────────────────
st.title("🍜 KostFood — Sistem Rekomendasi Makanan")
st.caption("Algoritma TOPSIS (Technique for Order Preference by Similarity to Ideal Solution) · Kelompok 9")
st.divider()

# ─────────────────────────────────────────────
#  SIDEBAR — INPUT PENGGUNA
# ─────────────────────────────────────────────
st.sidebar.header("⚙️ Preferensi Kamu")

# --- Filter ---
st.sidebar.subheader("Filter")
filter_jenis = st.sidebar.selectbox(
    "Jenis Makanan",
    ["Semua Jenis", "berat", "ringan", "cepat saji", "vegetarian"]
)
filter_harga = st.sidebar.selectbox(
    "Maks. Harga",
    {"Semua Harga": 999999, "≤ Rp 15.000": 15000, "≤ Rp 25.000": 25000, "≤ Rp 40.000": 40000}
)
# Ambil nilai numerik dari pilihan harga
harga_map = {"Semua Harga": 999999, "≤ Rp 15.000": 15000, "≤ Rp 25.000": 25000, "≤ Rp 40.000": 40000}
max_harga = harga_map[filter_harga]

# --- Bobot Kriteria ---
st.sidebar.divider()
st.sidebar.subheader("Bobot Kriteria (total = 100)")
st.sidebar.caption("Atur seberapa penting setiap kriteria bagimu.")

w_harga  = st.sidebar.slider("💰 Harga (cost — lebih murah lebih baik)",   0, 100, 30)
w_jarak  = st.sidebar.slider("📍 Jarak (cost — lebih dekat lebih baik)",   0, 100, 25)
w_rating = st.sidebar.slider("⭐ Rating (benefit — lebih tinggi lebih baik)", 0, 100, 30)
w_jenis  = st.sidebar.slider("🍽️ Jenis (benefit — sesuai preferensi)",      0, 100, 15)

total_bobot = w_harga + w_jarak + w_rating + w_jenis
st.sidebar.info(f"Total bobot: **{total_bobot}%**")

# Tombol jalankan
run_btn = st.sidebar.button("🔍 Cari Rekomendasi", type="primary", use_container_width=True)


# ─────────────────────────────────────────────
#  MAIN — HASIL
# ─────────────────────────────────────────────
if run_btn:
    # Validasi bobot
    if total_bobot == 0:
        st.warning("⚠️ Atur setidaknya satu bobot kriteria sebelum mencari.")
        st.stop()

    # Filter data
    filtered = [
        f for f in FOODS
        if (filter_jenis == "Semua Jenis" or f["jenis"] == filter_jenis)
        and f["harga"] <= max_harga
    ]

    if not filtered:
        st.error("😔 Tidak ada makanan yang sesuai filter. Coba perluas kriteria.")
        st.stop()

    # Normalisasi bobot agar total = 1
    total = w_harga + w_jarak + w_rating + w_jenis
    ranked = run_topsis(
        filtered,
        w_harga  / total,
        w_jarak  / total,
        w_rating / total,
        w_jenis  / total,
    )

    top = ranked[0]

    # ── Ringkasan ──
    st.subheader("🏆 Hasil Rekomendasi")
    st.metric("Rekomendasi Terbaik", top["nama"])
    col1, col2, col3 = st.columns(3)
    col1.metric("Skor CC", f"{top['CC']:.4f}")
    col2.metric("Harga",   f"Rp {top['harga']:,}".replace(",", "."))
    col3.metric("Jarak",   f"{top['jarak']} km")

    st.divider()

    # ── Tabel lengkap hasil ranking ──
    st.subheader("📊 Tabel Ranking Semua Alternatif")
    df_rank = pd.DataFrame([{
        "Rank":         r["rank"],
        "Nama Makanan": r["nama"],
        "Jenis":        r["jenis"],
        "Harga (Rp)":   r["harga"],
        "Jarak (km)":   r["jarak"],
        "Rating":       r["rating"],
        "CC":           r["CC"],
    } for r in ranked])
    st.dataframe(df_rank.set_index("Rank"), use_container_width=True)

    st.divider()

    # ── Detail langkah perhitungan TOPSIS ──
    st.subheader("📐 Detail Perhitungan TOPSIS")

    with st.expander("Langkah 1 — Matriks Keputusan (X)", expanded=False):
        st.markdown("""
        Setiap baris = satu alternatif makanan.
        Setiap kolom = satu kriteria (Harga, Jarak, Rating, Jenis).
        """)
        df_x = pd.DataFrame([{
            "Makanan":  r["nama"],
            "Harga":    r["harga"],
            "Jarak":    r["jarak"],
            "Rating":   r["rating"],
            "Jenis":    JENIS_SCORE[r["jenis"]],
        } for r in ranked], index=range(1, len(ranked)+1))
        st.dataframe(df_x, use_container_width=True)

    with st.expander("Langkah 2 — Normalisasi Matriks (R)", expanded=False):
        st.markdown(r"""
        Formula: $r_{ij} = \dfrac{x_{ij}}{\sqrt{\sum_{i} x_{ij}^2}}$

        Tujuan: menyamakan skala antar kriteria yang berbeda satuan.
        """)
        df_r = pd.DataFrame([{
            "Makanan":       r["nama"],
            "R_Harga":       r["R_harga"],
            "R_Jarak":       r["R_jarak"],
            "R_Rating":      r["R_rating"],
            "R_Jenis":       r["R_jenis"],
        } for r in ranked], index=range(1, len(ranked)+1))
        st.dataframe(df_r, use_container_width=True)

    with st.expander("Langkah 3 — Matriks Terbobot (V = R × W)", expanded=False):
        st.markdown(f"""
        Bobot yang digunakan (sudah dinormalisasi ke total 1):
        - 💰 Harga  : **{w_harga/total:.2f}**
        - 📍 Jarak  : **{w_jarak/total:.2f}**
        - ⭐ Rating : **{w_rating/total:.2f}**
        - 🍽️ Jenis  : **{w_jenis/total:.2f}**

        Formula: $v_{{ij}} = w_j \\times r_{{ij}}$
        """)

    with st.expander("Langkah 4 — Solusi Ideal A⁺ dan A⁻", expanded=False):
        st.markdown("""
        | Kriteria | Tipe   | A⁺ (terbaik)        | A⁻ (terburuk)       |
        |----------|--------|---------------------|---------------------|
        | Harga    | Cost   | nilai V terkecil    | nilai V terbesar    |
        | Jarak    | Cost   | nilai V terkecil    | nilai V terbesar    |
        | Rating   | Benefit| nilai V terbesar    | nilai V terkecil    |
        | Jenis    | Benefit| nilai V terbesar    | nilai V terkecil    |
        """)

    with st.expander("Langkah 5 & 6 — Jarak Euclidean & CC", expanded=False):
        st.markdown(r"""
        **D⁺** = jarak ke solusi ideal positif: $D_i^+ = \sqrt{\sum_j (v_{ij} - A_j^+)^2}$

        **D⁻** = jarak ke solusi ideal negatif: $D_i^- = \sqrt{\sum_j (v_{ij} - A_j^-)^2}$

        **CC** (Closeness Coefficient): $CC_i = \dfrac{D_i^-}{D_i^+ + D_i^-}$

        Semakin CC mendekati **1**, semakin baik alternatif tersebut.
        """)
        df_cc = pd.DataFrame([{
            "Makanan": r["nama"],
            "D⁺":      r["D_pos"],
            "D⁻":      r["D_neg"],
            "CC":      r["CC"],
            "Rank":    r["rank"],
        } for r in ranked]).set_index("Rank").sort_index()
        st.dataframe(df_cc, use_container_width=True)

    st.success(f"✅ Rekomendasi selesai! Ditemukan {len(ranked)} alternatif.")

else:
    # Tampilan awal sebelum tombol ditekan
    st.info("👈 Atur preferensi di sidebar, lalu tekan **Cari Rekomendasi**.")

    st.subheader("ℹ️ Tentang Metode TOPSIS")
    st.markdown("""
    **TOPSIS** (*Technique for Order Preference by Similarity to Ideal Solution*)
    adalah metode pengambilan keputusan multi-kriteria (MCDM) yang bekerja dengan cara:

    1. **Membuat matriks keputusan** dari semua alternatif dan kriteria.
    2. **Menormalisasi** matriks agar skala antar kriteria seragam.
    3. **Memberikan bobot** pada setiap kriteria sesuai preferensi pengguna.
    4. **Menentukan solusi ideal** — terbaik (A⁺) dan terburuk (A⁻).
    5. **Menghitung jarak** setiap alternatif ke A⁺ dan A⁻.
    6. **Menghitung CC** dan mengurutkan alternatif dari terbesar ke terkecil.

    ---
    **Kriteria yang digunakan:**

    | Kriteria | Tipe    | Keterangan                         |
    |----------|---------|------------------------------------|
    | Harga    | Cost    | Lebih murah → lebih baik           |
    | Jarak    | Cost    | Lebih dekat → lebih baik           |
    | Rating   | Benefit | Lebih tinggi → lebih baik          |
    | Jenis    | Benefit | Skor kecocokan jenis makanan       |
    """)

    st.subheader("📋 Data Makanan yang Tersedia")
    df_all = pd.DataFrame(FOODS)[["nama", "jenis", "harga", "jarak", "rating"]]
    df_all.columns = ["Nama Makanan", "Jenis", "Harga (Rp)", "Jarak (km)", "Rating"]
    st.dataframe(df_all, use_container_width=True, hide_index=True)
