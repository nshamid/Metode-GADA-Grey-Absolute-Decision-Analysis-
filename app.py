import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# -----------------------------------------------------------------------------
# KONFIGURASI HALAMAN
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="SPK Metode GADA - MOBA Games",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# DATA DAN FUNGSI PERHITUNGAN
# -----------------------------------------------------------------------------

@st.cache_data
def load_data():
    # Data Kriteria
    data_kriteria = {
        'Kode': [f'C{i}' for i in range(1, 14)],
        'Kriteria': [
            'Grafis & desain', 'Keseimbangan gameplay', 'Popularitas & komunitas',
            'Stabilitas server/koneksi', 'Fitur & mode', 'Kemudahan kontrol & UI',
            'Tingkat kompetitif', 'Sistem pembayaran', 'Ukuran aplikasi',
            'Lancar di HP menengah', 'Fairness F2P', 'Sosialisasi / menambah kenalan',
            'Rekomendasi untuk pemula'
        ],
        'Jenis': ['Benefit'] * 13
    }
    df_kriteria_info = pd.DataFrame(data_kriteria)

    # Data Alternatif
    data_alternatif = {
        'Kode': [f'A{i}' for i in range(1, 11)],
        'Alternatif': [
            'Mobile Legends: Bang Bang', 'Honor of Kings', 'Garena AOV',
            'League of Legends: Wild Rift', 'Pokémon Unite', 'Heroes Evolved',
            'Onmyoji Arena', 'Lokapala', 'Vainglory (IOS)', 'Marvel Super War'
        ]
    }
    df_alternatif_info = pd.DataFrame(data_alternatif)

    # Data Penilaian Awal (Raw Data)
    data_awal = {
        'Alternatif': [f'A{i}' for i in range(1, 11)],
        'C1': [4.207949381, 4.641588834, 3.55655882, 4.472135955, 2.828427125, 4, 4, 3, 2, 4],
        'C2': [3.78329541, 3.914867641, 3.55655882, 3.464101615, 2.449489743, 4, 4, 3, 2, 3.634241186],
        'C3': [4.295307446, 3.301927249, 2.91295063, 4, 2.449489743, 2, 3, 3, 3, 3.301927249],
        'C4': [3.590914704, 3.107232506, 2.91295063, 3.872983346, 2.828427125, 4, 4, 3, 3, 3.634241186],
        'C5': [4.278081229, 4.641588834, 3.13016816, 4.472135955, 2.449489743, 4, 4, 4, 3, 3.634241186],
        'C6': [4.344938193, 4.641588834, 2.91295063, 4.472135955, 2.828427125, 3, 4, 4, 2, 3.634241186],
        'C7': [4.257678109, 3.914867641, 3.55655882, 4.472135955, 2, 4, 4, 4, 2, 3.634241186],
        'C8': [4.065840375, 4.641588834, 3.55655882, 3, 2.828427125, 4, 4, 4, 3, 3.634241186],
        'C9': [4.0622705, 2.884999141, 2.632148026, 3.872983346, 2.828427125, 4, 4, 4, 3, 3.634241186],
        'C10': [3.650022501, 4.844999141, 2.91295063, 2.828427125, 2.449489743, 3, 4, 4, 3, 3.634241186],
        'C11': [3.799034928, 4.641588834, 3.080070098, 3.872983346, 2.449489743, 4, 4, 4, 3, 3.634241186],
        'C12': [3.702746903, 2.620741394, 1.861209718, 1.44213562, 2.449489743, 2, 3, 3, 3, 3.634241186],
        'C13': [3.456730406, 2.620741394, 2.51486659, 1.732050808, 2.828427125, 3, 3, 2, 3, 4]
    }
    df_awal = pd.DataFrame(data_awal)
    
    return df_kriteria_info, df_alternatif_info, df_awal

def calculate_gada(df_awal):
    # Pisahkan numerik
    df_alternatif_col = df_awal[['Alternatif']]
    df_kriteria = df_awal.drop(columns=['Alternatif'])
    
    # 1. Menghitung GM Max
    gm_max = df_kriteria.max()
    
    # 2. Normalisasi
    df_normalisasi = df_kriteria.div(gm_max, axis=1)
    df_hasil_normalisasi = pd.concat([df_alternatif_col, df_normalisasi], axis=1)
    
    # 3. Hitung Si
    df_calc = df_normalisasi.copy()
    si_values = (df_calc['C1'] / 2) + (df_calc['C2'] + df_calc['C12']) + (df_calc['C13'] / 2)
    df_si = pd.DataFrame({'Alternatif': df_awal['Alternatif'], 'Si': si_values})
    
    # 4. Matriks Perbandingan Berpasangan (Epsilon)
    n = len(si_values)
    epsilon_matrix = np.zeros((n, n))
    si_numpy = si_values.values
    
    for i in range(n):
        for j in range(n):
            Si = si_numpy[i]
            Sj = si_numpy[j]
            numerator = 1 + Si + Sj
            denominator = 1 + Sj + (Si - Sj)
            epsilon_matrix[i, j] = numerator / denominator
            
    alternatif_labels = df_awal['Alternatif'].values
    df_epsilon = pd.DataFrame(epsilon_matrix, index=alternatif_labels, columns=alternatif_labels)
    
    # 5. Bobot Akhir Alternatif
    hasil_kali = df_epsilon.prod(axis=1)
    gm_baris = hasil_kali**(1/n)
    total_gm_baris = gm_baris.sum()
    bobot_alternatif = gm_baris / total_gm_baris
    
    df_final_res = pd.DataFrame({
        'Alternatif': alternatif_labels,
        'Bobot Akhir': bobot_alternatif
    })
    
    return gm_max, df_hasil_normalisasi, df_si, df_epsilon, df_final_res

# Load Data
df_kriteria_info, df_alternatif_info, df_awal = load_data()
gm_max, df_normalisasi, df_si, df_epsilon, df_result = calculate_gada(df_awal)

# Gabungkan hasil ranking dengan nama lengkap alternatif
df_ranking = pd.merge(df_result, df_alternatif_info, left_on='Alternatif', right_on='Kode')

# Rename kolom sesuai permintaan
df_ranking = df_ranking[['Kode', 'Alternatif_y', 'Bobot Akhir']].rename(columns={
    'Alternatif_y': 'Nama Game',
    'Bobot Akhir': 'Nilai Bobot'
})
df_ranking = df_ranking.sort_values(by='Nilai Bobot', ascending=False).reset_index(drop=True)
df_ranking.index += 1  # Ranking mulai dari 1

# -----------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# -----------------------------------------------------------------------------
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih Halaman:", ["Metode GADA", "Hasil Perangkingan"])

st.sidebar.markdown("---")
st.sidebar.caption("© 2025 Kelompok 5 - TI Bilingual")

# -----------------------------------------------------------------------------
# HALAMAN 1: METODE GADA
# -----------------------------------------------------------------------------
if page == "Metode GADA":
    st.title("📚 Metode GADA (Grey Absolute Decision Analysis)")
    
    tab1, tab2 = st.tabs(["ℹ️ Informasi Umum", "🧮 Langkah Perhitungan"])
    
    # --- TAB 1: INFORMASI UMUM ---
    with tab1:
        st.subheader("Studi Kasus")
        st.markdown("""
        > **Pemilihan Game MOBA Mobile Terbaik**
        >
        > Banyaknya pilihan game MOBA Mobile dengan karakteristik dan fitur yang beragam membuat 
        > mahasiswa sering kesulitan menentukan game mana yang paling sesuai untuk dimainkan bersama teman.
        > Sistem ini membantu menentukan rekomendasi terbaik berdasarkan berbagai kriteria teknis dan sosial.
        """)
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.subheader("Data Alternatif")
            st.dataframe(df_alternatif_info, use_container_width=True, hide_index=True)
            
        with col_b:
            st.subheader("Data Kriteria")
            st.dataframe(df_kriteria_info, use_container_width=True, hide_index=True)

        st.subheader("Data Penilaian Awal")
        st.caption("Nilai numerik untuk setiap alternatif terhadap kriteria")
        st.dataframe(df_awal, use_container_width=True)

    # --- TAB 2: LANGKAH PERHITUNGAN ---
    with tab2:
        st.header("Langkah-langkah Perhitungan")
        
        # Step 1
        with st.expander("Tahap 1: Menghitung Nilai Geometric Mean (GM Max)", expanded=False):
            st.write("Mencari nilai maksimum geometric mean dari setiap kriteria.")
            st.dataframe(gm_max.to_frame(name="GM Max").transpose(), use_container_width=True)
            
        # Step 2
        with st.expander("Tahap 2: Normalisasi Matriks", expanded=False):
            st.write("Membagi nilai setiap kriteria dengan GM Max-nya.")
            # HANYA format kolom selain 'Alternatif'
            st.dataframe(df_normalisasi.style.format("{:.4f}", subset=df_normalisasi.columns.drop("Alternatif")), use_container_width=True)
            
        # Step 3
        with st.expander("Tahap 3: Menghitung Nilai Si", expanded=False):
            st.write("Menghitung komponen Si berdasarkan rumus yang ditentukan (kombinasi kolom C1, C2, C12, C13).")
            st.latex(r"Si = \frac{1}{2}C1 + (C2 - C12) + \frac{1}{2}C13")
            st.dataframe(df_si.style.format({"Si": "{:.4f}"}), use_container_width=True)
            
        # Step 4
        with st.expander("Tahap 4: Matriks Perbandingan Berpasangan (Epsilon)", expanded=False):
            st.write("Menghitung nilai perbandingan antar alternatif.")
            st.latex(r"\epsilon_{ij} = \frac{1 + Si + Sj}{1 + Sj + (Si - Sj)}")
            st.dataframe(df_epsilon.style.format("{:.4f}"), use_container_width=True)
            
        # Step 5
        with st.expander("Tahap 5: Perhitungan Bobot Akhir", expanded=True):
            st.write("Menghitung hasil kali baris, rata-rata geometrik baris, dan normalisasi bobot.")
            st.dataframe(df_result.style.format({"Bobot Akhir": "{:.8f}"}), use_container_width=True)

# -----------------------------------------------------------------------------
# HALAMAN 2: HASIL PERANGKINGAN
# -----------------------------------------------------------------------------
elif page == "Hasil Perangkingan":
    st.title("🏆 Hasil Perangkingan")
    
    # 3 Tab Terpisah: Tabel, Grafik, Credit
    tab_rank, tab_graph, tab_credit = st.tabs(["📋 Tabel Perangkingan", "📊 Grafik Visualisasi", "👥 Credit"])
    
    # --- TAB 1: TABEL PERANGKINGAN ---
    with tab_rank:
        st.subheader("Peringkat & Skor")
        
        # Highlight juara 1 dengan warna pastel agak gelap (Sage Green)
        def highlight_top(s):
            # Hex color #a3cfbb is a darker pastel green (Sage)
            return ['background-color: #a3cfbb' if s.name == 1 else '' for _ in s]

        st.dataframe(
            df_ranking.style.format({"Nilai Bobot": "{:.8f}"}).apply(highlight_top, axis=1),
            use_container_width=True,
            height=500
        )
        
        st.markdown("---")
        
        # Kesimpulan cards
        best_alt = df_ranking.iloc[0]
        worst_alt = df_ranking.iloc[-1]
        
        c1, c2, c3 = st.columns(3)
        
        with c1:
            st.success(f"**Rekomendasi Utama**\n\n### 🥇 {best_alt['Nama Game']}\nSkor: {best_alt['Nilai Bobot']:.6f}")
            
        with c2:
            st.warning(f"**Peringkat Kedua**\n\n### 🥈 {df_ranking.iloc[1]['Nama Game']}\nSkor: {df_ranking.iloc[1]['Nilai Bobot']:.6f}")

        with c3:
            st.error(f"**Peringkat Terbawah**\n\n### 📉 {worst_alt['Nama Game']}\nSkor: {worst_alt['Nilai Bobot']:.6f}")

    # --- TAB 2: GRAFIK VISUALISASI ---
    with tab_graph:
        st.subheader("Visualisasi Perbandingan Skor")
        
        fig = px.bar(
            df_ranking.sort_values(by="Nilai Bobot", ascending=True), 
            x="Nilai Bobot", 
            y="Nama Game", 
            orientation='h',
            text_auto='.4f',
            title="Perbandingan Nilai Bobot Alternatif",
            color="Nilai Bobot",
            color_continuous_scale="Viridis"
        )
        fig.update_layout(showlegend=False, xaxis_title="Nilai Bobot", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    # --- TAB 3: CREDIT ---
    with tab_credit:
        st.header("Credit Pengembang")
        
        # Warna background pastel agak gelap (Blue Grey)
        st.markdown("""
        <div style="background-color: #cfd8dc; padding: 20px; border-radius: 10px; border-left: 5px solid #455a64; color: #000000;">
            <h3>Kelompok 5 - Teknik Informatika Bilingual 2023</h3>
            <p><strong>Mata Kuliah:</strong> Sistem Pendukung Keputusan<br>
            <strong>Dosen Pengampu:</strong> Yunita, S.Si., M.Cs.</p>
        </div>
        <br>
        """, unsafe_allow_html=True)
        
        st.subheader("Anggota Kelompok")
        
        # Data Anggota
        anggota = [
            {"Nama": "Fransisca Stevanie Ekawati", "NIM": "09021382328127"},
            {"Nama": "Nabilah Shamid", "NIM": "09021382328147"},
            {"Nama": "Saravina Zharfa Kelana Putri", "NIM": "09021382328149"},
            {"Nama": "Azka Hukma Tsabita", "NIM": "09021382328159"},
            {"Nama": "Afny Chiara Wildani Nst", "NIM": "09021382328167"}
        ]
        
        df_anggota = pd.DataFrame(anggota)
        st.table(df_anggota)
