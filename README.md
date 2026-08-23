# Retail PO Intelligence — Revenue & Pareto Edition

Dashboard Streamlit untuk menganalisis transaksi mentah PO, barang datang, dan penjualan.

## Fitur utama

- Upload 3 dataset wajib: PO, Pembelian/Barang Datang, Penjualan.
- Upload snapshot stok opsional.
- Dashboard bar chart PO → Datang → Terjual.
- KPI revenue, fill rate, sell-through, overstock, understock, dan kandidat naik order.
- Filter Supplier dan Subdept; default daftar hanya **Top 20 berdasarkan revenue**.
- Revenue Ranking Top 20 untuk Supplier, Subdept, dan Product.
- Pareto Procurement 80/20 untuk Supplier dan Product.
- Membandingkan **Revenue Share** dengan **Order Share**.
- Flag `CORE 80% - ORDER KURANG`.
- Flag `NON-CORE - ORDER TINGGI`.
- Basis Order dapat dipilih antara **Nilai PO** dan **Qty PO**.
- Threshold mismatch share order vs revenue dapat diubah dari sidebar.
- Supplier lead time, fill rate, median lead time, dan P90 lead time.
- Safety Stock dan Reorder Point berbasis demand harian + variasi lead time.
- Days/Months of Stock dan estimasi recommended PO aktif bila snapshot stok tersedia.
- Export hasil analisis ke Excel termasuk sheet Pareto Supplier/Product dan Revenue Ranking.

## Format upload

### PO

```csv
no_po,tgl_po,sku,nama_barang,supplier,subdept,hrg_beli,qty,total
```

Alias `tgl` otomatis diterima sebagai `tgl_po`.

### Pembelian / Barang Datang

```csv
no_faktur_beli,no_po,tgl_beli,sku,nama_barang,supplier,subdept,hrg_beli,qty,total
```

Alias `tgl` otomatis diterima sebagai `tgl_beli`.

### Penjualan

```csv
tgl_jual,sku,nama_barang,supplier,subdept,hrg_jual,qty,total
```

Alias `tgl` otomatis diterima sebagai `tgl_jual`.

### Snapshot Stok — opsional

```csv
tanggal,sku,stok_akhir
```

atau multi lokasi:

```csv
tanggal,lokasi,sku,stok_akhir
```

## Menjalankan aplikasi

Windows: jalankan `install.bat` satu kali, lalu `run_dashboard.bat`.

Atau lewat terminal:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Interpretasi Pareto

Revenue diurutkan dari terbesar sampai kontribusi kumulatif mencapai 80%. Item yang membawa kumulatif melewati 80% tetap dimasukkan sebagai kelompok Core 80%.

`Order Index = Order Share / Revenue Share`

Default threshold mismatch 25%:

- `< 0.75` pada Core 80% → `CORE 80% - ORDER KURANG`.
- `> 1.25` pada Non-core → `NON-CORE - ORDER TINGGI`.

Default basis order adalah **Nilai PO** karena lebih comparable dengan revenue. User dapat mengganti ke Qty PO.

Pareto adalah indikator alokasi purchasing. Keputusan PO final tetap harus mempertimbangkan current stock, outstanding PO valid, cancel PO, MOQ, pack size, lead time, dan safety stock.

## V3 bugfix - StreamlitDuplicateElementId
Seluruh `st.plotly_chart()` sekarang memiliki `key=` unik. Ini mencegah error pada Streamlit versi baru ketika dua chart memiliki parameter/figure yang identik pada satu rerun.

## V3 FIX 2
- Semua `st.plotly_chart()` tetap menggunakan key unik.
- Revenue Ranking sekarang mendeduplikasi daftar kolom tabel sebelum dikirim ke Streamlit/PyArrow.
- Memperbaiki error `ValueError: Duplicate column names found` pada level Supplier/Subdept.

## Update V4 — Filter Global Subdept

- Subdept menjadi filter global utama: `SEMUA SUBDEPT` atau satu subdept tertentu.
- Seluruh tab mengikuti scope Subdept aktif.
- Supplier otomatis menjadi Top 20 Revenue di dalam Subdept terpilih.
- Opsi `Tampilkan semua supplier` tetap tersedia.
- Pilihan supplier otomatis di-reset saat berpindah Subdept untuk mencegah filter silang yang salah.

## V5 Performance Cache Edition

Optimasi utama:

- CSV mentah hanya diparse/clean ketika file berubah.
- Hasil preprocessing disimpan temporary pada `st.session_state`.
- Raw transaction dikompresi menjadi fact table SKU/Supplier/Subdept untuk filter interaktif.
- `monthly_flow`, lead time, demand statistics, dan safety-stock base dihitung sekali.
- Product metrics disimpan per window risiko (1–6 bulan) setelah pertama kali digunakan.
- Export Excel dibuat **on demand**, bukan pada setiap rerun/filter.
- Tombol **Clear Temporary Cache** tersedia di sidebar.

Cache bersifat temporary per Streamlit session dan akan hilang ketika session/server di-reset.


## V6 — Gemini AI Analyst

Fitur AI menggunakan official Google Gen AI Python SDK (`google-genai`).

### Cara pakai

1. Upload dataset PO, Pembelian, dan Penjualan seperti biasa.
2. Pilih Subdept / Supplier.
3. Di sidebar, masukkan **Gemini API Key** pada field password.
4. Default model: `gemini-3.7-flash`.
5. Buka tab **AI Analyst**.
6. Klik **Test Gemini API**.
7. Gunakan quick analysis atau ketik pertanyaan sendiri.

Contoh:
- `Kenapa kategori ini berpotensi overstock?`
- `Supplier mana yang revenue-nya tinggi tetapi ordernya kurang?`
- `Produk mana yang harus diprioritaskan untuk PO berikutnya?`
- `Jelaskan 10 product understock paling berisiko.`
- `Analisis SKU 33060001.`

### Security / privacy

- API key yang diketik tidak ditulis ke file.
- Aplikasi juga mendukung `GEMINI_API_KEY` dari environment variable atau Streamlit secrets.
- Raw CSV tidak dikirim seluruhnya ke Gemini.
- Saat user bertanya, aplikasi mengirim analytical context yang dibatasi: KPI, monthly flow,
  ranking, Pareto, risk tables, supplier performance, Safety Stock/ROP, dan product rows yang
  relevan dengan pertanyaan.
- Karena Gemini adalah layanan eksternal, analytical context yang dikirim akan diproses oleh
  layanan Gemini API sesuai pengaturan akun/API Google Anda.

### Production deployment

Lebih aman menyimpan key di environment:

Windows PowerShell:
```powershell
$env:GEMINI_API_KEY="YOUR_KEY"
streamlit run app.py
```

atau Streamlit secrets:

```toml
GEMINI_API_KEY="YOUR_KEY"
```

User masih dapat mengoverride key server dengan field password di sidebar.
