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
