# 🏡 Prediksi Harga Rumah dengan XGBoost & Streamlit

Aplikasi dashboard interaktif untuk memprediksi harga rumah berdasarkan fitur penting menggunakan algoritma XGBoost.

## 🚀 Fitur
- Prediksi harga rumah berdasarkan input pengguna.
- Visualisasi distribusi harga rumah.
- Korelasi fitur terhadap harga.
- Pentingnya fitur (feature importance).
- Tampilan data mentah.

## 🧠 Algoritma
Menggunakan model regresi `XGBoostRegressor` untuk memprediksi `SalePrice` berdasarkan transformasi logaritmik (`LogSalePrice`) dengan fitur:
- `OverallQual`
- `GrLivArea`
- `GarageCars`
- `TotalBsmtSF`
- `1stFlrSF`
- `FullBath`
- `YearBuilt`

## 📁 Struktur Proyek
PrediksiHargaRumah/
│
├── app.py # Aplikasi Streamlit
├── housing.csv # Dataset (bisa diganti)
├── xgb_model.pkl # Model yang telah dilatih
├── requirements.txt # Dependensi Python
└── README.md # Dokumentasi proyek

## 🛠️ Cara Menjalankan

1. **Clone Repository**
```bash
git clone https://github.com/username/PrediksiHargaRumah.git
cd PrediksiHargaRumah

2. **Install Depedency**
pip install -r requirements.txt

3. **Jalankan Streamlit**
streamlit run app.py


---


❤️ Author
Dibuat oleh Christian - Powered by Streamlit & XGBoost
### ✅ Selanjutnya:
Jika belum punya repo:

1. Buka: https://github.com/new
2. Buat repo: `PrediksiHargaRumah`
3. Jangan centang "Initialize this repository with a README"
4. Lalu lakukan push via terminal:

```bash
git init
git add .
git commit -m "Initial commit for house price prediction dashboard"
git remote add origin https://github.com/username/PrediksiHargaRumah.git
git branch -M main
git push -u origin main
