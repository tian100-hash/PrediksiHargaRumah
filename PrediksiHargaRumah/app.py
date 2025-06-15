import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# =======================================
# PAGE CONFIG (MUST BE AT THE TOP)
# =======================================
st.set_page_config(page_title="Prediksi Harga Rumah", layout="centered")

# =======================================
# LOAD DATASET & MODEL
# =======================================
@st.cache_data
def load_data():
    df = pd.read_csv("housing.csv")
    df.columns = df.columns.str.strip()  # Hapus spasi berlebih jika ada
    return df

data = load_data()

@st.cache_data
def train_model(df):
    df['LogSalePrice'] = np.log1p(df['SalePrice'])
    features = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 
                '1stFlrSF', 'FullBath', 'YearBuilt']

    df = df.dropna(subset=features + ['SalePrice'])
    df = df[features + ['SalePrice', 'LogSalePrice']]  # Pastikan semua kolom yang dibutuhkan tersedia

    X = df[features]
    y = df['LogSalePrice']

    model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)
    model.fit(X, y)
    joblib.dump(model, "xgb_model.pkl")
    return model, features

# Load or train model
try:
    model = joblib.load("xgb_model.pkl")
    features = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 
                '1stFlrSF', 'FullBath', 'YearBuilt']
except:
    model, features = train_model(data)

# =======================================
# DASHBOARD
# =======================================

st.title("🏠 Prediksi Harga Rumah dengan XGBoost")
st.markdown("Masukkan data rumah untuk memprediksi harga penjualannya.")

# Input form
user_input = {}
user_input['OverallQual'] = st.slider("Overall Quality (1-10)", 1, 10, 5)
user_input['GrLivArea'] = st.number_input("Ground Living Area (sq ft)", 200, 6000, 1500)
user_input['GarageCars'] = st.selectbox("Jumlah Mobil di Garasi", [0, 1, 2, 3, 4])
user_input['TotalBsmtSF'] = st.number_input("Total Basement SF", 0, 5000, 800)
user_input['1stFlrSF'] = st.number_input("First Floor SF", 200, 4000, 1200)
user_input['FullBath'] = st.selectbox("Jumlah Kamar Mandi", [0, 1, 2, 3])
user_input['YearBuilt'] = st.number_input("Tahun Dibangun", 1900, 2024, 2000)

input_df = pd.DataFrame([user_input])

if st.button("Prediksi Harga"):
    log_price_pred = model.predict(input_df)[0]
    price_pred = np.expm1(log_price_pred)
    st.subheader("💰 Estimasi Harga Rumah:")
    st.success(f"USD {price_pred:,.2f}")
    st.caption("Catatan: harga bersifat estimasi berdasarkan fitur yang dimasukkan.")

# =======================================
# ANALISIS TAMBAHAN
# =======================================

st.header("📊 Analisis Data")
if st.checkbox("Tampilkan Distribusi SalePrice"):
    fig, ax = plt.subplots()
    sns.histplot(data['SalePrice'], kde=True, ax=ax)
    st.pyplot(fig)

if st.checkbox("Tampilkan Pentingnya Fitur"):
    importance = pd.Series(model.feature_importances_, index=features).sort_values()
    st.bar_chart(importance)

if st.checkbox("Tampilkan Korelasi dengan SalePrice"):
    available_columns = [col for col in features + ['SalePrice'] if col in data.columns]
    corr = data[available_columns].corr()['SalePrice'].sort_values(ascending=False)
    st.write(corr)

if st.checkbox("Tampilkan Data Mentah"):
    st.dataframe(data.head(10)[[col for col in features + ['SalePrice'] if col in data.columns]])

st.markdown("---")
st.caption("Dibuat Oleh Christian")