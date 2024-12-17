import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.preprocessing import ColumnTransformer
import pickle

# Modeli ve diğer gerekli nesneleri yükle
model = joblib.load('eniyi.joblib')
with open('scaler_params.pkl', 'rb') as f:
    X_train_mean, X_train_scale = pickle.load(f)

# OneHotEncoder'ı başlat ve handle_unknown='ignore' parametresini kullan
encoder = OneHotEncoder(handle_unknown='ignore') 

# Streamlit uygulaması başlığı
st.title('Karpuz Fiyat Tahmini')

# Kullanıcıdan girdi al
boyut = st.selectbox('Boyut', ['küçük', 'orta', 'büyük'])
agirlik = st.number_input('Ağırlık (kg)', min_value=2.0, max_value=10.0, value=5.0, step=0.1)
renk = st.selectbox('Renk', ['açık yeşil', 'koyu yeşil'])
seker_orani = st.number_input('Şeker Oranı', min_value=8.0, max_value=15.0, value=10.0, step=0.1)
hasat_tarihi = st.date_input('Hasat Tarihi')

# Tarih bilgisini işleme
hasat_ayi = hasat_tarihi.month
hasat_gunu = hasat_tarihi.day

# ColumnTransformer kullanarak sütunları dönüştür
ct = ColumnTransformer(
    [('encoder', OneHotEncoder(handle_unknown='ignore'), ['boyut', 'renk']),
     ('scaler', StandardScaler(), ['agirlik', 'seker_orani', 'hasat_ayi', 'hasat_gunu'])],
    remainder='passthrough'
)

# Girdi verilerini DataFrame'e dönüştür
input_df = pd.DataFrame({'boyut': [boyut], 'renk': [renk], 'agirlik': [agirlik], 
                          'seker_orani': [seker_orani], 'hasat_ayi': [hasat_ayi], 
                          'hasat_gunu': [hasat_gunu]})

# Girdi verilerini dönüştür
transformed_input = ct.fit_transform(input_df)

# Tahmin yap
if st.button('Tahmin Et'):
    tahmin = model.predict(transformed_input)[0]
    st.success(f'Tahmini Karpuz Fiyatı: {tahmin:.2f} TL')