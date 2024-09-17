import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

model = load_model('D:\\Projects\\stockpro\\model.keras')

st.header('Stock Market Predictor')

stock = st.text_input('Enter Stock Symbol', 'BK')
start = '1984-09-07'
end = '2024-09-07'

data = yf.download(stock, start, end)

st.subheader('Stock Data')
st.write(data)

data_train = pd.DataFrame(data.Close[0: int(len(data) * 0.80)])
data_test = pd.DataFrame(data.Close[int(len(data) * 0.80): len(data)])

scaler = MinMaxScaler(feature_range=(0, 1))
data_train_scale = scaler.fit_transform(data_train)

pas_100_days = data_train.tail(100)
data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
data_test_scale = scaler.transform(data_test)

st.subheader('Price vs MA100')
ma_100_days = data.Close.rolling(100).mean()
fig1 = plt.figure(figsize=(8, 6))
plt.plot(ma_100_days, 'r', label='100-Day MA')
plt.plot(data.Close, 'g', label='Stock Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig1)

X = []
y = []

for i in range(100, data_test_scale.shape[0]):
    X.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i, 0])

X, y = np.array(X), np.array(y)

predict = model.predict(X)

y = scaler.inverse_transform(y.reshape(-1, 1)).reshape(-1)
predict = scaler.inverse_transform(predict)

st.subheader('Original Price vs Predicted Price')
fig2 = plt.figure(figsize=(8, 6))
plt.plot(predict, 'r', label='Predicted Price')
plt.plot(y, 'g', label='Original Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)
