import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

model = load_model('D:\\Projects\\stock_price_prediction\\model.keras')

st.header('Stock Market Predictor')

stock = st.text_input('Enter Stock Symbol', 'BK')
start = '1984-09-07'
end = '2024-09-07'

data = yf.download(stock, start, end)

st.subheader('Stock Data')
st.write(data)

data_train = pd.DataFrame(data.Close[0: int(len(data) * 0.80)])
data_test = pd.DataFrame(data.Close[int(len(data) * 0.80): len(data)])

# Fit scaler on training data
scaler = MinMaxScaler(feature_range=(0, 1))
data_train_scaled = scaler.fit_transform(data_train)  # Fitting scaler on training data only

# Prepare data for testing
past_100_days = data_train.tail(100)
data_test = pd.concat([past_100_days, data_test], ignore_index=True)
data_test_scaled = scaler.transform(data_test)  # Transforming test data with the fitted scaler

# Plot MA50
st.subheader('Price vs MA50')
ma_50_days = data.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(8, 6))
plt.plot(ma_50_days, 'r', label='50-Day MA')
plt.plot(data.Close, 'g', label='Stock Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig1)

# Plot MA50 vs MA100
st.subheader('Price vs MA50 vs MA100')
ma_100_days = data.Close.rolling(100).mean()
fig2 = plt.figure(figsize=(8, 6))
plt.plot(ma_50_days, 'r', label='50-Day MA')
plt.plot(ma_100_days, 'b', label='100-Day MA')
plt.plot(data.Close, 'g', label='Stock Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)

# Plot MA100 vs MA200
st.subheader('Price vs MA100 vs MA200')
ma_200_days = data.Close.rolling(200).mean()
fig3 = plt.figure(figsize=(8, 6))
plt.plot(ma_100_days, 'r', label='100-Day MA')
plt.plot(ma_200_days, 'b', label='200-Day MA')
plt.plot(data.Close, 'g', label='Stock Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig3)

# Prepare the data for prediction
X = []
y = []

for i in range(100, data_test_scaled.shape[0]):
    X.append(data_test_scaled[i-100:i])
    y.append(data_test_scaled[i, 0])

X, y = np.array(X), np.array(y)

# Predict
predict = model.predict(X)

# Inverse scale the predictions and actual values
y = scaler.inverse_transform(y.reshape(-1, 1)).reshape(-1)  # Correctly inverse transform y
predict = scaler.inverse_transform(predict)  # Correctly inverse transform predictions

# Plot Original Price vs Predicted Price
st.subheader('Original Price vs Predicted Price')
fig4 = plt.figure(figsize=(8, 6))
plt.plot(predict, 'r', label='Predicted Price')
plt.plot(y, 'g', label='Original Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig4)
