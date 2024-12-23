import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Load the trained model
model = load_model('model.keras')

# Streamlit header
st.header('Stock Market Predictor')

# Input for stock symbol and date range
stock = st.text_input('Enter Stock Symbol', 'BK')
start = '1984-09-07'
end = '2024-09-07'

# Download stock data
data = yf.download(stock, start, end)

st.subheader('Stock Data')
st.write(data)

# Split data into training and testing sets
data_train = pd.DataFrame(data.Close[0: int(len(data) * 0.80)])
data_test = pd.DataFrame(data.Close[int(len(data) * 0.80): len(data)])

# Scaling data
scaler = MinMaxScaler(feature_range=(0, 1))
data_train_scale = scaler.fit_transform(data_train)

# Prepare test data
pas_100_days = data_train.tail(100)
data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
data_test_scale = scaler.transform(data_test)

# Plot Price vs MA100 with Event Annotations
st.subheader('Price vs MA100 with Event Annotations')
ma_100_days = data.Close.rolling(100).mean()
fig1 = plt.figure(figsize=(12, 8))

# Plot stock prices and moving average
plt.plot(data.Close, 'g', label='Stock Price')
plt.plot(ma_100_days, 'r', label='100-Day MA')

# Define significant events
events = {
    'COVID-19 Start': '2020-02-01',
    'COVID-19 Recovery': '2020-06-01',
    'US Election 2020': '2020-11-03',
    'Interest Rate Hike': '2022-03-16',
    'Ukraine Conflict Start': '2022-02-24'
}

# Annotate events on the graph
for event, date in events.items():
    event_date = pd.Timestamp(date)
    plt.axvline(event_date, color='blue', linestyle='--', alpha=0.7)
    plt.text(event_date, data.Close.min(), event, fontsize=9, color='blue', rotation=90)

# Add labels and legend
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Stock Price with Major Events Highlighted')
plt.legend()
st.pyplot(fig1)

# Prepare data for model prediction
X = []
y = []

for i in range(100, data_test_scale.shape[0]):
    X.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i, 0])

X, y = np.array(X), np.array(y)

# Predict using the loaded model
predict = model.predict(X)

# Inverse transform the predictions and actual prices
y = scaler.inverse_transform(y.reshape(-1, 1)).reshape(-1)
predict = scaler.inverse_transform(predict)

# Plot Original vs Predicted Prices
st.subheader('Original Price vs Predicted Price')
fig2 = plt.figure(figsize=(12, 8))
plt.plot(predict, 'r', label='Predicted Price')
plt.plot(y, 'g', label='Original Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('Original vs Predicted Stock Prices')
plt.legend()
st.pyplot(fig2)
