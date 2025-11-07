import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from binance import Client

client = Client()

@st.cache_resource(show_spinner=False)
def load_model_and_scaler():
    model = load_model('lstm_model.h5')
    scaler = joblib.load('scaler.save')
    return model, scaler

def fetch_live_binance_data(symbol="BTCUSDT", window=60):
    klines = client.get_klines(symbol=symbol, interval=Client.KLINE_INTERVAL_1DAY, limit=window)
    closes = np.array([float(k[4]) for k in klines])
    return closes

def predict_next_price(prices, model, scaler, time_step=10):
    scaled = scaler.transform(prices.reshape(-1,1))
    X = np.array([scaled[-time_step:,0]])
    X = X.reshape((X.shape[0], X.shape[1], 1))
    pred_scaled = model.predict(X)
    pred = scaler.inverse_transform(pred_scaled)
    return pred[0][0]

st.title("Live LSTM BTCUSDT Price Prediction")

symbol = st.text_input("Enter Binance symbol (for BTCUSDT live demo, use default):", "BTCUSDT")
model, scaler = load_model_and_scaler()

if st.button("Predict Next Close Price"):
    try:
        prices = fetch_live_binance_data(symbol)
        if len(prices) < 10:
            st.warning("Not enough data to predict")
        else:
            prediction = predict_next_price(prices, model, scaler)
            st.write(f"Predicted next close price for {symbol}: {prediction:.4f}")
            st.line_chart(np.append(prices, prediction))
    except Exception as e:
        st.error(f"Error fetching or predicting: {e}")
