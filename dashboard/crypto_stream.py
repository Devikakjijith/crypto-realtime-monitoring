import time
import numpy as np
import pandas as pd
from datetime import datetime
from binance import Client
from influxdb_client import InfluxDBClient, Point, WriteOptions
from tensorflow.keras.models import load_model
import joblib


BINANCE_SYMBOL = "BTCUSDT"
INTERVAL = Client.KLINE_INTERVAL_1MINUTE
WINDOW_SIZE = 60   # how many minutes to cache for rolling metrics
INFLUX_URL = "https://us-east-1-1.aws.cloud2.influxdata.com/"
INFLUX_TOKEN = "fdfMIyO1iGd79rTF5nZG6qrpX51MtyU7qQvb_gnXaafvCGAZK0Rw6lkSmPJcflMDIypfAwJ0qq_dw38CqsLeGg=="
INFLUX_ORG = "della"
INFLUX_BUCKET = "crypto-realtime-data"

INF_CLIENT = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
BINANCE_CLIENT = Client()
WRITE_API = INF_CLIENT.write_api(write_options=WriteOptions(batch_size=1))

MODEL = load_model('lstm_model.h5')
SCALER = joblib.load('scaler.save')
TIME_STEP = 10

def fetch_bars(symbol=BINANCE_SYMBOL, interval=INTERVAL, window=WINDOW_SIZE):
    # fetches the latest k bars (candlesticks) for your symbol
    klines = BINANCE_CLIENT.get_klines(symbol=symbol, interval=interval, limit=window)
    df = pd.DataFrame(klines, columns=[
        "Open time", "Open", "High", "Low", "Close", "Volume", "Close time",
        "Quote asset volume", "Number of trades", "Taker buy base", "Taker buy quote", "Ignore"
    ])
    df["time"] = pd.to_datetime(df["Close time"], unit="ms")
    df["Open"] = df["Open"].astype(float)
    df["High"] = df["High"].astype(float)
    df["Low"] = df["Low"].astype(float)
    df["Close"] = df["Close"].astype(float)
    df["Volume"] = df["Volume"].astype(float)
    return df

def rolling_metrics(df):
    metrics = {}
    close = df["Close"]
    metrics['close'] = float(close.iloc[-1])
    metrics['open'] = float(df["Open"].iloc[-1])
    metrics['high'] = float(df["High"].max())
    metrics['low'] = float(df["Low"].min())
    metrics['volume'] = float(df["Volume"].sum())
    metrics['ma7'] = float(close.rolling(7).mean().iloc[-1])
    metrics['ma21'] = float(close.rolling(21).mean().iloc[-1])
    metrics['ema21'] = float(close.ewm(span=21).mean().iloc[-1])
    metrics['stddev'] = float(close.rolling(10).std().iloc[-1])
    metrics['pct_change_1m'] = float(close.pct_change(1).iloc[-1]) * 100
    metrics['pct_change_10m'] = float(close.pct_change(10).iloc[-1]) * 100
    metrics['trade_count'] = int(df["Number of trades"].astype(int).sum())
    return metrics

def model_prediction(close_prices):
    if len(close_prices) < TIME_STEP:
        return None
    scaled = SCALER.transform(close_prices[-TIME_STEP:].reshape(-1, 1))
    X_input = np.array([scaled[:, 0]]).reshape(1, TIME_STEP, 1)
    pred_scaled = MODEL.predict(X_input)
    pred = SCALER.inverse_transform(pred_scaled)
    return float(pred[0, 0])

while True:
    df = fetch_bars()
    metrics = rolling_metrics(df)
    prediction = model_prediction(df['Close'].values)
    now = datetime.utcnow()

    point = Point("crypto") \
        .tag("symbol", BINANCE_SYMBOL) \
        .field("close", metrics["close"]) \
        .field("open", metrics["open"]) \
        .field("high", metrics["high"]) \
        .field("low", metrics["low"]) \
        .field("volume", metrics["volume"]) \
        .field("ma7", metrics["ma7"]) \
        .field("ma21", metrics["ma21"]) \
        .field("ema21", metrics["ema21"]) \
        .field("stddev", metrics["stddev"]) \
        .field("pct_change_1m", metrics["pct_change_1m"]) \
        .field("pct_change_10m", metrics["pct_change_10m"]) \
        .field("trade_count", metrics["trade_count"])

    if prediction is not None:
        point = point.field("lstm_prediction", prediction)
    
    point = point.time(now)

   
    try:
        WRITE_API.write(bucket=INFLUX_BUCKET, record=point)
        print(f"Sent data at {now}, Close: {metrics['close']}, Prediction: {prediction}")
    except Exception as e:
        print(f"InfluxDB write error: {e}")

    time.sleep(30)   