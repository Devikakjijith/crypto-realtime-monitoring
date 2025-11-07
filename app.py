import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import streamlit as st

@st.cache_data
def load_data():
    df = pd.read_csv('BTCUSDT_binance.csv')    
    df['date'] = pd.to_datetime(df['Date'])
    df = df.set_index('date')
    all_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
    df = df.reindex(all_idx)
    df['Close'] = df['Close'].fillna(method='ffill')
    return df

data = load_data()

st.title("BTCUSDT Price Prediction (Binance Data)")
st.subheader("Dataset Overview")
st.write(f"Rows: {data.shape[0]} Columns: {data.shape[1]}")
st.dataframe(data.head(10))
st.dataframe(data.describe())

st.subheader("Historical Closing Prices")
st.line_chart(data['Close'])

close_data = data['Close']
train_size = int(len(close_data) * 0.8)
train, test = close_data[:train_size], close_data[train_size:]

def evaluate(y_true, y_pred):
    return {
        'RMSE': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'MAE': float(mean_absolute_error(y_true, y_pred)),
        'R2': float(r2_score(y_true, y_pred))
    }

# ARIMA
arima_model = ARIMA(train, order=(5, 1, 0))
arima_result = arima_model.fit()
arima_forecast = arima_result.forecast(steps=len(test))
arima_pred_full = pd.concat([train, pd.Series(arima_forecast, index=test.index)])
arima_metrics = evaluate(test, arima_forecast)

# SARIMAX
sarimax_model = SARIMAX(train, order=(5,1,0), seasonal_order=(1,1,1,12))
sarimax_result = sarimax_model.fit(disp=False)
sarimax_forecast = sarimax_result.forecast(steps=len(test))
sarimax_pred_full = pd.concat([train, pd.Series(sarimax_forecast, index=test.index)])
sarimax_metrics = evaluate(test, sarimax_forecast)

# Prophet
prophet_df = pd.DataFrame({'ds': train.index, 'y': train.values})
prophet_model = Prophet()
prophet_model.fit(prophet_df)
future = prophet_model.make_future_dataframe(periods=len(test))
forecast = prophet_model.predict(future)
prophet_pred_full = pd.Series(forecast['yhat'].values, index=future['ds'])
prophet_pred_test = forecast.set_index('ds')['yhat'].iloc[-len(test):]
prophet_metrics = evaluate(test, prophet_pred_test)

# LSTM
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train.values.reshape(-1, 1))
time_step = 10
X_train, y_train = [], []
for i in range(len(train_scaled) - time_step):
    X_train.append(train_scaled[i:i+time_step, 0])
    y_train.append(train_scaled[i+time_step, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

test_scaled = scaler.transform(test.values.reshape(-1,1))
X_test, y_test = [], []
for i in range(len(test_scaled) - time_step):
    X_test.append(test_scaled[i:i+time_step, 0])
    y_test.append(test_scaled[i+time_step, 0])
X_test, y_test = np.array(X_test), np.array(y_test)
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

lstm_model = Sequential()
lstm_model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
lstm_model.add(LSTM(50))
lstm_model.add(Dense(1))
lstm_model.compile(loss='mean_squared_error', optimizer='adam')
lstm_model.fit(X_train, y_train, epochs=10, batch_size=64, verbose=0)

lstm_pred = lstm_model.predict(X_test)
y_test_inv = scaler.inverse_transform(y_test.reshape(-1,1))
lstm_pred_inv = scaler.inverse_transform(lstm_pred)
lstm_metrics = evaluate(y_test_inv, lstm_pred_inv)

lstm_model.save('lstm_model.h5')
joblib.dump(scaler, 'scaler.save')

lstm_full_pred = np.full_like(close_data, fill_value=np.nan, dtype=np.float64)
lstm_full_pred[-len(lstm_pred_inv):] = lstm_pred_inv[:,0]

metrics_df = pd.DataFrame({
    'ARIMA': arima_metrics,
    'SARIMAX': sarimax_metrics,
    'PROPHET': prophet_metrics,
    'LSTM': lstm_metrics
}).T

st.header("Model Evaluation Metrics")
st.dataframe(metrics_df, use_container_width=True)

def plot_full_forecast(full_dates, full_actuals, pred_full, title, pred_label):
    fig, ax = plt.subplots()
    ax.plot(full_dates, full_actuals, label='Actual', color='blue')
    ax.plot(full_dates, pred_full, label=pred_label, color='orange')
    ax.set_title(title)
    ax.legend()
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    fig.autofmt_xdate(rotation=45)
    return fig

st.subheader("Forecast Visualizations")
st.pyplot(plot_full_forecast(data.index, data['Close'], arima_pred_full, "ARIMA Forecast", "ARIMA"))
st.pyplot(plot_full_forecast(data.index, data['Close'], sarimax_pred_full, "SARIMAX Forecast", "SARIMAX"))
st.pyplot(plot_full_forecast(data.index, data['Close'], lstm_full_pred, "LSTM Forecast", "LSTM"))

fig3, ax3 = plt.subplots()
ax3.plot(data.index, data['Close'], label='Actual', color='blue')
ax3.plot(prophet_pred_full.index, prophet_pred_full, label='Prophet Forecast', color='orange') 
ax3.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='orange', alpha=0.2, label='Conf. Interval')
ax3.set_title("Prophet Forecast")
ax3.legend()
ax3.xaxis.set_major_locator(mdates.MonthLocator())
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
fig3.autofmt_xdate(rotation=45)
st.pyplot(fig3)
