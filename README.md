# crypto-realtime-monitoring
This project delivers a robust framework for cryptocurrency price prediction by leveraging a combination of classical time series models and deep learning techniques, specifically the Long Short-Term Memory (LSTM) neural network. The architecture is complemented by a real-time data streaming system and dynamic visualization dashboards. The objective is to accurately forecast cryptocurrency prices such as BTC/USDT and provide actionable insights through continuous monitoring, supporting traders and analysts in making informed decisions in volatile market environments.

# Objectives:
Collect and preprocess comprehensive historical cryptocurrency price data to form a clean and continuous dataset for modeling.

Develop and compare multiple forecasting models such as ARIMA, SARIMAX, Facebook Prophet, and LSTM deep learning, to identify best performance.

Enable live streaming of real-time price data with ongoing forecasting using the trained models for near real-time price predictions.

Store live and forecasted data in a time-series database (InfluxDB) and visualize insights dynamically using Grafana dashboards.

Implement interactive dashboard slicers for flexible filtering by cryptocurrency symbols and time ranges.

Support extendability for additional coins, ensemble forecasting methods, and alerting mechanisms to aid timely decision-making.


# Notes:
While running code to stream data to dashboard make sure to replace influx token,bucket,url,etc with your own.
