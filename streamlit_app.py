#!/usr/bin/env python
# coding: utf-8

# download this is at .py then run streamlit run streamlit_app.py in your command line (video in appendix of app running if donloading is an issue

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from keras.regularizers import l2
import streamlit as st
import plotly.graph_objs as go
import plotly.express as px

# Load the cleaned dataset
file_path = 'feeder_clean.csv'
data = pd.read_csv(file_path)

# Convert the 'TLIST(M1)' column to a datetime format
data['Date'] = pd.to_datetime(data['TLIST(M1)'], format='%Y%m')
data.set_index('Date', inplace=True)

# Normalize data preparation and model training functions
def prepare_data(feedstuff_data):
    feedstuff_data['VALUE'].fillna(feedstuff_data['VALUE'].mean(), inplace=True)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(feedstuff_data['VALUE'].values.reshape(-1, 1))
    return scaled_data, scaler

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i + seq_length]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def build_and_train_model(X_train, y_train, X_test, y_test, seq_length):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(seq_length, 1), kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.2))
    model.add(LSTM(50, kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])
    return model

def forecast_lstm(model, last_sequence, steps):
    forecast = []
    current_sequence = last_sequence
    for _ in range(steps):
        current_sequence = current_sequence.reshape((1, seq_length, 1))
        next_value = model.predict(current_sequence)
        forecast.append(next_value[0, 0])
        current_sequence = np.append(current_sequence[:, 1:, :], next_value.reshape((1, 1, 1)), axis=1)
    return np.array(forecast)

# Streamlit app
st.title("Feedstuff Price Prediction Dashboard")

feedstuff_options = data['Type of Feedstuff'].unique()
feedstuff = st.selectbox("Select Feedstuff:", feedstuff_options)

feedstuff_data = data[data['Type of Feedstuff'] == feedstuff].sort_index()
scaled_data, scaler = prepare_data(feedstuff_data)
seq_length = 12
X, y = create_sequences(scaled_data, seq_length)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

model = build_and_train_model(X_train, y_train, X_test, y_test, seq_length)

predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

last_sequence = scaled_data[-seq_length:]
last_date = feedstuff_data.index[-1]
forecast_end_date = pd.to_datetime('2025-01-01')
steps_to_forecast = (forecast_end_date.year - last_date.year) * 12 + (forecast_end_date.month - last_date.month)
future_forecast = forecast_lstm(model, last_sequence, steps_to_forecast)
future_forecast = scaler.inverse_transform(future_forecast.reshape(-1, 1))

forecast_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=steps_to_forecast, freq='M')

# Plot predictions
pred_df = pd.DataFrame({'Date': feedstuff_data.index[-len(y_test):], 'Actual': y_test.flatten(), 'Predicted': predictions.flatten()})
fig_pred = px.line(pred_df, x='Date', y=['Actual', 'Predicted'], title=f'{feedstuff} Price Prediction using LSTM')
st.plotly_chart(fig_pred)

# Plot forecast
forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecast': future_forecast.flatten()})
actual_df = feedstuff_data[['VALUE']]
actual_df.columns = ['Actual']
actual_df.reset_index(inplace=True)
forecast_df.reset_index(inplace=True)

fig_forecast = go.Figure()
fig_forecast.add_trace(go.Scatter(x=actual_df['Date'], y=actual_df['Actual'], mode='lines', name='Actual'))
fig_forecast.add_trace(go.Scatter(x=pred_df['Date'], y=pred_df['Predicted'], mode='lines', name='Predicted'))
fig_forecast.add_trace(go.Scatter(x=forecast_df['Date'], y=forecast_df['Forecast'], mode='lines', name='Forecast'))
fig_forecast.update_layout(title=f'{feedstuff} Price Forecast using LSTM', xaxis_title='Date', yaxis_title='Price (Euro per Tonne)')
st.plotly_chart(fig_forecast)

# Print forecasted values
forecast_df.set_index('Date', inplace=True)
st.write(forecast_df)


# In[ ]:




