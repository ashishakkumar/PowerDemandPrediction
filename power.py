import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import urllib.request
from math import sqrt
from sklearn.metrics import mean_squared_error
import plotly.express as px


# Download the model file
url = "https://github.com/ashishakkumar/PowerDemandPrediction/raw/main/multivariate_lstm.h5"
filename = "multivariate_lstm.h5"
urllib.request.urlretrieve(url, filename)


def load_model():
    # Load the model
    multivariate_lstm = tf.keras.models.load_model(url)
    return multivariate_lstm

def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
    data = []
    labels = []
    if start_index == 0:
        start_index = start_index + history_size
    else:
        start_index = start_index
    if end_index is None:
        end_index = len(dataset) - target_size
        
    for i in range(start_index, end_index):
        indices = range(i - history_size, i, step)
        data.append(dataset[indices])
        
        if single_step:
            labels.append(target[i + target_size])
        else:
            labels.append(target[i : i + target_size])

    return np.array(data), np.array(labels)

def main():
    st.set_page_config(
        page_title="Power Demand Prediction",
        layout='wide'
    )

    st.title('Power Demand Prediction')
    selected_date = st.date_input("Select a date for prediction")
    merged_df = pd.read_csv("https://raw.githubusercontent.com/ashishakkumar/PowerDemandPrediction/main/ap_dataset/APDATA.csv",
                            parse_dates=True)
    merged_df["Date"] = pd.to_datetime(merged_df["Date"])

    merged_df["Day"] = merged_df["Date"].dt.dayofweek
    merged_df.drop("day", axis=1, inplace=True)
    weather_df = pd.read_csv("https://raw.githubusercontent.com/ashishakkumar/PowerDemandPrediction/main/ap_dataset/finWeatherData.csv",
                             index_col=0, parse_dates=True)
    weather_df = weather_df.rename(columns={"date": "Date", "tmax": "Tmax", "tmin": "Tmin", "rain": "Rain"})

    weather_df["Date"] = pd.to_datetime(weather_df["Date"])
    merged_df = pd.merge(merged_df, weather_df, on='Date', how='inner')
    merged_df.drop(["rain", "temp"], axis=1, inplace=True)
    merged_df = merged_df.set_index("Date")
    merged_df.index = pd.date_range(start=merged_df.index[0], periods=len(merged_df), freq='D')


    fig = px.line(merged_df, x=merged_df.index, y="Energy Required (MU)")
    fig.update_xaxes(tickangle=90)
    st.plotly_chart(fig)
    columns = ['Energy Required (MU)', 'Rain', 'Tmax', 'Tmin', 'inflation', 'Day']
    merged_df = merged_df.loc[:, columns]

    X = merged_df.values
    y = merged_df['Energy Required (MU)'].values
    y = y.reshape(-1, 1)

    train_end_idx = 1202
    cv_end_idx = 1412
    test_end_idx = 1594

    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))

    scaler_X.fit(X[:train_end_idx])
    scaler_y.fit(y[:train_end_idx])

    X_norm = scaler_X.transform(X)
    y_norm = scaler_y.transform(y)

    dataset_norm = np.concatenate((X_norm, y_norm), axis=1)
    n_features = 6
    past_history = 15
    future_target = 1

    X_train, y_train = multivariate_data(dataset_norm[:, 0:-1], dataset_norm[:, -1],
                                         0, train_end_idx, past_history, 
                                         future_target, step=1, single_step=True)

    X_val, y_val = multivariate_data(dataset_norm[:, 0:-1], dataset_norm[:, 0],
                                     train_end_idx, cv_end_idx, past_history, 
                                     future_target, step=1, single_step=True)

    X_test, y_test = multivariate_data(dataset_norm[:, 0:-1], dataset_norm[:, 0],
                                       cv_end_idx, test_end_idx, past_history, 
                                       future_target, step=1, single_step=True)

    batch_size = 32
    buffer_size = 1184

    train = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train = train.cache().shuffle(buffer_size).batch(batch_size).prefetch(1)

    validation = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    validation = validation.batch(batch_size).prefetch(1)

    input_shape = X_train.shape[-2:]
    loss = tf.keras.losses.MeanSquaredError()
    metric = [tf.keras.metrics.RootMeanSquaredError()]
    lr_schedule = tf.keras.callbacks.LearningRateScheduler(
                  lambda epoch: 1e-4 * 10 ** (epoch / 10))
    early_stopping = tf.keras.callbacks.EarlyStopping(patience=10)

    # Load the model
    multivariate_lstm = load_model()

    y_test = y_test.reshape(-1, 1)
    y_test_inv = scaler_y.inverse_transform(y_test)
    test_forecast = multivariate_lstm.predict(X_test)
    lstm_forecast = scaler_y.inverse_transform(test_forecast)
    rmse_lstm = sqrt(mean_squared_error(y_test_inv, lstm_forecast))
    st.write(f"RMSE of day ahead power demand LSTM forecast: {round(rmse_lstm, 3)}")

    train_forecast = multivariate_lstm.predict(X_train)
    train_forecast_inverse = scaler_y.inverse_transform(train_forecast)

    valid_forecast = multivariate_lstm.predict(X_val)
    valid_forecast_inverse = scaler_y.inverse_transform(valid_forecast)

    start_idx = X_train.shape[0] + X_val.shape[0] + X_test.shape[0] + 1
    end_idx = start_idx + past_history
    X_test2 = scaler_X.transform(X[start_idx:end_idx, :])
    X_test2 = X_test2.reshape(1, past_history, n_features)
    forecast_unseen = multivariate_lstm.predict(X_test2)
    forecast_unseen_inverse = scaler_y.inverse_transform(forecast_unseen)
    st.write("Unseen data forecast:", forecast_unseen_inverse)

if __name__ == "__main__":
    main()

