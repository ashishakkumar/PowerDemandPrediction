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
import urllib.request
import datetime
url = "https://github.com/ashishakkumar/PowerDemandPrediction/raw/main/multivariate_lstm2.h5"
filename = "multivariate_lstm.h5"
urllib.request.urlretrieve(url, filename)

 
page_bg_img = f'''
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://raw.githubusercontent.com/ashishakkumar/PowerDemandPrediction/main/340966496_570385771749371_2312951338874449866_n%20(1).jpg");
background-size: cover;
}}
[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}
[data-testid="stToolbar"] {{
right: 2rem;
}}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)






multivariate_lstm = tf.keras.models.load_model('multivariate_lstm.h5',compile = False)



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

    
    st.title(':black[Power Demand Prediction (AI4ICPS)]')
    date_str = "2023-05-14"
    last_date = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()

    st.markdown("_Tips : Click on the same date twice in the calendar interface to see the prediction for single day_")

    selected_date = st.date_input(
        "Select a date for prediction",
        value=(last_date, last_date)
       
    )

    if isinstance(selected_date, tuple):
        start_date, end_date = selected_date
    else:
        start_date, end_date = selected_date, selected_date
     

    start_date = datetime.datetime.strptime(str(start_date), "%Y-%m-%d").date()
    end_date = datetime.datetime.strptime(str(end_date), "%Y-%m-%d").date()
    num_days = (end_date - start_date).days +1
    num_days_2 = (end_date - last_date).days +1
    st.markdown(f"You have selected: **{start_date}** to **{end_date}** for prediction.")

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


    st.line_chart(merged_df, x="Date", y="Energy Required (MU)",use_container_width=True)
 
    columns = ['Energy Required (MU)', 'Rain', 'Tmax', 'Tmin', 'inflation', 'Day']
    merged_df = merged_df.loc[:, columns]

    X = merged_df.values
    y = merged_df['Energy Required (MU)'].values
    y = y.reshape(-1, 1)

    #defining the end indices of train validation and test set
    train_end_idx = 1202
    cv_end_idx = 1367
    test_end_idx = 1565

    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))

    scaler_X.fit(X[:train_end_idx])
    scaler_y.fit(y[:train_end_idx])

    X_norm = scaler_X.transform(X)
    y_norm = scaler_y.transform(y)

    dataset_norm = np.concatenate((X_norm, y_norm), axis=1)
    n_features = 6
    past_history = 60
    future_target = 30

    X_train, y_train = multivariate_data(dataset_norm[:, 0:-1], dataset_norm[:, -1],
                                         0, train_end_idx, past_history, 
                                         future_target, step=1, single_step=False)

    X_val, y_val = multivariate_data(dataset_norm[:, 0:-1], dataset_norm[:, 0],
                                     train_end_idx, cv_end_idx, past_history, 
                                     future_target, step=1, single_step=False)

    X_test, y_test = multivariate_data(dataset_norm[:, 0:-1], dataset_norm[:, 0],
                                       cv_end_idx, test_end_idx, past_history, 
                                       future_target, step=1, single_step=False)

    batch_size = 32
    buffer_size = 1184

    train = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train = train.cache().shuffle(buffer_size).batch(batch_size).prefetch(1)

    validation = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    validation = validation.batch(batch_size).prefetch(1)


    
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
    forecast_unseen_inverse = scaler_y.inverse_transform(forecast_unseen)[0]
    date_range = pd.date_range(start=start_date, periods=(end_date -start_date).days+1, freq='D')
    # Create an empty DataFrame
    df = pd.DataFrame()

    # Add the Date column to the DataFrame
    df['Date'] = date_range
    df['Forecasted Demand (MU)'] = forecast_unseen_inverse[(start_date-last_date).days:(end_date-last_date).days+1]
    st.bar_chart(data=df, x='Date', y='Forecasted Demand (MU)')

if __name__ == "__main__":
    main()


