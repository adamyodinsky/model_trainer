import argparse
import yaml
from munch import DefaultMunch
from data_downloader import timescale, helper

# Importing Preprocessing tools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

# Importing the Keras libraries and packages for training the model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--symbol", help="Symbol name")
    parser.add_argument("-t", "--time_period", help="Time period")

    return parser.parse_args()


def load_config():
    config_dict = yaml.safe_load(open("config.yaml"))
    return DefaultMunch.fromDict(config_dict)


def main():
    args = args_parser()
    config = load_config()

    db = timescale.TmDB(config)
    train_model(db, config)


def train_model(db, config):
    # Importing the training set
    ticker = 'MSFT'
    start = "2018-11-01"
    end = "2020-11-01"
    query = f"""
    SELECT close, volume
    FROM {config.db.prices_table}
    WHERE ticker LIKE '{ticker}' AND date BETWEEN '{start}' AND '{end}'"""

    df_total = pd.read_sql_query(query, db.conn)
    df_train, df_test = train_test_split(df_total, test_size=0.2, shuffle=False)
    training_set = df_train.iloc[:, 0:2].values


    # Feature Scaling
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler(feature_range=(0, 1))
    training_set_scaled = scaler.fit_transform(training_set)

    # Creating a data structure with 60 timesteps and 1 output

    set_length = len(training_set)
    time_steps = 60
    epochs = 3

    x_train = []
    y_train = []
    for i in range(time_steps, set_length):
        x_train.append(training_set_scaled[i - time_steps:i, 0:2])
        y_train.append(training_set_scaled[i, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)

    # Reshaping
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 2))

    # Part 2 - Building the RNN

    # Initialising the RNN
    regressor_model = Sequential()

    # Adding the first LSTM layer and some Dropout regularisation
    regressor_model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 2)))
    regressor_model.add(Dropout(0.2))
    #
    # # Adding a second LSTM layer and some Dropout regularisation
    regressor_model.add(LSTM(units=50, return_sequences=True))
    regressor_model.add(Dropout(0.2))

    # # Adding a third LSTM layer and some Dropout regularisation
    regressor_model.add(LSTM(units=50, return_sequences=True))
    regressor_model.add(Dropout(0.2))

    # # # Adding a fourth LSTM layer and some Dropout regularisation
    regressor_model.add(LSTM(units=50))
    regressor_model.add(Dropout(0.2))

    # # Adding the output layer
    regressor_model.add(Dense(units=1))

    # # Compiling the RNN
    regressor_model.compile(optimizer='adam', loss='mean_squared_error')

    # # Fitting the RNN to the Training set
    regressor_model.fit(x_train, y_train, epochs=epochs, batch_size=32)

    # # Part 3 - Making the predictions and visualising the results

    # # Getting the real stock price
    real_stock_price = df_test.iloc[:, 0].values

    # # Getting the predicted stock price
    inputs = df_total[len(df_total) - len(df_test) - time_steps:].values
    inputs = inputs.reshape(-1, 2)
    inputs = scaler.transform(inputs)

    x_test = []
    set_length = len(df_test)

    for i in range(time_steps, set_length + time_steps):
        x_test.append(inputs[i - time_steps:i, 0:2])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 2))
    predicted_stock_price = regressor_model.predict(x_test)

    print("real_price_stocks length:", len(real_stock_price))
    print("inputs length:", len(inputs))
    print("predicted_price_length:", len(predicted_stock_price))

    # create empty table with 12 fields
    predicted_stock_price_like_shape = np.zeros(shape=(len(predicted_stock_price), 2))
    # put the predicted values in the right field
    predicted_stock_price_like_shape[:, 0] = predicted_stock_price[:, 0]
    # inverse transform and then select the right field
    predicted_stock_price = scaler.inverse_transform(predicted_stock_price_like_shape)[:, 0]

    # # Visualising the results
    plt.plot(real_stock_price, color='red', label='Real Google Stock Price')
    plt.plot(predicted_stock_price, color='blue', label='Predicted Google Stock Price')
    plt.title('Google Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Google Stock Price')
    plt.legend()
    plt.show()


main()
