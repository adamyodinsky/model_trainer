import argparse

import yaml
from datetime import datetime
from munch import DefaultMunch
from data_downloader import timescale, helper

# Importing Preprocessing tools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

# Importing the Keras libraries and packages for training the model
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--symbol", help="Symbol name")
    parser.add_argument("-t", "--time_period", help="Time period")

    return parser.parse_args()


def load_config():
    config_dict = yaml.safe_load(open("config.yaml"))
    return DefaultMunch.fromDict(config_dict)


def present(real_stock_price, predicted_stock_price):
    # # Visualising the results
    plt.plot(real_stock_price, color='red', label='Real Google Stock Price')
    plt.plot(predicted_stock_price, color='blue', label='Predicted Google Stock Price')
    plt.title('Google Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Google Stock Price')
    plt.legend()
    plt.show()


class ModelTrainer:
    model = None
    raw_df = None
    df_train = None
    df_test = None
    training_set = None
    x_train = []
    y_train = []

    def __init__(self, time_steps: int = 60):
        self.args = args_parser()
        self.config = load_config()
        self.db = timescale.TmDB(self.config)
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.time_steps = time_steps

    def get_data(self, ticker: str, start: str, end: str):
        # Importing the training set
        query = f"""
        SELECT close, volume
        FROM {self.config.db.prices_table}
        WHERE ticker LIKE '{ticker}' AND date BETWEEN '{start}' AND '{end}'"""

        self.raw_df = pd.read_sql_query(query, self.db.conn)

    def preprocessing(self, test_size_ratio, shuffle: bool = False):
        self.df_train, self.df_test = train_test_split(self.raw_df, test_size=test_size_ratio, shuffle=shuffle)
        self.training_set = self.df_train.iloc[:, 0:2].values

        # Feature Scaling
        self.training_set = self.scaler.fit_transform(self.training_set)

        # Creating a data structure with self.timesteps inputs and 1 output
        set_length = len(self.training_set)

        for i in range(self.time_steps, set_length):
            self.x_train.append(self.training_set[i - self.time_steps:i, 0:2])
            self.y_train.append(self.training_set[i, 0])
        self.x_train, self.y_train = np.array(self.x_train), np.array(self.y_train)

        # Reshaping
        self.x_train = np.reshape(self.x_train, (self.x_train.shape[0], self.x_train.shape[1], 2))

    def create_model(self, dropout: float, neurons: int, mid_layers: int):
        # Building the RNN

        # Initialising the RNN
        self.model = Sequential()

        # Adding the first LSTM layer and Dropout regularisation
        self.model.add(LSTM(units=neurons, return_sequences=True, input_shape=(self.x_train.shape[1], 2)))
        self.model.add(Dropout(dropout))

        for i in range(mid_layers - 1):
            # Adding an LSTM layer and Dropout regularisation
            self.model.add(LSTM(units=neurons, return_sequences=True))
            self.model.add(Dropout(dropout))

        # Adding a last middle LSTM layer
        self.model.add(LSTM(units=neurons))
        self.model.add(Dropout(dropout))

        # Adding the output layer
        self.model.add(Dense(units=1))

        # Compiling the RNN
        self.model.compile(optimizer='adam', loss='mean_squared_error')

    def train_model(self, epochs, batch_size):
        # # Fitting the RNN to the Training set
        self.model.fit(self.x_train, self.y_train, epochs=epochs, batch_size=32)

    def test_model(self):
        # # Getting the real stock price
        real_stock_price = self.df_test.iloc[:, 0].values

        # # Getting the predicted stock price
        inputs = self.raw_df[len(self.raw_df) - len(self.df_test) - self.time_steps:].values
        inputs = inputs.reshape(-1, 2)
        inputs = self.scaler.transform(inputs)

        x_test = []
        set_length = len(self.df_test)

        for i in range(self.time_steps, set_length + self.time_steps):
            x_test.append(inputs[i - self.time_steps:i, 0:2])

        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 2))
        predicted_stock_price = self.model.predict(x_test)

        # create empty table with 12 fields
        predicted_stock_price_like_shape = np.zeros(shape=(len(predicted_stock_price), 2))
        # put the predicted values in the right field
        predicted_stock_price_like_shape[:, 0] = predicted_stock_price[:, 0]
        # inverse transform and then select the right field
        predicted_stock_price = self.scaler.inverse_transform(predicted_stock_price_like_shape)[:, 0]

        return predicted_stock_price, real_stock_price

    def load_model(self, path):
        self.model = load_model(path)


def main():
    ticker = 'MSFT'
    start = "2015-01-01"
    end = datetime.today().strftime('%Y-%m-%d')

    model_trainer = ModelTrainer()
    model_trainer.get_data(ticker, start, end)
    model_trainer.preprocessing(test_size_ratio=0.1)
    # model_trainer.create_model(dropout=0.2, neurons=50, mid_layers=4)
    # model_trainer.train_model(epochs=100, batch_size=32)
    # model_trainer.model.save("model.h5")
    model_trainer.load_model("model.h5")

    predicted_stock_price, real_stock_price = model_trainer.test_model()
    present(real_stock_price, predicted_stock_price)
    score = model_trainer.model.evaluate(model_trainer.x_train, model_trainer.y_train, verbose=0)
    print(score)


main()
