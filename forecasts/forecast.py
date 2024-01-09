import numpy as np

import time

from keras.src.losses import Loss
import tensorflow as tf


class Forecast:
    def __init__(self, data):
        self.data = data
        self.name = "Generic Forecast - You should change this name."
        # Statistics
        self.statistics = None
        self.training_start_time = None
        self.prediction_start_time = None

    def forecast(self, validation_data, forecast_attribute, time_horizon):
        raise NotImplementedError("forecast must me implemented by the subclass.")

    def start_training(self):
        self.training_start_time = time.time()

    def end_training(self):
        if self.training_start_time is not None:
            end_time = time.time()
            elapsed_time = end_time - self.training_start_time
            self.statistics["amount_training"] += 1
            self.statistics["training_time"] += elapsed_time
            self.training_start_time = None
            print(f"Training {self.name} completed in {elapsed_time} seconds.")

    def start_predicting(self):
        self.prediction_start_time = time.time()

    def end_predicting(self):
        if self.prediction_start_time is not None:
            end_time = time.time()
            elapsed_time = end_time - self.prediction_start_time
            self.statistics["amount_predictions"] += 1
            self.statistics["prediction_time"] += elapsed_time
            self.prediction_start_time = None
            print(f"Prediction {self.name} completed in {elapsed_time} seconds.")

    def start_statistics(self):
        self.statistics = {
            "amount_training": 0,
            "amount_predictions": 0,
            "training_time": 0,
            "prediction_time": 0,
            "avr_training_time": 0,
            "avr_prediction_time": 0
        }

    def get_statistics(self):
        statistics = {}
        if self.statistics["amount_predictions"] > 0:
            statistics["avr_prediction_time"] = self.statistics["prediction_time"] / self.statistics[
                "amount_predictions"]
        else:
            statistics["avr_prediction_time"] = 0
        if self.statistics["amount_training"] > 0:
            statistics["avr_training_time"] = self.statistics["training_time"] / self.statistics[
                "amount_training"]
        else:
            statistics["avr_training_time"] = 0
        statistics["avr_total_time"] = statistics["avr_prediction_time"] + statistics["avr_training_time"]
        return statistics

    def get_actual_and_predicted_values(self, forecasts, forecast_attribute, time_horizon, start=0):
        # Initialize lists to store actual and forecasted values
        actual_values = []
        forecasted_values = []

        for i in range(start, time_horizon):
            # Get the actual values and the forecasted values for the current time step
            actual = self.data.iloc[forecasts.TimestepID + i][forecast_attribute].values
            forecast_column_name = f"Predicted_Values_{i}H"
            forecast_values = forecasts[forecast_column_name].values

            # Append actual and forecasted values to their respective lists
            actual_values.extend(actual)
            forecasted_values.extend(forecast_values)

        # Convert lists to NumPy arrays
        actual_values = np.array(actual_values)
        forecasted_values = np.array(forecasted_values)

        return actual_values, forecasted_values

    def mae(self, forecasts, forecast_attribute, time_horizon, start=0):
        actual, forecast = self.get_actual_and_predicted_values(forecasts, forecast_attribute, time_horizon, start)

        # Calculate the absolute errors
        absolute_errors = np.abs(actual - forecast)

        # Calculate the mean absolute error
        mae_value = np.mean(absolute_errors)

        return mae_value

    def mape(self, forecasts, forecast_attribute, time_horizon, start=0):
        actual, forecast = self.get_actual_and_predicted_values(forecasts, forecast_attribute, time_horizon, start)
        # Calculate the absolute percentage errors
        absolute_percentage_errors = np.abs((actual - forecast) / actual)
        return np.mean(absolute_percentage_errors) * 100

    def rmse(self, forecasts, forecast_attribute, time_horizon, start=0):
        actual, forecast = self.get_actual_and_predicted_values(forecasts, forecast_attribute, time_horizon,
                                                                start)
        # calculate RMSE
        return np.sqrt(np.mean(np.square(actual - forecast)))

    def r_squared(self, forecasts, forecast_attribute, time_horizon, start=0):
        actual, forecast = self.get_actual_and_predicted_values(forecasts, forecast_attribute, time_horizon,
                                                                start)
        # Calculate the mean of actual values
        mean_actual = np.mean(actual)

        # Calculate RSS
        rss = np.sum((actual - forecast) ** 2)

        # Calculate TSS
        tss = np.sum((mean_actual - actual) ** 2)

        # Calculate R-squared
        r_squared = 1 - (rss / tss)

        return r_squared

    def SD(self, forecasts, forecast_attribute, time_horizon, start=0):
        actual, forecast = self.get_actual_and_predicted_values(forecasts, forecast_attribute, time_horizon, start)

        # Calculate the absolute errors
        absolute_errors = np.abs(actual - forecast)

        # Calculate the standard deviation of the absolute errors
        std_absolute_error_value = np.std(absolute_errors)

        return std_absolute_error_value

    def get_error(self, forecasts, forecast_attribute, time_horizon, error_function, start=0):
        if error_function == "MAPE":
            return self.mape(forecasts, forecast_attribute, time_horizon, start)
        elif error_function == "RMSE":
            return self.rmse(forecasts, forecast_attribute, time_horizon, start)
        elif error_function == "R²":
            return self.r_squared(forecasts, forecast_attribute, time_horizon, start)
        elif error_function == "MAE":
            return self.mae(forecasts, forecast_attribute, time_horizon, start)
        elif error_function == "SD":
            return self.SD(forecasts, forecast_attribute, time_horizon, start)

    def get_error_per_Hour(self, forecasts, forecast_attribute, time_horizon, error_function, start=0):
        average_error = self.get_error(forecasts, forecast_attribute, time_horizon, error_function, start)
        average_per_hour = []
        for i in range(0, time_horizon):
            average_per_hour.append(self.get_error(forecasts, forecast_attribute, i + 1, error_function, start + i))
        return average_error, average_per_hour


    def error_rmse(self):
        return RootMeanSquaredError()


class RootMeanSquaredError(Loss):
    def call(self, y_true, y_pred):
        return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))
