import pandas as pd
import numpy as np
import tensorflow as tf
from forecast import Forecast
from tensorflow import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout


class LSTMForecast(Forecast):
    def __init__(self, data):
        super().__init__(data)
        self.name = "LSTM Forecast"
        self.amn_hours_looking_back = 24
        self.correlations = {
            "pv": "direct_radiation",
            "wind": "windspeed_100m",
            # "demand": "hour_of_day"
        }

    def forecast(self, validation_data, training_data, forecast_attribute, time_horizon, hybrid=False):
        if hybrid:
            self.name = "LSTM-Hybrid Forecast"
        self.start_statistics()
        forecasts = pd.DataFrame({'TimestepID': validation_data['TimestepID']})
        model_data = self.data[48:-48]
        # Add the last amn_hours_looking_back values of the forecast_attribute as input features
        for i in range(self.amn_hours_looking_back):
            column_name = f"{forecast_attribute} - {i + 1} hours ago"
            model_data[column_name] = self.data.iloc[model_data["TimestepID"] - i - 2][forecast_attribute].values
        # Add correlated attribute if exists
        if forecast_attribute in self.correlations:
            correlated_attribute = self.correlations[forecast_attribute]
            for i in range(self.amn_hours_looking_back):
                column_name = f"{correlated_attribute} - {i} hours ago"
                model_data[column_name] = self.data.iloc[model_data["TimestepID"] - i - 1][correlated_attribute].values
            for i in range(self.amn_hours_looking_back):
                column_name = f"{correlated_attribute} - in {i + 1} hours"
                model_data[column_name] = self.data.iloc[model_data["TimestepID"] + i][correlated_attribute].values

        # Create an empty DataFrame to store the target data
        y = pd.DataFrame(index=training_data.index)

        # Loop through the time horizon to create columns for each time step
        for t in range(time_horizon):
            y[f'Target_{t}'] = model_data[forecast_attribute].shift(-t)

        # Get the index values from the training_data DataFrame
        common_index = training_data.index

        # Use .loc to select the rows from model_data that have the same index
        model_data_subset = model_data.loc[common_index]

        columns_to_drop = model_data_subset.columns[range(15)]  # Names of columns to drop
        x = model_data_subset.drop(columns=columns_to_drop)  # Convert to NumPy array

        x_np = x.values.reshape(-1, len(x.columns))
        y_np = y.values.reshape(-1, len(y.columns))
        # Create and train a new model
        model = self.build_model(len(x.columns), output_dim=time_horizon)
        self.start_training()
        model.fit(x_np, y_np, epochs=100, batch_size=36)  # Modify epochs and batch size as needed
        self.end_training()

        # Make predictions
        for run_nr, (index, row) in enumerate(validation_data.iterrows(), start=1):
            self.start_predicting()
            input_data = model_data.loc[index].drop(columns_to_drop).values.reshape(1, -1)
            prediction = model.predict(input_data)

            for hour in range(time_horizon):
                column_name = f"Predicted_Values_{hour}H"
                # Store the prediction in the DataFrame
                forecasts.loc[index, column_name] = prediction[0][hour]
                if hybrid and forecast_attribute == "pv" and (hour <= 6 or hour >= 20):
                    forecasts.loc[index, column_name] = 0
            self.end_predicting()
            # Print progress
            print(f"Predicting {run_nr}/{len(validation_data)} ...")
        return forecasts

    def build_model_bamberg(self, input_dim, output_dim):
        # Define the hyperparameters
        hidden_layer_size = 64
        num_layers = 2
        dropout = 0.2
        batch_size = 10
        learning_rate = 0.01

        # Create a Sequential model
        model = Sequential()

        # Add the first LSTM layer with dropout
        model.add(LSTM(hidden_layer_size, return_sequences=True, input_shape=(input_dim, 1)))
        model.add(Dropout(dropout))

        model.add(LSTM(hidden_layer_size, return_sequences=False))
        model.add(Dropout(dropout))

        # Add a fully connected layer for output
        model.add(Dense(output_dim))

        # Compile the model
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss=self.error_rmse())  # Change the loss function as needed

        # Print the model summary
        model.summary()
        return model

    def build_model(self, input_dim, output_dim):
        # Create your ANN model here
        model = keras.Sequential([
            keras.layers.LSTM(64, return_sequences=True, activation='relu', input_shape=(input_dim, 1)),
            keras.layers.Dropout(0.5),  # Add dropout layer
            keras.layers.LSTM(32),
            keras.layers.Dropout(0.5),  # Add dropout layer
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(output_dim)
        ])

        model.compile(optimizer='adam', loss=self.error_rmse())
        return model
