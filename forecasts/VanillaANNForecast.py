import pandas as pd
import tensorflow as tf
from forecasts.forecast import Forecast
from tensorflow import keras
from keras.losses import Loss
from keras.optimizers import Adam


class VanillaANNForecast(Forecast):
    def __init__(self, data):
        super().__init__(data)
        self.name = "ANN Forecast"
        self.amn_hours_looking_back = 24
        self.correlations = {
            "pv": "direct_radiation",
            "wind": "windspeed_100m",
        }

    def forecast(self, validation_data, training_data, forecast_attribute, time_horizon):
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
        model.fit(x_np, y_np, epochs=200, batch_size=24)  # Modify epochs and batch size as needed
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
            self.end_predicting()
            # Print progress
            print(f"Predicting {run_nr}/{len(validation_data)} ...")
        return forecasts

    def build_model(self, input_dim, output_dim):
        # Create your ANN model here
        model = keras.Sequential([
            keras.layers.Dense(32, activation='relu', input_shape=(input_dim,)),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(output_dim)
        ])
        learning_rate = 0.0005  # Set your desired learning rate
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss=self.error_rmse())  # Adjust loss function as needed
        return model


class RootMeanSquaredError(Loss):
    def call(self, y_true, y_pred):
        return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))
