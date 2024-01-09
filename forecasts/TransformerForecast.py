import pandas as pd
from forecasts.forecast import Forecast
import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Dense, MultiHeadAttention, Embedding, Dropout, LayerNormalization


class TransformerForecast(Forecast):
    def __init__(self, data):
        super().__init__(data)
        self.name = "Transformer Forecast"
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
        model = self.build_model_old(len(x.columns), output_dim=time_horizon)
        self.start_training()
        model.fit(x_np, y_np, epochs=100, batch_size=32)  # Modify epochs and batch size as needed
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

    def build_model(self, input_dim, output_dim, num_heads=4, hidden_size=32, dropout_rate=0.1):
        # Input layer
        inputs = Input(shape=(input_dim,))

        # Token Embedding
        embedding_layer = Embedding(input_dim, hidden_size)(inputs)

        # LSTM Layer
        lstm_layer = tf.keras.layers.LSTM(hidden_size, return_sequences=True)(embedding_layer)

        # Transformer Blocks (Multi-Head Attention)
        for _ in range(num_heads):
            attention = MultiHeadAttention(num_heads=1, key_dim=hidden_size)(lstm_layer, lstm_layer)
            attention = Dropout(dropout_rate)(attention)
            res = LayerNormalization(epsilon=1e-6)(tf.math.add(lstm_layer, attention))
            lstm_layer = res

        # Global Average Pooling
        avg_pool = tf.reduce_mean(lstm_layer, axis=1)

        # Output layer
        outputs = Dense(output_dim, activation="ReLU")(avg_pool)

        model = keras.Model(inputs, outputs)
        model.compile(optimizer='adam', loss=self.error_rmse())
        return model

    def build_model_old(self, input_dim, output_dim, num_heads=8, ff_dim=32, num_transformer_blocks=1, mlp_units=128,
                    dropout_rate=0.1):
        # Input layer
        inputs = Input(shape=(input_dim,))

        # Token Embedding
        embedding_layer = Embedding(input_dim, mlp_units)(inputs)

        # Positional Embedding
        positional_encoding = Embedding(input_dim, mlp_units)(tf.range(start=0, limit=input_dim, delta=1))
        embeddings = tf.math.add(embedding_layer, positional_encoding)

        # Transformer Blocks
        for _ in range(num_transformer_blocks):
            attention = MultiHeadAttention(num_heads=num_heads, key_dim=mlp_units // num_heads)(embeddings, embeddings)
            attention = Dropout(dropout_rate)(attention)
            res = LayerNormalization(epsilon=1e-6)(tf.math.add(embeddings, attention))
            feed_forward = keras.Sequential([
                Dense(ff_dim, activation='relu'),
                Dense(mlp_units),
            ])(res)
            feed_forward = Dropout(dropout_rate)(feed_forward)
            embeddings = LayerNormalization(epsilon=1e-6)(tf.math.add(res, feed_forward))

        # Global Average Pooling
        avg_pool = tf.reduce_mean(embeddings, axis=1)

        # Output layer
        outputs = Dense(output_dim)(avg_pool)

        model = keras.Model(inputs, outputs)
        model.compile(optimizer='adam', loss=self.error_rmse())
        return model


