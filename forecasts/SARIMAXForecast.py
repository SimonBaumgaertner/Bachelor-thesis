import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from forecasts.forecast import Forecast
import numpy as np
import itertools


class SARIMAXForecast(Forecast):
    def __init__(self, data):
        super().__init__(data)
        self.name = "SARIMAX Forecast"
        self.parameterDict = {"pv": (0, 1, 0), "wind": (2, 0, 2), "demand": (8, 0, 8)}

    def forecast(self, validation_data, training_data, forecast_attribute, time_horizon, hybrid=False):
        if hybrid:
            self.name = "SARIMAX Hybrid Forecast"
        self.start_statistics()
        forecasts = pd.DataFrame({'TimestepID': validation_data['TimestepID']})

        for run_nr, (index, row) in enumerate(validation_data.iterrows(), start=1):
            self.start_training()
            # Create and train a new ARIMAX model using the existing data up to the current index
            #p, d, q = 1, 1, 2
            p, d, q = self.parameterDict[forecast_attribute]
            P, D, Q, s = 1, 1, 1, 24
            # Calculate the start date as one month before the index
            start_date = index - pd.DateOffset(months=2)

            # Slice the data to include only records from start_date to index
            model_data = self.data.loc[start_date:index]

            # Define exogenous variables
            exog_variables = ['temperature', 'windspeed_100m', 'direct_radiation']  # Add your exogenous variables here

            # Create and fit the ARIMAX model
            model = SARIMAX(endog=model_data[forecast_attribute], exog=model_data[exog_variables],
                            order=(p, d, q), seasonal_order=(P, D, Q, s))
            model = model.fit()
            self.end_training()
            self.start_predicting()

            # Initialize an empty array to store exogenous variables for each step
            exog_values = np.zeros((time_horizon, len(exog_variables)))

            # Iterate over the next time_horizon - 1 steps to get exogenous values
            for i in range(0, time_horizon):
                next_time_step = index + pd.Timedelta(hours=i)
                exog_values[i, :] = self.data.loc[next_time_step, exog_variables].values

            # Make an out-of-sample forecast with exogenous variables
            forecast_steps = model.get_forecast(steps=time_horizon, exog=exog_values)

            for hour in range(time_horizon):
                column_name = f"Predicted_Values_{hour}H"
                # Store the prediction in the DataFrame
                forecasts.loc[index, column_name] = forecast_steps.predicted_mean[hour]
                if hybrid and forecast_attribute == "pv" and (hour <= 6 or hour >= 20):
                    forecasts.loc[index, column_name] = 0
            self.end_predicting()

            # Print progress
            print(f"Predicting {run_nr}/{len(validation_data)} ...")

        return forecasts

    def hyper_train(self, forecast_attribute, index, time_horizon):
        # Define hyperparameter search space
        p_values = [1, 2, 3]
        d_values = [1, 2]
        q_values = [1, 2, 3]
        P_values = [1, 2]
        D_values = [1, 2]
        Q_values = [1, 2]
        s_values = [24]  # Assuming daily seasonality

        # Create a list of all possible combinations of p, d, and q
        param_combinations = list(itertools.product(p_values, d_values, q_values, P_values, D_values, Q_values, s_values))

        # Calculate the start date as one month before the index
        start_date = index - pd.DateOffset(months=1)

        # Slice the data to include only records from start_date to index
        model_data = self.data.loc[start_date:index]

        # Define exogenous variables
        exog_variables = ['temperature', 'windspeed_10', 'precipitation',
                          'windgust']  # Add your exogenous variables here
        for params in param_combinations:
            p, d, q, P, D, Q, s = params
            # Create and fit the ARIMAX model
            # Create and fit the ARIMAX model
            model = SARIMAX(endog=model_data[forecast_attribute], exog=model_data[exog_variables],
                            order=(p, d, q), seasonal_order=(P, D, Q, s))
            model = model.fit()
            best_aic = float('inf')
            best_params = None
            aic = model.aic
            print(f"finished for combination {params} with aic {aic}", end="\r")
        print("Best AIC:", best_aic)
        print("Best Parameters (p, d, q, P, D, Q, s):", best_params)




