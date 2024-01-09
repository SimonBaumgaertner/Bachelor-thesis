import pandas as pd
import numpy as np
import statsmodels.api as sm
from forecast import Forecast
import itertools


class ARIMAForecast(Forecast):
    def __init__(self, data):
        super().__init__(data)
        self.name = "ARIMA Forecast"
        self.parameterDict = {"pv": (0, 1, 0), "wind":  (2, 0, 2), "demand": (8, 0, 24)}

    def forecast(self, validation_data, training_data, forecast_attribute, time_horizon):
        self.start_statistics()
        forecasts = pd.DataFrame({'TimestepID': validation_data['TimestepID']})

        for run_nr, (index, row) in enumerate(validation_data.iterrows(), start=1):
            self.start_training()
            # Create and train a new model using the existing data up to the current index
            p, d, q = self.parameterDict[forecast_attribute]  # Example parameters, adjust accordingly
            start_date = index - pd.DateOffset(months=2)

            # Slice the data to include only records from start_date to index
            model_data = self.data.loc[start_date:index]
            model = sm.tsa.ARIMA(model_data[forecast_attribute], order=(p, d, q))
            model = model.fit()
            self.end_training()
            self.start_predicting()
            # Make an Out-of-sample forecast
            forecast_steps = model.forecast(steps=time_horizon)

            for hour in range(time_horizon):
                column_name = f"Predicted_Values_{hour}H"
                # Store the prediction in the DataFrame
                forecasts.loc[index, column_name] = forecast_steps[hour]
            self.end_predicting()
            # Print progress
            print(f"Predicting {run_nr}/{len(validation_data)} ...")

        return forecasts

    def hyper_train(self, training_data, forecast_attribute):
        # Define ranges for p, d, and q
        p_range = [0, 1, 2, 8, 24]
        d_range = range(0, 2)
        q_range = [0, 1, 2, 8, 24]

        # Create a list of all possible combinations of p, d, and q
        param_combinations = list(itertools.product(p_range, d_range, q_range))

        best_aic = float('inf')
        best_params = None

        # Iterate through all parameter combinations
        for params in param_combinations:
            p, d, q = params
            model = sm.tsa.ARIMA(training_data[forecast_attribute], order=(p, d, q))
            try:
                results = model.fit()
                aic = results.aic
                if aic < best_aic:
                    best_aic = aic
                    best_params = params
            except:
                continue
            print(f"finished for combination {params} with aic {aic}", end="\r")
        print("Best AIC:", best_aic)
        print("Best Parameters (p, d, q):", best_params)

    # ...
