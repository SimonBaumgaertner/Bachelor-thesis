from forecasts.forecast import Forecast
import pandas as pd


class NaiveHourForecast(Forecast):
    def __init__(self, data):
        super().__init__(data)
        self.name = "Naive Hour Forecast"

    def forecast(self, validation_data, training_data, forecast_attribute, time_horizon):
        self.start_statistics()
        self.start_predicting()
        forecasts = pd.DataFrame({'TimestepID': validation_data['TimestepID']})

        for i in range(time_horizon):
            column_name = f"Predicted_Values_{i}H"
            forecasts[column_name] = self.data.iloc[validation_data["TimestepID"] - 2][forecast_attribute].values
        self.end_predicting()
        return forecasts
