from forecast import Forecast
import pandas as pd


class NaiveDayForecast(Forecast):
    def __init__(self, data):
        super().__init__(data)
        self.name = "Naive Day Forecast"

    def forecast(self, validation_data, training_data, forecast_attribute, time_horizon):
        self.start_statistics()
        self.start_predicting()
        forecasts = pd.DataFrame({'TimestepID': validation_data['TimestepID']})

        for i in range(time_horizon):
            column_name = f"Predicted_Values_{i}H"
            forecasts[column_name] = self.data.iloc[validation_data["TimestepID"] - (time_horizon - i)][forecast_attribute].values
        self.end_predicting()
        return forecasts
