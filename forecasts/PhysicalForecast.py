from forecasts.forecast import Forecast
import pandas as pd

from powerCurveCalculation import power_curve_model


class PhysicalForecast(Forecast):
    def __init__(self, data):
        super().__init__(data)
        self.name = "Physical Forecast"

    min_wind_speed = {
        "E70-E4": 2.5,
        "N117-Gamma": 3.0,
        "N117-3600": 3.0,
    }
    max_power_at_speed = {
        "E70-E4": 15.0,
        "N117-Gamma": 11.5,
        "N117-3600": 13.0,
    }
    max_wind_speed = {
        "E70-E4": 25.0,
        "N117-Gamma": 20.0,
        "N117-3600": 25.0,
    }

    def forecast(self, validation_data, training_data, forecast_attribute, time_horizon):
        self.start_statistics()
        self.start_predicting()

        turbine_name = "N117-Gamma"
        forecasts = pd.DataFrame({'TimestepID': validation_data['TimestepID']})

        for i in range(time_horizon):
            column_name = f"Predicted_Values_{i}H"
            prediction = self.data.iloc[validation_data["TimestepID"] - 1 + i][
                "windspeed_100m"].apply(lambda x: power_curve_model(turbine_name, x))
            forecasts[column_name] = prediction.values
        self.end_predicting()
        return forecasts
