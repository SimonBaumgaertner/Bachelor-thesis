import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd


def power_curve_model(turbine_name, wind_speed):
    # wind speed must be given in m/s
    wind_speed = wind_speed * 19.61 # max wind speed in data is 19.61 m/s
    if wind_speed < min_wind_speed[turbine_name]:
        return 0.0
    if wind_speed > max_wind_speed[turbine_name]:
        return 0.0
    if wind_speed > max_power_at_speed[turbine_name]:
        return 1.0
    #
    c = fitted_coefs[turbine_name]
    result = np.sum([c[n] * np.power(wind_speed, fitted_degree - n) for n in range(fitted_degree + 1)])
    return result


real_max_windspeed = 30279.75
power_curve_data_abs = {

    "N117-Gamma":

    # power curve data from https://www.wind-turbine-models.com/turbines/96-nordex-n117-gamma

        np.array([

            [0, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14,
             14.5, 15, 15.5, 16, 16.5, 17, 17.5, 18, 18.5, 19, 19.5, 20, 21],

            [0, 0.0, 23.00, 81.00, 154.00, 245.00, 356.00, 488.00, 644.00, 826.00, 1037.00, 1273.00, 1528.00, 1797.00,
             2039.00, 2212.00, 2325.00, 2385.00, 2400.00, 2400.00, 2400.00, 2400.00, 2400.00, 2400.00, 2400.00, 2400.00,
             2400.00, 2400.00, 2400.00, 2400.00, 2400.00, 2400.00, 2400.00, 2400.00, 2400.00, 2400.00, 2400.00, 0.0]

        ])

}

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
power_curve_data_rel = {}
for turbine_name, turbine_data in power_curve_data_abs.items():
    turbine_data = turbine_data.copy()
    turbine_data[1, :] = turbine_data[1, :] / turbine_data[1, :].max()
    power_curve_data_rel[turbine_name] = turbine_data
# generation of the fitted model
fitted_coefs = {}
fitted_degree = 6
for turbine_name, turbine_data in power_curve_data_rel.items():
    mask = (turbine_data[0] >= min_wind_speed[turbine_name] - 1) & \
           (turbine_data[0] <= max_power_at_speed[turbine_name] + 1)
    fitted_coefs[turbine_name] = np.polyfit(turbine_data[0, mask], turbine_data[1, mask], deg=fitted_degree)

