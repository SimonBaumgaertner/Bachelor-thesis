import pandas as pd

from forecasts.ARIMAForecast import ARIMAForecast
from forecasts.LSTMForecast import LSTMForecast
from forecasts.NaiveDayForecast import NaiveDayForecast
from forecasts.NaiveHourForecast import NaiveHourForecast
from forecasts.PhysicalForecast import PhysicalForecast
from forecasts.SARIMAXForecast import SARIMAXForecast
import matplotlib.pyplot as plt

from forecasts.TransformerForecast import TransformerForecast
from forecasts.VanillaANNForecast import VanillaANNForecast

root_directory = "C:\\Users\\Simon\\Desktop\\Bachelorarbeit\\assets"
# Define a function to plot and save average per hour graphs
def plot_and_save_average_per_hour(average_per_hour, average, error_function, model_name, attribute, directory,
                                   file_name, dpi=100,
                                   font_size=20,
                                   figure_width=12, figure_height=6, axis_label_font_size=18, tick_label_font_size=18):
    plt.figure(figsize=(figure_width, figure_height),
               dpi=dpi)  # Create a new figure with the specified width and DPI

    plt.plot(range(0, time_horizon), average_per_hour, marker='o', linestyle='-', label=f'Hourly {error_function}')

    average_line = [average] * time_horizon
    plt.plot(range(0, time_horizon), average_line, linestyle='--', label=f'Average {error_function}', color='red')

    plt.text(0, average, f'Avg: {average:.4f}', fontsize=font_size, color='red', verticalalignment='bottom')

    plt.title(f"{model_name} for {attribute} hourly {error_function}", fontsize=font_size)

    plt.xlabel("Hour", fontsize=axis_label_font_size)
    plt.ylabel(f"{error_function} Value", fontsize=axis_label_font_size)

    plt.xticks(fontsize=tick_label_font_size)
    plt.yticks(fontsize=tick_label_font_size)

    plt.grid(True)
    plt.legend(fontsize=font_size)


    # Define the file name based on the key (forecast_model name + attribute)
    file_name = f"{root_directory}\\{directory}\\{error_function}\\{file_name}.png"
    plt.savefig(file_name, dpi=dpi)  # Save the plot as an image file with the specified DPI
    print(f"Saved {file_name}")
    plt.close()  # Close the figure to save memory


# Example usage
if __name__ == "__main__":
    # Read your CSV data into a DataFrame
    data = pd.read_csv("data.csv")

    # Convert the 'local_time' column to a datetime index in the training_data DataFrame
    data['local_time'] = pd.to_datetime(data['local_time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    data = data.set_index('local_time').asfreq('H')
    # Remove the first 48 and last entries to ensure historical data is available for all forecasts and forecasts can be made for all
    prediction_data = data[48:-48]

    # Create training and validation sets
    split_percentage = 0.4  # first 40% for training, part of last 60% for validation
    split_index = int(len(prediction_data) * split_percentage)
    training_data = prediction_data[:split_index]
    all_validation_data = prediction_data[split_index:]

    # Filter validation data where hour_of_day is 0
    validation_data_hour_0 = all_validation_data[all_validation_data['hour_of_day'] == 0]
    training_data_hour_0 = training_data[training_data['hour_of_day'] == 0]
    # Define the number of validation samples to use
    validation_samples_to_use = 1  # For example, use 200 samples for validation

    # Create a random sample of the validation data with the given number of samples AND hour_of_day = 0
    random_validation_samples = validation_data_hour_0.sample(validation_samples_to_use)

    forecast_models = [NaiveHourForecast(data), NaiveDayForecast(data), ARIMAForecast(data), SARIMAXForecast(data), VanillaANNForecast(data), LSTMForecast(data), TransformerForecast(data), PhysicalForecast(data)]
    attributes_to_forecast = ["pv", "wind", "demand"]
    time_horizon = 24  # 24 hours
    # Create an empty list to store the results as lists
    results_table = []
    average_error_per_hour_dict = {}
    average_error_dict = {}

for forecast_model in forecast_models:
    for attribute in attributes_to_forecast:
        print(f"Forecasting {attribute} with {forecast_model.name}")
        # Make predictions for the current attribute
        forecast = forecast_model.forecast(random_validation_samples, training_data_hour_0, attribute, time_horizon)

        error_functions = ["RMSE", "MAPE", "R²", "MAE", "SD"]
        for error_function in error_functions:
            # Calculate the average error for the current attribute and error function
            (average, averages_per_hour) = forecast_model.get_error_per_Hour(forecast, attribute, time_horizon,
                                                                             error_function)
            plot_and_save_average_per_hour(averages_per_hour, average, error_function, forecast_model.name, attribute,
                                           "errors",
                                           f"{attribute}_{forecast_model.name}_{error_function}_{validation_samples_to_use}_samples")
        print(f"Finished forecasting {forecast_model.name} for {attribute}")
        statistics = forecast_model.get_statistics()
        print(f"Statistics for {forecast_model.name} for {attribute}:")
        print(f"Average prediction time: {statistics['avr_prediction_time']}")
        print(f"Average training time: {statistics['avr_training_time']}")