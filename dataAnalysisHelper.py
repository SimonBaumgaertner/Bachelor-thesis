import csv
import sqlite3
import csv

import matplotlib as matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

dpi = 100
font_size = 20
figure_width = 12
figure_height = 6
axis_label_font_size = 18
tick_label_font_size = 18
tick_size = 15  # Adjust this value as needed


def printTables(cursor):
    # Tabellennamen in der Datenbank abrufen
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    # Tabellennamen ausgeben
    for table in tables:
        print("Tabelle:", table[0])


def printTableDetails(cursor, table_name):
    cursor.execute("PRAGMA table_info(" + table_name + ");")
    columns_info = cursor.fetchall()

    # Spalteninformationen ausgeben
    for column in columns_info:
        print("Spaltenname:", column[1])
        print("Datentyp:", column[2])


def printSelectAll(cursor, table_name):
    # Fetch all data from the table
    cursor.execute(f"SELECT * FROM {table_name}")
    data = cursor.fetchall()
    # Print the data
    for row in data:
        print(row)


def aggregateTimestepIDs():
    # script for correcting timestepIDs (starting from 1)
    # Read the CSV file into a DataFrame
    df = pd.read_csv('data_og_timestamps.csv')

    # Reset the 'TimestepID' column starting from 1
    df['TimestepID'] = range(1, len(df) + 1)

    # Save the updated DataFrame back to the CSV file
    df.to_csv('data.csv', index=False)

    print("TimestepID values have been updated and the updated data has been saved to '.csv'.")


def printInfos(filePath, table_name, selectAll=False):
    db_file_path = filePath
    connection = sqlite3.connect(db_file_path)
    cursor = connection.cursor()

    printTables(cursor)

    printTableDetails(cursor, table_name)
    if selectAll:
        printSelectAll(cursor, table_name)
    # Close the connection
    connection.close()


def writeOutput(filePath, outputCSV, sql_query):
    db_file_path = filePath
    connection = sqlite3.connect(db_file_path)
    cursor = connection.cursor()

    # Execute the query
    cursor.execute(sql_query)

    # Get column names from cursor description
    column_names = [description[0] for description in cursor.description]

    # Fetch all the rows from the query result
    rows = cursor.fetchall()

    # Replace 'output.csv' with the desired name of your output CSV file.
    csv_file = f'{outputCSV}.csv'

    # Open a CSV file for writing
    with open(csv_file, 'w', newline='') as file:
        # Create a CSV writer
        csv_writer = csv.writer(file)

        # Write the column headers as the first row
        csv_writer.writerow(column_names)

        # Write the data rows to the CSV file
        csv_writer.writerows(rows)

    # Close the connection
    connection.close()


def addAttributes():
    # Read the CSV file into a DataFrame
    df = pd.read_csv('normalized_data.csv')

    # Convert the 'local_time' column to datetime
    df['local_time'] = pd.to_datetime(df['local_time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')

    # Add hour of day
    df['hour_of_day'] = df['local_time'].dt.hour

    # Add day of week (Monday=0, Sunday=6)
    df['day_of_week'] = df['local_time'].dt.dayofweek + 1

    # Add day of year
    df['day_of_year'] = df['local_time'].dt.dayofyear

    # Save the DataFrame with added attributes to a new CSV file
    df.to_csv('data.csv', index=False)


def calculateCorrelation(inputCsv):
    # Load your DataFrame from the CSV file
    df = pd.read_csv(f"{inputCsv}.csv")
    # Display the DataFrame's structure
    print(df.head())

    # Select only the numeric columns for correlation analysis
    numeric_columns = df.select_dtypes(include=['float64', 'int64'])

    # Calculate the correlation matrix
    correlation_matrix = numeric_columns.corr()

    # Get the top N correlations
    N = 20  # You can change N to get a different number of top correlations
    top_correlations = correlation_matrix.unstack().sort_values(ascending=False).drop_duplicates()

    # Print the top N correlations
    print(top_correlations.head(N))


def plotAttributes():
    # Read the CSV file into a DataFrame
    df = pd.read_csv('normalized_data.csv')

    # List of attributes to plot
    attributes_to_plot = [
        'pv', 'wind'
    ]

    # Iterate through the attributes and create plots
    for attribute in attributes_to_plot:
        plt.figure(figsize=(12, 6))
        sns.lineplot(x='local_time', y=attribute, data=df)
        plt.title(f'{attribute} Over Time')
        plt.xlabel('Time')
        plt.ylabel(attribute)
        plt.xticks(rotation=45)
        plt.show()


def plotStuff():
    # Read the CSV file into a DataFrame
    df = pd.read_csv('data.csv')

    # Extract the date part from the 'local_time' column
    df['date'] = pd.to_datetime(df['local_time']).dt.date

    # Define the specific indices you want to use as x-ticks
    x_ticks_indices = [0, 8762, 17521, 26305, 35065]

    # Create the plot
    plt.figure(figsize=(figure_width, figure_height), dpi=dpi)
    plt.plot(df.index, df['demand'])  # Use DataFrame index as x-values

    # Customize your plot
    plt.xlabel('Time', fontsize=axis_label_font_size)
    plt.ylabel('Demand', fontsize=axis_label_font_size)
    plt.title('Demand over dataset', fontsize=20)

    # Set the specific indices as x-axis ticks and format them
    plt.xticks(x_ticks_indices, df['date'].iloc[x_ticks_indices], rotation=0,
               fontsize=16)  # Adjust the fontsize as needed
    plt.yticks(fontsize=16)  # Adjust the fontsize as needed
    # Show the plot
    plt.show()


def findGapsAndDuplicates():
    # Read your data into a DataFrame
    data = pd.read_csv('data.csv')

    # Convert the 'local_time' column to datetime objects
    data['local_time'] = pd.to_datetime(data['local_time'])

    # Sort the DataFrame by 'local_time' to ensure it's in chronological order
    data = data.sort_values(by='local_time')

    # Find duplicates in 'local_time'
    duplicate_rows = data[data.duplicated(subset=['local_time'], keep=False)]

    # Find gaps in 'local_time'
    time_diff = data['local_time'].diff()
    gaps = time_diff[time_diff > pd.Timedelta(hours=1)]

    # Print duplicates and gaps
    print("Duplicate Rows:")
    print(duplicate_rows)
    print("\nGaps in 'local_time':")
    print(gaps)


def normalizeData():
    # Read your data into a DataFrame
    data = pd.read_csv('data.csv')

    # Find the peak values for each numeric column (excluding 'TimestepID' and 'local_time')
    peak_values = data.drop(['TimestepID', 'local_time'], axis=1).max()

    # Normalize the data by dividing each numeric column by its respective peak value
    normalized_data = data.copy()
    normalized_data[data.columns.difference(['TimestepID', 'local_time'])] /= peak_values

    # Save the normalized data to a new CSV file or perform further analysis as needed
    normalized_data.to_csv('normalized_data.csv', index=False)


def addPredictorToDataFrame(data, attribute):
    if attribute == 'wind':
        X = data['windgust'].values.reshape(-1, 1)  # Input feature
    elif attribute == 'pv':
        X = data['direct_radiation'].values.reshape(-1, 1)
    y = data['Value_Feedin']  # Target variable

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = Sequential()
    model.add(Dense(64, input_dim=1, activation='relu'))
    model.add(Dense(1, activation='linear'))

    model.compile(loss='mean_squared_error', optimizer='adam')

    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))
    predictions = model.predict(X)
    data['Predicted_Value_Feedin'] = predictions
    return data


def createAbnormalityData(csv_number, attribute):
    wind_data = pd.read_csv(f'data\\AllSubstations_2018_2022\\SeparatedSmartMeterData\\{csv_number}.csv')
    weather_data = pd.read_csv('weather.csv')
    merged_data = wind_data.merge(weather_data, on='TimestepID')
    merged_data['Value_Feedin_absolute'] = merged_data[['Value_Feedin']]
    # Scale 'Value_Feedin' and 'windgust' columns
    # Initialize the scaler
    scaler = MinMaxScaler()
    if attribute == 'wind':
        merged_data['windgust'] = scaler.fit_transform(merged_data[['windgust']])
    elif attribute == 'pv':
        merged_data['Value_Feedin'] = scaler.fit_transform(merged_data[['direct_radiation']])
    scaler.fit(merged_data[['Value_Feedin']])
    merged_data['Value_Feedin'] = scaler.fit_transform(merged_data[['Value_Feedin']])

    # create predictor and add prediction to dataframe
    # merged_data = addPredictorToDataFrame(merged_data, attribute)
    merged_data['Predicted_Value_Feedin'] = pd.read_csv(f'analysis_{csv_number}.csv')['Predicted_Value_Feedin']

    # Inverse transform the 'Predicted_Value_Feedin' to get 'Predicted_Value_Feedin_absolute'
    merged_data['Predicted_Value_Feedin_absolute'] = scaler.inverse_transform(merged_data[['Predicted_Value_Feedin']])

    # merged_data = pd.read_csv(f'analysis_{csv_number}.csv')
    merged_data['Prediction_Difference'] = merged_data['Predicted_Value_Feedin'] - merged_data['Value_Feedin']

    abnormality_threshold = 0.225
    merged_data['abnormality'] = (merged_data['Prediction_Difference'] >= abnormality_threshold).astype(int)
    min_duration = 4  # or 4 for wind

    abnormal_periods = []  # Initialize a list to store abnormal periods

    current_abnormal_period = None
    for index, row in merged_data.iterrows():
        if row['abnormality'] == 1:
            if current_abnormal_period is None:
                current_abnormal_period = [index]
            else:
                current_abnormal_period.append(index)
        else:
            if current_abnormal_period is not None:
                abnormal_periods.append(current_abnormal_period)
                current_abnormal_period = None

    # Filter out abnormal periods that are at least min_duration timesteps long
    filtered_abnormal_periods = [period for period in abnormal_periods if len(period) >= min_duration]

    # Flatten the list of indices within each abnormal period
    filtered_indices = [index for period in filtered_abnormal_periods for index in period]
    filtered_indices = [x + 1 for x in filtered_indices]  # add 1 to each index to account for the shift to TimeStepID

    # merged_data.drop(columns=["Abnormality"], inplace=True)

    merged_data['abnormality'] = 0
    # Set Abnormality to 1 for the timesteps in filtered_indices
    merged_data.loc[merged_data['TimestepID'].isin(filtered_indices), 'abnormality'] = 1

    # Create a new column 'critical_abnormality' with the specified condition
    merged_data['critical_abnormality'] = (merged_data['abnormality'] == 1) & (merged_data['Value_Feedin'] < 0.01)

    merged_data.to_csv(f'analysis_{csv_number}.csv', index=False)


def analyzAnomalities(csv_number, window_size, abnormality_type):
    merged_data = pd.read_csv(f'analysis_{csv_number}.csv')
    # Define the timestep window (e.g., 1000 timesteps)

    # Initialize variables to store counts and time periods with the most abnormalities
    abnormality_counts = []
    abnormal_periods = []

    # Iterate through the data in windows
    for start in range(0, len(merged_data), window_size):
        end = start + window_size
        window_data = merged_data[start:end]

        # Count abnormalities in the window
        abnormality_count = window_data[abnormality_type].sum()
        abnormality_counts.append(abnormality_count)

        # Store the start and end time periods of the window
        time_period_start = window_data['TimestepID'].iloc[0]
        time_period_end = window_data['TimestepID'].iloc[-1]
        abnormal_periods.append((time_period_start, time_period_end))

    # Find the indices of the windows with the most abnormalities
    top_n = 10  # Change this to the number of top windows you want to display
    top_indices = sorted(range(len(abnormality_counts)), key=lambda i: abnormality_counts[i], reverse=True)[:top_n]

    # Print the time periods with the most abnormalities
    for i in top_indices:
        count = abnormality_counts[i]
        start, end = abnormal_periods[i]
        print(f"Time Period: {start} to {end}, {abnormality_type}: {count}")
    return merged_data


def printPVAbnormilies(csv_name, fromTime, toTime, abnormality_type):
    merged_data = pd.read_csv(f'analysis_{csv_name}.csv')

    # Filter the data based on 'TimestepID' within the specified range
    merged_data = merged_data[(merged_data['TimestepID'] >= fromTime) & (merged_data['TimestepID'] <= toTime)]

    # Plot the data
    plt.figure(figsize=(12, 6))  # Set the figure size

    # Plot 'Value_Feedin'
    plt.plot(merged_data['TimestepID'], merged_data['Value_Feedin'], label='Value_Feedin', linewidth=2)

    # Plot 'windgust'
    plt.plot(merged_data['TimestepID'], merged_data['windgust'], label='Direct Radiation', linewidth=2)

    # Plot 'Predicted_Value_Feedin'
    plt.plot(merged_data['TimestepID'], merged_data['Predicted_Value_Feedin'], label='Predicted Value_Feedin',
             linewidth=2)

    # Mark abnormalities (Abnormality == 1)
    abnormalities = merged_data[merged_data[abnormality_type] == 1]
    plt.scatter(abnormalities['TimestepID'], abnormalities['Value_Feedin'], c='red', marker='o', label='Abnormality',
                s=100)

    # Set labels and title
    plt.xlabel('TimestepID')
    plt.ylabel('Scaled Value')
    plt.title('Value_Feedin, Direct Radiation, and Predicted Value_Feedin with Abnormalities')

    # Add legend
    plt.legend()

    # Show grid
    plt.grid(True)

    # Show the plot
    plt.show()


def printWindAbnormilies(csv_name, fromTime, toTime, abnormality_type):
    merged_data = pd.read_csv(f'analysis_{csv_name}.csv')

    # Filter the data based on 'TimestepID' within the specified range
    merged_data = merged_data[(merged_data['TimestepID'] >= fromTime) & (merged_data['TimestepID'] <= toTime)]

    # Create the plot
    plt.figure(figsize=(figure_width, figure_height), dpi=dpi)

    # Plot 'Value_Feedin'
    plt.plot(merged_data['TimestepID'], merged_data['Value_Feedin'], label='Energy generation', linewidth=2)

    # Plot 'windgust'
    plt.plot(merged_data['TimestepID'], merged_data['windgust'], label='Wind speed', linewidth=2)

    # Plot 'Predicted_Value_Feedin'
    plt.plot(merged_data['TimestepID'], merged_data['Predicted_Value_Feedin'], label='Predicted energy generation',
             linewidth=2)

    # Mark abnormalities (Abnormality == 1)
    abnormalities = merged_data[merged_data[abnormality_type] == 1]
    plt.scatter(abnormalities['TimestepID'], abnormalities['Value_Feedin'], c='red', marker='o', label='Abnormality',
                s=100)

    # Set labels and title
    plt.xlabel('Time', fontsize=axis_label_font_size)
    plt.ylabel('Scaled Value', fontsize=axis_label_font_size)
    plt.title('Actual vs predicted wind energy generation', fontsize=font_size)  # Add your title here

    # Set the tick size
    plt.xticks(fontsize=tick_label_font_size)
    plt.yticks(fontsize=tick_label_font_size)
    plt.xticks([])

    # Add legend
    plt.legend(fontsize=font_size)

    # Show grid
    plt.grid(True)

    # Show the plot
    plt.show()


def sumPredictedOfAbnormalities(attribute):
    csv_numbers_wind = [6614, 6638, 6748, 6749]
    csv_numbers_pv = [6615, 6639, 6746, 6747]
    if attribute == 'wind':
        csv_numbers = csv_numbers_wind
    elif attribute == 'pv':
        csv_numbers = csv_numbers_pv
    windowSize = 72
    abnormality_type = 'critical_abnormality'

    # Initialize a variable to store the sum
    total_predicted_value_feedin_absolute = 0
    total_feedin_absolute = 0
    amount_of_abnormalities = 0

    for csv_number in csv_numbers:
        # Assuming this loop updates the `merged_data` DataFrame with predictions
        merged_data = analyzAnomalities(attribute, csv_number, windowSize, abnormality_type)

        # Calculate the sum for the current csv_number
        current_sum = merged_data[merged_data[abnormality_type] == 1]['Predicted_Value_Feedin_absolute'].sum()

        total_feedin_absolute += merged_data['Value_Feedin_absolute'].sum()

        # Add the amount of abnormalities
        amount_of_abnormalities += merged_data['critical_abnormality'].sum()

        # Add the current sum to the total sum
        total_predicted_value_feedin_absolute += current_sum

    # print the amount of abnormality
    print(f"amount of abnormalities = {amount_of_abnormalities}")

    # print the sum of predicted values of abnormalities
    print(f"sum of predicted values of abnormalities = {total_predicted_value_feedin_absolute}kwH")

    # print the sum of values of feedin
    print(f"sum of total feedin = {total_feedin_absolute}kwH")


filePath = "C:\\Users\\Simon\\Desktop\\Bachelorarbeit\\Implementation\\data\\AllSubstations_2018_2022\\SystemStructure.db"
table_name = "list_of_measurement_units"
sql_query_2018_2022 = f'SELECT DISTINCT time_indices.TimestepID, time_indices.local_time, var1 as temperature, var18 as direct_radiation, var19 as diffuse_radiation, var13 as windspeed_10, var14 as windspeed_100m, var15 as cloudcover, var17 as windgust, var4 as precipitation ' \
                      f'FROM global_profile_wind, time_indices, global_profiles_pv, weather_data_om_ts, residual_grid_load ' \
                      f'WHERE global_profile_wind.TimestepID = time_indices.TimestepID AND time_indices.TimestepID = global_profiles_pv.TimestepID AND time_indices.TimestepID = weather_data_om_ts.TimestepID AND time_indices.TimestepID = residual_grid_load.TimestepID'
# printInfos(filePath, table_name, selectAll=False)
# writeOutput(filePath, "weather", sql_query_2018_2022)
# calculateCorrelation("data")
# aggregateTimestepIDs()
# addAttributes()
# plotAttributes()
# plotStuff()
# normalizeData()
# findGapsAndDuplicates()
# sumPredictedWindOfAbnormalities()


'''
Preprocessing:
    - Use query to get all data from the database
    - Use demandSummary.py to get the demand per TimestepID
    - Use solarSummary.py to get the solar data per TimestepID
    - Manually eliminate duplicates (daylight saving time)
    - (Combine data from different years if necessary)
    - Overwrite the TimestepIDs with numbers 1 - len(dataPoints)
'''

'''
Tables
Tabelle: time_indices
Tabelle: list_of_measurement_units
Tabelle: list_of_control_units
Tabelle: address_data
Tabelle: general_data_information
Tabelle: list_of_substations
Tabelle: global_profile_wind
Tabelle: global_profiles_pv
Tabelle: global_profiles_pv_info
Tabelle: global_profiles_heatpumps
Tabelle: global_profiles_heatpumps_info
Tabelle: address_roof_data
Tabelle: residual_grid_load
Tabelle: weather_data_ts
Tabelle: weather_data_description
Tabelle: heat_demand_per_location
'''
'''
Weather Data 2021
('var1', 'Temperature', '°C', '2 m elevation corrected')
('var2', 'Growing Degree Days', 'GDDc', '2 m elevation corrected')
('var3', 'Temperature', '°C', '1000 mb')
('var4', 'Temperature', '°C', '850 mb')
('var5', 'Temperature', '°C', '700 mb')
('var6', 'Sunshine Duration', 'min', 'sfc')
('var7', 'Shortwave Radiation', 'W/m²', 'sfc')
('var8', 'Direct Shortwave Radiation', 'W/m²', 'sfc')
('var9', 'Diffuse Shortwave Radiation', 'W/m²', 'sfc')
('var10', 'Precipitation Total', 'mm', 'sfc')
('var11', 'Snowfall Amount', 'cm', 'sfc')
('var12', 'Relative Humidity', '%', '2 m')
('var13', 'Cloud Cover Total', '%', 'sfc')
('var14', 'Cloud Cover High', '%', 'high cld lay')
('var15', 'Cloud Cover Medium', '%', 'mid cld lay')
('var16', 'Cloud Cover Low', '%', 'low cld lay')
('var17', 'CAPE', 'J/kg', '180-0 mb above gnd')
('var18', 'Mean Sea Level Pressure', 'hPa', 'MSL')
('var19', 'Geopotential Height', 'Gpm', '1000 mb')
('var20', 'Geopotential Height', 'Gpm', '850 mb')
('var21', 'Geopotential Height', 'Gpm', '700 mb')
('var22', 'Geopotential Height', 'Gpm', '500 mb')
('var23', 'Evapotranspiration', 'mm', 'sfc')
('var24', 'FAO Reference Evapotranspiration', 'mm', '2 m')
('var25', 'Temperature', '°C', 'sfc')
('var26', 'Soil Temperature', '°C', '0-10 cm down')
('var27', 'Soil Moisture', 'm³/m³', '0-10 cm down')
('var28', 'Vapor Pressure Deficit', 'hPa', '2 m')
('var29', 'Wind Speed', 'km/h', '10 m')
('var30', 'Wind Direction', '°', '10 m')
('var31', 'Wind Speed', 'km/h', '80 m')
('var32', 'Wind Direction', '°', '80 m')
('var33', 'Wind Gust', 'km/h', 'sfc')
('var34', 'Wind Speed', 'km/h', '900 mb')
('var35', 'Wind Direction', '°', '900 mb')
('var36', 'Wind Speed', 'km/h', '850 mb')
('var37', 'Wind Direction', '°', '850 mb')
('var38', 'Wind Speed', 'km/h', '700 mb')
('var39', 'Wind Direction', '°', '700 mb')
('var40', 'Wind Speed', 'km/h', '500 mb')
('var41', 'Wind Direction', '°', '500 mb')

Weather Data 2022
('var1', 'temperature_2m', '°C')
('var2', 'relativehumidity_2m', '%')
('var3', 'dewpoint_2m', '°C')
('var4', 'rain', 'mm')
('var5', 'snowfall', 'cm')
('var6', 'weathercode', 'wmo code')
('var7', 'pressure_msl', 'hPa')
('var8', 'surface_pressure', 'hPa')
('var9', 'cloudcover', '%')
('var10', 'cloudcover_low', '%')
('var11', 'cloudcover_mid', '%')
('var12', 'cloudcover_high', '%')
('var13', 'windspeed_10m', 'm/s')
('var14', 'windspeed_100m', 'm/s')
('var15', 'winddirection_10m', '°')
('var16', 'winddirection_100m', '°')
('var17', 'windgusts_10m', 'm/s')
('var18', 'direct_radiation', 'W/m²')
('var19', 'diffuse_radiation', 'W/m²')
('var20', 'is_day', '')

Data:
(solar)
(wind)
(demand)
temperature
direct_radiation
diffuse_radiation
windspeed_10m
windspeed_100m
winddirection / cloudcover?
windgusts ?? check if data has same unit
precipitation





sql_query_2022 = f'SELECT time_indices.TimestepID, time_indices.local_time, global_profile_wind.wind_profile_value as pv, global_profile_wind.wind_profile_value as wind, residual_grid_load.P_residual_gridload as demand, var1 as temperature, var18 as direct_radiation, var19 as diffuse__radiation, var13 as windspeed_10, var14 as windspeed_100m, var15 as winddirection, var17 as windgust, var4 as precipitation' \
                f'FROM global_profile_wind, time_indices, global_profiles_pv, weather_data_om, residual_grid_load ' \
                f'WHERE global_profile_wind.TimestepID = time_indices.TimestepID AND time_indices.TimestepID = global_profiles_pv.TimestepID AND time_indices.TimestepID = weather_data_om.TimestepID AND time_indices.TimestepID = residual_grid_load.TimestepID' \
 \
 
 sql_query_2021 = f'SELECT time_indices.TimestepID, time_indices.local_time, global_profile_wind.wind_profile_value as pv, global_profile_wind.wind_profile_value as wind, residual_grid_load.P_residual_gridload as demand, var1 as temperature, var8 as direct_radiation, var9 as diffuse__radiation, var29 as windspeed_10, var31 as windspeed_100m, var30 as winddirection, var33 as windgust, var10 as precipitation' \
                f'FROM global_profile_wind, time_indices, global_profiles_pv, weather_data_ts, residual_grid_load ' \
                f'WHERE global_profile_wind.TimestepID = time_indices.TimestepID AND time_indices.TimestepID = global_profiles_pv.TimestepID AND time_indices.TimestepID = weather_data_ts.TimestepID AND time_indices.TimestepID = residual_grid_load.TimestepID' \
 \

 sql_query_2021_old = f'SELECT time_indices.TimestepID, time_indices.local_time, global_profile_wind.wind_profile_value as wind, global_profiles_pv.Value_Feedin as pv, residual_grid_load.P_residual_gridload as demand, var25 as Temparature, var1 as Temperature_corrected, var6 as Sunshine_Duration, var7 as shortwave_radiation, var8 as direct_shortwave_radiation, var9 as diffuse_shortwave_radiation, \
                 var10 as Precipitation, var13 as Cloud_Cover_Total, var29 as Wind_Speed, var30 as Wind_Direction, var33 as Wind_Gust, var34 as Wind_Speed_900mb, var35 as Wind_Direction_900mb ' \
                f'FROM global_profile_wind, time_indices, global_profiles_pv, weather_data_ts, residual_grid_load ' \
                f'WHERE global_profiles_pv.Orientation = \'S\' ' \
                f'AND global_profile_wind.TimestepID = time_indices.TimestepID AND time_indices.TimestepID = global_profiles_pv.TimestepID AND time_indices.TimestepID = weather_data_ts.TimestepID AND time_indices.TimestepID = residual_grid_load.TimestepID' \
 \

'''
# SUM list_of_measurement_units.Sum Feedin kWh where hasWind == 1 => 53028637.750016004kWh
