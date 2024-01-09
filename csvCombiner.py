import pandas as pd

# Read the CSV files into Pandas DataFrames
data = pd.read_csv('weather_pv_wind.csv')
sum = pd.read_csv('demand_meter_ids_sum.csv')

# Perform an inner join on the "TimestepID" column
merged_data = pd.merge(data, sum, on='TimestepID')

# Rename "Value" column to "Value_PV"
merged_data.rename(columns={'Value': 'demand'}, inplace=True)

# Save the merged and summed data to a new CSV file
merged_data.to_csv('weather_pv_wind_demand.csv', index=False)
