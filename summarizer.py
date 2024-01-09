import csv
import sqlite3
import pandas as pd

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


pv_sql_query = "Select MeUID FROM list_of_measurement_units Where has_pv_open_space = 1"
wind_sql_query = "Select MeUID FROM list_of_measurement_units Where has_wind = 1"
demand_sql_query = "Select DISTINCT MeUID FROM list_of_measurement_units"
pv_output_csv = "pv_meter_ids"
wind_output_csv = "wind_meter_ids"
demand_output_csv = "demand_meter_ids"
filePath = "C:\\Users\\Simon\\Desktop\\Bachelorarbeit\\Implementation\\data\\AllSubstations_2018_2022\\"


# choose what do use
ouput_csv = demand_output_csv
sql_query = demand_sql_query
value = "Value_Demand" # or Value_Feedin

# Write the output CSV file
writeOutput(filePath + "SystemStructure.db", ouput_csv, sql_query)

# Read the MeUID values from the just written CSV file
meuid_df = pd.read_csv(f'{ouput_csv}.csv')
meuids = meuid_df['MeUID']

# Initialize a dictionary to store the sums for each TimestepID
timestep_sums = {}

# Loop through the MeUID values and calculate the sum for each
for meuid in meuids:
    if (meuid == 5694 or meuid == 6621 or meuid == 6732 or meuid==6750):
        print(f"skipping {meuid} because of problematic data")
        continue
    # Construct the CSV file name
    csv_filename = f'{filePath}SeparatedSmartMeterData\\{meuid}.csv'

    try:
        # Read the CSV file
        df = pd.read_csv(csv_filename)

        # Group the dataframe by "TimestepID" and sum the "Value_Feedin" for each group
        grouped_df = df.groupby('TimestepID')[value].sum().reset_index()

        # Iterate through the grouped dataframe and update the timestep_sums dictionary
        for index, row in grouped_df.iterrows():
            timestep_id = row['TimestepID']
            feedin_value = row[value]
            if timestep_id in timestep_sums:
                timestep_sums[timestep_id] += feedin_value
            else:
                timestep_sums[timestep_id] = feedin_value
    except FileNotFoundError:
        # Handle the case where the CSV file doesn't exist
        pass
    except pd.errors.ParserError:
        # Handle parsing errors
        print(f"Skipping file {csv_filename} due to parsing error.")
    except Exception as e:
        # Handle other exceptions
        print(f"Error processing file {csv_filename}: {str(e)}")

   # Print the progress
    print(f"Processed {meuid}/{len(meuids)} = {round((meuid / len(meuids)) * 100, 2)}%")


if timestep_sums:
    # Convert the result_dict to a dataframe
    result_df = pd.DataFrame(list(timestep_sums.items()), columns=['TimestepID', 'Value'])

    # Print the result dataframe
    print("\nResult:")
    print(result_df)

    # Specify the path for the output CSV file
    output_csv_file = ouput_csv + "_sum.csv"

    # Write the result to the output CSV file with only "TimestepID" and "Demand" fields
    result_df.to_csv(output_csv_file, index=False)

    print(f"\nResult has been saved to {output_csv_file}")
else:
    print("No valid data found.")

import matplotlib.pyplot as plt

csv_file_path = f'{output_csv_file}'

# Read the CSV file using pandas
df = pd.read_csv(csv_file_path)

# Extract the TimestepID and Value columns
timestep_id = df['TimestepID']
value = df['Value']

# Create a line plot
plt.plot(timestep_id, value)
plt.xlabel('TimestepID')
plt.ylabel('Value')
plt.title('Data from CSV File')
plt.grid(True)

plt.show()