import os
import pandas as pd
import folium
from folium.plugins import HeatMap

# Set the correct path to the data folder
data_folder = os.path.join(os.getcwd(), '..', 'data')

# Specify the file name of the data to be loaded
file_name = 'uber-raw-data-apr24.csv'
file_path = os.path.join(data_folder, file_name)

# Check if the file exists in the directory
if os.path.exists(file_path):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    print(f"Loaded {file_name} with shape: {df.shape}")
else:
    print(f"File '{file_name}' not found in '{data_folder}'")
    exit()

# Convert 'Date/Time' column to pandas datetime
df['Date/Time'] = pd.to_datetime(df['Date/Time'])

# Derive 'Hour' and 'Weekday' columns for analysis
df['Hour'] = df['Date/Time'].dt.hour
df['Weekday'] = df['Date/Time'].dt.day_name()

# Check for missing values and duplicates
missing_values = df.isnull().sum()
duplicates = df.duplicated().sum()

print(f"Missing values:\n{missing_values}")
print(f"Duplicate rows: {duplicates}")

# Drop duplicates
df_cleaned = df.drop_duplicates()

# Verify duplicates are removed
duplicates_after = df_cleaned.duplicated().sum()
print(f"Number of duplicate rows after removal: {duplicates_after}")

# Display the shape of the cleaned dataset
print(f"Shape of the cleaned dataset: {df_cleaned.shape}")

# Round Lat/Lon to 6 decimal places for consistency
df_cleaned.loc[:, 'Lat'] = df_cleaned['Lat'].round(6)
df_cleaned.loc[:, 'Lon'] = df_cleaned['Lon'].round(6)

# Save the cleaned dataset
df_cleaned.to_csv('final_uber_apr14_dataset.csv', index=False)
print("Final cleaned dataset saved.")

# Create the visuals folder if it doesn't exist
visuals_folder = os.path.join(os.getcwd(), 'visuals')
if not os.path.exists(visuals_folder):
    os.makedirs(visuals_folder)

# Create a base map centered around NYC (latitude, longitude)
nyc_map = folium.Map(location=[40.7128, -74.0060], zoom_start=11)

# Create a list of pickup points (lat, lon) to use for the heatmap
heat_data = [[row['Lat'], row['Lon']] for index, row in df_cleaned.iterrows()]

# Add a HeatMap layer to highlight areas of high activity
HeatMap(heat_data).add_to(nyc_map)

# Save the map in the 'visuals' folder
heatmap_path = os.path.join(visuals_folder, 'nyc_uber_heatmap.html')
nyc_map.save(heatmap_path)
print(f"Heatmap saved to {heatmap_path}")

# Verify the first few rows of the dataset
print(df_cleaned.head())
