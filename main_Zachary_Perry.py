# Okay this is the script file for the data mining project

import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from scipy.stats import shapiro

# Load the dataset
file_path = r'C:\Users\Zac\Desktop\Spring 2024 Semester\Visualization\Project_Zachary_Perry\Moving_Violations_Issued_in_August_2023.csv'
data = pd.read_csv(file_path)

############################################################
# EDA and descriptive statistics
############################################################

# Display basic information about the dataset
print(data.info())
print(data.head())

# Calculate the number of missing values in each row
missing_values_per_row = data.isnull().sum(axis=1)
missing_values_summary = missing_values_per_row.describe()

# Compute descriptive statistics for each column
descriptive_stats = data.describe(include='all', datetime_is_numeric=True)

# Adjusted normality tests to handle columns with insufficient non-null values or variance
adjusted_normality_results = {}
for column in data.select_dtypes(include=['float64', 'int64']).columns:
    # Dropping null values and ensuring at least two unique values for the test
    non_null_sample = data[column].dropna()
    if non_null_sample.nunique() > 1:
        # Limiting the sample size to 5000 for practicality (its alot of rows we can just randomley sample)
        sample = non_null_sample.sample(n=min(5000, len(non_null_sample)), random_state=1)
        stat, p_value = shapiro(sample)
        adjusted_normality_results[column] = {'Statistic': stat, 'p-value': p_value}

# Output the results
print("Missing Values per Row Summary:\n", missing_values_summary)
print("\nDescriptive Statistics:\n", descriptive_stats)
print("\nNormality Test Results:\n", adjusted_normality_results)

############################################################
# This is the cleaning section established as necessary by the EDA that showed it was not clean at all
############################################################

# Replace missing values with np.nan for specified geospatial columns
columns_to_nan = ['XCOORD', 'YCOORD', 'LATITUDE', 'LONGITUDE']
data[columns_to_nan] = data[columns_to_nan].replace('', np.nan).astype(float)

# Dropping sparse and empty columns
data = data.drop(columns=['DISPOSITION_DATE', 'BODY_STYLE'])

# Handling missing values in 'ACCIDENT_INDICATOR', converting 'N/A' to np.nan
# Assuming missing values should be treated as missing information, not as False
data['ACCIDENT_INDICATOR'] = data['ACCIDENT_INDICATOR'].replace({'Y': True, 'N': False, 'N/A': np.nan})

# Converting dates to datetime format
data['ISSUE_DATE'] = pd.to_datetime(data['ISSUE_DATE'])
data['GIS_LAST_MOD_DTTM'] = pd.to_datetime(data['GIS_LAST_MOD_DTTM'])

# Ensuring 'OBJECTID' is int64
data['OBJECTID'] = data['OBJECTID'].astype('int64')

# Ensuring 'LOCATION', 'ISSUING_AGENCY_CODE', 'ISSUING_AGENCY_NAME', and 'ISSUING_AGENCY_SHORT' are string
data['LOCATION'] = data['LOCATION'].astype(str)
data['ISSUING_AGENCY_CODE'] = data['ISSUING_AGENCY_CODE'].astype(str)
data['ISSUING_AGENCY_NAME'] = data['ISSUING_AGENCY_NAME'].astype(str)
data['ISSUING_AGENCY_SHORT'] = data['ISSUING_AGENCY_SHORT'].astype(str)

# Ensuring 'VIOLATION_CODE', 'VIOLATION_PROCESS_DESC', and 'PLATE_STATE' are categorical
data['VIOLATION_CODE'] = data['VIOLATION_CODE'].astype('category')
data['VIOLATION_PROCESS_DESC'] = data['VIOLATION_PROCESS_DESC'].astype('category')
data['PLATE_STATE'] = data['PLATE_STATE'].astype('category')

# Numeric conversions with error coercion
data['FINE_AMOUNT'] = pd.to_numeric(data['FINE_AMOUNT'], errors='coerce')
data['TOTAL_PAID'] = pd.to_numeric(data['TOTAL_PAID'], errors='coerce')
data['RP_MULT_OWNER_NO'] = pd.to_numeric(data['RP_MULT_OWNER_NO'], errors='coerce')

# Handling 'ISSUE_TIME' conversion with padding and to datetime
data['ISSUE_TIME'] = data['ISSUE_TIME'].astype(str).str.zfill(4)
data['ISSUE_TIME'] = pd.to_datetime(data['ISSUE_TIME'], format='%H%M', errors='coerce')

# GeoDataFrame creation for spatial data
geometry = [Point(xy) for xy in zip(data['LONGITUDE'], data['LATITUDE'])]
gdf = gpd.GeoDataFrame(data, geometry=geometry)
gdf.drop(columns=['LATITUDE', 'LONGITUDE'], inplace=True)

# Converting 'MAR_ID' to string and 'GIS_LAST_MOD_DTTM' to date
data['MAR_ID'] = data['MAR_ID'].astype(str)
data['GIS_LAST_MOD_DTTM'] = data['GIS_LAST_MOD_DTTM'].dt.date

# Drop ACCIDENT_INDICATOR column
data = data.drop(columns=['ACCIDENT_INDICATOR'])
# Remove rows where XCOORD, YCOORD, LATITUDE, or LONGITUDE is null
data = data.dropna(subset=['XCOORD', 'YCOORD', 'LATITUDE', 'LONGITUDE'])

print("Data cleaning and type conversion completed.")

print(data.columns)

# Define a function to determine the classification of each column
def classify_column(column):
    dtype = column.dtype
    if dtype == 'object' or dtype.name == 'category':
        return 'Categorical'
    elif dtype == 'int64' or dtype == 'float64':
        return 'Numerical'
    elif 'datetime' in str(dtype):  # Check if dtype contains 'datetime'
        if 'timezone' in str(dtype):  # Check if timezone information is present
            return 'Datetime (Timezone-aware)'
        else:
            return 'Datetime'
    else:
        return 'Other'

# Create a dictionary to store column classifications
column_classifications = {}

# Apply the classification function to each column and store the result in the dictionary
for column in data.columns:
    column_classifications[column] = classify_column(data[column])

# Display the column classifications
for column, classification in column_classifications.items():
    print(f"Column '{column}' has classification '{classification}'")

############################################################
# EDA and descriptive statistics on cleaned data
############################################################
# This should have no missing values but we did not make the data normal so they should all still fail

# Ensure proper data types for datetime columns
data['ISSUE_DATE'] = pd.to_datetime(data['ISSUE_DATE'])
data['GIS_LAST_MOD_DTTM'] = pd.to_datetime(data['GIS_LAST_MOD_DTTM']).dt.date
data['ISSUE_TIME'] = pd.to_datetime(data['ISSUE_TIME'], format='%H%M', errors='coerce')

# Re-compute the EDA and descriptive statistics on the cleaned dataset
missing_values_per_row_cleaned = data.isnull().sum(axis=1)
missing_values_summary_cleaned = missing_values_per_row_cleaned.describe()
descriptive_stats_cleaned = data.describe(include='all', datetime_is_numeric=True)

# Adjusted normality tests on the cleaned data
adjusted_normality_results_cleaned = {}
for column in data.select_dtypes(include=['float64', 'int64']).columns:
    non_null_sample = data[column].dropna()
    if non_null_sample.nunique() > 1:
        sample = non_null_sample.sample(n=min(5000, len(non_null_sample)), random_state=1)
        stat, p_value = shapiro(sample)
        adjusted_normality_results_cleaned[column] = {'Statistic': stat, 'p-value': p_value}

# Output the results
print("Missing Values per Row Summary (Cleaned):\n", missing_values_summary_cleaned)
print("\nDescriptive Statistics (Cleaned):\n", descriptive_stats_cleaned)
print("\nNormality Test Results (Cleaned):\n", adjusted_normality_results_cleaned)

############################################################
# Next section EDA Question one : How many moving violations were issued in each district in August 2023?
############################################################

##################
# # First I simply want them over a map as this is good geospatial data
#
# violations_per_district = data['ISSUING_AGENCY_NAME'].value_counts()
# print(violations_per_district)
#
# import geopandas as gpd
# import contextily as ctx
# import matplotlib.pyplot as plt
#
# # Ensure LATITUDE and LONGITUDE are floats
# data['LATITUDE'] = data['LATITUDE'].astype(float)
# data['LONGITUDE'] = data['LONGITUDE'].astype(float)
#
# # Create a GeoDataFrame
# gdf = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data.LONGITUDE, data.LATITUDE))
#
# # Convert your DataFrame to a GeoDataFrame
# gdf = gpd.GeoDataFrame(
#     data, geometry=gpd.points_from_xy(data.LONGITUDE, data.LATITUDE))
#
# # Set the CRS for WGS84 (lat/long)
# gdf.crs = 'epsg:4326'
#
# # Convert to Web Mercator for contextily
# gdf = gdf.to_crs(epsg=3857)
#
# # Plotting
# fig, ax = plt.subplots(figsize=(10, 10))
# gdf.plot(ax=ax, color='red', markersize=5, alpha=0.5)
#
# # Add basemap
# ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
#
# # Adjust the map extent to your data points
# ax.set_xlim(gdf.total_bounds[[0, 2]])
# ax.set_ylim(gdf.total_bounds[[1, 3]])
#
# plt.title('Moving Violations in Washington D.C., August 2023')
# plt.axis('off')
# plt.show()

###############
# Okay and now we can aggregate them to zip code

# from tqdm import tqdm
# from uszipcode import SearchEngine
#
# # Initialize the SearchEngine
# search = SearchEngine()
#
# # Pre-load Washington D.C. zip codes
# dc_zip_codes = [zipcode.zipcode for zipcode in search.by_city_and_state(city="Washington", state="DC")]
#
# # Function to get zip codes from latitude and longitude
# def get_zipcode(lat, lon):
#     # Narrow down the search to a 5-mile radius, assuming we're centralized around D.C.
#     possible_zipcodes = search.by_coordinates(lat, lon, radius=5, returns=None)
#     # Filter the results to include only D.C. zip codes
#     for zipcode in possible_zipcodes:
#         if zipcode.zipcode in dc_zip_codes:
#             return zipcode.zipcode
#     return None
#
# # Apply the function to your data with a progress bar
# tqdm.pandas(desc="Finding Zip Codes")  # Initialize tqdm with pandas
# data['zipcode'] = data.apply(lambda row: get_zipcode(row['LATITUDE'], row['LONGITUDE']), axis=1)
#
# # Calculate the average fine amount by zip code
# average_fines_by_zip = data.groupby('zipcode')['FINE_AMOUNT'].mean()
#
# # Display the results
# print(average_fines_by_zip)

############################################################
# EDA Question Two : Is there correlation present in average moving violations and GDP per capita of the ward?
############################################################

import matplotlib.pyplot as plt

# Convert your DataFrame to a GeoDataFrame with the appropriate coordinates
gdf = gpd.GeoDataFrame(
    data,
    geometry=gpd.points_from_xy(data.LONGITUDE, data.LATITUDE),
    crs='epsg:4326'
)

# Load the ward boundary shapefile
ward_shapefile_path = 'C:\\Users\\Zac\\Desktop\\Spring 2024 Semester\\Data Mining\\Data_Mining_Project\\Wards_from_2022.shp'
wards = gpd.read_file(ward_shapefile_path)

# Perform the spatial join to determine which ward each violation occurred in
data_with_wards = gpd.sjoin(gdf, wards, how='left', op='within')

# Assuming the ward identifier column in your shapefile is named 'WARD_ID'
average_fines_by_ward = data_with_wards.groupby('WARD_ID')['FINE_AMOUNT'].mean().reset_index()

# Merge the average fines back onto the ward shapefile for mapping
wards_with_fines = wards.merge(average_fines_by_ward, on='WARD_ID', how='left')

# Plotting the gradient map
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
wards_with_fines.plot(column='FINE_AMOUNT', ax=ax, legend=True,
                      legend_kwds={'label': "Average Fine Amount by Ward",
                                   'orientation': "horizontal"})
plt.title('Gradient Map of Average Fine Amounts by Ward in Washington D.C.')
plt.show()

# # Now we run a correlation test on the aggregated FINE_AMOUNT and the per capita income data for each ward
# # First we have to web scrape since I cannot find the ward income data in a nice tableimport requests
# # import pandas as pd
#
# import requests
#
#
# def get_income_data_by_wards():
#     url = "https://api.censusreporter.org/1.0/data/show/latest"
#     table_id = "B19301"
#     ward_geo_ids = ["61000US11001", "61000US11002", "61000US11003", "61000US11004",
#                     "61000US11005", "61000US11006", "61000US11007", "61000US11008"]
#     params = {
#         'table_ids': table_id,
#         'geo_ids': ','.join(ward_geo_ids)
#     }
#     response = requests.get(url, params=params)
#     incomes = []
#     if response.status_code == 200:
#         data = response.json()
#         for geo_id in ward_geo_ids:
#             income = data['data'][geo_id][table_id]['estimate']['B19301001']
#             ward_number = geo_id[-2:]  # Assuming the last two characters represent the ward number
#             incomes.append({'WARD_ID': 'Ward ' + ward_number, 'PER_CAPITA_INCOME': income})
#         return incomes
#     else:
#         print("Failed to retrieve data:", response.status_code)
#         return None
#
# # Save the incomes in a DataFrame
# income_data = pd.DataFrame(get_income_data_by_wards())
#
# import geopandas as gpd
# import matplotlib.pyplot as plt
#
# # Assume 'income_data' is the DataFrame obtained from the previous step
# gdf = gpd.GeoDataFrame(
#     income_data,
#     geometry=gpd.points_from_xy(income_data.LONGITUDE, income_data.LATITUDE),
#     crs='epsg:4326'
# )
#
# ward_shapefile_path = 'C:\\Users\\Zac\\Desktop\\Spring 2024 Semester\\Data Mining\\Data_Mining_Project\\Wards_from_2022.shp'
# wards = gpd.read_file(ward_shapefile_path)
#
# # Merge the income data back onto the ward shapefile for mapping
# wards_with_income = wards.merge(income_data, on='WARD_ID', how='left')
#
# # Plotting the gradient map for Per Capita Income
# fig, ax = plt.subplots(1, 1, figsize=(10, 6))
# wards_with_income.plot(column='PER_CAPITA_INCOME', ax=ax, legend=True,
#                        legend_kwds={'label': "Per Capita Income by Ward",
#                                     'orientation': "horizontal"})
# plt.title('Gradient Map of Per Capita Income by Ward in Washington D.C.')
# plt.show()
