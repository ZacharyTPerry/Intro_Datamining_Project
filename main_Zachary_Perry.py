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

#################
# First I simply want them over a map as this is good geospatial data

violations_per_district = data['ISSUING_AGENCY_NAME'].value_counts()
print(violations_per_district)

import geopandas as gpd
import contextily as ctx
import matplotlib.pyplot as plt

# Ensure LATITUDE and LONGITUDE are floats
data['LATITUDE'] = data['LATITUDE'].astype(float)
data['LONGITUDE'] = data['LONGITUDE'].astype(float)

# Create a GeoDataFrame
gdf = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data.LONGITUDE, data.LATITUDE))

# Convert your DataFrame to a GeoDataFrame
gdf = gpd.GeoDataFrame(
    data, geometry=gpd.points_from_xy(data.LONGITUDE, data.LATITUDE))

# Set the CRS for WGS84 (lat/long)
gdf.crs = 'epsg:4326'

# Convert to Web Mercator for contextily
gdf = gdf.to_crs(epsg=3857)

# Plotting
fig, ax = plt.subplots(figsize=(10, 10))
gdf.plot(ax=ax, color='red', markersize=5, alpha=0.5)

# Add basemap
ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

# Adjust the map extent to your data points
ax.set_xlim(gdf.total_bounds[[0, 2]])
ax.set_ylim(gdf.total_bounds[[1, 3]])

plt.title('Moving Violations in Washington D.C., August 2023')
plt.axis('off')
plt.show()

##############
#Okay and now we can aggregate them to zip code

from tqdm import tqdm
from uszipcode import SearchEngine

# Initialize the SearchEngine
search = SearchEngine()

# Pre-load Washington D.C. zip codes
dc_zip_codes = [zipcode.zipcode for zipcode in search.by_city_and_state(city="Washington", state="DC")]

# Function to get zip codes from latitude and longitude
def get_zipcode(lat, lon):
    # Narrow down the search to a 5-mile radius, assuming we're centralized around D.C.
    possible_zipcodes = search.by_coordinates(lat, lon, radius=5, returns=None)
    # Filter the results to include only D.C. zip codes
    for zipcode in possible_zipcodes:
        if zipcode.zipcode in dc_zip_codes:
            return zipcode.zipcode
    return None

# Apply the function to your data with a progress bar
tqdm.pandas(desc="Finding Zip Codes")  # Initialize tqdm with pandas
data['zipcode'] = data.apply(lambda row: get_zipcode(row['LATITUDE'], row['LONGITUDE']), axis=1)

# Calculate the average fine amount by zip code
average_fines_by_zip = data.groupby('zipcode')['FINE_AMOUNT'].mean()

# Display the results
print(average_fines_by_zip)

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

average_fines_by_ward['WARD_ID'] = 'Ward ' + average_fines_by_ward['WARD_ID'].astype(str)

# Merge the average fines back onto the ward shapefile for mapping
wards_with_fines = wards.merge(average_fines_by_ward, on='WARD_ID', how='left')

# Plotting the gradient map
# fig, ax = plt.subplots(1, 1, figsize=(10, 6))
# wards_with_fines.plot(column='FINE_AMOUNT', ax=ax, legend=True,
#                       legend_kwds={'label': "Average Fine Amount by Ward",
#                                    'orientation': "horizontal"})
# plt.title('Gradient Map of Average Fine Amounts by Ward in Washington D.C.')
# plt.show()

# I can web scrape the incomes and I remember doing it so idk where it is
# Function to fetch income data by wards

import requests
import pandas as pd
def get_income_data_by_wards():
    url = "https://api.censusreporter.org/1.0/data/show/latest"
    table_id = "B19301"
    ward_geo_ids = ["61000US11001", "61000US11002", "61000US11003", "61000US11004",
                    "61000US11005", "61000US11006", "61000US11007", "61000US11008"]
    params = {
        'table_ids': table_id,
        'geo_ids': ','.join(ward_geo_ids)
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        incomes = []
        for geo_id in ward_geo_ids:
            income = data['data'][geo_id][table_id]['estimate']['B19301001']
            ward_number = 'Ward ' + geo_id[-1]  # Extracting ward number from geo_id
            incomes.append({'WARD_ID': ward_number, 'PER_CAPITA_INCOME': income})
        return pd.DataFrame(incomes)
    else:
        print("Failed to retrieve data:", response.status_code)
        return pd.DataFrame()

# Fetch the income data
income_data = get_income_data_by_wards()

# Load the ward shapefile
ward_shapefile_path = 'C:\\Users\\Zac\\Desktop\\Spring 2024 Semester\\Data Mining\\Data_Mining_Project\\Wards_from_2022.shp'
wards = gpd.read_file(ward_shapefile_path)

wards['WARD_ID'] = wards['NAME']
# Merge the income data onto the ward shapefile for mapping
wards_with_income = wards.merge(income_data, on='WARD_ID', how='left')

# # Plotting the gradient map for Per Capita Income
# fig, ax = plt.subplots(1, 1, figsize=(10, 6))
# wards_with_income.plot(column='PER_CAPITA_INCOME', ax=ax, legend=True,
#                        legend_kwds={'label': "Per Capita Income by Ward",
#                                     'orientation': "horizontal"})
# plt.title('Gradient Map of Per Capita Income by Ward in Washington D.C.')
# plt.show()
#

#################
# Okay here we make the 2x1 plot
#
# # Create a figure with two subplots, arranged vertically
# fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 12))
#
# # Plotting the gradient map for fines
# wards_with_fines.plot(column='FINE_AMOUNT', ax=axes[0], legend=True,
#                       legend_kwds={'label': "Average Fine Amount by Ward",
#                                    'orientation': "horizontal"})
# axes[0].set_title('Gradient Map of Average Fine Amounts by Ward in Washington D.C.')
# axes[0].axis('off')  # Optional: Turn off the axis.
#
# # Plotting the gradient map for Per Capita Income
# wards_with_income.plot(column='PER_CAPITA_INCOME', ax=axes[1], legend=True,
#                        legend_kwds={'label': "Per Capita Income by Ward",
#                                     'orientation': "horizontal"})
# axes[1].set_title('Gradient Map of Per Capita Income by Ward in Washington D.C.')
# axes[1].axis('off')  # Optional: Turn off the axis.
#
# plt.tight_layout()  # Adjust layout to fit both subplots neatly
# plt.show()

###############
# Okay and now we get the density of violations by ward
# Calculate the number of fines per ward
fines_per_ward = data_with_wards.groupby('WARD_ID').size().reset_index(name='Number of Violations')

# Adjust 'WARD_ID' in fines_per_ward to match the format in wards
fines_per_ward['WARD_ID'] = 'Ward ' + fines_per_ward['WARD_ID'].astype(str)

# Now merge the data
wards_with_fines_violations = wards.merge(fines_per_ward, on='WARD_ID', how='left')
#
# # Plotting the number of violations per ward
# fig, ax = plt.subplots(1, 1, figsize=(10, 10))
# wards_with_fines_violations.plot(column='Number of Violations', ax=ax, legend=True,
#                       legend_kwds={'label': "Number of Violations by Ward",
#                                    'orientation': "horizontal"},
#                       missing_kwds={'color': 'lightgrey', 'label': 'Missing data'})
#
# ax.set_title('Number of Moving Violations by Ward in Washington D.C.')
# ax.axis('off')  # Hide the axis
#
# plt.show()

#################
# New 1x3 plot
# Create a single-row subplot layout with three columns
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))

# Plotting the gradient map for fines
wards_with_fines.plot(column='FINE_AMOUNT', ax=axes[0], legend=True,
                      legend_kwds={'label': "Average Fine Amount by Ward",
                                   'orientation': "horizontal"})
axes[0].set_title('Average Fine Amounts by Ward')
axes[0].axis('off')  # Hide the axis

# Plotting the gradient map for Per Capita Income
wards_with_income.plot(column='PER_CAPITA_INCOME', ax=axes[1], legend=True,
                       legend_kwds={'label': "Per Capita Income by Ward",
                                    'orientation': "horizontal"})
axes[1].set_title('Per Capita Income by Ward')
axes[1].axis('off')  # Hide the axis

# Plotting the number of violations per ward
wards_with_fines_violations.plot(column='Number of Violations', ax=axes[2], legend=True,
                      legend_kwds={'label': "Number of Violations by Ward",
                                   'orientation': "horizontal"},
                      missing_kwds={'color': 'lightgrey', 'label': 'Missing data'})
axes[2].set_title('Number of Moving Violations by Ward')
axes[2].axis('off')  # Hide the axis

plt.tight_layout()  # Adjust layout to fit all subplots neatly
plt.show()


##############################
# Section
##############################
# Already done above
# # Now we bring in the per capita of each ward and plot them 2x1
# # Create a figure with 1x2 subplots
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
#
# # This needs a fix
# # Plotting the gradient map for Fine Amounts
# fine_plot = wards_with_fines.plot(column='FINE_AMOUNT', ax=ax1, legend=True)
# # Set the title for the plot
# ax1.set_title('Gradient Map of Average Fine Amounts by Ward in Washington D.C.')
#
# # Plotting the gradient map for Per Capita Income
# income_plot = wards_with_income.plot(column='PER_CAPITA_INCOME', ax=ax2, legend=True)
# # Manually setting legend properties after plotting
# income_plot.legend(title="Per Capita Income by Ward", loc='lower left')
# ax2.set_title('Gradient Map of Per Capita Income by Ward in Washington D.C.')
#
# plt.tight_layout()
# plt.show()

# Now we find the correllation between income and fine amounts

# Merge the two datasets on 'WARD_ID'
data_merged = pd.merge(wards_with_fines[['WARD_ID', 'FINE_AMOUNT']],
                       wards_with_income[['WARD_ID', 'PER_CAPITA_INCOME']],
                       on='WARD_ID')

# Calculate the correlation
correlation = data_merged['FINE_AMOUNT'].corr(data_merged['PER_CAPITA_INCOME'])
print("Correlation between fine amounts and per capita income:", correlation)

###########################################################
#EDA Question Three : Is there a solid model to predict where the moving violation will be?
###########################################################

#A random forest attempt

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Data preprocessing

# Checking for NaN values in the 'WARD' column
nan_in_ward = data_with_wards['WARD'].isnull().any()

print("Are there NaN values in the 'WARD' column?", nan_in_ward)
# drop out of ward values there are 7 of 100000
if nan_in_ward:
    print("Number of NaN values in 'WARD':", data_with_wards['WARD'].isnull().sum())
    # Optional: Drop rows with NaN in 'WARD' if you decide to remove these entries
    data_with_wards = data_with_wards.dropna(subset=['WARD'])

# Ensure ISSUE_TIME is an integer or categorical
data_with_wards['ISSUE_TIME'] = pd.to_datetime(data_with_wards['ISSUE_TIME'], format='%H%M', errors='coerce').dt.hour
data_with_wards['ISSUE_TIME'] = data_with_wards['ISSUE_TIME'].fillna(data_with_wards['ISSUE_TIME'].mode()[0])

# Assuming 'data' is already loaded and you've identified which columns are categorical
le = LabelEncoder()
categorical_cols = ['ISSUING_AGENCY_NAME', 'VIOLATION_CODE', 'VIOLATION_PROCESS_DESC', 'PLATE_STATE']  # Update as necessary
for col in categorical_cols:
    data_with_wards[col] = le.fit_transform(data_with_wards[col].astype(str))

# Feature selection (excluding direct geographical features)
features = data_with_wards[['ISSUING_AGENCY_NAME', 'ISSUE_TIME', 'DISPOSITION_CODE', 'FINE_AMOUNT', 'TOTAL_PAID', 'VIOLATION_CODE', 'VIOLATION_PROCESS_DESC', 'PLATE_STATE']]

# Target variable
target = data_with_wards['WARD']  # Ensure 'WARD' is your target column

# Ensure no NaNs across features
features = features.dropna()

# Synchronize indices of features and target
features, target = features.align(target, join='inner', axis=0)

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Grid Search for Hyperparameter Tuning
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42),
                           param_grid=param_grid,
                           scoring='accuracy',
                           cv=3,  # Using 3-fold cross-validation
                           verbose=1,
                           n_jobs=-1)

grid_search.fit(X_train, y_train)

# Output the best parameters and the best score
print("Best parameters:", grid_search.best_params_)
print("Best cross-validation accuracy: {:.2f}".format(grid_search.best_score_))

# Evaluate on the test set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
print("Test set accuracy: {:.2f}".format(accuracy_score(y_test, y_pred)))