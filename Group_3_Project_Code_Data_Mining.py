# Okay this is the script file for the data mining project

#########################################################################################################################
# Zac Code
#########################################################################################################################


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

#Display basic information about the dataset
print(data.info())
print(data.head())

missing_values = data.isnull().sum()
missing_values_table = missing_values.reset_index()
missing_values_table.columns = ['Column', 'Missing Values']
print(missing_values_table)

# Calculate the number of missing values in each row
missing_values_per_row = data.isnull().sum(axis=1)
missing_values_summary = missing_values_per_row.describe()

#Compute descriptive statistics for each column
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

# compute the EDA and descriptive statistics on the cleaned dataset
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

# Fixing the ward id column
wards['WARD_ID'] = wards['NAME']

# Merge the average fines back onto the ward shapefile for mapping
# I need to check to make sure that this plots because the plot is empty for the tripleplot
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

#############################
# spatial data eats memeory
#############################
# I need to show the cons of spatial data. it eats memory
import sys
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

# Dictionary to hold variable names and their sizes
memory_usage = {}

# Create a static list of (name, value) pairs to prevent RuntimeError during iteration
items = list(globals().items())

# Check each variable in the copied list
for var_name, var_value in items:
    if isinstance(var_value, pd.DataFrame) or isinstance(var_value, gpd.GeoDataFrame):
        # Get memory size in bytes and convert to megabytes
        memory_usage[var_name] = sys.getsizeof(var_value) / (1024**2)

# Plotting the memory usage
plt.figure(figsize=(10, 5))
plt.bar(memory_usage.keys(), memory_usage.values(), color='skyblue')
plt.ylabel('Memory Usage (MB)')
plt.title('Memory Usage of DataFrames and GeoDataFrames')
plt.xticks(rotation=45)
plt.tight_layout()  # Adjust layout to make room for rotated x-axis labels
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

correlation = data_with_wards['YCOORD'].corr(data_with_wards['FINE_AMOUNT'])
print("Correlation between fine amounts and latitude:", correlation)

correlation = data_with_wards['XCOORD'].corr(data_with_wards['FINE_AMOUNT'])
print("Correlation between fine amounts and longitude:", correlation)

print("This is just to show the spatial abstraction is not only recommeded it is necessary to show the complex relationship")
###########################################################
#EDA Question Three : Is there a regressive relationship between the ward and the FINE_AMOUNT
###########################################################
#A random forest attempt

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


# Data preprocessing I want to make the counts so I can see if they should be included or excluded
issuing_agency_counts = data_with_wards['ISSUING_AGENCY_NAME'].value_counts()
violation_code_counts = data_with_wards['VIOLATION_CODE'].value_counts()
WARD_counts = data_with_wards['WARD'].value_counts()
fine_amount_counts = data_with_wards['FINE_AMOUNT'].value_counts()

print("Counts for ISSUING_AGENCY_NAME:")
print(issuing_agency_counts)
print("\nCounts for VIOLATION_CODE:")
print(violation_code_counts)
print("\nCounts for WARD:")
print(WARD_counts)
print("\nCounts for FINE_AMOUNT:")
print(fine_amount_counts)

print("ISSUING_AGENCY_NAME is dominated it needs to be broken down or abstracted if it is to be included")

# Checking for NaN values in the 'WARD' column. need to do this because it is one of the features it should have no nan
nan_in_ward = data_with_wards['WARD'].isnull().any()
print("Are there NaN values in the 'WARD' column?", nan_in_ward)

# drop out of ward values there are 7 of 100000 the first time you run this =
if nan_in_ward:
    print("Number of NaN values in 'WARD':", data_with_wards['WARD'].isnull().sum())
    data_with_wards = data_with_wards.dropna(subset=['WARD'])

# Ensure ISSUE_TIME is an integer and fill the nans its a feature
data_with_wards['ISSUE_TIME'] = pd.to_datetime(data_with_wards['ISSUE_TIME'], format='%H%M', errors='coerce').dt.hour
data_with_wards['ISSUE_TIME'] = data_with_wards['ISSUE_TIME'].fillna(data_with_wards['ISSUE_TIME'].mode()[0])

le = LabelEncoder()
# declare our two catagotical variables
categorical_cols = ['VIOLATION_CODE', 'WARD']
for col in categorical_cols:
    data_with_wards[col] = le.fit_transform(data_with_wards[col].astype(str))

# Feature selection
features = data_with_wards[['ISSUE_TIME', 'WARD', 'VIOLATION_CODE','LONGITUDE', 'LATITUDE']]
# Target variable
target = data_with_wards['FINE_AMOUNT']
# Ensure no NaNs across features. im doing this to be safe even though i explicitly checked earlier
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

# I am using MSE instead of accuracy as the metric for model diagnosis is MSE not accuracy.
# an accurate model with an MSE of infinity is a horrible model
grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42),
                           param_grid=param_grid,
                           scoring='neg_mean_squared_error',
                           cv=3,
                           verbose=1,
                           n_jobs=-1)
grid_search.fit(X_train, y_train)

# Output the best parameters and the best score
print("Best parameters:", grid_search.best_params_)
print("Best neg MSE: {:.2f}".format(grid_search.best_score_))

# Evaluate on the test set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Calculate regression metrics
mse = mean_squared_error(y_test, y_pred)
rmse = mse**0.5
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Output the metrics
print("Test set MSE: {:.2f}".format(mse))
print("Test set RMSE: {:.2f}".format(rmse))
print("Test set MAE: {:.2f}".format(mae))
print("Test set R-squared: {:.2f}".format(r2))

print(data_with_wards['FINE_AMOUNT'].mean())
print("R^2 of 92 is not bad and MSE is like 204 with mean of like 100 this is pretty bad it should not be used ")


############################################
# We can diagnose the model here use vif scores, plot the residuals, check for multicollinearity

# First residuals
residuals = y_test - y_pred
plt.scatter(y_pred, residuals)
plt.title('Residuals vs Predicted')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.axhline(y=0, color='r', linestyle='-')
plt.show()
print("Not randomley distributed not good at all")

# feture importance
importances = best_model.feature_importances_
indices = np.argsort(importances)[::-1]
print("Feature ranking:")
for f in range(features.shape[1]):
    print(f"{f + 1}. feature {features.columns[indices[f]]} ({importances[indices[f]]})")
print("The code looks to take up a majority of our breathing room. This makes sense since it correllated directly with the fine")

# vif heatmap
import seaborn as sns
corr_matrix = features.corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f")
plt.tight_layout()
plt.show()
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif_data = pd.DataFrame()
vif_data["feature"] = features.columns
vif_data["VIF"] = [variance_inflation_factor(features.values, i) for i in range(len(features.columns))]
print(vif_data)

print("Okay there is high multicollinearity between the ward and the lat and long which makes perfect sense."
      "If i were using any linear regression I would have to do something. Random forest so it handles it well"
      "An interaction term needs to be added to the the equation to show the spatial effect of ward on the violation code"
      "VIF scores are pretty good considering multicollinearity is allowed with forest")

print("It seems that there is potential in the research in the regressive nature of overpolicing in DC"
      ", but the model needs work.")

# I clear the envirment to make sure I start from a clean slate
def clear_all():
    # Caution: This will delete all the variables that have been defined.
    globals().clear()

#########################################################################################################################
# Danalii Code
#########################################################################################################################

#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

data =pd.read_csv("processed.csv")

#%%
#What is the number of each moving violation by category in August 2023?

data.head()

# %%
unique_violations = data['VIOLATION_PROCESS_DESC'].unique()
print(unique_violations)

unique_violations_code = data['VIOLATION_CODE'].unique()
print
# %%
unique_violations_counts = data['VIOLATION_PROCESS_DESC'].value_counts()
print(unique_violations_counts)

# %%
unique_violations_code = data['VIOLATION_CODE'].value_counts()
print(unique_violations_code)
# %%
import pandas as pd
import matplotlib.pyplot as plt


violation_counts = pd.Series({
    "SPEED 11-15 MPH OVER THE SPEED LIMIT": 67969,
    "SPEED 16-20 MPH OVER THE SPEED LIMIT": 13078,
    "FAIL TO STOP PER REGULATIONS FACING RED SIGNAL": 7627,
    "PASSING STOP SIGN WITHOUT COMING TO A FULL STOP": 5302,
    "SPEED 21-25 MPH OVER THE SPEED LIMIT": 2735
})

# Sort the violation counts and select the top 5
top_5_violations = violation_counts.nlargest(5)

# Create the horizontal bar chart
plt.figure(figsize=(10, 6))
top_5_violations.plot(kind='barh', color='skyblue')
plt.xlabel('Frequency')
plt.ylabel('Violation Type')
plt.title('Top 5 Traffic Violations')
plt.gca().invert_yaxis()
plt.show()

# %%
unique_tags = data['PLATE_STATE'].value_counts()
print(unique_tags)
# %%
import matplotlib.pyplot as plt

unique_tags.pop(' ')

# Create a bar chart
plt.figure(figsize=(10, 6))
plt.bar(unique_tags.index, unique_tags.values, color='skyblue')
plt.xlabel('State Tag')
plt.ylabel('Count')
plt.title('Distribution of Vehicle State Tags')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()




# %%
print(data.dtypes)
# %%
data.isnull().sum()
# %%
import seaborn as sns
import matplotlib.pyplot as plt



# Calculate Pearson correlation coefficient
correlation = data['ISSUING_AGENCY_CODE'].corr(data['FINE_AMOUNT'])
print("Pearson correlation coefficient:", correlation)

# Create a box plot
plt.figure(figsize=(10, 6))
sns.boxplot(x='ISSUING_AGENCY_CODE', y='FINE_AMOUNT', data=data)
plt.title('Box Plot of Fine Amount by Issuing Agency')
plt.xlabel('Issuing Agency Code')
plt.ylabel('Fine Amount')
plt.show()

# Create a scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='ISSUING_AGENCY_CODE', y='FINE_AMOUNT', data=data)
plt.title('Scatter Plot of Fine Amount by Issuing Agency')
plt.xlabel('Issuing Agency Code')
plt.ylabel('Fine Amount')
plt.show()



#%%
import pandas as pd
import matplotlib.pyplot as plt


violation_counts = data['VIOLATION_CODE'].value_counts()
violation_counts.plot(kind='bar')
plt.title('Distribution of Violation Codes')
plt.xlabel('Violation Code')
plt.ylabel('Frequency')
plt.show()

# %%
plt.hist(data['FINE_AMOUNT'], bins=20, edgecolor='black')
plt.title('Distribution of Fine Amounts')
plt.xlabel('Fine Amount')
plt.ylabel('Frequency')
plt.show()



# %%
plt.scatter(data['FINE_AMOUNT'], data['TOTAL_PAID'])
plt.title('Fine Amount vs Total Paid')
plt.xlabel('Fine Amount')
plt.ylabel('Total Paid')
plt.show()

# %%
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.boxplot(x='PLATE_STATE', y='FINE_AMOUNT', data=data)
plt.title('Fine Amount Distribution by Plate State')
plt.xlabel('Plate State')
plt.ylabel('Fine Amount')
plt.xticks(rotation=45)
plt.show()

# %%
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
sns.boxplot(data['PLATE_STATE'], data['FINE_AMOUNT'])
plt.title('Fine Amount Distribution by Plate State')
plt.xlabel('Plate State')
plt.ylabel('Fine Amount')
plt.xticks(rotation=45)
plt.show()



# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the data

# Select relevant features and encode categorical variables
#'PLATE_STATE' and 'VIOLATION_PROCESS_DESC' are relevant features
# Encode 'PLATE_STATE' using LabelEncoder
label_encoder = LabelEncoder()
data['PLATE_STATE'] = label_encoder.fit_transform(data['PLATE_STATE'])

# One-hot encode 'VIOLATION_PROCESS_DESC'
data = pd.get_dummies(data, columns=['VIOLATION_PROCESS_DESC'])

# Select predictor variables (X) and the outcome variable (y)
X = data.drop(columns=['OBJECTID', 'LOCATION', 'ISSUE_DATE', 'ISSUE_TIME', 'ISSUING_AGENCY_CODE',
                       'ISSUING_AGENCY_NAME', 'ISSUING_AGENCY_SHORT', 'DISPOSITION_CODE',
                       'DISPOSITION_TYPE', 'TOTAL_PAID', 'PENALTY_1', 'PENALTY_2',
                       'PENALTY_3', 'PENALTY_4', 'PENALTY_5', 'RP_MULT_OWNER_NO',
                       'LATITUDE', 'LONGITUDE', 'MAR_ID', 'GIS_LAST_MOD_DTTM', 'VIOLATION_CODE'])
y = data['FINE_AMOUNT']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize logistic regression model
logreg = LogisticRegression()

# Fit the model on the training data
logreg.fit(X_train, y_train)

# Predict on the testing data
y_pred = logreg.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Get classification report and confusion matrix
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


#%%
coefficients = logreg.coef_
intercept = logreg.intercept_

# Get the names of the features
feature_names = X.columns

# Create a DataFrame to store feature names and their coefficients
coefficients_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients[0]})

# Print the DataFrame
print(coefficients_df)



# %%


# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


features = ['XCOORD', 'YCOORD', 'ISSUE_DATE', 'ISSUE_TIME', 'ISSUING_AGENCY_NAME', 'VIOLATION_CODE']
X = data[features]
y = data['FINE_AMOUNT']


X = pd.get_dummies(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)

# %%

#########################################################################################################################
# Ramana Code
#########################################################################################################################

#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

#%%

# Importing the processed data

data = pd.read_csv('processed.csv')

#%%
# Question : "What is the trend in the number of violations over the observed period, and are there any significant spikes or dips in violation frequency?

# Constructing a Line plot to see how violations vary over time in the month of August (2023)
import matplotlib.pyplot as plt
# Group by 'ISSUE_DATE' and count the frequency of violations for each date
violations_over_time = data.groupby('ISSUE_DATE').size()

# Plotting the line plot
plt.figure(figsize=(10, 6))
violations_over_time.plot(kind='line')
plt.title('Violations Over Time')
plt.xlabel('Date')
plt.ylabel('Count')
plt


# %%

# Question: How does the frequency of violations issued change throughout different times of a day?

import matplotlib.dates as mdates

# Converting 'ISSUE_TIME' to datetime format because right now it is in object datatype format
data['ISSUE_TIME'] = pd.to_datetime(data['ISSUE_TIME'])

# Extracting the hour component
data['HOUR'] = data['ISSUE_TIME'].dt.hour

# Creating bins of 4-hour time windows
bins = [0, 4, 8, 12, 16, 20, 24]  # 4-hour time windows in hours

# Grouping the data by time window and count violations using 'OBJECTID'
time_window_counts = data.groupby(pd.cut(data['HOUR'], bins=bins))['OBJECTID'].count()


# Plotting the bar chart
plt.figure(figsize=(10, 6))
time_window_counts.plot(kind='bar', color=['blue', 'green', 'orange', 'red', 'purple', 'cyan'])
plt.title('Frequency of violations issued based on time of a day')
plt.xlabel('Time Window (AM/PM)')
plt.ylabel('Violations issued')

# Formatting x-axis labels to display time in AM/PM format horizontally
plt.xticks(range(len(time_window_counts)), ['12 AM - 4 AM', '4 AM - 8 AM', '8 AM - 12 PM', '12 PM - 4 PM', '4 PM - 8 PM', '8 PM - 12 AM'], rotation=0)

plt.tight_layout()
plt.show()

#%%
# Question: Can we predict the fine amount using certain features in the dataset and see what features are influencing the fine amount significantly?

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error, r2_score


df = pd.DataFrame(data)


# Preprocess categorical variables using Label Encoding
categorical_features = ['VIOLATION_CODE','ISSUE_TIME', 'ISSUE_DATE']
label_encoders = {}

for feature in categorical_features:
    le = LabelEncoder()
    df[feature] = le.fit_transform(df[feature])
    label_encoders[feature] = le  # Store label encoder for each feature

# Define the features and the target variable
X = df[['XCOORD', 'YCOORD', 'LATITUDE', 'LONGITUDE'] + categorical_features]
y = df['FINE_AMOUNT']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the DecisionTreeRegressor
dt_regressor = DecisionTreeRegressor(random_state=42)

# Fit the model on the training data
dt_regressor.fit(X_train, y_train)

# Make predictions on the test data
y_pred = dt_regressor.predict(X_test)

# Calculate the Mean Squared Error (MSE) on the test data
mse = mean_squared_error(y_test, y_pred)

# Calculate  the Root Mean Squared Error (RMSE) on the test data
rmse = np.sqrt(mse)

# Calculate Mean Absolute Error (MAE) 
mae = mean_absolute_error(y_test, y_pred)

# Calculate R-squared (R^2) score
r_squared = r2_score(y_test, y_pred)

# Print the MSE,RMSE, MAE and R-squared score
print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R-squared (R^2) Score: {r_squared}")
