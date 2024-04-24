#%%
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from scipy.stats import shapiro
import datetime as datetime

#%%
# Load the dataset
file_path = r'/Users/ramanabhaskarkosuru/Documents/GitHub/Intro_Datamining_Project/Moving_Violations_Issued_in_August_2023.csv'
data = pd.read_csv(file_path)

#%%
############################################################
# EDA and descriptive statistics
############################################################

# Display basic information about the dataset
print(data.info())
print(data.head())

# Calculate the number of missing values in each row
missing_values_per_row = data.isnull().sum(axis=1)
missing_values_summary = missing_values_per_row.describe()



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
#%%
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
data.to_csv("processed_dataset.csv", index=False)

# %%
