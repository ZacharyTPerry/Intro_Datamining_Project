#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

#%%

# Importing the processed data 

data = pd.read_csv('processed_dataset.csv')

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
