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
plt.gca().invert_yaxis()  # Invert y-axis to display the highest count at the top
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
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability if needed
plt.tight_layout()
plt.show()


# %%
import pandas as pd
import statsmodels.api as sm

# Load the dataset
#data = pd.read_csv('your_dataset.csv')  # Replace 'your_dataset.csv' with the actual filename

# Creating dummy variables for 'PLATE_STATE'
plate_state_dummies = pd.get_dummies(data['PLATE_STATE'], prefix='PLATE_STATE', drop_first=True)

# Concatenate dummy variables with the original dataframe
data = pd.concat([data, plate_state_dummies], axis=1)

# Creating a binary column indicating whether the violation was paid or not
data['VIOLATION_PAID'] = (data['DISPOSITION_TYPE'] == 'PAID').astype(int)

# Define predictor variables including dummy variables
predictors = ['FINE_AMOUNT', 'VIOLATION_CODE'] + list(plate_state_dummies.columns)

# Add constant term for the intercept
data['intercept'] = 1

# Fit logistic regression model
logit_model = sm.Logit(data['VIOLATION_PAID'], data[predictors + ['intercept']])
result = logit_model.fit()

# Print summary of the model
print(result.summary())


# %%
print(data.dtypes)
# %%
data.isnull().sum()
# %%
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming 'ISSUING_AGENCY_CODE' is the column name for issuing agency code
# Assuming 'FINE_AMOUNT' is the column name for fine amount
# Assuming 'data' is your DataFrame

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

# Create a scatter plot (optional, for visualization)
plt.figure(figsize=(10, 6))
sns.scatterplot(x='ISSUING_AGENCY_CODE', y='FINE_AMOUNT', data=data)
plt.title('Scatter Plot of Fine Amount by Issuing Agency')
plt.xlabel('Issuing Agency Code')
plt.ylabel('Fine Amount')
plt.show()

# %%
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming 'PLATE_STATE' is the column name for state and 'FINE_AMOUNT' is for fine amount
# Assuming 'data' is your DataFrame

# Calculate Pearson correlation coefficient
correlation = data['PLATE_STATE'].corr(data['FINE_AMOUNT'])
print("Pearson correlation coefficient:", correlation)

# Create a scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='PLATE_STATE', y='FINE_AMOUNT', data=data)
plt.title('Scatter Plot of Fine Amount by State')
plt.xlabel('State')
plt.ylabel('Fine Amount')
plt.xticks(rotation=90)  # Rotate x-axis labels for better visibility
plt.show()


#%%
import pandas as pd
import matplotlib.pyplot as plt

# Assuming data is loaded into a DataFrame called df
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
import pandas as pd
import matplotlib.pyplot as plt

# Assuming data is loaded into a DataFrame called data
violation_process_counts = data['VIOLATION_PROCESS_DESC'].value_counts()

# Select the top 5 categories
top_5 = violation_process_counts.head(5)

# Sum up counts of all other categories
others_count = violation_process_counts.iloc[5:].sum()

# Create a new Series with the top 5 categories and the 'Others' category
modified_counts = top_5.append(pd.Series(others_count, index=['Others']))

# Plot the pie chart
plt.pie(modified_counts, labels=modified_counts.index, autopct='%1.1f%%')
plt.title('Percentage of Top 5 Violation Process Descriptions')
plt.axis('equal')
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
#data = pd.read_csv("your_data.csv")

# Select relevant features and encode categorical variables
# For example, let's assume 'PLATE_STATE' and 'VIOLATION_PROCESS_DESC' are relevant features
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
