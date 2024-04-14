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
