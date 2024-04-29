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
