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
