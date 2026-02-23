#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


#%%
# Load the data
data = pd.read_csv('../Data/mystery_virus_daily_active_counts_RELEASE#1.csv', parse_dates=['date'], header=0, index_col=None)

#%%
#Extract the information from data. The x-variable will be the day progressed with the y variable being the number of active cases

x = data['day']
y = data['active reported daily cases']


# Plotting the scatter graph
plt.scatter(x, y, alpha=0.6, color='green')
plt.title('Progression of Mystery Illness by Dates')
plt.xlabel('Days')
plt.ylabel('Active Reported Daily Cases')

plt.show()
# %%
