#!/usr/bin/env python
# coding: utf-8

#  <h1 style ='text-align: center'>
#      Uber - Fare Price Prediction System
#  </h1>

# ### Dataset Used -> **[Kaggle - Uber Fares Dataset](https://www.kaggle.com/datasets/yasserh/uber-fares-dataset)**

# <img src = 'https://www.kindpng.com/picc/m/20-205458_uber-logo-2018-uber-logo-2018-png-transparent.png' width = '4200'>

# #### **Description:**
# Uber is one of the world's largest ridesharing company, providing millions of rides to customers daily. In this project, we will analyze Uber's historical transactional data to build a model for predicting ride fares.
# 
# #### **Goal:** 🎯
# Our goal is to identify key features that correlate with ride fares and build a model to predict **fare prices**. We will train algorithms on this data to create a robust fare price prediction system.

# ---

# # I. Importing Required Dependencies

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings(action = 'ignore')

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import r2_score

import joblib


# ---

# ## II. Loading the Dataset

# In[2]:


uber = pd.read_csv('uber.csv', parse_dates = ['pickup_datetime'])
uber.head()


# In[3]:


uber.info()


# ---

# # III. Data Wrangling

# #### 1. Dropping Unwanted Fields

# ```Dropping the fields 'Unnamed: 0' and 'key' as they are not required.```

# In[4]:


uber.drop(['Unnamed: 0', 'key'], axis = 'columns' , inplace = True)

# At first glance the dataset has almost no null values
uber.isnull().sum()


# #### 2. Dealing with Null Values

# ##### Caveats found in Dataset
# - Null Values are represented by <ins>**zeroes**</ins>.
# 

# ```As per the dataset, uber transactions are done at 0 Latitude and Longitude, which is not correct```

# ![Atlantic Ocean](https://nowiknow.com/wp-content/uploads/Screen-Shot-2017-05-31-at-10.12.15-PM-650x276.png)

# In[5]:


uber.replace(0, None, inplace = True)
uber.isnull().sum()


# In[6]:


print(uber.info())

uber['fare_amount'] = uber['fare_amount'].astype(float)
uber['passenger_count'] = uber['passenger_count'].astype(float)


# ```The pickup and dropoff locations are crucial in order to predict the fare_amount.```  
# ```Dropping the rows with Null Location Values.  ```  
# ```Dropping the rows with Null Passenger Count Values.```

# In[7]:


uber.dropna(subset = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'passenger_count'], inplace = True)
uber.isnull().sum()


# #### 3. Data Preprocessing

# In[8]:


uber['fare_amount'].fillna(uber['fare_amount'].median(), inplace = True)
uber.isnull().sum()


# ```We have 2 sets of Latitudes and Longitudes. Using these, we can calculate the distance travelled```

# ![Formula for Calculating Distance](https://www.auraq.com/wp-content/uploads/2019/02/charlie2.jpg)

# In[9]:


# Using numpy arrays for computation
# Converting the Latitudes and Longitude to Radians
pickup_lat = np.radians(np.array(uber.pickup_latitude.astype(float)))
pickup_lon = np.radians(np.array(uber.pickup_longitude.astype(float)))
dropoff_lat = np.radians(np.array(uber.dropoff_latitude.astype(float)))
dropoff_lon = np.radians(np.array(uber.dropoff_longitude.astype(float)))


# In[10]:


# Computing dlat and dlon 
dlat = pickup_lat - dropoff_lat
dlon = pickup_lon - dropoff_lon


# In[11]:


a = np.sin(dlat / 2)**2 + np.cos(pickup_lat) * np.cos(dropoff_lat) * np.sin(dlon / 2)**2
c = 2 * np.arcsin(np.sqrt(a))


# In[12]:


# Radius of the Earth
R = 6371 # Kilometers
distance_travelled = R * c


# In[13]:


uber['distance_travelled'] = distance_travelled
uber.head(2)


# ```Now that we have distance travelled. Latitudes and Longitudes are no longer required.```

# In[14]:


uber.drop(['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude'],axis = 'columns' ,inplace = True)
uber.head()


# In[15]:


print(f'Found {(uber.distance_travelled == 0).sum()} no. of invalid values in Distance Travelled.\n')

# In order to prevent divide by zero error
uber.drop(uber[uber['distance_travelled'] == 0].index, inplace = True)

uber['fare_per_km'] = uber['fare_amount'] / uber['distance_travelled']
uber.head()


# In[16]:


uber.describe()


# #### 4. Dealing with Outliers and Anomalies

# In[17]:


def return_kmslab(km):
    if km < 5: return '0-5'
    if (km >= 5) & (km < 10): return '5-10'
    if (km >= 10) & (km < 15): return '10-15'
    if (km >= 15) & (km < 20): return '15-20'
    if (km >= 20) & (km < 30): return '20-30'
    if (km >= 30): return '>30'

plt.figure(figsize = (14,4))
plt.hist(uber.distance_travelled.apply(return_kmslab), align ='left')
plt.title('Distance Travelled Slabs')
plt.show()


# In[18]:


def return_fareslab(fare):
    if fare < 5: return '0-5'
    if (fare >= 5) & (fare < 10): return '5-10'
    if (fare >= 10) & (fare < 15): return '10-15'
    if (fare >= 15) & (fare < 20): return '15-20'
    if (fare >= 20) & (fare < 30): return '20-30'
    if (fare >= 30): return '>30'

plt.figure(figsize = (14,4))
plt.hist(uber.fare_per_km.apply(return_fareslab), color = 'g')
plt.title('Fare Per Km Slabs')
plt.show()


# ```We will be dropping rows that meets the below conditions. Outliers are found to be most concentrated in these regions```
# 
# - Distance Travelled less than 0.5 Km and greater than 30 Km
# - Fare Per Km less than 0.5 US dollars and greater than 15 US dollars

# In[19]:


uber.drop(uber[(uber.fare_per_km < 0.5) | (uber.fare_per_km > 15)].index, inplace = True)
uber.drop(uber[(uber.distance_travelled < 0.5) | (uber.distance_travelled > 30)].index, inplace = True)


# #### 5. Data Rearranging

# In[20]:


# Creating a field with pickup year for analysing
uber['pickup_year'] = uber['pickup_datetime'].apply(lambda x: x.year)


# In[21]:


uber['fare/km/passenger'] = uber['fare_per_km'] / uber['passenger_count'] 


# In[22]:


uber = uber[['pickup_year', 'pickup_datetime', 'passenger_count', 'distance_travelled', 'fare_amount' ,'fare/km/passenger', 'fare_per_km']]
uber.drop('pickup_datetime', axis = 'columns', inplace = True)
print(f"The dataset after wrangling has {uber.shape[0]} rows and {uber.shape[1]} columns\n")
uber.head()


# ---

# ## IV. Data Visualization and Insights

# In[23]:


plt.figure(figsize = (14,6))
sns.set_style('ticks')

sns.boxplot(data = uber, x = 'distance_travelled', palette = 'YlGnBu')

plt.title('Distribution of Distance Travelled')
plt.xlabel('KMs Travelled')
plt.show()


# In[24]:


plt.figure(figsize = (14,6))
sns.barplot(x = uber.describe(include = ['float']).loc['50%'].index, y = uber.describe(include = ['float']).loc['50%'], palette = 'muted')
plt.title('Median Across the Dataset')
plt.xticks([0,1,2,3,4], ['Passenger Count', 'Distance Travelled', 'Fare Amount', 'Fare/Km/Passenger', 'Fare Per KM'])
plt.show()


# In[25]:


year = uber.groupby('pickup_year')
plt.style.use('ggplot')


# In[26]:


x = year.agg('mean')['passenger_count'].index
y = year.agg('mean')['passenger_count']

plt.figure(figsize = (12,6))
plt.plot(x, y, marker = 'o', linestyle = "--", label = 'Avg Passenger Count')
plt.xlabel('Year')
plt.ylabel('Average Passenger Count')
plt.legend(loc = 'lower right', fontsize = 15)
plt.title('Avg. Passenger Count Over The Years')

plt.tight_layout()
plt.show()


# In[27]:


x = year.agg('mean')['distance_travelled'].index
y = year.agg('mean')['distance_travelled']

plt.figure(figsize = (12,6))
plt.plot(x, y, marker = 'o', linestyle = "--", color = 'g', label = 'Avg. Distance Travelled')
plt.xlabel('Year')
plt.ylabel('Average Distance Travelled')
plt.legend(loc = 'lower right', fontsize = 15)
plt.title('Avg. Distance Travelled Over The Years')

plt.tight_layout()
plt.show()


# In[28]:


x = year.agg('mean')['fare/km/passenger'].index
y = year.agg('mean')['fare/km/passenger']

plt.figure(figsize = (12,6))
plt.plot(x, y, marker = 'o', linestyle = "--", color = 'purple', label = 'Avg. Fare/Km per Passenger')
plt.xlabel('Year')
plt.ylabel('Average Fare Per Km Per Passenger')
plt.legend(loc = 'lower right', fontsize = 15)
plt.title('Avg. Fare/Km Per Passenger Over The Years')

plt.tight_layout()
plt.show()


# In[29]:


plt.figure(figsize = (14, 8))
sns.heatmap(uber.corr(), annot = True, cmap = 'BuPu')
plt.yticks(rotation = 0)
plt.title('Correlation Matrix Heat-Map')
plt.show()


# ---

# ## V. Machine Learning

# #### Splitting the Dataset

# In[30]:


uber.head()


# In[31]:


X = uber.drop(['fare_amount', 'fare/km/passenger', 'fare_per_km'], axis = 'columns')
y = uber['fare_amount']


# In[32]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


# #### Using Linear Regression Algorithm

# In[33]:


LR = LinearRegression()

LR.fit(X_train, y_train)
y_pred1 = LR.predict(X_test)

LR_score = r2_score(y_test, y_pred1)
print(f'The R\u00B2 score for Linear Regression is {(LR_score * 100).round(2)}%')


# ---

# #### Using Decision Tree Regressor Algorithm

# In[34]:


DT = DecisionTreeRegressor()

DT.fit(X_train, y_train)
y_pred2 = DT.predict(X_test)

DT_score = r2_score(y_test, y_pred2)
print(f'The R\u00B2 score for Linear Regression is {(DT_score * 100).round(2)}%')


# ---

# #### Using Random Forest Regressor Algorithm

# In[35]:


RF = RandomForestRegressor()

RF.fit(X_train, y_train)
y_pred3 = RF.predict(X_test)

RF_score = r2_score(y_test, y_pred3)
print(f'The R\u00B2 score for Linear Regression is {(RF_score * 100).round(2)}%')


# ---

# #### Using Gradient Boosting Regressor Algorithm

# In[36]:


GB = GradientBoostingRegressor()

GB.fit(X_train, y_train)
y_pred4 = GB.predict(X_test)

GB_score = r2_score(y_test, y_pred4)
print(f'The R\u00B2 score for Linear Regression is {(GB_score * 100).round(2)}%')


# ---

# ## VI. Comparing Model Performance

# In[37]:


plt.figure(figsize = (12,6))
plt.title('R\u00B2 Score Across Algorithms')
plt.plot(["Linear Regression", "Decision Tree", "Random Forest", "Gradient Boost"], np.array([LR_score, DT_score, RF_score, GB_score]) * 100, marker = 'o', linestyle = '--')
plt.xlabel('ML Algorithm')
plt.ylabel('R\u00B2 Score')


plt.savefig('Charts/Rsquared score.png', dpi = 96)
plt.tight_layout()
plt.show()


# ### Model Implementation

# In[38]:


model = joblib.dump(GB, 'fare_predictor')


# In[39]:


fare_predictor = joblib.load(model[0])


# ### Predicting Fare Price

# In[ ]:


pickup_year = int(input("Enter the Pickup Year: "))
passenger_count = int(input("Enter the Passenger Count: "))
distance_travelled = float(input("Enter the Distance Travelled(in Kms): "))

fare_amount =  fare_predictor.predict(pd.DataFrame({
    'pickup_year': pickup_year,
    'passenger_count': passenger_count,
    'distance_travelled': distance_travelled
}, index = [0]))

print(f'The Fare Price Prediction for the given parameters is {fare_amount[0].round(2)}$\n')
pd.DataFrame({
    'pickup_year': pickup_year,
    'passenger_count': passenger_count,
    'distance_travelled': distance_travelled,
    'fare_amount': fare_amount
}, index = [0])


# In[ ]:




