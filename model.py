#!/usr/bin/env python
# coding: utf-8

# In[88]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import requests
import json
from sklearn import preprocessing 


# In[89]:


df = pd.read_csv('/Users/pratikguram/Downloads/data.csv')


# In[90]:


df.head(10)


# In[91]:


df.shape


# In[92]:


df.drop(["data/location/latitude", "data/location/longitude","data/location/zip_code", "data/session_id","data/amount","data/location/state", "data/location/city","data/client_time"], axis = 1, inplace = True) 


# In[93]:


df.shape


# In[94]:


df.head(5)


# In[95]:


df.replace(['View Project', 'Fund Project'], 
             ['<=1', '>1'], inplace = True) 
  


# In[96]:


df.head(5)


# In[97]:


print(df['data/category'].unique())
print(df['data/event_name'].unique())
print(df['data/gender'].unique())
print(df['data/device'].unique())


# In[98]:


df.isnull().sum()


# In[99]:


df.isna().sum()


# In[100]:


df.columns


# In[101]:


category_col =['data/category', 'data/gender', 'data/age', 'data/marital_status', 
               'data/device','data/event_name']  
labelEncoder = preprocessing.LabelEncoder() 
  
mapping_dict ={} 
for col in category_col: 
    df[col] = labelEncoder.fit_transform(df[col]) 
  
    le_name_mapping = dict(zip(labelEncoder.classes_, 
                        labelEncoder.transform(labelEncoder.classes_))) 
  
    mapping_dict[col]= le_name_mapping 
print(mapping_dict) 


# In[ ]:





# In[102]:


df.head(5)


# In[103]:


df.columns


# In[104]:


df.corr()


# In[105]:


x=df.iloc[:,1:]
y=df.iloc[:,0]


# In[106]:


from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score 


# In[107]:


X=df.iloc[:,1:]
Y=df.iloc[:,0]


# In[108]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
from sklearn.ensemble import RandomForestRegressor
rf_random=RandomForestRegressor()
###Hyperparameter Tuning - Randomize search CV
import numpy as np
n_estimators=[int(x) for x in np.linspace(start=100, stop=300, num=15)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 50, num = 10)]
# max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2]
from sklearn.model_selection import RandomizedSearchCV
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

print(random_grid)
#Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor()
# Random search of parameters, using 5 fold cross validation, 
# search across 15 different combinations
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 15, cv = 5, verbose=2, random_state=42, n_jobs = 1)
rf_random.fit(x_train,y_train)


# In[109]:


pickle.dump(rf_random, open('model.pkl','wb'))
model = pickle.load( open('model.pkl','rb'))

