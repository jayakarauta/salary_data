#!/usr/bin/env python
# coding: utf-8

# In[1]:


machine learning is divided into two categerious
1) supervised learning 
2) unsupervised learning


# In[ ]:


supervused leraning we have to methods
1)reggresion
2)classification


# In[ ]:


###Importing Libraries


# In[5]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[8]:


###Importing the data


# In[4]:


dataset = pd.read_csv(r"J:\1.ML_PROJECT\1.ML_REGGRESION_PROJECTS\1.SALARY LEANIAR REGGRESION _P\Salary_Data.csv")


# In[ ]:


### Read the first 5 rows & colums in the data 


# In[7]:


dataset.head()


# In[ ]:


### data reprocssing 


# In[9]:


dataset.isna().sum()


# In[ ]:


here we are checking null values in the dataset


# In[ ]:


###featuer matrix


# In[17]:


X = dataset.iloc[:, :1].values
y = dataset.iloc[:, -1].values


# In[ ]:


### Splitting the dataset into the Training set and Test set


# In[25]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[ ]:


# Fitting Simple Linear Regression to the Training set


# In[26]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# In[ ]:


# Predicting the Test set results


# In[ ]:


we habve to check how model is predicting


# In[27]:


y_pred = regressor.predict(X_test)


# In[ ]:


# Visualising the Training set results


# In[ ]:


lets check how model is behaving in train data by using visuliztion methods


# In[28]:


plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


# In[ ]:


# Visualising the Test set results


# In[ ]:


lets check how model is behaving in test data by using visuliztion methods


# In[29]:


plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


# In[ ]:


in prediction graph now all the points are in the same line excpet one.
if we bring that one point closer to the line our model works much more better.

