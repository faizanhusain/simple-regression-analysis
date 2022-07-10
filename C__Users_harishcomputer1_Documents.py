#!/usr/bin/env python
# coding: utf-8

# # GRIP: The Sparks Foundation
# Data science and Business Analytics Intern
# Task 1: Prediction Using Supervised ML

# In this task we have to predict the percentage score of a student on the number of hours studied. The task two variables where the feature is the number of hours studied and the target value is the percentage. This can be solved using simple linear regression.

# # Importing the dataset

# In[5]:


#importing all the necessary modules
import pandas as nd #reading the csv file and creating a dataframe
import numpy as np
import matplotlib.pyplot as plt     #for plotting data from url and trained data
from sklearn.linear_model import LinearRegression


# In[4]:


#importing the data set
dataset=nd.read_csv("http://bit.ly/w-data")
print(dataset.shape)
dataset.head()


# In[7]:


#to find the no. of columns and rows
dataset.shape


#  Discover and visulize the data

# In[8]:


# to find the information about our data set
dataset.info()


# In[9]:


# now we will check if our dataset contains null or missing values
dataset.isnull().sum()


# In[18]:


dataset.describe()


# # visulizing the dataset
# 

# In[15]:


#plotting the given data in 2-D 

dataset.plot(x='Hours', y='Scores', style='*')
plt.title('Hours vs Persantage')
plt.xlabel('Hours Studied')
plt.ylabel('Persantage score')
plt.grid()
plt.show()


# From the above graph, we can observe that there is a linear relationship between "hours studied" and "percentage score". so we can use the linear regression supervised machine model on it to predict futher values.

# In[16]:


# we can also use .corr to dertermine the correlation b/w the variables.
dataset.corr()


# # Preparing Data

# In[17]:


dataset.head()


# In[18]:


#using iloc function we will divide the data
x= dataset.iloc[:,:1].values
y= dataset.iloc[:,1:].values


# In[19]:


x


# In[20]:


y


# 
# # Training the Algorithm
# we have to split our data into training and testing and then we will train our model

# In[22]:


#splting the data values obtained into training and testing samples:
from sklearn.model_selection import train_test_split
x_train, x_test,y_train, y_test = train_test_split(x,y,test_size=0.25, random_state=0)


# In[25]:


#obtaing the datataken fir traning are trained using linear Regression model's algorithm
from sklearn.model_selection import train_test_split
model= LinearRegression()
model.fit(x_train, y_train)
print("data trained is successfully")


# # Visualizing the model
# after training the model, now its time to visualize it.

# In[30]:


line= model.coef_*x+ model.intercept_
#plotting for the training data, using a line equation(ax+b)
plt.scatter(x_train, y_train)
plt.plot(x, line)
plt.xlabel('HOURS STUDIED')
plt.ylabel('PERCENTAGE SCORE')
plt.grid()
plt.show()


# In[32]:


#ploting for the testing data
plt.scatter(x_test, y_test)
plt.plot(x, line)
plt.xlabel("Hours studied")
plt.ylabel("percentage Score")
plt.grid()
plt.show()


# # Making predictions
# 

# In[34]:


print(x_test)
y_pred= model.predict(x_test)


# In[35]:


#comparing actual vs prediction
y_test


# In[36]:


y_pred


# In[38]:


#comparing Actual vs predicted
compare= nd.DataFrame({'actual':[y_test], 'predicted':[y_pred]})
compare


# In[42]:


#testing with our own data
hours= 9
own_pred= model.predict([[hours]])
own_pred


# # Evaluating the Model

# In[43]:


from sklearn import metrics

print('mean square error:', metrics.mean_absolute_error(y_test, y_pred))


# In[ ]:




