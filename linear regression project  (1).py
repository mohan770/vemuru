#!/usr/bin/env python
# coding: utf-8

# # Importing libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# # Importing Dataset.

# In[2]:


data=pd.read_csv('C:/Users/USER/Desktop/insurance.csv')
data


# In[3]:


data.head()


# # Total no: of Rows and Columns

# In[4]:


data.shape


# # Describing Numerical Data.

# In[5]:


data.describe()


# # seeing the dependent variable

# In[6]:


sns.distplot(data.charges)


# In[7]:


target=data['charges']
target


# In[8]:


sns.distplot(target)


# In[ ]:





# # Apply log transformation to convert into normal distributuion

# In[9]:


target_log=np.log1p(target)
sns.distplot(target_log,hist=True)
plt.show()


# # convert children column into categorical.

# In[10]:


data['children']=data['children'].astype(str)


# # Division of categorical and numerical datas

# In[11]:


categorical_columns=[col for col in data.columns.values if data[col].dtype=='object']
data_cat=data[categorical_columns]
data_num=data.drop(categorical_columns,axis=1)


# In[12]:


data_cat


# In[13]:


data_num


# In[14]:


data_cat.isnull().sum()


# In[15]:


data_num.isnull().sum()


# # Histogram for numerical data

# In[16]:


data_num.hist(figsize=(16,20),xlabelsize=8,ylabelsize=8)
plt.show()


# In[ ]:





# In[17]:


data_cat


# In[18]:


data_cat.dtypes


# In[19]:


object_bol = data.dtypes == 'object'
object_bol


# In[ ]:





# # Bar plot for categorical data

# In[20]:


sns.countplot(y=data.smoker)

plt.show()


# In[21]:


sns.countplot(y=data.region)

plt.show()


# In[22]:


sns.countplot(data.children)


# In[23]:


sns.countplot(y=data.sex)

plt.show()


# In[24]:



data_num


# # Correlation for numerical data.

# In[25]:


data_num.corr()


# In[26]:


data_num=data_num.drop('charges',axis=1)


# In[27]:


data_num


# In[28]:


data_cat


# # normalisation

# a=data_num.min()
# b=data_num.max()
# data_num=(data_num-a)/(b-a)
# data_num

# # create dummies for categorical data.

# data_cat_dummies=pd.get_dummies(data_cat,drop_first=True)
# data_cat_dummies.head()

# In[ ]:





# # concatinate data_num,data_cat_dummies except target variable

# newdata=pd.concat([data_num,data_cat_dummies],axis=1)
# newdata.shape

# # splitting data into training,testing

# In[36]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(newdata,target_log,test_size=0.30,random_state=0)
print("x_train",x_train.shape)
print("x_test",x_test.shape)
print("y_train",y_train.shape)
print("y_test",y_train.shape)
y_train
y_test


# # building model

# In[35]:


import statsmodels.api as sm
model1=sm.OLS(y_train,x_train).fit()
model1.summary()


# In[37]:


y_pred=model1.predict(x_test)


# In[38]:


y_pred


# In[39]:


y_test


# In[ ]:





# In[ ]:





# In[ ]:




