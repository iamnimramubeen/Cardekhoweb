#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install mysql-connector')



# In[2]:


get_ipython().system('pip install pandas pymysql')


# In[3]:


get_ipython().system('pip install jupyter')


# In[4]:


import pandas as pd
import mysql.connector
import pymysql


# In[5]:


host="localhost"
user="root"
password="Learn@123"
database="car_dekho"


# In[6]:


connection = pymysql.connect(host=host,user=user,password=password,db=database)


# In[7]:


query="SELECT * from `car details`"


# In[8]:


df=pd.read_sql(query,connection)


# In[9]:


df


# In[10]:


df1= df[df.duplicated()]


# In[11]:


df1


# In[12]:


df


# In[13]:


df = df.drop_duplicates()
df


# In[14]:


df.head()


# In[15]:


df.describe()


# In[ ]:





# In[16]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[17]:


sns.boxplot(data = df)
plt.show()


# In[18]:


df.describe()


# In[19]:


Q1=df.selling_price.quantile(0.25)


# In[20]:


Q1


# In[21]:


Q3=df.selling_price.quantile(0.75)


# In[22]:


Q3


# In[23]:


IQR=Q3-Q1


# In[24]:


IQR


# In[25]:


highest_outlier=Q3+1.5*IQR
lowest_outlier=Q1-1.5*IQR

highest_outlier,lowest_outlier


# In[26]:


df1=df[df.selling_price<highest_outlier]


# In[27]:


df1


# In[28]:


df1=df1.drop(columns='name')


# In[29]:


Q1=df1.km_driven.quantile(0.25)


# In[30]:


Q1


# In[31]:


Q3=df1.km_driven.quantile(0.75)


# In[32]:


Q3


# In[33]:


IQR=Q3-Q1


# In[34]:


IQR


# In[35]:


highest_outlier=Q3+1.5*IQR
lowest_outlier=Q1-1.5*IQR

highest_outlier,lowest_outlier


# In[36]:


df2=df1[df1.km_driven<highest_outlier]


# In[37]:


df2


# In[38]:


sns.boxplot(data = df2)
plt.show()


# In[39]:


X=df2.drop(columns='selling_price')
y=df2['selling_price']


# In[40]:


X


# In[41]:


y


# In[42]:


X


# In[43]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[44]:


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn import set_config
from sklearn.preprocessing import OneHotEncoder
#from sklearn.compose import make_column_transform
#from sklearn.pipeline import make_pipeline


# In[45]:


lr=LinearRegression()


# In[46]:


num_cols= X.select_dtypes(include=['int64','float64']).columns
cat_cols= X.select_dtypes(include=['object','category', 'bool']).columns
X[cat_cols]=X[cat_cols].astype('category')
df2['selling_price']=df2['selling_price'].astype('category')


# In[47]:


# Preprocessing for numerical features
numeric_features = X.select_dtypes(include=['float64', 'int64']).columns
numeric_transformer = StandardScaler()

# Preprocessing for categorical features
categorical_features = X.select_dtypes(include=['object']).columns
categorical_transformer = OneHotEncoder()


# In[48]:


# Combine the preprocessing steps
from sklearn.compose import ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num_cols', numeric_transformer, numeric_features),
        ('cat_cols', categorical_transformer, categorical_features)
    ])


# In[49]:


# Create a pipeline with preprocessing and linear regression
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regression', LinearRegression())
])


# In[50]:


RF_pipe= Pipeline(steps=[('preprocessor', preprocessor),
                         ('classifier', LinearRegression())])


# In[51]:


pipeline


# In[52]:


RF_pipe


# In[53]:


RF_pipe.fit(X_train,y_train)


# In[54]:


# Fit the pipeline on the training data
pipeline.fit(X_train, y_train)

# Make predictions on the test data
y_pred = pipeline.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)


# In[55]:


y_pred=RF_pipe.predict(X_test)


# In[56]:


from sklearn import set_config
set_config(display='diagram')


# In[57]:


RF_pipe.fit(X_train,y_train)


# In[58]:


preprocessor


# In[59]:


train_pred_RFpipe= RF_pipe.predict(X_train)
test_pred_RFpipe=RF_pipe.predict(X_test)


# In[60]:


from sklearn.metrics import r2_score

# Assuming y_true is the array of actual values and y_pred is the array of predicted values
r2 = r2_score(y_test, y_pred)
print("R2 score:", r2)


# In[61]:


from sklearn.metrics import confusion_matrix,accuracy_score,recall_score,precision_score,f1_score


# In[62]:


import pickle


# In[63]:


import numpy as np


# In[64]:


import pickle
pickle.dump(RF_pipe, open('pipe.pkl','wb'))


# In[65]:


pipe=pickle.load(open('pipe.pkl', 'rb'))


# In[66]:


import pickle

car_names = df.name

# Specify the file path and name to save the pickle file
file_path = r"C:\Users\easil\car_name.pkl"

# Save the car names to the pickle file
with open(file_path, 'wb') as file:
    pickle.dump(car_names, file)


# In[68]:


pwd


# In[ ]:




