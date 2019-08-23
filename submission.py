
# coding: utf-8

# ## India-Ml-Hiring-Hackathon:

# In[1]:


# To handle datasets
import pandas as pd
import numpy as np

# for plotting
import matplotlib.pyplot as plt


# In[2]:


# to display all the columns of the dataframe in the notebook
pd.pandas.set_option('display.max_columns', None)


# In[3]:


# Import data 
df_train = pd.read_csv('Downloads/India Hackathon/train.csv') 
df_test = pd.read_csv('Downloads/India Hackathon/test.csv')
df_train.head()


# In[4]:


# rows and columns of the data
print(df_train.shape)
print(df_test.shape)


# In[5]:


df_train = df_train.drop(["financial_institution","loan_id","origination_date","first_payment_date"],axis=1)
df_test = df_test.drop(["financial_institution","loan_id","origination_date","first_payment_date"],axis=1)


# In[6]:


df_train.head()


# In[7]:


# rows and columns of the data
print(df_train.shape)
print(df_test.shape)


# In[8]:


# One hot encoding
df_train=pd.get_dummies(df_train, prefix=['source','loan_purpose',],drop_first=True)
df_test=pd.get_dummies(df_test, prefix=['source','loan_purpose',],drop_first=True)


# In[9]:


df_train.head()


# In[10]:


# rows and columns of the data
print(df_train.shape)
print(df_test.shape)


# In[11]:


# # Make dummy variables for rank
# data = pd.concat([df, pd.get_dummies(df['source'], prefix='source',drop_first=True)], axis=1)
# data = data.drop('source', axis=1)
# data.head()


# In[12]:


# split into features ans Labels
X = df_train.drop(['m13'],axis=1).values
y = df_train['m13'].values


# In[13]:


df_test=df_test.values


# ## Over Sampling:

# In[14]:


from imblearn.over_sampling import SMOTE
X,y=SMOTE().fit_sample(X,y)


# In[15]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state=0)


# ## Random Forest:

# In[16]:


from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train,y_train)


# In[17]:


y_pred = random_forest.predict(X_test)
y_pred


# In[18]:


# confusion_matrix
from sklearn.metrics import confusion_matrix
cnf_matrix = confusion_matrix(y_test,y_pred)
cnf_matrix


# In[19]:


# Accuracy:
random_forest.score(X_test,y_test)


# In[20]:


from sklearn.metrics import classification_report,f1_score
f1_score(y_test, y_pred,average="binary")


# In[21]:


# heatmap
import seaborn
seaborn.heatmap(cnf_matrix,annot=True)


# ## Sample Submission:

# In[22]:


sample=random_forest.predict(df_test)


# In[23]:


sample = pd.DataFrame(data=sample)
sample[0].value_counts()


# In[24]:


test=pd.read_csv('Downloads/India Hackathon/test.csv')



# In[25]:


loan_id=test["loan_id"]


# In[26]:


data = pd.concat([loan_id,sample], axis=1)


# In[27]:


data=data.rename(columns={0:"m13"})


# In[28]:


data.to_csv('Downloads/sample1.csv')

