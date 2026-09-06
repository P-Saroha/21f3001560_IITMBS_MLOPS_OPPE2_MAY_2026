#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

df = pd.read_csv('data/data.csv')
df.head(5)


# In[13]:


df['gender']=pd.factorize(df['gender'])[0]
df.isna().sum()


# In[18]:


cleaned_df=df.dropna()
cleaned_df.shape[0]


# In[20]:


x = cleaned_df.drop("target", axis=1)
y = cleaned_df["target"]

np.random.seed(42)

# Split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


# In[22]:


from sklearn.model_selection import RandomizedSearchCV

log_reg_grid = {"C": np.logspace(-4, 4, 20),
                "solver": ["liblinear"]}

rs_log_reg = RandomizedSearchCV(LogisticRegression(),
                                param_distributions=log_reg_grid,
                                cv=5,
                                n_iter=20,
                                verbose=True)
rs_log_reg.fit(x_train, y_train)
rs_log_reg.best_params_


# In[23]:


rs_log_reg.score(x_test, y_test)


# In[28]:


x_test.iloc[0]


# In[ ]:


rs_log_reg.predict(x_test.iloc[0])


# In[ ]:




