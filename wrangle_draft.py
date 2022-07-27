#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import env

from sklearn.model_selection import train_test_split


# new_zillow_data() \
# get_zillow_data() \
# prep_zillow() \
# wrangle_zillow()
# 

# In[2]:


print('Files from wrangle: \nnew_zillow_data \nget_zillow_data \nprep_zillow \nsplit_zillow_data \nwrangle_zillow')


# In[3]:


print('df, train, validate, test = wrangle_draft.wrangle_zillow()')


# # 1 Acquire:
# 
# bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt, taxamount, and fips 
# 
# from the zillow database for all 'Single Family Residential' properties.

# In[4]:


url = f'mysql+pymysql://{env.user}:{env.password}@{env.host}/zillow'


# In[ ]:





# # 1 Answer

# In[5]:


def new_zillow_data():
    return pd.read_sql('''SELECT
    p.bedroomcnt,
    p.bathroomcnt,
    p.calculatedfinishedsquarefeet,
    p.taxvaluedollarcnt,
    p.yearbuilt,
    p.fips,
    p.taxamount,
    p.regionidzip,
    t.propertylandusedesc,
    pr.transactiondate
FROM properties_2017 p
LEFT JOIN propertylandusetype t USING (propertylandusetypeid)
LEFT JOIN airconditioningtype a USING (airconditioningtypeid)
RIGHT JOIN predictions_2017 pr USING (parcelid)
WHERE t.propertylandusedesc = 'Single Family Residential'
AND pr.transactiondate LIKE "2017%%";

''', url)


import os

def get_zillow_data():
    filename = "zillow_d.csv"
    
    # if file is available locally, read it
    if os.path.isfile(filename):
        return pd.read_csv(filename, index_col = 0)
    
    # if file not available locally, acquire data from SQL database
    # and write it as csv locally for future use
    else:
        # read the SQL query into a dataframe
        df_zillow = new_zillow_data()
        
        # Write that dataframe to disk for later. Called "caching" the data for later.
        df_zillow.to_csv(filename)

        # Return the dataframe to the calling code
        return df_zillow


# CONSIDERED:
# 
# bulidingclassdesc\
# architecturalstyledesc\
# yardbuildingsqft26\
# decktypeid\
# fireplace\
# poolcnt\
# regionidneighborhood\
# airconditioningdesc\
# numberofstories\
# buildingqualitytypeid\
# airconditioningdesc\
# garagecarcnt\
# garagetotalsqft\
# 
# BUT TOO MANY NULLS (over 50%)
# 
# dropped because redundant-three different columns deal with location
# regionidcity
# fips

# In[6]:


#df = new_zillow_data()


# In[7]:


#df.to_csv('zillow.csv')


# In[8]:


df= get_zillow_data()


# In[9]:


#cols =  df.columns


# In[10]:


# for col in cols:
#     print(df[col].value_counts())
#     print('----------------------')


# In[11]:


df = new_zillow_data()


# # 2 Prep
# 
# Using your acquired Zillow data, walk through the summarization and cleaning steps in your wrangle.ipynb file like we did above. You may handle the missing values however you feel is appropriate and meaninful; remember to document your process and decisions using markdown and code commenting where helpful.
# 

# In[12]:


# df.info()

#i checked it here in my original code. 
#But to prevent it from popping up whenver I import wrangle 
#I commeted out after verifying it worked


# In[13]:


df.shape


# In[14]:


df.isnull().sum()


# # 2 Answer

# In[15]:


def prep_zillow(df):
    #drop nulls
    df.dropna(subset = ['bedroomcnt', 'bathroomcnt', 'calculatedfinishedsquarefeet',
       'taxvaluedollarcnt', 'yearbuilt', 'fips', 'taxamount', 'regionidzip'], inplace = True)

    df.drop(columns = ['taxamount', 'propertylandusedesc', 'transactiondate'],  inplace = True)
    
    df.rename(columns = {'bedroomcnt': 'bed', 'bathroomcnt': 'bath', 'calculatedfinishedsquarefeet' : 'sqft',
   'taxvaluedollarcnt': 'tax_value', 'regionidzip':'zip'}, inplace = True)
    return df
    
    #total data loss from nulls: 3.0%


# In[ ]:





# In[16]:


df.shape


# In[17]:


df = prep_zillow(df)


# In[18]:


df.shape


# In[19]:


52441-50871


# In[20]:


1570/52441


# In[ ]:





# In[ ]:





# In[ ]:





# # 3 
# Store all of the necessary functions to automate your process from acquiring the data to returning a cleaned dataframe witn no missing values in your wrangle.py file. Name your final function wrangle_zillow.

# In[21]:


def split_zillow_data(df):

    train_validate, test = train_test_split(df, test_size=.2, 
                                        random_state=123)
    train, validate = train_test_split(train_validate, test_size=.3, 
                                   random_state=123)
    return train, validate, test


# In[ ]:





# In[22]:


def wrangle_zillow():
    df = get_zillow_data()
    df = prep_zillow(df)
    train, validate, test = split_zillow_data(df)
    return df, train, validate, test


# In[23]:


df, train, validate, test = wrangle_zillow()


# In[24]:


df.shape, train.shape, validate.shape, test.shape


# In[25]:


df


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




