#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import env

from sklearn.model_selection import train_test_split


# In[2]:


print('Files from wrangle: \nnew_zillow_data \nget_zillow_data \nprep_zillow \nsplit_zillow_data \nwrangle_zillow')


# In[3]:


print('df, train, validate, test = wrangle_final.wrangle_zillow()')


# # 1 Acquire:
# 
# from the zillow database for all 'Single Family Residential' properties with transactions in 2017

# In[4]:


url = f'mysql+pymysql://{env.user}:{env.password}@{env.host}/zillow'


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
    filename = "zillow.csv"
    
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


# CONSIDERED THE FOLLOWING COLUMNS:
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
# BUT THE ABOVE HAVE TOO MANY NULLS (over 50%)
# THEREFORE THEY ARE NOT INCLUDED IN THE DATASET
# 
# 
# 
# REDUNDANCY NEEDS ADDRESSING-three different columns deal with location: \
# regionidcity \
# fips \
# regionzip

# In[6]:


#df = new_zillow_data()


# In[7]:


#df.to_csv('zillow.csv')


# In[8]:


#df = get_zillow_data()


# In[9]:


#cols =  df.columns


# In[10]:


# for col in cols:
#     print(df[col].value_counts())
#     print('----------------------')


# In[ ]:





# # 2 Prep
# 
# Using your acquired Zillow data, walk through the summarization and cleaning steps in your wrangle.ipynb file like we did above. You may handle the missing values however you feel is appropriate and meaninful; remember to document your process and decisions using markdown and code commenting where helpful.
# 

# In[11]:


#df.shape


# In[12]:


#df.isnull().sum()


# # 2 

# In[13]:


def prep_zillow(df):
    #drop nulls
    df.dropna(subset = ['bedroomcnt', 'bathroomcnt', 'calculatedfinishedsquarefeet',
       'taxvaluedollarcnt', 'yearbuilt', 'taxamount', 'fips', 'regionidzip'], inplace = True)
    
    
    #deal with outliers
    df = df[(df.bathroomcnt < 7) & (df.bathroomcnt > 0)]
    df = df[(df.bedroomcnt < 7) & (df.bedroomcnt > 0)]
    df = df[df.taxamount < 25_000]
    df = df[df.calculatedfinishedsquarefeet < 7_000]
    df = df[df.yearbuilt>=1890]
    df = df[df.taxvaluedollarcnt < 3_000_000]
    df = df[(df.regionidzip < 150_000)]
    
    #drop data leakage columns and search by columns that are no longer needed
    df.drop(columns = ['transactiondate', 'propertylandusedesc', 'taxamount', 'regionidzip', 'bathroomcnt'], inplace = True)
    
    #rename columns for convenience
    df.rename(columns = {'bedroomcnt': 'bed', 'calculatedfinishedsquarefeet' : 'sqft',
   'taxvaluedollarcnt': 'tax_value'}, inplace = True)
    
    #set up columns to make dummies
    df.fips = df.fips.astype('str')
    #df.bed = df.bed.astype('str')
    
    #create dummy columns for catagorical varaibles
    dummy_df = pd.get_dummies(df['fips'], dummy_na=False, drop_first= False)
    df = pd.concat([df, dummy_df], axis=1)
    
    #dummy columns were mostly zeros, and created sparse matrix effect. I think. so i kept beds as numbers. 
    #Turned out, it still didn't make a difference. at least not to the first 3 decimal places. 
    
    return df
    
    #total data loss from nulls and outliers: 3.07%


# In[14]:


df = get_zillow_data()


# In[15]:


df.shape


# In[16]:


df = prep_zillow(df)


# In[17]:


df.shape


# In[18]:


52441-50831


# In[19]:


1610/52441


# In[20]:


#losing 3% of data


# In[ ]:





# In[ ]:





# # 3

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




