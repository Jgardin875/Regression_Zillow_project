#!/usr/bin/env python
# coding: utf-8

# In[1]:


# personally made imports
import pandas as pd
import env
import wrangle_draft
import wrangle_final
#import explore_py

# typical imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split

# modeling methods
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.preprocessing import PolynomialFeatures

import warnings
warnings.filterwarnings("ignore")


# In[2]:


def plot_categorical_and_continuous_vars(df, cat_col, cont_col):

    for i, col in enumerate(cat_col):
        plt.figure(figsize=(16, 3))
        # i starts at 0, but plot nos should start at 1
        plot_number = i + 1

        l= len(cat_col)

        plt.subplot(1,l,plot_number)

        # Title with column name.
        plt.title(col)
        
        sns.stripplot(x = df[col], y = df.tax_value)
        #--------------------------------------------------------------
        plt.figure(figsize=(16, 3))
        # i starts at 0, but plot nos should start at 1
        plot_number = i + 1

        l= len(cat_col)

        plt.subplot(1,l,plot_number)

        # Title with column name.
        plt.title(col)
        
        sns.boxplot(df[col], y = df.tax_value)
        
        #--------------------------------------------------------------
        plt.figure(figsize=(16, 3))
        # i starts at 0, but plot nos should start at 1
        plot_number = i + 1

        l= len(cat_col)

        plt.subplot(1,l,plot_number)

        # Title with column name.
        plt.title(col)
        
        sns.barplot(df[col], y = df.tax_value)


# In[ ]:





# In[3]:


from sklearn.preprocessing import MinMaxScaler

#Define function to scale all data based on the train subset
def mms_scale_data(train, validate, test):
    
    mms_cols = ['sqft', 'yearbuilt']
    
    train_mms = train.copy()
    validate_mms = validate.copy()
    test_mms = test.copy()
    
    mms = MinMaxScaler()
    
    mms.fit(train[mms_cols])
    
    train_mms[mms_cols] = mms.transform(train[mms_cols])
    validate_mms[mms_cols] = mms.transform(validate[mms_cols])
    test_mms[mms_cols] = mms.transform(test[mms_cols])
    
    return train_mms, validate_mms, test_mms


# In[ ]:





# In[4]:


from sklearn.preprocessing import StandardScaler

#Define function to scale all data based on the train subset
def ss_scale_data(train, validate, test):
    
    ss_cols = ['sqft', 'yearbuilt']
    
    train_ss = train.copy()
    validate_ss = validate.copy()
    test_ss = test.copy()
    
    ss = StandardScaler()
    
    ss.fit(train[ss_cols])
    
    train_ss[ss_cols] = ss.transform(train[ss_cols])
    validate_ss[ss_cols] = ss.transform(validate[ss_cols])
    test_ss[ss_cols] = ss.transform(test[ss_cols])
    
    return train_ss, validate_ss, test_ss


# In[ ]:





# In[5]:


def lin_reg(x_tr_data, y_tr_data, x_val_data, y_val_data):

    # create the model object
    lm = LinearRegression(normalize=True)

    # fit the model ONLY to our training data.  

    lm.fit(x_tr_data, y_tr_data)

    # predict train
    y_tr_predict_lr = lm.predict(x_tr_data)

    # evaluate: rmse
    rmse_train = mean_squared_error(y_tr_data, y_tr_predict_lr)**(1/2)

    # predict validate
    y_val_predict_lr = lm.predict(x_val_data)

    # evaluate: rmse
    rmse_validate = mean_squared_error(y_val_data, y_val_predict_lr)**(1/2)

    print("RMSE for OLS using LinearRegression\nTraining/In-Sample: ", rmse_train.round(2), 
          "\nValidation/Out-of-Sample: ", rmse_validate.round(2))
    return y_tr_predict_lr, y_val_predict_lr


# In[ ]:





# In[6]:


def lasso_lars(x_tr_data, y_tr_data, x_val_data, y_val_data):

    # create the model object
    lm = LassoLars(alpha=1.0)

    # fit the model ONLY to our training data.  

    lm.fit(x_tr_data, y_tr_data)

    # predict train
    y_tr_predict_lass = lm.predict(x_tr_data)

    # evaluate: rmse
    rmse_train = mean_squared_error(y_tr_data, y_tr_predict_lass)**(1/2)

    # predict validate
    y_val_predict_lass = lm.predict(x_val_data)

    # evaluate: rmse
    rmse_validate = mean_squared_error(y_val_data, y_val_predict_lass)**(1/2)

    print("RMSE for OLS using LassoLars\nTraining/In-Sample: ", rmse_train.round(2), 
          "\nValidation/Out-of-Sample: ", rmse_validate.round(2))
    return y_tr_predict_lass, y_val_predict_lass


# In[ ]:





# In[7]:


def glm(x_tr_data, y_tr_data, x_val_data, y_val_data):

    # create the model object
    lm = TweedieRegressor(power=1, alpha=0)

    # fit the model ONLY to our training data.  

    lm.fit(x_tr_data, y_tr_data)

    # predict train
    y_tr_predict_glm = lm.predict(x_tr_data)

    # evaluate: rmse
    rmse_train = mean_squared_error(y_tr_data, y_tr_predict_glm)**(1/2)

    # predict validate
    y_val_predict_glm = lm.predict(x_val_data)

    # evaluate: rmse
    rmse_validate = mean_squared_error(y_val_data, y_val_predict_glm)**(1/2)

    print("RMSE for OLS using GLM\nTraining/In-Sample: ", rmse_train.round(2), 
          "\nValidation/Out-of-Sample: ", rmse_validate.round(2))
    return y_tr_predict_glm, y_val_predict_glm


# In[ ]:





# In[8]:


def polynomial_transform(x_tr_data, y_tr_data, x_val_data, y_val_data):
    # make the polynomial features to get a new set of features
    pf = PolynomialFeatures(degree=2)

    # fit and transform X_train_scaled ONLY training gets fit, even for learning transformation!!!
    x_tr_data_deg2 = pf.fit_transform(x_tr_data)

    # transform X_validate_scaled & X_test_scaled
    x_val_data_deg2 = pf.transform(x_val_data)

    # create the model object
    lm2 = LinearRegression(normalize=True)

    # fit the model to our training data
    lm2.fit(x_tr_data_deg2, y_tr_data)

    # predict train
    y_tr_data_deg2 = lm2.predict(x_tr_data_deg2)

    # evaluate: rmse
    rmse_train = mean_squared_error(y_tr_data, y_tr_data_deg2)**(1/2)

    # predict validate
    y_val_data_deg2 = lm2.predict(x_val_data_deg2)

    # evaluate: rmse
    rmse_validate = mean_squared_error(y_val_data, y_val_data_deg2)**(1/2)

    print("RMSE for Polynomial Model, degrees=2\nTraining/In-Sample: ", rmse_train.round(2), 
          "\nValidation/Out-of-Sample: ", rmse_validate.round(2))

    return y_tr_data_deg2, y_val_data_deg2 


# In[ ]:





# In[9]:


# didnt use this loop, but i liked it and so i put t here for future use


# #Create a looping statement to evaluate all models
# models = [lm, lars, glm]
# 
# for model in models:
#     y_train[str(model)] = model.predict(X_train)
#     rmse_train = sqrt(mean_squared_error(y_train['tip'],
#                                          y_train[str(model)]))
#     
#     y_valid[str(model)] = model.predict(X_valid)
#     rmse_valid = sqrt(mean_squared_error(y_valid['tip'],
#                                          y_valid[str(model)]))
#     
#     print('RMSE for {} model on the train dataset: {}.'.format(model, round(rmse_train, 2)))
#     print('RMSE for {} model on the validate dataset: {}.'.format(model, round(rmse_valid, 2)))
#     print()

# In[ ]:





# In[ ]:





# In[ ]:




