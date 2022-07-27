# Regression_Zillow_project


### Project Planning

Mon  - Explore
Tues - MVP
Wed  - Further Exploration and updates
     - Practice delivery
Wed afternoon - Present



### Data Dictionary



### Steps to reproduce

You will need an env.py file that contains the hostname, username and password of the mySQL database that contains the zillow db. Store that env file locally in the repository.

Clone my repo and don't forget to confirm .gitignore is hiding your env.py file in case you accidently uplaod something

Library imports are pandas, matplotlib.pyplot, seaborn, scipy, sklearn('model_selection, 'metrics', 'linear_model', 'preprocessing') 


### Project Goals

The goal is to beat the baseline model in predicting housing prices for Single-Family homes that had transactions in 2017


### Project Description

Housing is a billion dollar industry for investors and possibly the biggest purchase of a lifetime for some homeowners. Determining where and when to buy can have huge financial repercussions. We aim to help you make an informed decision!


### Initial Testing and Hypotheses

- Locationlocationlocataion. Is it really all about location?

- What catagorical valiables correlate with housing prices?

- What continous variables correlate with  housing prices?

### Report findings

Stong correlation in:
        sqft
        bath


### Recommendations

Larger homes sell for more!

### Future work
- I would like to incorporate having bathrooms as a feature. 

### Detailed Project Plan

### Acquire

Requires:

    env.user
    env.password
    env.host

get_zillow_data()

    Function gets telco_data first from save file inside folder, then, if no file exits, it will pull directly from mysql.

new_zillow_data()

    pulls data from mysql

### Cleaning

- Dropped all nulls (less than 1% of data was lost)
- Dropped outliers (less than 3% of data was lost)

Columns considered, but had to be dropped due to quantity of nulls:

bulidingclassdesc\
architecturalstyledesc\
yardbuildingsqft26\
decktypeid\
fireplace\
poolcnt\
regionidneighborhood\
airconditioningdesc\
numberofstories\
buildingqualitytypeid\
airconditioningdesc\
garagecarcnt\
garagetotalsqft

Split Telco

train, validate, test = prepare.split_telco_data()

    20% of data into test group
    30% of remaining data into validate group (30% of 80% = 24% of total data)
    70% of remaining data into train group (70% of 80% = 56% of total data)

target leakage

    data is further split to avoid target leakage

    x_train = train.drop(columns=['churn'])
    y_train = train.churn

    x_validate = validate.drop(columns=['churn'])
    y_validate = validate.churn

    x_test = test.drop(columns=['churn'])
    y_test = test.churn


### Prep Zillow

- Cleans data
- Applies filter for outliers
- Renames columns for ease
- Drops unnecessary columns


###  Split Zillow 

Data split:

    20% of data into test group
    30% of remaining data into validate group (30% of 80% = 24% of total data)
    70% of remaining data into train group (70% of 80% = 56% of total data)

Data is further split to avoid target leakage:

    x_train = train.drop(columns=['tax_value'])
    y_train = train.tax_value

    x_validate = validate.drop(columns=['tax_value'])
    y_validate = validate.tax_value

    x_test = test.drop(columns=['tax_value'])
    y_test = test.tax_value



### Explore

- Locationlocationlocataion. Is it really all about location?

- What catagorical valiables correlate with housing prices?

- What continous variables correlate with  housing prices?


For example:

Visual:
    created stripplot, scatterplot, barplot, boxplot, and more
    
Statistical: 
      heatmap and pearsons testing


Summary

Features based on visual analysis:

- There are large outliers in nearly every group
- trends along bed, bath, sqft, yearbuilt, and fips   
    
    
 Features based on statistical analysis:

- bath and sqft had strongest correlation
- bed and yearbuilt had weak correlations
- sqft and bath had strong correlation with each other. I decided to use only sqft

Data loss:
- Only ~3% data loss in removing outliers


### Scaling:

    I have scaled data set using two different methods. 

    x-data MinMax Scaling (range 0-1) 
    x_train_mms, x_validate_mms, x_test_mms

    x-data Standard Scaling (mean of data = 0) 
    x_train_ss, x_validate_ss, x_test_ss

    y-data is unscaled and remains the same 
    y_train, y_validate, y_test

### Modeling

Select Evaluation Metric

- minimize rmse for tax_value

Evaluate Baseline

    The baseline is the mean of the target value.

Develop 3 Models:
Evaluate on Train
Evaluate on Validate 


Linear Regression
Polynomial Regression (degree = 2)
Lasso Lars (alpha = 1.0)
TweedieRegression (power=1, alpha=0)


### Evaluate Top Model on Test

Polynomial Regression (degree = 2) on un-scaled data
























































































































