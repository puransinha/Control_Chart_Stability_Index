#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
**** NOTE ****
1. Kindly refer cell no. 16 for pH prediction and Excel file 'Model_prediction_for_pH.xlsx' for stored prediction results.
   The prediction is done through model.predict() fuction of the respective Model
2. Kindly refer cell no. 15 for Weight prediction and Excel file 'Model_prediction_for_Weight.xlsx' for stored prediction results.
   The prediction is done through model.predict() fuction of the respective Model   
"""



import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler


"""
# Simulating Data for analysis
import csv
from faker import Faker
NUMBER_OF_EXP=100
RECORD_COUNT=9
fake=Faker()
# All ideal parameter values to simulate successful experiment considering 9 Hour cycle, pH value taken at every 3 Hour
def create_csv_file():
    with open('chemical_reactor_regression_dataset.csv','w',newline='') as csvfile:
        fieldnames=['Temperature_of_Mass','Jacket_Return_Temperature','Chemical_Reactor_Pressure','Chemical_Reactor_RPM','pH','Weight'] 

        writer=csv.DictWriter(csvfile,fieldnames=fieldnames)
        writer.writeheader()
        
        for i in range(NUMBER_OF_EXP):
            for j in range(RECORD_COUNT):
                writer.writerow(
                {
                    'Temperature_of_Mass':fake.random.uniform(20,28),
                    'Jacket_Return_Temperature':fake.random.uniform(50,80),
                    'Chemical_Reactor_Pressure':fake.random.uniform(100,200),
                    'Chemical_Reactor_RPM':fake.random_int(min=850,max=1250),
                    'pH':fake.random.uniform(6,8),
                    'Weight':fake.random.uniform(12,15)
                }
                )
create_csv_file()
"""



data=pd.read_csv(r'D:\Assignments\Control Chart and Reactor Stability Index\For pH and Weight Prediction\chemical_reactor_regression_dataset.csv')
data.head()
# Calculating Mean of Parameter for every 3 hour for 9 hour cycle
hour_3_mean=[]
for l in range(1,int(len(data)/9)+1):
    temp=data.iloc[:9*l,]
    hour_3_mean.append(list(temp.mean()))
mean_every_3_hour=pd.DataFrame(hour_3_mean)
# Creating whole data set
mean_every_3_hour.columns=['Temperature_of_Mass','Jacket_Return_Temperature','Chemical_Reactor_Pressure','Chemical_Reactor_RPM','pH','Weight']
mean_every_3_hour.to_excel('every_1_hour_mean_ph_and_weight.xlsx')
hour_3_mean_ph_and_weight=mean_every_3_hour



hour_3_mean_ph_and_weight



# No significant Correlation between target and feature
hour_3_mean_ph_and_weight.corr()



# **************************************** For predicting pH *****************************************
X = hour_3_mean_ph_and_weight.drop(['pH','Weight'],axis=1)
y = hour_3_mean_ph_and_weight['pH']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 10)
print(X.shape)
print(X_test.shape)
print(y.shape)
print(y_test.shape)



# Function to calculate mean absolute error
def mae(y_true, y_pred):
    return np.mean(abs(y_true - y_pred))

# Takes in a model, trains the model, and evaluates the model on the test set
def fit_and_evaluate(model):
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions and evalute
    model_pred = model.predict(X_test)
    model_mae = mae(y_test, model_pred)
    r2=r2_score(y_test, model_pred)
    # Return the performance metric
    # Ploting test results
    print('---------------  Linear Regression  ------------------')
    plt.plot(y_test,'r*',label='Original')
    plt.plot(model_pred,label='Predicted')
    plt.legend()
    plt.show()
    return 'Test MAE - ',model_mae,' Test R2 - ',r2   



# Random Forest
random_forest = RandomForestRegressor(random_state=60)
random_forest_mae = fit_and_evaluate(random_forest)
print('Random Forest Regression Performance on the test set:',random_forest_mae)



# Gradient Boosted
gradient_boosted = GradientBoostingRegressor(random_state=60)
gradient_boosted_mae = fit_and_evaluate(gradient_boosted)
print('Gradient Boosted Regression Performance on the test set:',gradient_boosted_mae)



# K-Nearest Neighbour
knn = KNeighborsRegressor(n_neighbors=10)
knn_mae = fit_and_evaluate(knn)
print('K-Nearest Neighbors Regression Performance on the test set:',knn_mae)


# Linear Regression
lr = LinearRegression()
lr_mae=fit_and_evaluate(lr)
print('Linear Regression Performance on the test set:',lr_mae)



# Support Vector Machine
svm = SVR(C = 1000, gamma = 0.1)
svm_mae = fit_and_evaluate(svm)
print('Support Vector Machine Regression Performance on the test set:',svm_mae)


# **************************************** For predicting Weight ****************************************
X = hour_3_mean_ph_and_weight.drop(['pH','Weight'],axis=1)
y = hour_3_mean_ph_and_weight['Weight']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 10)
print(X.shape)
print(X_test.shape)
print(y.shape)
print(y_test.shape)



# Function to calculate mean absolute error
def mae(y_true, y_pred):
    return np.mean(abs(y_true - y_pred))

# Takes in a model, trains the model, and evaluates the model on the test set
def fit_and_evaluate(model):
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions and evalute
    model_pred = model.predict(X_test)
    model_mae = mae(y_test, model_pred)
    r2=r2_score(y_test, model_pred)
    # Return the performance metric
    # Ploting test results
    print('---------------  Linear Regression  ------------------')
    plt.plot(y_test,'r*',label='Original')
    plt.plot(model_pred,label='Predicted')
    plt.legend()
    plt.show()
    return 'Test MAE - ',model_mae,' Test R2 - ',r2 



# Random Forest
random_forest = RandomForestRegressor(random_state=60)
random_forest_mae = fit_and_evaluate(random_forest)
print('Random Forest Regression Performance on the test set:',random_forest_mae)



# Gradient Boosted
gradient_boosted = GradientBoostingRegressor(random_state=60)
gradient_boosted_mae = fit_and_evaluate(gradient_boosted)
print('Gradient Boosted Regression Performance on the test set:',gradient_boosted_mae)



# K-Nearest Neighbour
knn = KNeighborsRegressor(n_neighbors=10)
knn_mae = fit_and_evaluate(knn)
print('K-Nearest Neighbors Regression Performance on the test set:',knn_mae)



# Linear Regression
lr = LinearRegression()
lr_mae=fit_and_evaluate(lr)
print('Linear Regression Performance on the test set:',lr_mae)



# Support Vector Machine
svm = SVR(C = 1000, gamma = 0.1)
svm_mae = fit_and_evaluate(svm)
print('Support Vector Machine Regression Performance on the test set:',svm_mae)



# ************************************* Models Which Perforemed Best *******************************************
X = hour_3_mean_ph_and_weight.drop(['pH','Weight'],axis=1)
y = hour_3_mean_ph_and_weight['pH']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 10)
# Linear Regression
model=lr = LinearRegression()
# Train the model
model.fit(X_train, y_train)
# Make predictions and evalute
model_pred = model.predict(X_test)
# Return the performance metric
# Ploting test results
print('--------------- Test Set Linear Regression  ------------------')
plt.plot(y_test,'r*',label='Original')
plt.plot(model_pred,label='Predicted')
plt.legend()
plt.show()
# Predictions on whole Dataset 
model_pred = model.predict(X)
# Return the performance metric
# Ploting test results
print('---------------  Whole Data Set Linear Regression  ------------------')
plt.plot(y,'r*',label='Original')
plt.plot(model_pred,label='Predicted')
plt.legend()
plt.show()
# Storing Reuslts in Excel file
model_pred_dict={'True y for pH':y,'Predicted y for pH':model_pred}
df_pH_predictions=pd.DataFrame(model_pred_dict)
df_pH_predictions.to_excel('Model_prediction_for_pH.xlsx')



# ************************************* Models Which Perforemed Best *******************************************
X = hour_3_mean_ph_and_weight.drop(['pH','Weight'],axis=1)
y = hour_3_mean_ph_and_weight['Weight']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 10)
# Linear Regression
model=random_forest = RandomForestRegressor(random_state=60)
# Train the model
model.fit(X_train, y_train)
# Make predictions and evalute
model_pred = model.predict(X_test)

# Return the performance metric
# Ploting test results
print('---------------  Test Set Random Forest  ------------------')
plt.plot(y_test,'r*',label='Original')
plt.plot(model_pred,label='Predicted')
plt.legend()
plt.show()
# Predictions on whole Dataset 
model_pred = model.predict(X)
# Return the performance metric
# Ploting test results
print('--------------- Whole Data Set Random Forest  ------------------')
plt.plot(y,'r*',label='Original')
plt.plot(model_pred,label='Predicted')
plt.legend()
plt.show()
# Storing Reuslts in Excel file
model_pred_dict={'True y for pH':y,'Predicted y for pH':model_pred}
df_pH_predictions=pd.DataFrame(model_pred_dict)
df_pH_predictions.to_excel('Model_prediction_for_weight.xlsx')




