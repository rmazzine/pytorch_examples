# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 15:59:59 2019

@author: rmazzine
"""

''' 

This code uses a Neural Network (made with PyTorch package) to predict
house prices from a dataset of Kaggle, it did not get a good score Gradient Boosting
frameworks like XGBoost got much better results, however, this still could be
an useful code on how use PyTorch in a regression problem.

'''

import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import StandardScaler

train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')


# Function to get the dummies of categorical features and separate from
# the numerical
def get_dummies_and_numericals(data):
    
    df_categorical = pd.DataFrame()
    df_numerical = pd.DataFrame()
    
    for col in data.columns:
        # The object type columns have categorical datatype, so we will
        # one-hot encode those columns
        if data[col].dtypes=='object':
            df_categorical = pd.concat((df_categorical,pd.get_dummies(data[col],prefix=col)),
                                       sort=False, axis=1)
        else:
            df_numerical = pd.concat((df_numerical,data[col]),sort=False,axis=1)
            
    return df_categorical,df_numerical

# Get the categorical and numerical features from train_data
train_df_categorical, train_df_numerical = get_dummies_and_numericals(train_data)
# Concatenate both dataframes
train = pd.concat((train_df_numerical,train_df_categorical),sort=False,axis=1)
# Replace all NaN for 0
train = train.fillna(0)

# Get all columns names, to use in the test_data Dataframe construction
train_col = train.columns

# Get the target (y) and train features
y = torch.tensor(train['SalePrice']).float()
train = train.drop(['SalePrice'],axis=1)

# Create a scaler to use in train and test data 
scaler = StandardScaler()
# Fit and transfor the train data
train = scaler.fit_transform(train)
# Convert the train features to PyTorch tensor
X = torch.tensor(train).float()

# TEST_DATA
# Create Dataframe for the test data
X_df = pd.DataFrame(columns=train_col)

# Get the categorical and numerical features from test_data
test_df_categorical, test_df_numerical = get_dummies_and_numericals(test_data)
features_x = pd.concat((test_df_categorical,test_df_numerical),sort=False,axis=1)

# Append to the dataframe that contains all required columns (to be the same
# as the train features)
X_test = X_df.append(features_x,sort=False)
# Remove the SalePrice column
X_test = X_test.drop('SalePrice',axis=1)
# Replace all NaN values for 0 and transform using the same scale used with training data
X_test = X_test.fillna(0)
X_test = scaler.transform(X_test)


# Network architecture
model = nn.Sequential(
        nn.Linear(289,200),
        nn.ReLU(),
        nn.Linear(200,100),
        nn.ReLU(),
        nn.Linear(100,50),
        nn.ReLU(),
        nn.Linear(50,25),
        nn.ReLU(),
        nn.Linear(25,12),
        nn.ReLU(),
        nn.Linear(12,1)
        )

# Create a loss function, in this case we will use the mean squared error
critetion = nn.MSELoss()

# Create the optimization function, it will need the parameters of the
# NN model and the learning rate.
optimizer = torch.optim.Adagrad(params=model.parameters(), lr=0.01)

# Define the number of epochs to run
epoch = 1000

for e in range(epoch):
    
    
    
    # Forward step
    output = model.forward(X)
    # Calculate the loss of the prediction
    loss = critetion(output,y)
    # Reset the loss for tracking and in the optimizer
    optimizer.zero_grad() # This is extremely important, if not called the code will not run properly
    # Backpropagation step
    loss.backward()
    # Update weights
    optimizer.step()
    
    # Loss of the current step
    current_loss = loss.item()
    
    # Print the loss every 100 epochs
    if e%100==0:
        print(current_loss)


# Predict using the PyTorch model
        
# Convert X_test np array to torch tensor
X_test = torch.tensor(X_test).float()
# Enter in the prediction model
model.eval()
# Disable gradients
with torch.no_grad():
    predictions = model.forward(X_test)
model.train()
predictions = predictions.tolist()
df_pred = pd.DataFrame(predictions)
df_pred.to_csv('pred.csv')
