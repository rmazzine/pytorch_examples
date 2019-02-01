# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 15:59:59 2019

@author: rmazzine
"""

''' 

This is a simple code applying a neural network in a supervised learning 
task to solve a linear equation (make a linear regression)

'''
import torch
import torch.nn as nn

# Here we create the data that will be given to the Neural Network
# We will use a linear equation y = 2*X +5


# Create a X torch tensor from 1 to 1000 in 1.0 steps
X = torch.unsqueeze(torch.arange(1,1000,1.0), dim=1)
# Create y following: y = 2*X + 5
y = X*2+5


# Network architecture
model = nn.Sequential(
        nn.Linear(1,1)
        )

# Create a loss function, in this case we will use the mean squared error
critetion = nn.MSELoss()

# Create the optimization function, it will need the parameters of the
# NN model and the learning rate.
optimizer = torch.optim.RMSprop(params=model.parameters(), lr=0.001, weight_decay=0.01)

# Define the number of epochs to run
epoch = 10000

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
    
