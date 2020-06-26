# Imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# See myTorch.py for my basic implementation of neural net
from myTorch import Node, FC_Layer, Net

# Read data, clean in data, extract features and response variable
df = pd.read_csv("Housing_Data.csv")
badrows = df.index[df['floorArea'] == "sqft"].tolist()
df.drop(badrows, inplace=True)
floor_area = pd.to_numeric(df.floorArea).to_numpy()
bedrooms = df.bedrooms.to_numpy()
price = df.value.to_numpy()

# Drop observations with floor area > 7500
idx = np.where(floor_area > 7500)
floor_area = np.delete(floor_area, idx)
bedrooms = np.delete(bedrooms, idx)
price = np.delete(price, idx)

# Put data into X & y numpy arrays and scale
scaler = StandardScaler()
X = np.zeros([price.shape[0], 2])
X[:,0] = floor_area
X[:,1] = bedrooms
scaler.fit(X)
X = scaler.transform(X)
y = price / 1000

# Create train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=89)

# Create NN model using myTorch implementation
model = Net(n_inputs=2, hidden_dim=2)

# Fit the model
# Get MB GD to work
# Add validation data
# Add history
# Move back to notebook with plotting
model.fit(X_train=X_train, y_train=y_train, n_epochs=100, lr=.0002, batch_size=128)