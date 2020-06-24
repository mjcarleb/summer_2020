"""
CSCI S-89, Introduction to Deep Learning
Mark Carlebach
Homework 1, problem 1

Due:  June 28, 2020

Usage:  $python
"""

# Imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras import layers
from keras import models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Read data, clean in data, extract features and response variable
df = pd.read_csv("Housing_Data.csv")
badrows = df.index[df['floorArea'] == "sqft"].tolist()
df.drop(badrows, inplace=True)
floor_area = pd.to_numeric(df.floorArea).to_numpy()
bedrooms = df.bedrooms.to_numpy()
price = pd.to_numeric(df.pricePerSqFt).to_numpy()
assert floor_area.shape[0] == price.shape[0]
assert bedrooms.shape[0] == price.shape[0]

# Shape and scale data
scaler = StandardScaler()
scaler.fit(np.array([floor_area, bedrooms]).reshape(-1, 2))
X = scaler.transform(np.array([floor_area, bedrooms]).reshape(-1, 2))
y = price

# Create train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=89)


# Create NN model
model = models.Sequential()
model.add(layers.Dense(2, activation='relu', input_shape=(2,)))
model.add(layers.Dense(1, activation='linear'))
print(model.summary())

# Compile and fit the model
epochs = 2
model.compile(loss="mean_squared_error",
              metrics=["accuracy", "mse"],
              optimizer="adam")

history = model.fit(X_train, y_train, epochs=epochs,
                    verbose=1,
                    validation_data=(X_test, y_test))

# Plot results
