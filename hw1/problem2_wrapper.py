# Imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# See myTorch.py for my basic implementation of neural net
from hw1.myTorch import SequentialModel

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
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=.2, random_state=89)

# Create NN model using myTorch implementation
model = SequentialModel(n_inputs=2, hidden_dim=5)
print(model.summary())

# Fit the model
history = model.fit(X_train=X_train, y_train=y_train,
                    X_val=X_val, y_val=y_val,
                    n_epochs=100, lr=.0002,
                    batch_size=128, verbose=True)

# Plot results
fig, ax = plt.subplots(1, 1, figsize=(10,8))
ax.set_title("Training and Validation Loss per Epoch", fontsize=18)
ax.set_xlabel("epoch", fontsize=14)
ax.set_ylabel("mse", fontsize=14)
ax.plot(history["loss"], label = "train loss")
ax.plot(history["val_loss"], label = "val loss")
ax.legend(fontsize=14)
ax.grid(True)
plt.show()
