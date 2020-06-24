# Imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Read data, clean in data, extract features and response variable
df = pd.read_csv("Housing_Data.csv")
badrows = df.index[df['floorArea'] == "sqft"].tolist()
df.drop(badrows, inplace=True)
floor_area = pd.to_numeric(df.floorArea).to_numpy()
bedrooms = df.bedrooms.to_numpy()
price = floor_area * pd.to_numeric(df.pricePerSqFt).to_numpy()

# Look at data before we model
# Plot price vs. floor area
fig, ax = plt.subplots(1, 1, figsize=(10,8))
ax.set_title("Price vs. floor area", fontsize=18)
ax.set_xlabel("floor area", fontsize=14)
ax.set_ylabel("price", fontsize=14)
ax.scatter(floor_area, price)
ax.grid(True)
#plt.show()

# Drop observations with floor area > 7500
idx = np.where(floor_area > 7500)
floor_area = np.delete(floor_area, idx)
bedrooms = np.delete(bedrooms, idx)
price = np.delete(price, idx)

# Look again
fig, ax = plt.subplots(1, 1, figsize=(10,8))
ax.set_title("Price vs. floor area", fontsize=18)
ax.set_xlabel("floor area", fontsize=14)
ax.set_ylabel("price", fontsize=14)
ax.scatter(floor_area, price)
ax.grid(True)
#plt.show()

# Put data into X & y numpy arrays and scale
scaler = StandardScaler()
X = np.zeros([price.shape[0], 2])
X[:,0] = floor_area
X[:,1] = bedrooms
scaler.fit(X)
X = scaler.transform(X)
y = price / 1000

# Create train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=89)

# Create torch tensor version of train/test data numpy arrays
X_train_t = torch.from_numpy(X_train)
X_test_t = torch.from_numpy(X_test)
y_train_t = torch.from_numpy(y_train)
y_test_t = torch.from_numpy(y_test)


# Cites:
# https://livebook.manning.com/book/deep-learning-with-pytorch/chapter-6/v-13/153
# https://pytorch.org/tutorials/beginner/pytorch_with_examples.html

class TwoLayerModel(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        super().__init__()

        self.hidden_linear = nn.Linear(self.input_dim, self.hidden_dim)

        self.output_linear = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, input):
        hidden_t = self.hidden_linear(input)
        activated_t = torch.relu(hidden_t)
        output_t = self.output_linear(activated_t)

        return output_t


# Create and show model
model = TwoLayerModel(input_dim=2, hidden_dim=2, output_dim=1)

# Define Adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)


def mini_batch_indices(X_train, batch_size):
    # Permute index to train data
    idx = np.random.permutation(X_train.shape[0])

    # Split data into mini batches
    n_batches = X_train.shape[0] // batch_size

    indices = []
    for n_batch in range(n_batches):
        indices.append(idx[n_batch * batch_size:(n_batch + 1) * batch_size])

    return indices


epochs = 200
batch_size = 128
train_loss_history = []
val_loss_history = []

for epoch in range(epochs):

    model.train()
    train_mse = 0

    # Permute and split into mini batches
    mb_indices = mini_batch_indices(X_train=X_train, batch_size=batch_size)

    # Process data for each mini batch
    for i, mb_index in enumerate(mb_indices):
        optimizer.zero_grad()
        model.zero_grad()

        # Create tensors for mini batch
        x_t = torch.from_numpy(X_train[mb_index]).float()
        y_t = torch.from_numpy(y_train[mb_index]).float()

        # Forward pass: Compute predicted y by passing x to the model
        y_hat = model(x_t)

        mse = ((y_hat - y_t) ** 2).mean()
        mse.backward()
        optimizer.step()
        train_mse += mse

    # Calcuate train loss after an epoch
    train_mse = train_mse / i

    model.eval()
    val_mse = 0

    # Permute and split into mini batches
    mb_indices = mini_batch_indices(X_train=X_test, batch_size=batch_size)

    # Process data for each mini batch
    for i, mb_index in enumerate(mb_indices):

        # Create tensors for mini batch
        x_t = torch.from_numpy(X_test[mb_index]).float()
        y_t = torch.from_numpy(y_test[mb_index]).float()

        # Forward pass: Compute predicted y by passing x to the model
        y_hat = model(x_t)

        mse = ((y_hat - y_t) ** 2).mean()
        val_mse += mse

    val_mse = val_mse/i

    print(f"Epoch {epoch + 1}:  train_rmse={train_mse ** .5 :3.0f},000, val_rmse={val_mse ** .5 :3.0f},000")

    train_loss_history.append(train_mse)
    val_loss_history.append(val_mse)

# Plot results
fig, ax = plt.subplots(1, 1, figsize=(10,8))
ax.set_title("Training and Validation Loss per Epoch", fontsize=18)
ax.set_xlabel("epoch", fontsize=14)
ax.set_ylabel("mse", fontsize=14)
ax.plot(train_loss_history, label = "train loss")
ax.plot(val_loss_history, label = "val loss")
ax.legend(fontsize=14)
ax.grid(True)
plt.show()

# Calcluate mean of prices
price_mean = price.mean() / 1000

# Calculate rmse
rmse = np.min(val_loss_history) ** .5

# Show how good our ability to predict is
print(f"Mean price:  ${price_mean :3.0f},000")
print(f"RMSE:  ${rmse :3.0f},000")