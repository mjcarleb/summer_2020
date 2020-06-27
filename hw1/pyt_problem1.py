# Imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Use seed consistently to random processes
my_random_state = 17

# Read data, clean in data, extract features and response variable
df = pd.read_csv("Housing_Data.csv")
badrows = df.index[df['floorArea'] == "sqft"].tolist()
df.drop(badrows, inplace=True)
floor_area = pd.to_numeric(df.floorArea).to_numpy()
bedrooms = df.bedrooms.to_numpy()
price = floor_area * pd.to_numeric(df.pricePerSqFt).to_numpy()

"""
# Look at data before we model
# Plot price vs. floor area
fig, ax = plt.subplots(1, 1, figsize=(10,8))
ax.set_title("Price vs. floor area", fontsize=18)
ax.set_xlabel("floor area", fontsize=14)
ax.set_ylabel("price", fontsize=14)
ax.scatter(floor_area, price)
ax.grid(True)
#plt.show()
"""

# Drop observations with floor area > 7500
idx = np.where(floor_area > 7500)
floor_area = np.delete(floor_area, idx)
bedrooms = np.delete(bedrooms, idx)
price = np.delete(price, idx)

# Look again
"""
fig, ax = plt.subplots(1, 1, figsize=(10,8))
ax.set_title("Price vs. floor area", fontsize=18)
ax.set_xlabel("floor area", fontsize=14)
ax.set_ylabel("price", fontsize=14)
ax.scatter(floor_area, price)
ax.grid(True)
#plt.show()
"""

# Scale X data
scaler = StandardScaler()
X = np.zeros([price.shape[0], 2])
X[:,0] = floor_area
X[:,1] = bedrooms
scaler.fit(X)
X = scaler.transform(X)

# Response variable is in 000 units
y = price / 1000

# Create train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=my_random_state)

# Cites:
# https://livebook.manning.com/book/deep-learning-with-pytorch/chapter-6/v-13/153
# https://pytorch.org/tutorials/beginner/pytorch_with_examples.html

class TwoLayerModel(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        #super(TwoLayerModel, self).__init__()
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.hidden_linear = nn.Linear(self.input_dim, self.hidden_dim)
        self.output_linear = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, input):
        z0 = self.hidden_linear(input)
        u0 = torch.relu(z0)
        y_hat = self.output_linear(u0)

        return y_hat


# Create and show model
model = TwoLayerModel(input_dim=2, hidden_dim=2, output_dim=1)

# https://stackoverflow.com/questions/49433936/how-to-initialize-weights-in-pytorch
def init_all(model, init_func, *params, **kwargs):
    for p in model.parameters():
        init_func(p, *params, **kwargs)

#init_all(model, torch.nn.init.normal_, mean=0., std=1)
init_all(model, torch.nn.init.constant_, 0.1)

with torch.no_grad():
    p_count = 0
    for i, p in enumerate(model.parameters()):
        try:
            p_count += p.size()[0] * p.size()[1]
        except:
            p_count += p.size()[0]
        print(f"{type(p.data)}:  {p.size()}")
    print(f"Total param count = {p_count}")

# Convert model and data to gpu/cpu tensors
if torch.cuda.is_available():
    model.cuda()
    X_train_t = torch.from_numpy(X_train).cuda().float()
    y_train_t = torch.from_numpy(y_train).cuda().float()
    X_test_t = torch.from_numpy(X_test).cuda().float()
    y_test_t = torch.from_numpy(y_test).cuda().float()
else:
    model.cpu()
    X_train_t = torch.from_numpy(X_train).float()
    y_train_t = torch.from_numpy(y_train).float()
    X_test_t = torch.from_numpy(X_test).float()
    y_test_t = torch.from_numpy(y_test).float()


def mini_batch_indices(X, batch_size):

    # Permute the X index to effect shuffling
    idx = np.random.permutation(X.shape[0])

    # Number of mini batches
    # Ignoring "left over data" that do not fit into full batch
    n_batches = X.shape[0] // batch_size

    # Create indices for each mini-batch
    indices = []
    for n_batch in range(n_batches):
        indices.append(idx[n_batch * batch_size:(n_batch + 1) * batch_size])

    return indices


epochs = 100
lr = 2e-3
batch_size = 128
train_loss_history = []
val_loss_history = []
np.random.seed(my_random_state)

# Define Adam optimizer
#optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for epoch in range(epochs):

    # Starting new epoch
    model.train()
    train_mse = 0

    # Permute and split into training data into mini batches
    mb_indices = mini_batch_indices(X=X_train, batch_size=batch_size)

    # Process data for each mini batch in training data
    for i_mb, mb_index in enumerate(mb_indices):

        # New mini-batch
        # Always call zero_grad first to start each time afresh
        model.zero_grad()
        optimizer.zero_grad()

        # Select mini batch
        x_t = X_train_t[mb_index]
        y_t = y_train_t[mb_index]

        # Forward pass: Compute predicted y by passing x to the model
        y_hat = model(x_t)

        # Update mse for epoch
        # mse in squared units
        # mse is mean for the entire mini-batch
        mse = ((y_hat - y_t) ** 2).mean()
        train_mse += mse

        # Back propagate errors and partials
        mse.backward()

        # Update the weights based on partials
        # Always call zero_grad first to start each time afresh
        optimizer.step()

    # Normalize the epoch's train mse by dividing by number of batches
    # Each mse added to train_mse was already the mean for its mini-batch
    train_mse = train_mse / i_mb

    # Now evaluate on validation data
    model.eval()
    val_mse = 0

    # Permute and split validation data into mini batches
    mb_indices = mini_batch_indices(X=X_test, batch_size=batch_size)

    # Process data for each mini batch in validation data
    for i_mb, mb_index in enumerate(mb_indices):

        # Select mini batch
        x_t = X_test_t[mb_index]
        y_t = y_test_t[mb_index]

        # Forward pass: Compute predicted y by passing x to the model
        y_hat = model(x_t)

        mse = ((y_hat - y_t) ** 2).mean()
        val_mse += mse

    # Normalize the validation mse by dividing by number of batches
    # Each mse added to val_mse was already the mean for its mini-batch
    val_mse = val_mse / i_mb

    print(f"Epoch {epoch + 1}:  train_rmse={train_mse ** .5 :3.0f},000, val_rmse={val_mse ** .5 :3.0f},000")

    train_loss_history.append(train_mse)
    val_loss_history.append(val_mse)

# Plot results
fig, ax = plt.subplots(1, 1, figsize=(10,8))
ax.set_title("Training and Validation Loss per Epoch", fontsize=18)
ax.set_xlabel("epoch", fontsize=14)
ax.set_ylabel("rmse", fontsize=14)
ax.plot(np.array(train_loss_history) **.5, label = "train rmse")
ax.plot(np.array(val_loss_history) ** .5, label = "val rmse")
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