import numpy as np

class Node:

    """
    Basic building block of neural network
    Each node contains the following:
        - node_id:  unique id
        - inputs:  list of values passed in the forward direction
        - weights_in:  list of weights for each incoming connection (including bias)
        - z:  sum of (inputs * weights_in)
        - u:  act(z)
        - weights_out:  list of weights for each outgoing connection
        - partials:  partials of the loss with respect to each weight_in (for each node in layer and weight_in)
    """

    # Class attribute to give id to each node created
    class_id = 0

    def __init__(self, weights_in):

        Node.class_id += 1
        self.node_id = Node.class_id
        self.inputs = None
        self.weights_in = weights_in

        # Calculated on forward propagation
        self.z = None              # single sum of inputs * weights_in
        self.u = None              # single u to each outgoing node

        # Added after calculating all weights_in for network, just for convenience
        self.weights_out = None

        # Calculate on backward propagation
        self.fprime = None
        self.error = None           # partial L wrt u
        self.partials = None        # list for each incoming weight

    def forward(self, inputs, eval_mode=False):

        if not eval_mode:
            self.inputs = inputs

        self.z = (inputs * self.weights_in).sum()
        self.u = self.z if self.z > 0 else 0

class FC_Layer:

    """
    Layer is list of nodes
    This is a fully connected layer so each node has weight from each node in preceding layer

    Each layer contains the following:
        - layer_id:  unique id
        - nodes:  list of nnodes nodes (initialized with random weights) where each node as weight from bias
                  and each node of prior input layer (e.g., fully connected)
    """

    # Class attribute to give id to each layer created
    class_id = 0

    def __init__(self, nnodes, input_dim):
        FC_Layer.class_id += 1
        self.layer_id = FC_Layer.class_id

        self.nodes = []
        for i in range(nnodes):
            weights_in = np.random.random(input_dim + 1)
            self.nodes.append(Node(weights_in=weights_in))

    def forward(self, inputs, eval_mode=False):

        # For each node in the layer...
        for node in self.nodes:

            # Calculate z and u for each node and store in node
            # node.z = sum(weights * inputs)
            # node.u = act(node.z)
            node.forward(inputs=inputs, eval_mode=eval_mode)


def mini_batch_indices(X, batch_size):
    # Permute index to train data
    idx = np.random.permutation(X.shape[0])

    # Split data into mini batches
    n_batches = X.shape[0] // batch_size

    indices = []
    for n_batch in range(n_batches):
        indices.append(idx[n_batch * batch_size:(n_batch + 1) * batch_size])

    return indices


class Net:

    """
    A list of fully connected layers

    The first layer has n_inputs
    The last layer has 1 output (to keep this simple)
    There are n_hidden_layers in between that are fully connected with hidden_dim nodes each
    """

    def __init__(self, n_inputs, hidden_dim, n_hidden_layers=1):

        # Start with first hidden layer
        self.layers = []
        self.layers.append(FC_Layer(nnodes=hidden_dim, input_dim=n_inputs))

        # Add other hidden layers (if there are any)
        for n in range(1, n_hidden_layers):
            self.layers.append(FC_Layer(nnodes=hidden_dim, input_dim=hidden_dim))

        # End with output layer
        # Assumes single value output
        self.layers.append(FC_Layer(nnodes=1, input_dim=hidden_dim))

        # Above only adds weights_in
        # Now add the weights_out
        for i, layer in enumerate(self.layers[:-1]):

            next_layer = self.layers[i + 1]
            for j, node in enumerate(layer.nodes):
                node.weights_out = [n.weights_in[j+1] for n in next_layer.nodes]

    def predict(self, X, y):

        # Reset to 0
        predict_mse = 0

        # For each observation...
        for i_obs, observation in enumerate(X):

            # Add bias to input
            X_plus = np.array([1] + list(observation))

            # Forward propagation
            for layer in self.layers:

                # Do not save inputs in each node while evaluating
                layer.forward(inputs=X_plus, eval_mode=True)

                # The input to the next layer is the u_values from the current layer
                # Be sure to include the bias
                inputs = np.array([1] + [node.u for node in layer.nodes])

            # Final result of forward propagation
            # Start with special case for single output
            node = self.layers[-1].nodes[0]
            y_hat = node.u

            predict_mse += (y_hat - y[i_obs]) ** 2

        # Return average mse
        return predict_mse / len(y)

    def fit(self, X_train, y_train, X_val, y_val, n_epochs, lr, batch_size=128):

        # Initialize history to store results per epoch
        history = dict()
        history["loss"] = []
        history["val_loss"] = []

        # For each epoch...
        for epoch in range(n_epochs):

            # Reset for each epoch
            train_mse = 0

            # Permute and split into mini batches
            mb_indices = mini_batch_indices(X=X_train, batch_size=batch_size)

            # For each mini batch...
            for i_mb, mb_index in enumerate(mb_indices):

                # X and y are a mini batch
                X_mb = X_train[mb_index]
                y_mb = y_train[mb_index]

                # For each observation of the mini batch...
                for i_obs, observation in enumerate(X_mb):

                    # Add bias to input
                    X_plus = np.array([1] + list(observation))

                    # Forward propagation
                    for layer in self.layers:

                        layer.forward(inputs=X_plus)

                        # The input to the next layer is the u_values from the current layer
                        # Be sure to include the bias
                        inputs = np.array([1] + [node.u for node in layer.nodes])

                    # Final result of forward propagation
                    # Start with special case for single output
                    node = self.layers[-1].nodes[0]
                    y_hat = node.u

                    # For reporting, sum up train_mse for epoch
                    train_mse += (y_hat - y_mb[i_obs]) ** 2

                    # Calculate node.error for final node (special case)
                    error_2 = 2 * (y_hat - y_mb[i_obs])
                    fprime_2 = 1 if node.z > 0 else 0
                    node.error = error_2
                    node.fprime = fprime_2

                    # Now calculate errors on the hidden layer
                    # Assumes only single weight connect hidden layer to single output
                    layer = self.layers[-2]
                    for node in layer.nodes:
                        node.error = error_2 * fprime_2 * node.weights_out[0]
                        node.fprime = 1 if node.z > 0 else 0

                    # For each node in each layer...
                    for layer in self.layers:
                        for node in layer.nodes:

                            # Partials for latest observation
                            partials = [node.error * node.fprime * inp for inp in node.inputs]

                            # Add latest partials to partials for other observations in the mini batch
                            # We will average latger
                            sum_partials = []
                            for i_partial, partial in enumerate(partials):
                                if node.partials is None:
                                    sum_partials = partials
                                else:
                                    sum_partials.append(partials[i_partial] + node.partials[i_partial])
                            node.partials = sum_partials

                # Now update weights at end of mini-batch
                # Divide sum of gradients by batch_size to get average gradient
                # Be sure to reset node.partials to None prior to next mini-batch
                for layer in self.layers:
                    for node in layer.nodes:
                        w_partials = np.array(node.partials) / batch_size
                        w_old = np.array(node.weights_in)
                        w_new = w_old - lr * w_partials
                        node.weights_in = list(w_new)
                        node.partials = None

                # Above only adds weights_in
                # Now add the weights_out
                for j, layer in enumerate(self.layers[:-1]):
                    next_layer = self.layers[j + 1]
                    for k, node in enumerate(layer.nodes):
                        node.weights_out = [n.weights_in[k+1] for n in next_layer.nodes]

            # Report at end of epoch
            n_batches = len(mb_indices)
            train_mse = train_mse / (batch_size * n_batches)
            train_rmse =  train_mse ** .5

            val_mse = self.predict(X=X_val, y=y_val)
            val_rmse = val_mse ** .5
            print(f"Epoch {epoch}:  training rmse={train_rmse :3.2f}   -   validation rmse={val_rmse :3.2f}")

            # Update  history with losses
            history["loss"].append(train_mse)
            history["val_loss"].append(val_mse)

        return history

if __name__ == "__main__":

    model = Net(n_inputs=2, hidden_dim=20)

    # Make data
    n = 10
    X1 = np.random.permutation(n)
    X2 = np.random.permutation(n)
    X = np.zeros([n, 2])
    for i in range(n):
        X[i,0] = X1[i]
        X[i,1] = X2[i]
    y = np.array(X1) * np.array(X2) + np.random.random()*5
    model.train(X = X, y=y , n_epochs=100, lr=.0002)