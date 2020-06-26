"""
CSIC E-S89, Introduction to Deep Learning

Mark Carlebach
June 26, 2020

My implementation of a neural network is loosely inspired by Joel Grus youtube (where he does much better job!):
https://www.youtube.com/watch?v=o64FV-ez6Gw

I also used this post to get clear about taking average of gradients to perform mini-batch gradient descent:
https://stats.stackexchange.com/questions/183840/sum-or-average-of-gradients-in-mini-batch-gradient-decent
"""

import numpy as np

class Node:

    """
    Basic computational building block of neural network

    Each node contains the following:
        - node_id:  unique id
        - inputs:  list of values passed in the forward direction
        - weights_in:  list of weights for each incoming connection (including bias)
        - z:  sum of (inputs * weights_in)
        - u:  act(z)
        - weights_out:  list of weights for each outgoing connection
        - fprime: for relu, 1 if z > 0 else 0
        - error:  partial L wrt u
        - partials:  partials L wrt each weight_in

    Methods:
        - forward():  calculates z & u based on inputs and weights
        - NOTE:  I implement backward in the SequentialModel.train() method
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

        weights_in = np.array(self.weights_in)

        if not eval_mode:
            self.inputs = inputs

        self.z = (inputs * weights_in).sum()
        self.u = self.z if self.z > 0 else 0

    def backward(self, error, fprime):

        # Partial wrt to output for this node
        # Implementation limited to 1 hidden layer with output layer being single node
        self.error = error * fprime * self.weights_out[0]
        self.fprime = 1 if self.z > 0 else 0


class FC_Layer:

    """
    Layer is list of nodes.  This is a fully connected layer so each node has weight from each node in preceding layer.

    Each layer contains the following:
        - layer_id:  unique id
        - nodes:  list of nnodes nodes (initialized with random weights) where each node as weight from bias
                  and each node of prior input layer (e.g., fully connected)

    Methods:
        - forward():  calculates z & u for all nodes in layer, based on inputs and weights
        - NOTE:  I implement backward in the SequentialModel.train() method
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

    def backward(self, error, fprime):

        # For each node in the layer...
        for node in self.nodes:

            # Calculate derivative of activation and the partial wrt loss
            # Store in node.error and node.fprime
            node.backward(error=error, fprime=fprime)


def mini_batch_indices(X, batch_size):

    """
    Produces a a list of indices, 1 per mini batch
    :param X:  The data to be split into mini-batches
    :param batch_size: The size of each mini batch
    :return: A list of indices, 1 per mini batch

    NOTE:  The data is shuffled before splitting, but the final observations < batch_size are not included in indices
    """

    # Permute index to train data
    idx = np.random.permutation(X.shape[0])

    # Split data into mini batches
    n_batches = X.shape[0] // batch_size

    indices = []
    for n_batch in range(n_batches):
        indices.append(idx[n_batch * batch_size:(n_batch + 1) * batch_size])

    return indices


class SequentialModel:

    """
    A list of fully connected layers
        - The first layer has n_inputs nodes
        - There is only 1 single hidden layer (limitation of this implementation) with hidden_dim nodes
        - The last layer has 1 output node (limitation of this implementation)
    """

    def __init__(self, n_inputs, hidden_dim):

        # Start with first hidden layer
        self.layers = []
        self.layers.append(FC_Layer(nnodes=hidden_dim, input_dim=n_inputs))

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

        """
        Uses networks current weights to make predictions and calculate prediction mse
        :param X: The value from which to predict
        :param y: The true values of y
        :return: the predictions and prediction mse
        """

        # Reset to 0
        predict_mse = 0
        predictions = []

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
                X_plus = np.array([1] + [node.u for node in layer.nodes])

            # Final result of forward propagation
            # Start with special case for single output
            node = self.layers[-1].nodes[0]
            y_hat = node.u

            predict_mse += (y_hat - y[i_obs]) ** 2
            predictions.append(y_hat)

        # Return average mse
        return predictions, predict_mse / len(y)

    def summary(self):

        """
        Return simple string describing the network

        :return: description
        """
        # Collect data for description
        input_dim = self.layers[0].nodes[0].weights_in.shape[0] - 1
        hidden_layers = len(self.layers) - 1
        hidden_dim = len(self.layers[0].nodes)

        # Return description

        desc = f"myTorch Model Summary:\n"
        desc += f">>fully connected\n"
        desc += f"  input_dim={input_dim} :: hidden_layers={hidden_layers} :: hidden_dim={hidden_dim} :: output_dim=1"
        return desc

    def fit(self, X_train, y_train, X_val, y_val, n_epochs, lr, batch_size=128, verbose=True):

        """
        Train the network on train data, and provide training and validation loss per epoch.  Use
        mini batch gradient descent.

        :param X_train: training dependent variables
        :param y_train: training response variable
        :param X_val: validation dependent variables
        :param y_val: validation response variable
        :param n_epochs: number of epochs
        :param lr: learning rate
        :param batch_size: mini batch batch size
        :param verbose: print losses after epoch (or not)
        :return: history (dictionary with training loss and validation loss
        """
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
                        X_plus = np.array([1] + [node.u for node in layer.nodes])

                    # Final result of forward propagation
                    # Start with special case for single output
                    node = self.layers[-1].nodes[0]
                    y_hat = node.u

                    # For reporting, sum up train_mse for epoch
                    train_mse += (y_hat - y_mb[i_obs]) ** 2

                    # Calculate node.error for final node (special case)
                    error = 2 * (y_hat - y_mb[i_obs])
                    fprime = 1 if node.z > 0 else 0
                    node.error = error
                    node.fprime = fprime

                    # Back propagate the error
                    for layer in self.layers[-2:-1]:
                        layer.backward(error=error, fprime=fprime)

                    # Now that errors are back propagated, calculate the partials for each node in each layer...
                    for layer in self.layers:
                        for node in layer.nodes:

                            # Partials for latest observation
                            partials = [node.error * node.fprime * inp for inp in node.inputs]

                            # Add latest partials to partials for other observations in the mini batch
                            # We will average later
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
            _, val_mse = self.predict(X=X_val, y=y_val)

            # Optional show RMSE
            if verbose:
                train_rmse =  train_mse ** .5
                val_rmse = val_mse ** .5
                print(f"Epoch {epoch}:  training rmse={train_rmse :3.2f}   -   validation rmse={val_rmse :3.2f}")

            # Update  history with losses
            history["loss"].append(train_mse)
            history["val_loss"].append(val_mse)

        return history