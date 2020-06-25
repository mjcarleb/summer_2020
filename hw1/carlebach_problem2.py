import numpy as np

class Node:

    """
    Basic building block of neural network
    Each node contains the following:
        - node_id:  unique id
        - act:  specifies type of activation function (e.g, "relu")
        - inputs:  list of values passed in the forward direction
        - weights_in:  list of weights for each incoming connection (including bias)
        - z:  sum of (inputs * weights_in)
        - u:  act(z)
        - weights_out:  list of weights for each outgoing connection
        - error:  partials of the loss with respect to output (e.g., u)
        - partials:  partials of the loss with respect to each weight_in (for each node in layer and weight_in)
    """

    # Class attribute to give id to each node created
    class_id = 0

    def __init__(self, weights_in):

        Node.class_id += 1
        self.node_id = Node.class_id
        self.act = "relu"
        self.inputs = None
        self.weights_in = weights_in

        # Calculated on forward propagation
        self.z = None              # single sum of inputs * weights_in
        self.u = None              # single u to each outgoing node

        # Added after calculating all weights_in for network, just for convenience
        self.weights_out = []

        # Calculate on backward propagation
        self.error = None           # partial L wrt u
        self.partials = None     # list for each incoming weight

    def forward(self, inputs):

        self.inputs = inputs
        self.z = (np.array(inputs) * np.array(self.weights_in)).sum()

        if self.act == "relu":

            if self.z < 0:
                self.u = 0
            else:
                self.u = self.z

        else:
            raise NotImplemented

    def backward(self, error, fprime):

        self.error = error * fprime * self.weights_out[0]

        if self.act == "relu":
            if self.z >= 0:
                fprime = 1
            else:
                fprime = 0
        else:
            raise NotImplemented

        self.partials = [self.error * fprime * i for i in self.inputs]

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

    def forward(self, inputs):

        # For each node in the layer...
        for node in self.nodes:

            # Calculate z and u for each node and store in node
            # node.z = sum(weights * inputs)
            # node.u = act(node.z)
            node.forward(inputs=inputs)

    def backward(self, error):

        # For each node in the layer...
        for node in self.nodes:

            # Calculate z and u for each node and store in node
            # node.error = partial of loss with respect to output (u)
            # node.partials = partials of loss with respect to each weight coming into node

            if node.act == "relu":
                if node.z >= 0:
                    fprime = 1
                else:
                    fprime = 0
            else:
                raise NotImplemented

            node.backward(error=error, fprime=fprime)

class Net:

    """
    A list of fully connected layers

    The first layer has n_inputs
    The last layer has 1 output (to keep this simple)
    There are n_hidden_layers in between that are fully connected with hidden_dim nodes each
    """

    def __init__(self, n_inputs, n_hidden_layers, hidden_dim):

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

    def train(self, X = [[1]], y= [2], n_epochs=1, batch_size=128, lr=.1):

        # For each epoch...
        for epoch in range(n_epochs):
            print()
            print(f"######## Epoch:  {epoch}")

            # SGD:  for each row, sample or observation of data
            for i, observation in enumerate(X):

                # Add bias to inputs
                inputs = [1] + observation

                # Forward propagation
                for layer in self.layers:

                    layer.forward(inputs=inputs)

                    # The input to the next layer is the u_values from the current layer
                    # Be sure to include the bias
                    inputs = [1] + [node.u for node in layer.nodes]

                # Final result of forward propagation
                # Special case single output
                node = self.layers[-1].nodes[0]
                y_hat = node.u
                node.error = 2 * (y_hat - y[i])
                if node.act == "relu":
                    if node.z >= 0:
                        fprime = 1
                    else:
                        fprime = 0
                else:
                    raise NotImplemented

                node.partials = [node.error * fprime * i for i in node.inputs]

                print(f"obs({i}):  prediction error = {node.error}")

                # Backward propagation
                for layer_i in range(len(self.layers)-1, 0, -1):
                    layer = self.layers[layer_i - 1]

                    layer.backward(error=node.error)

                # Now update weights
                for layer in self.layers:
                    for node in layer.nodes:
                        w_partials = np.array(node.partials)
                        w_old = np.array(node.weights_in)
                        w_new = w_old - lr * w_partials
                        node.weights_in = list(w_new)

                # Above only adds weights_in
                # Now add the weights_out
                for j, layer in enumerate(self.layers[:-1]):
                    next_layer = self.layers[j + 1]
                    for k, node in enumerate(layer.nodes):
                        node.weights_out = [n.weights_in[k+1] for n in next_layer.nodes]


model = Net(n_inputs=1, n_hidden_layers=1, hidden_dim=1)
model.train(n_epochs=30)