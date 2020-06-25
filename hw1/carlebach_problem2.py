import numpy as np

class Node:

    # Class attribute to give id to each node created
    class_id = 0

    def __init__(self, weights):

        Node.class_id += 1
        self.node_id = Node.class_id

        self.weights = weights
        self.act = "relu"
        self.z = None
        self.u = None
        self.error = None
        self.partials = []

    def forward(self, inputs):

        self.z = (np.array(inputs) * np.array(self.weights)).sum()

        if self.act == "relu":

            if self.z < 0:
                self.u = 0
            else:
                self.u = self.z

        else:
            raise NotImplemented

class FC_Layer:

    # Class attribute to give id to each layer created
    class_id = 0

    def __init__(self, nnodes, input_dim):
        FC_Layer.class_id += 1
        self.layer_id = FC_Layer.class_id

        self.nodes = []
        for i in range(nnodes):
            weights = np.random.random(input_dim + 1)
            self.nodes.append(Node(weights=weights))

    def forward(self, inputs):

        # For each node in the layer...
        for node in self.nodes:

            # Calculate z and u for each node
            node.forward(inputs=inputs)


class Net:

    def __init__(self, n_inputs, n_hidden_layers, hidden_dim):

        # Start with first layer
        self.layers = []
        self.layers.append(FC_Layer(nnodes=hidden_dim, input_dim=n_inputs))

        # Add other hidden layers if specified
        for n in range(1, n_hidden_layers):
            self.layers.append(FC_Layer(nnodes=hidden_dim, input_dim=hidden_dim))

        # End with output layer
        # Assumes single value output
        self.layers.append(FC_Layer(nnodes=1, input_dim=hidden_dim))

    def train(self, X = [[1, 2], [2, 3]], y= [5, 13], n_epochs=1, batch_size=128, lr=.05):

        # For each epoch...
        for epoch in range(n_epochs):
            print()
            print(f"######## Epoch:  {epoch}")

            # SGD:  for each row, sample or observation of data
            for i, observation in enumerate(X):

                # Let's start the forward process with input to first layer being and observation of X
                inputs = [1] + observation

                # Now let's for forward through each layer
                for layer in self.layers:

                    layer.forward(inputs=inputs)

                    # The input to the next layer is the u_values from the current layer
                    inputs = [1] + [node.u for node in layer.nodes]

                # Final result of forward propagation
                # Special case single output
                node = self.layers[-1].nodes[0]
                y_hat = node.u
                node.error = 2 * (y_hat - y[i])
                print(f"obs({i}):  prediction error = {node.error}")

                # Go backwards through the layers
                for layer_i in range(len(self.layers), 0, -1):
                    layer = self.layers[layer_i - 1]

                    for node in layer.nodes:

                        node.partials = []
                        if node.z > 0:
                            fprime_z = 1
                        else:
                            fprime_z = 0

                        for weight in node.weights:
                            node.partials.append(node.error * fprime_z * weight)

                    # Move errors backward
                    if layer_i > 1:
                        pr_layer = self.layers[layer_i - 2]
                        for j, pr_node in enumerate(pr_layer.nodes):
                            pr_node.error = node.partials[i + 1]

                # Now update weights
                for layer in self.layers:
                    for node in layer.nodes:
                        w_partials = np.array(node.partials)
                        w_old = np.array(node.weights)
                        w_new = w_old - lr * w_partials
                        node.weights = list(w_new)


model = Net(n_inputs=2, n_hidden_layers=1, hidden_dim=2)
model.train(n_epochs=100)