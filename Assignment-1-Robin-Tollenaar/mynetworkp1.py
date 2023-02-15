import numpy as np

class MyNetwork1:
    def __init__(self, weights, biases, epochs, batch_size, momentum=0.9, learning_rate=0.1) -> None:
        """
        Parameters
        ----------
        weights : ndarray or list of ndarray
            A list of ndarrays (or a single ndarray if there is only one dimension)
            corresponding to the weights of the network.
            Each weight has the same shape as f.
        
        biases : ndarray or list of ndarray
            A list of ndarrays (or a single ndarray if there is only one dimension)
            corresponding to the biases of the network.
            Each bias has the same shape as f.
        
        epochs : int
            The number of epochs to train for.
        
        batch_size : int
            The size of the batches to use for training.
        
        momentum : float, optional
            The momentum to use for training.
            Standard is 0.9
        
        learning_rate : float, optional
            The learning rate to use for training.
            Standard is 0.1
        """
        #One input, one output -> one layer
        self.layers = 1
        self.weights = np.array(weights)
        self.biases = biases
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.momentum = momentum
        #Velocity starts out at 0
        self.velocity = 0.0


    def train(self, training_data) -> None:
        """Updates the weights and biases of the network using backpropagation."""
        #looping through the epochs
        for i in range(self.epochs):
            #looping through every scenario in the training data
            for scenario in training_data:
                #backpropagation starts
                self.backpropagation(np.array(scenario[0]), np.array(scenario[1]))
                print("weights: ", self.weights)

    def sigmoid(self, x) -> float:
        """A quick calculation of the sigmoid function."""
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x) -> float:
        """A quick calculation of the derivative of the sigmoid function."""
        return self.sigmoid(x) * (1.0 - self.sigmoid(x))

    def loss(self, output, target) -> float:
        """A quick calculation of the loss function using the sun of squares method."""
        return np.sum(np.square(output - target))

    def forwardspropagation(self, input_matrix)-> np.ndarray:
        """Forwardspropagation of the network using the sigmoid function.
        
        Parameters
        ----------
        input_matrix : ndarray
            The input matrix to the network.
        
        Returns
        -------
        ndarray
            The output of the network.
        """
        for i in range(self.layers):
            input_matrix = self.sigmoid(np.dot(self.weights[i], input_matrix) + self.biases[i])
        return input_matrix

    def backpropagation(self, input_matrix, target) -> None:
        """Backpropagation of the network using the sigmoid function.
        The weights of the network are updated using gradient descent with momentum.

        Parameters
        ----------
        input_matrix : ndarray
            The input matrix to the network.
        target : ndarray
            The target output of the network.
        """
        #get the predicion y'
        prediction_y= self.forwardspropagation(input_matrix)
        #calculate the loss
        loss = self.loss(prediction_y, target)
        
        #calculate the gradients
        for i in range(self.layers):
            #calculate the gradient for the weights
            gradient_weights = self.learning_rate * np.gradient([loss, self.weights[i]]) + self.momentum * self.velocity
            self.weights[i] -= gradient_weights[0]
            #update the velocity
            self.velocity = gradient_weights