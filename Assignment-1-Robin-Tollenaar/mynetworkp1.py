import numpy as np

class MyNetwork1:
    def __init__(self, weights, biases, epochs, batch_size, momentum=0.9, learning_rate=0.1):
        self.layers = 1
        self.weights = np.array(weights)
        self.biases = biases
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.momentum = momentum
        self.velocity = 0.0


    def train(self, training_data):
        #random batch from training data
        
        for i in range(self.epochs):
            for scenario in training_data:
                self.backpropagation(np.array(scenario[0]), np.array(scenario[1]))
                print("weights: ", self.weights)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1.0 - self.sigmoid(x))

    def loss(self, output, target):
        return np.sum(np.square(output - target))
        #loss function used: sum of squares
    #backpropagation

    def forwardspropagation(self, input_matrix):
        for i in range(self.layers):
            input_matrix = self.sigmoid(np.dot(self.weights[i], input_matrix) + self.biases[i])
        return input_matrix

    def backpropagation(self, input_matrix, target):
        #get the predicion y'
        prediction_y= self.forwardspropagation(input_matrix)
        #calculate the loss
        loss = self.loss(prediction_y, target)
        print("loss: ", loss)
        #calculate the gradients
        for i in range(self.layers):
            #calculate the gradient for the weights
            gradient_weights = self.learning_rate * np.gradient([loss, self.weights[i]]) + self.momentum * self.velocity
            self.weights[i] -= gradient_weights[0]
            #update the velocity
            self.velocity = gradient_weights
            #print("velocity: ", self.velocity)