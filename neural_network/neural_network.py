import numpy as np

import activation_functions as af

class NeuralNetwork:
    __image_data = None
    __layers = None
    __weights = None
    __biases = None

    def __init__(
            self,
            amount_input_nodes,
            amount_hidden_layers,
            hidden_activation_function,
            output_activation_function,
            targets
    ):
        print("Initializing neural network ...")
        self.__amount_input_nodes = amount_input_nodes
        self.__amount_hidden_layers = amount_hidden_layers
        self.__hidden_activation_function = hidden_activation_function
        self.__output_activation_function = output_activation_function
        self.__targets = targets
        self.__initialize_network()

    def set_image_data(self, image_data):
        self.__image_data = image_data
        self.__layers[0] = [af.sigmoid_activation(input_value) for input_value in np.ndarray.flatten(image_data)]

    def evaluate(self) -> list:
        return self.__propagate_forward()

    def train(self, label, learning_rate, training_iterations):
        print(f'Training neural network for label {label} in {training_iterations} iterations ...')
        expected_outputs = self.__retrieve_expected_outputs(label)
        for iteration in range(training_iterations + 1):
            outputs = self.evaluate()
            if iteration % 10 == 0:
                print(f'Iteration {iteration} output for label {label}: {outputs}')
            self.__propagate_backward(expected_outputs, outputs, learning_rate)

    def __propagate_forward(self) -> list:
        for i in range(1, len(self.__layers)):
            for j in range(len(self.__layers[i])):
                sum_prev_nodes = sum(prev_node * self.__weights[i - 1][k][j]
                                     for k, prev_node in enumerate(self.__layers[i - 1]))
                self.__layers[i][j] = af.sigmoid_activation(sum_prev_nodes + self.__biases[i - 1][j])
        return self.__layers[len(self.__layers) - 1]

    def __propagate_backward(self, expected_outputs, outputs, learning_rate):
        # Calculate errors
        errors = [[(expected_outputs[i] - outputs[i]) for i in range(len(outputs))]]

        start_layer = len(self.__layers) - 2
        for i in range(start_layer, 0, -1):
            errors.insert(0, [sum(errors[0][k] * self.__weights[i][j][k] for k in range(len(self.__layers[i + 1])))
                              for j in range(len(self.__layers[i]))])

        # Adjust weights and biases
        for i in range(start_layer, -1, -1):
            for j in range(len(errors[i])):
                for k in range(len(self.__weights[i])):
                    self.__weights[i][k][j] += learning_rate * errors[i][j] * af.sigmoid_derivative(self.__layers[i][k])

                for k in range(len(self.__biases[i])):
                    self.__biases[i][k] += learning_rate * errors[i][j]

        self.__reset_layers()

    def __initialize_network(self):
        np.random.seed(42)
        self.__initialize_layers()
        self.__initialize_weights()
        self.__initialize_biases()

    def __initialize_layers(self):
        self.__layers = []
        self.__layers.append([0 for _ in range(self.__amount_input_nodes)])
        for _ in range(self.__amount_hidden_layers): self.__layers.append([0 for _ in range(len(self.__targets) * 2)])
        self.__layers.append([0 for _ in range(len(self.__targets))])

    def __reset_layers(self):
        for layerIndex in range(1, len(self.__layers) - 1):
            self.__layers[layerIndex] = [0 for _ in range(len(self.__targets) * 2)]
        self.__layers[-1] = [0 for _ in range(len(self.__targets))]

    def __initialize_weights(self):
        # TODO retrieve from database or file
        # weights[layer][prev_node][next_node]
        self.__weights = [[[np.random.uniform(-1, 1) for _ in range(len(self.__layers[i + 1]))]
                           for _ in range(len(self.__layers[i]))] for i in range(len(self.__layers) - 1)]

    def __initialize_biases(self):
        # TODO retrieve from database or file
        # biases[layer][next_node]
        self.__biases = [[np.random.uniform(-1, 1) for _ in range(len(self.__layers[i]))]
                         for i in range(1, len(self.__layers))]

    def __retrieve_expected_outputs(self, label):
        expected_outputs = [0 for _ in range(len(self.__targets))]
        expected_outputs[label] = 1
        return expected_outputs
