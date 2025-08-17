import pickle

import numpy as np

from utils import activation_functions as af

class NeuralNetwork:
    __layers: list = None
    __weights: list = None
    __biases: list = None

    def __init__(
            self,
            amount_input_nodes,
            amount_hidden_layers,
            hidden_activation_function,
            output_activation_function,
            targets
    ):
        self.__amount_input_nodes = amount_input_nodes
        self.__amount_hidden_layers: int = amount_hidden_layers
        self.__hidden_activation_function: str = hidden_activation_function
        self.__output_activation_function: str = output_activation_function
        self.__targets = targets

        self.__initialize_network()

    def __initialize_network(self):
        print(f'Initializing neural network with following properties:\n'
              f'    - Amount input nodes: {self.__amount_input_nodes}\n'
              f'    - Amount hidden layers: {self.__amount_hidden_layers}\n'
              f'    - Activation function for hidden layers: {self.__hidden_activation_function}\n'
              f'    - Activation function for output layer: {self.__output_activation_function}\n'
              f'    - Targets: {', '.join(str(target) for target in self.__targets)}')

        np.random.seed(42)
        self.__initialize_layers()
        self.__initialize_weights()
        self.__initialize_biases()

    def predict(self, image) -> int:
        self.__layers[0] = af.sigmoid_activation(image[1][1])
        output = self.__propagate_forward()
        return np.argmax(output, 0)

    def train(self, images, learning_rate, training_iterations, batch_size):
        print(f'Training neural network with {len(images)} images in {training_iterations} iterations ...')
        for iteration in range(1, training_iterations):
            # Shuffle images every iteration for better training results
            np.random.shuffle(images)

            for i in range(0, len(images), batch_size):
                image_label: str = images[i][1][0]
                image_data = np.array(images[i][1][1])
                expected_outputs = self.__retrieve_expected_outputs(image_label)
                self.__layers[0] = af.sigmoid_activation(image_data)

                outputs = self.__propagate_forward()

                if iteration % batch_size == 0:
                    outputs_string = ', '.join(f'{self.__targets[i]}={output}' for i, output in enumerate(outputs))
                    print(f'Iteration {iteration} - Outputs for label {image_label}: {outputs_string}')

                self.__propagate_backward(expected_outputs, outputs, learning_rate)

    def __propagate_forward(self) -> list:
        for i in range(len(self.__layers) - 1):
            sums_prev_nodes = np.dot(self.__layers[i], self.__weights[i]) + self.__biases[i]
            self.__layers[i + 1] = af.sigmoid_activation(sums_prev_nodes)
        return self.__layers[len(self.__layers) - 1]

    def __propagate_backward(self, expected_outputs, outputs, learning_rate):
        # Calculate errors
        errors = [np.subtract(expected_outputs, outputs)]
        start_layer = len(self.__layers) - 2
        for i in range(start_layer, 0, -1):
            errors.insert(0, self.__weights[i].dot(errors[0]))

        # Adjust biases and weights
        for i in range(start_layer, -1, -1):
            self.__biases[i] += errors[i] * learning_rate
            for j in range(len(self.__layers[i])):
                self.__weights[i][j] += errors[i] * af.sigmoid_derivative(self.__layers[i][j] * learning_rate)

        self.__reset_layers()

    def __initialize_layers(self):
        self.__layers = [np.zeros(self.__amount_input_nodes)]
        for i in range(self.__amount_hidden_layers):
            amount_hidden_layer_nodes: int = int(len(self.__layers[i]) / (i + 2) + len(self.__targets))
            self.__layers.append(np.zeros(amount_hidden_layer_nodes))
        self.__layers.append(np.zeros(len(self.__targets)))

    def __reset_layers(self):
        for i in range(1, len(self.__layers)):
            self.__layers[i] = np.zeros(len(self.__layers[i]))

    def __initialize_weights(self):
        # TODO retrieve from database or file
        # weights[layer starting from 0][current_layer_node][next_layer_node]
        self.__weights = []
        for i in range(len(self.__layers) - 1):
            self.__weights.append(np.random.rand(len(self.__layers[i]), len(self.__layers[i + 1])) - 0.5)

    def __initialize_biases(self):
        # TODO retrieve from database or file
        # biases[layer starting from 1][current_layer_node]
        self.__biases = []
        for i in range(1, len(self.__layers)):
            self.__biases.append(np.random.rand(len(self.__layers[i])) - 0.5)

    def __retrieve_expected_outputs(self, label):
        expected_outputs = np.zeros(len(self.__targets))
        expected_outputs[np.where(self.__targets == label)] = 1
        return expected_outputs
