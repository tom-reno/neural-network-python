import numpy as np

from utils import activation_functions as af

class NeuralNetwork:
    _layers = None
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
        self.__amount_input_nodes = amount_input_nodes
        self.__amount_hidden_layers = amount_hidden_layers
        self.__amount_hidden_layer_nodes = int(len(targets) * 4) # TODO: how to find out best size?
        self.__hidden_activation_function = hidden_activation_function
        self.__output_activation_function = output_activation_function
        self.__targets = targets
        print(f'Initializing neural network with following properties:\n'
              f'    - Amount input nodes: {amount_input_nodes}\n'
              f'    - Amount hidden layers: {amount_hidden_layers}\n'
              f'    - Amount hidden nodes: {self.__amount_hidden_layer_nodes}\n'
              f'    - Activation function for hidden layers: {hidden_activation_function}\n'
              f'    - Activation function for output layer: {output_activation_function}\n'
              f'    - Targets: {targets}')
        self.__initialize_network()

    def predict(self, image_data) -> int:
        self._layers[0] = [af.sigmoid_activation(input_value) for input_value in np.ndarray.flatten(image_data)]
        output = self.__propagate_forward()
        return output.index(np.max(output))

    def train(self, images, learning_rate, training_iterations):
        print(f'Training neural network with {len(images)} images in {training_iterations} iterations ...')
        for iteration in range(training_iterations + 1):
            for image in images:
                image_label = image[0]
                image_data = image[1]
                expected_output = self.__retrieve_expected_outputs(image_label)
                self._layers[0] = [af.sigmoid_activation(input_value) for input_value in np.ndarray.flatten(image_data)]
                output = self.__propagate_forward()
                if iteration % 10 == 0:
                    print(f'Iteration {iteration} output for image label {image_label}: {[output]}')
                self.__propagate_backward(expected_output, output, learning_rate)

    def __propagate_forward(self) -> list:
        for i in range(1, len(self._layers)):
            for j in range(len(self._layers[i])):
                sum_prev_nodes = sum(prev_node * self.__weights[i - 1][k][j]
                                     for k, prev_node in enumerate(self._layers[i - 1]))
                self._layers[i][j] = af.sigmoid_activation(sum_prev_nodes + self.__biases[i - 1][j])
        return list(self._layers[len(self._layers) - 1])

    def __propagate_backward(self, expected_output, output, learning_rate):
        # Calculate errors
        errors = [[(expected_output[i] - output[i]) for i in range(len(output))]]

        start_layer = len(self._layers) - 2
        for i in range(start_layer, 0, -1):
            errors.insert(0, [sum(errors[0][k] * self.__weights[i][j][k] for k in range(len(self._layers[i + 1])))
                              for j in range(len(self._layers[i]))])

        # Adjust weights and biases
        for i in range(start_layer, -1, -1):
            for j in range(len(errors[i])):
                for k in range(len(self.__weights[i])):
                    self.__weights[i][k][j] += learning_rate * errors[i][j] * af.sigmoid_derivative(self._layers[i][k])

                for k in range(len(self.__biases[i])):
                    self.__biases[i][k] += learning_rate * errors[i][j]

        self.__reset_layers()

    def __initialize_network(self):
        np.random.seed(42)
        self.__initialize_layers()
        self.__initialize_weights()
        self.__initialize_biases()

    def __initialize_layers(self):
        self._layers = [np.zeros(self.__amount_input_nodes)]
        for i in range(self.__amount_hidden_layers):
            self._layers.append(np.zeros(int(len(self._layers[i]) / 3)))
        self._layers.append(np.zeros(len(self.__targets)))

    def __reset_layers(self):
        for i in range(1, len(self._layers)):
            self._layers[i] = np.zeros(len(self._layers[i]))

    def __initialize_weights(self):
        # TODO retrieve from database or file
        # weights[layer beginning from 0][current_layer_node][next_layer_node]
        self.__weights = [[[np.random.uniform(-1, 1) for _ in range(len(self._layers[i + 1]))]
                           for _ in range(len(self._layers[i]))] for i in range(len(self._layers) - 1)]

    def __initialize_biases(self):
        # TODO retrieve from database or file
        # biases[layer beginning from 1][current_layer_node]
        self.__biases = [[np.random.uniform(-1, 1) for _ in range(len(self._layers[i]))]
                         for i in range(1, len(self._layers))]

    def __retrieve_expected_outputs(self, label):
        expected_outputs = [0 for _ in range(len(self.__targets))]
        expected_outputs[label] = 1
        return expected_outputs
