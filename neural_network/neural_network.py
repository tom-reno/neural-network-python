import numpy as np

from utils import activation_functions as af

class NeuralNetwork:
    _layers: list = None
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
        self.__amount_input_nodes: int = amount_input_nodes
        self.__amount_hidden_layers: int = amount_hidden_layers
        self.__hidden_activation_function: str = hidden_activation_function
        self.__output_activation_function: str = output_activation_function
        self.__targets: tuple[int] = targets
        print(f'Initializing neural network with following properties:\n'
              f'    - Amount input nodes: {amount_input_nodes}\n'
              f'    - Amount hidden layers: {amount_hidden_layers}\n'
              f'    - Activation function for hidden layers: {hidden_activation_function}\n'
              f'    - Activation function for output layer: {output_activation_function}\n'
              f'    - Targets: {', '.join(str(target) for target in targets)}')
        self.__initialize_network()

    def predict(self, image_data) -> int:
        self._layers[0]: list = [af.sigmoid_activation(value) for value in np.ndarray.flatten(image_data)]
        output = self.__propagate_forward()
        return np.argmax(output, 0)

    def train(self, images, learning_rate, training_iterations):
        print(f'Training neural network with {len(images)} images in {training_iterations} iterations ...')
        for iteration in range(training_iterations + 1):
            for image in images:
                image_label: str = image[0]
                image_data = image[1]
                expected_outputs = self.__retrieve_expected_outputs(image_label)
                self._layers[0]: list = af.sigmoid_activation(np.ndarray.flatten(image_data))

                outputs: list = self.__propagate_forward()

                if iteration % 10 == 0:
                    outputs_string = ', '.join(f'{i}={output}' for i, output in enumerate(outputs))
                    print(f'Iteration {iteration} - Outputs for image label {image_label}: {outputs_string}')

                self.__propagate_backward(expected_outputs, outputs, learning_rate)

    def __propagate_forward(self) -> list:
        for i in range(len(self._layers) - 1):
            sums_prev_nodes = np.dot(self._layers[i], self.__weights[i]) + self.__biases[i]
            self._layers[i + 1] = af.sigmoid_activation(sums_prev_nodes)
        return self._layers[len(self._layers) - 1]

    def __propagate_backward(self, expected_outputs, outputs, learning_rate):
        # Calculate errors
        # errors = [np.subtract(expected_outputs, outputs)]
        errors = [np.round(expected_outputs - outputs, 5)]

        start_layer = len(self._layers) - 2

        for i in range(start_layer, 0, -1):
            errors.insert(0, np.round(self.__weights[i].dot(errors[0]), 5))

        # Adjust biases and weights
        for i in range(start_layer, -1, -1):
            self.__biases[i] += np.round(errors[i] * learning_rate, 5)
            for j in range(len(self._layers[i])):
                self.__weights[i][j] += np.round(errors[i] * af.sigmoid_derivative(self._layers[i][j] * learning_rate),
                                                 5)

        self.__reset_layers()

    def __initialize_network(self):
        np.random.seed(42)
        self.__initialize_layers()
        self.__initialize_weights()
        self.__initialize_biases()

    def __initialize_layers(self):
        self._layers: list = [list(np.zeros(self.__amount_input_nodes))]
        for i in range(self.__amount_hidden_layers):
            amount_hidden_layer_nodes: int = int(len(self._layers[i]) / (i + 2))
            self._layers.append(list(np.zeros(amount_hidden_layer_nodes)))
        self._layers.append(list(np.zeros(len(self.__targets))))

    def __reset_layers(self):
        for i in range(1, len(self._layers)):
            self._layers[i]: list = np.zeros(len(self._layers[i]))

    def __initialize_weights(self):
        # TODO retrieve from database or file
        # weights[layer starting from 0][current_layer_node][next_layer_node]
        self.__weights = []
        for i in range(len(self._layers) - 1):
            self.__weights.append(np.random.rand(len(self._layers[i]), len(self._layers[i + 1])) - 0.5)

    def __initialize_biases(self):
        # TODO retrieve from database or file
        # biases[layer starting from 1][current_layer_node]
        self.__biases = []
        for i in range(1, len(self._layers)):
            self.__biases.append(np.random.rand(len(self._layers[i])) - 0.5)

    def __retrieve_expected_outputs(self, label):
        expected_outputs: list = list(np.zeros(len(self.__targets)))
        expected_outputs[label] = 1
        return expected_outputs
