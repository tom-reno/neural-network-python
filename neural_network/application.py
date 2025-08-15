import numpy as np

from neural_network import activation_functions as af

class NeuralNetwork:
    layers = None

    def __init__(self, amount_hidden_layers, hidden_activation_function, output_activation_function, inputs, targets):
        print("Initializing neural network ...")
        self.amount_hidden_layers = amount_hidden_layers
        self.hidden_activation_function = hidden_activation_function
        self.output_activation_function = output_activation_function
        self.inputs = inputs
        self.targets = targets
        self._initialize_layers()
        np.random.seed(42)
        self.weights = self._retrieve_weights()
        self.biases = self._retrieve_biases()

    def evaluate(self) -> list:
        return self._propagate_forward()

    def train(self, label, learning_rate, training_iterations):
        print(f'Training neural network for label {label} in {training_iterations} iterations ...')
        expected_outputs = self._retrieve_expected_outputs(label)
        for iteration in range(training_iterations):
            outputs = self.evaluate()
            if iteration % 10 == 0:
                print(f'Iteration {iteration} output {outputs}')
            self._propagate_backward(expected_outputs, outputs, learning_rate)

    def _propagate_forward(self) -> list:
        for i in range(1, len(self.layers)):
            for j in range(len(self.layers[i])):
                sum_prev_nodes = sum(prev_node * self.weights[i - 1][k][j]
                                     for k, prev_node in enumerate(self.layers[i - 1]))
                self.layers[i][j] = af.sigmoid_activation(sum_prev_nodes + self.biases[i - 1][j])
        return self.layers[len(self.layers) - 1]

    def _propagate_backward(self, expected_outputs, outputs, learning_rate):
        # Calculate errors
        errors = [[(expected_outputs[i] - outputs[i]) for i in range(len(outputs))]]

        start_layer = len(self.layers) - 2
        for i in range(start_layer, 0, -1):
            errors.insert(0, [sum(errors[0][k] * self.weights[i][j][k] for k in range(len(self.layers[i + 1])))
                              for j in range(len(self.layers[i]))])

        # Adjust weights and biases
        for i in range(start_layer, -1, -1):
            for j in range(len(errors[i])):
                for k in range(len(self.weights[i])):
                    self.weights[i][k][j] += learning_rate * errors[i][j] * af.sigmoid_derivative(self.layers[i][k])

                for k in range(len(self.biases[i])):
                    self.biases[i][k] += learning_rate * errors[i][j]

        # Reset layers
        self._initialize_layers()

    def _initialize_layers(self):
        self.layers = []
        self.layers.append([af.sigmoid_activation(input_value)
                            for input_value in np.ndarray.flatten(self.inputs).tolist()])
        for _ in range(self.amount_hidden_layers): self.layers.append([0 for _ in range(len(self.targets) * 2)])
        self.layers.append([0 for _ in range(len(self.targets))])

    def _retrieve_weights(self):
        # TODO retrieve from database or file
        # weights[layer][prev_node][next_node]
        return [[[np.random.uniform(-1, 1) for _ in range(len(self.layers[i + 1]))] for _ in range(len(self.layers[i]))]
                for i in range(len(self.layers) - 1)]

    def _retrieve_biases(self):
        # TODO retrieve from database or file
        # biases[layer][next_node]
        return [[np.random.uniform(-1, 1) for _ in range(len(self.layers[i]))] for i in range(1, len(self.layers))]

    def _retrieve_expected_outputs(self, label):
        expected_outputs = [0 for _ in range(len(self.targets))]
        expected_outputs[label] = 1
        return expected_outputs
