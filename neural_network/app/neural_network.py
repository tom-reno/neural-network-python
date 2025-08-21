import os.path
import random

import numpy as np

from utils import activation_functions as af
from utils import file_utils as fu

class NeuralNetwork:
    _layers: list = None
    _weights: list = None
    _biases: list = None

    def __init__(
            self,
            amount_input_nodes,
            amount_hidden_layers,
            hidden_activation_function,
            output_activation_function,
            targets
    ):
        self.__amount_hidden_layers: int = amount_hidden_layers
        self._targets = np.array(targets)
        self.__activation_function = hidden_activation_function
        self.__output_activation_function = output_activation_function
        self.__activate = af.sigmoid_activation
        self.__derive = af.sigmoid_derivative
        self.__activate_output = af.sigmoid_activation
        self.__derive_output = af.sigmoid_derivative

        self.__amount_nodes = [amount_input_nodes]
        for i in range(self.__amount_hidden_layers):
            self.__amount_nodes.append(int(self.__amount_nodes[i] / (i + 2) + len(self._targets)))
        self.__amount_nodes.append(len(self._targets))

        self.__initialize_network()

    def __initialize_network(self):
        print(f'Initializing neural network with following properties:\n'
              f'    - Amount input nodes: {self.__amount_nodes[0]}\n'
              f'    - Amount hidden layers: {self.__amount_hidden_layers}\n'
              f'    - Activation function for hidden layers: {self.__activation_function}\n'
              f'    - Activation function for output layer: {self.__output_activation_function}\n'
              f'    - Targets: {', '.join(str(target) for target in self._targets)}')

        np.random.seed(42)
        self.__initialize_weights()
        self.__initialize_biases()

    def __initialize_weights(self):
        # weights[layer starting from 0][current_layer_node][next_layer_node]
        filename = self.__retrieve_weights_config_filename()
        if os.path.exists(filename):
            self._weights = fu.load(filename)
        else:
            self._weights = []
            for i in range(len(self.__amount_nodes) - 1):
                self._weights.append(np.random.rand(self.__amount_nodes[i], self.__amount_nodes[i + 1]) - 0.5)

    def __initialize_biases(self):
        # biases[layer starting from 1][current_layer_node]
        filename = self.__retrieve_biases_config_filename()
        if os.path.exists(filename):
            self._biases = fu.load(filename)
        else:
            self._biases = []
            for i in range(1, len(self.__amount_nodes)):
                self._biases.append(np.random.rand(self.__amount_nodes[i]) - 0.5)

    def predict(self, image) -> int:
        output = self.__propagate_forward(image)
        return self._targets[np.argmax(output, 0)]

    def train(self, images, labels, learning_rate, training_iterations, batch_size):
        print(f'Training neural network with {len(images)} images in {training_iterations} iterations ...')

        for iteration in range(1, training_iterations):
            # Permute images every iteration for better training results
            self.__permute_images(images, labels)

            iteration_result = dict()
            iteration_result_max = dict()
            for i in range(0, len(images), batch_size):
                images_batch = np.array(images[i:i + batch_size])
                labels_batch = np.array(labels[i:i + batch_size])

                batch_output = self.__propagate_forward(images_batch)

                self.__track_results(batch_output, labels_batch, iteration_result, iteration_result_max)

                expected_output = self.__retrieve_expected_output(labels_batch)
                self.__propagate_backward(expected_output, batch_output, learning_rate)

            if iteration % batch_size == 0:
                # self.__save_configs(f'_cp_{iteration}')
                print('#######################################################')
                for label, result in iteration_result.items():
                    print(f'Iteration {iteration} - Output label \'{label}\': {result:.5f} '
                          f'(max \'{iteration_result_max[label][0]}\': {iteration_result_max[label][1]:.5f})'
                          f'{f'    {u'\u2713'}' if label == iteration_result_max[label][0] else ''}')

        self.__save_configs()
        # fu.delete('./data/config/*_cp_*.pkl')

    def __propagate_forward(self, image) -> list:
        self._layers = [self.__activate(image)]
        for i in range(len(self.__amount_nodes) - 1):
            sums_prev_nodes = np.dot(self._layers[i], self._weights[i]) + self._biases[i]
            if len(self._layers) <= self.__amount_hidden_layers:
                self._layers.append(self.__activate(sums_prev_nodes))
            else:
                self._layers.append(self.__activate_output(sums_prev_nodes))
        return self._layers[-1]

    def __propagate_backward(self, expected_outputs, outputs, learning_rate):
        # Calculate errors
        errors = [np.subtract(expected_outputs, outputs).T]
        start_layer = len(self._layers) - 2
        for i in range(start_layer, 0, -1):
            errors.insert(0, self._weights[i].dot(errors[0]))

        # Adjust biases and weights
        for i in range(start_layer, -1, -1):
            source_nodes = self._layers[i].T
            target_nodes = self._layers[i + 1].T
            target_errors = errors[i]
            if i < start_layer:
                target_errors *= self.__derive(target_nodes)
            self._biases[i] = self._biases[i] + (learning_rate * sum(target_errors.T))
            self._weights[i] = self._weights[i] + (learning_rate * np.dot(target_errors, source_nodes.T)).T

    def __retrieve_expected_output(self, labels):
        expected_outputs = np.zeros((len(labels), self._targets.size))
        for i, label in enumerate(labels):
            expected_outputs[i][np.where(self._targets == label)] = 1
        return expected_outputs

    def __save_configs(self, postfix=''):
        fu.save(self._weights, self.__retrieve_weights_config_filename(postfix))
        fu.save(self._biases, self.__retrieve_biases_config_filename(postfix))

    def __track_results(self, batch_output, labels_batch, iteration_result, iteration_result_max):
        for j, label in enumerate(labels_batch):
            iteration_result.setdefault(label, 0)
            iteration_result[label] = batch_output[j][np.where(self._targets == label)][0] # G=16=0.004845659027819352
            iteration_result_max.setdefault(label, 0)
            max_target_index = np.argmax(batch_output[j]) # max=0.3532016060938982=63=A
            max_target_label = self._targets[max_target_index]
            iteration_result_max[label] = (max_target_label, batch_output[j][max_target_index])

    def __retrieve_weights_config_filename(self, postfix=''):
        return (
            f'./data/config/weights_'
            f'{''.join(f'({self.__amount_nodes[i]},{self.__amount_nodes[i + 1]})' for i in range(len(self.__amount_nodes) - 1))}'
            f'_{self.__activation_function}_{self.__output_activation_function}{postfix}.pkl'
        )

    def __retrieve_biases_config_filename(self, postfix=''):
        return (
            f'./data/config/biases_'
            f'{''.join(f'({amount})' for amount in self.__amount_nodes)}'
            f'_{self.__activation_function}_{self.__output_activation_function}{postfix}.pkl'
        )

    @staticmethod
    def __permute_images(images: list, labels: list):
        zipped_lists = list(zip(images.copy(), labels.copy()))
        random.shuffle(zipped_lists)
        shuffled_images, shuffled_labels = zip(*zipped_lists)

        label_dict = dict()
        for i in range(len(shuffled_labels)):
            label = shuffled_labels[i]
            label_dict.setdefault(label, [])
            label_dict[label].append(shuffled_images[i])

        lengths = set((np.unique(shuffled_labels, return_counts=True)[1]).tolist())
        if len(lengths) > 1:
            raise ValueError('Failed to permute list: inhomogeneous amount of labels')

        images.clear()
        labels.clear()
        for i in range(lengths.pop()):
            for k, v in label_dict.items():
                images.append(v[i])
                labels.append(k)
