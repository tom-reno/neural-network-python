import argparse
import os
from datetime import datetime

import numpy as np
from PIL import Image

from neural_network.application import NeuralNetwork

ACTIVATION_FUNCTIONS = ('sigmoid', 'softmax', 'tanh', 'relu')
MODES = ('evaluation', 'training')
DEFAULT_ACTIVATION_FUNCTION = 'sigmoid'
DEFAULT_AMOUNT_HIDDEN_LAYERS = 1
DEFAULT_MODE = 'training'
DEFAULT_LEARNING_RATE = 0.01
DEFAULT_TRAINING_ITERATIONS = 1000
SUPPORTED_FILETYPES = ('.jpg', '.jpeg', '.png')
TARGETS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)

def _retrieve_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=MODES, type=str, default=DEFAULT_MODE)
    parser.add_argument('--data', type=str)
    parser.add_argument('--amount-hidden-layers', type=int, default=DEFAULT_AMOUNT_HIDDEN_LAYERS)
    parser.add_argument('--hidden-activation-function', type=str, choices=ACTIVATION_FUNCTIONS,
                        default=DEFAULT_ACTIVATION_FUNCTION)
    parser.add_argument('--output-activation-function', type=str, choices=ACTIVATION_FUNCTIONS,
                        default=DEFAULT_ACTIVATION_FUNCTION)
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--training-iterations', type=int, default=DEFAULT_TRAINING_ITERATIONS)
    return parser.parse_args()

if __name__ == '__main__':
    start_time = datetime.now()
    args = _retrieve_args()

    mode = args.mode
    data = args.data or f'./data/{mode}'
    amount_hidden_layers = args.amount_hidden_layers
    hidden_activation_function = args.hidden_activation_function
    output_activation_function = args.output_activation_function
    learning_rate = args.learning_rate
    training_iterations = args.training_iterations

    print(f'Running neural network in {mode} mode with data in directory {data} ...')

    for file in os.listdir(data):
        if not file.endswith(SUPPORTED_FILETYPES):
            continue

        image = Image.open(f'{data}/{file}')
        inputs = np.array(image)

        neural_network = NeuralNetwork(amount_hidden_layers, hidden_activation_function, output_activation_function,
                                       inputs, TARGETS)

        if mode == 'evaluation':
            output = neural_network.evaluate()
            print(f'Evaluation result: {output.index(np.max(output))}')
        else:
            label = int((file.split('/')[-1]).split('-')[0])
            neural_network.train(label, learning_rate, training_iterations)

        print(f'Finished in {(datetime.now() - start_time).total_seconds()} seconds')
