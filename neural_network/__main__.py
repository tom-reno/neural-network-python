import argparse
import glob
from datetime import datetime

import numpy as np
from PIL import Image

from neural_network import NeuralNetwork

SUPPORTED_FILETYPES = ('jpg', 'jpeg', 'png')
TARGETS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
ACTIVATION_FUNCTIONS = ('sigmoid', 'softmax', 'tanh', 'relu')
MODES = ('prediction', 'training')

DEFAULT_ACTIVATION_FUNCTION = 'sigmoid'
DEFAULT_AMOUNT_HIDDEN_LAYERS = 1
DEFAULT_MODE = 'training'
DEFAULT_LEARNING_RATE = 0.01
DEFAULT_TRAINING_ITERATIONS = 1000
DEFAULT_IMAGE_SIZE = (28, 28)

def __retrieve_args():
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
    args = __retrieve_args()
    mode = args.mode
    data = args.data or f'./data/{mode}'

    print(f'Running neural network in {mode} mode with data in directory {data} ...')

    neural_network = NeuralNetwork(
        np.prod(DEFAULT_IMAGE_SIZE),
        args.amount_hidden_layers,
        args.hidden_activation_function,
        args.output_activation_function,
        TARGETS
    )

    filenames = []
    for filetype in SUPPORTED_FILETYPES:
        filenames.extend(glob.glob(f'{data}/*.{filetype}'))
    np.random.shuffle(filenames)

    if mode == 'training':
        training_start_time = datetime.now()
        images = [
            [int((filename.split('/')[-1]).split('-')[0]), np.array(Image.open(filename).resize(DEFAULT_IMAGE_SIZE))]
            for filename in filenames]
        neural_network.train(images, args.learning_rate, args.training_iterations)
        print(f'Finished training in {(datetime.now() - training_start_time).total_seconds()} seconds.')
        mode = 'prediction'

    if mode == 'prediction':
        prediction_start_time = datetime.now()
        for filename in filenames:
            image = np.array(Image.open(filename).resize(DEFAULT_IMAGE_SIZE))
            prediction = neural_network.predict(image)
            print(f'Prediction result for file {filename}: {prediction}')
            print(f'Finished prediction in {(datetime.now() - prediction_start_time).total_seconds()} seconds.')
