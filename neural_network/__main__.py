import argparse
import glob
import os.path
from datetime import datetime
import pandas as pd

import numpy as np
from PIL import Image

from neural_network import NeuralNetwork

TYPES = ('csv', 'img')
SUPPORTED_IMG_FILES = ('jpg', 'jpeg', 'png')
SUPPORTED_FILETYPES = ('csv', 'jpg', 'jpeg', 'png')
MODES = ('prediction', 'training')
ACTIVATION_FUNCTIONS = ('sigmoid', 'softmax', 'tanh', 'relu')
TARGETS_FILE = './data/config/targets.npy'

DEFAULT_IMAGE_SIZE = (40, 40)
DEFAULT_MODE = 'training'
DEFAULT_DATATYPE = 'csv'
DEFAULT_ACTIVATION_FUNCTION = 'sigmoid'
DEFAULT_AMOUNT_HIDDEN_LAYERS = 2
DEFAULT_LEARNING_RATE = 0.01
DEFAULT_TRAINING_ITERATIONS = 1000
DEFAULT_BATCH_SIZE = 10

def retrieve_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=MODES, type=str, default=DEFAULT_MODE)
    parser.add_argument('--datatype', choices=TYPES, type=str, default=DEFAULT_DATATYPE)
    parser.add_argument('--data', type=str)
    parser.add_argument('--amount-hidden-layers', type=int, default=DEFAULT_AMOUNT_HIDDEN_LAYERS)
    parser.add_argument('--hidden-activation-function', type=str, choices=ACTIVATION_FUNCTIONS,
                        default=DEFAULT_ACTIVATION_FUNCTION)
    parser.add_argument('--output-activation-function', type=str, choices=ACTIVATION_FUNCTIONS,
                        default=DEFAULT_ACTIVATION_FUNCTION)
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--training-iterations', type=int, default=DEFAULT_TRAINING_ITERATIONS)
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE)
    return parser.parse_args()

def retrieve_targets(image_data):
    if not os.path.exists(TARGETS_FILE):
        np.save(TARGETS_FILE, [])

    loaded_targets = np.load(TARGETS_FILE, allow_pickle=True)
    image_targets = set([entry[1][0] for entry in image_data])

    target_diffs = image_targets.difference(loaded_targets)
    if target_diffs:
        loaded_targets = np.concatenate((loaded_targets, list(target_diffs)), axis=0)
        np.save(TARGETS_FILE, loaded_targets)

    return loaded_targets

def retrieve_image_data(file_name) -> list:
    if file_name.endswith('.csv'):
        csv = np.array(pd.read_csv(file_name, dtype=str, header=None))
        return [[file_name, [entry[:1][0], [int(value) for value in entry[1:]]]] for entry in csv]
    elif file_name.endswith(SUPPORTED_IMG_FILES):
        image_data = np.array(Image.open(file_name).resize(DEFAULT_IMAGE_SIZE))
        return [[file_name, [(file_name.split('/')[-1]).split('_')[0], np.ndarray.flatten(image_data)]]]
    else:
        raise ValueError(f'Filetype {file_name.split('.')[-1]} is not supported')

if __name__ == '__main__':
    args = retrieve_args()
    mode = args.mode
    datatype = args.datatype
    data = args.data
    if not data:
        data = f'./data/{mode}' + ('' if mode == 'prediction' else f'/{datatype}')

    print(f'Running neural network in {mode} mode with files in {data} ...')

    filenames = []
    for filetype in SUPPORTED_FILETYPES:
        filenames.extend(glob.glob(f'{data}/*.{filetype}'))

    images = []
    for filename in filenames:
        images.extend(retrieve_image_data(filename))

    targets = retrieve_targets(images)

    neural_network = NeuralNetwork(
        np.prod(DEFAULT_IMAGE_SIZE),
        args.amount_hidden_layers,
        args.hidden_activation_function,
        args.output_activation_function,
        targets
    )

    start_time = datetime.now()

    if mode == 'training':
        neural_network.train(images, args.learning_rate, args.training_iterations, args.batch_size)
        mode = 'prediction' # just for checking in

    if mode == 'prediction':
        for image in images:
            prediction = neural_network.predict(image[1])
            print(f'Prediction result for file {image[0]}: {prediction}')

    print(f'Finished {mode} in {(datetime.now() - start_time).total_seconds()} seconds.')
