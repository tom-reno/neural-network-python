import argparse
import glob
from datetime import datetime

import numpy as np
import pandas as pd
from PIL import Image

from app.neural_network import NeuralNetwork
from utils import file_utils as fu

DATATYPES = ('csv', 'img')
SUPPORTED_IMG_FILES = ('jpg', 'jpeg', 'png')
SUPPORTED_FILETYPES = ('csv', 'jpg', 'jpeg', 'png')
MODES = ('prediction', 'training')
ACTIVATION_FUNCTIONS = ('sigmoid', 'softmax', 'tanh', 'relu')
TARGETS_FILE = './data/config/targets.pkl'

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
    parser.add_argument('--datatype', choices=DATATYPES, type=str, default=DEFAULT_DATATYPE)
    parser.add_argument('--directory', type=str)
    parser.add_argument('--amount-hidden-layers', type=int, default=DEFAULT_AMOUNT_HIDDEN_LAYERS)
    parser.add_argument('--hidden-activation-function', type=str, choices=ACTIVATION_FUNCTIONS,
                        default=DEFAULT_ACTIVATION_FUNCTION)
    parser.add_argument('--output-activation-function', type=str, choices=ACTIVATION_FUNCTIONS,
                        default=DEFAULT_ACTIVATION_FUNCTION)
    parser.add_argument('--learning-rate', type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument('--training-iterations', type=int, default=DEFAULT_TRAINING_ITERATIONS)
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE)
    return parser.parse_args()

def retrieve_targets(new_labels) -> list:
    loaded_targets: list = fu.load(TARGETS_FILE, True)
    new_targets: set = set(new_labels)

    targets_diffs: set = new_targets.difference(loaded_targets)
    if targets_diffs:
        loaded_targets += list(targets_diffs)
        fu.save(loaded_targets, TARGETS_FILE)

    return loaded_targets

def retrieve_image_data(files_names: list) -> tuple:
    names = []
    labels = []
    data = []
    for file_name in files_names:
        file = file_name.split('/')[-1]
        if file_name.endswith('.csv'):
            csv = np.array(pd.read_csv(file_name, dtype=str, header=None))
            names.extend([file for _ in range(len(csv))])
            labels.extend(csv[:,0].tolist())
            data.extend(csv[:,1:].astype(np.int16))
        elif file_name.endswith(SUPPORTED_IMG_FILES):
            names.append(file_name)
            labels.append(file.split('_')[0])
            data.append(np.array(Image.open(file_name).resize(DEFAULT_IMAGE_SIZE)))
        else:
            raise ValueError(f'Filetype {file.split('.')[-1]} is not supported')
    return names, labels, data


if __name__ == '__main__':
    args = retrieve_args()
    mode = args.mode
    datatype = args.datatype
    directory = args.directory
    if not directory:
        directory = f'./data/{mode}' + ('' if mode == 'prediction' else f'/{datatype}')

    print(f'Running neural network in {mode} mode with files from {directory} ...')

    filenames = []
    for filetype in SUPPORTED_FILETYPES:
        filenames.extend(glob.glob(f'{directory}/*.{filetype}'))

    images_filenames, images_labels, images_data = retrieve_image_data(filenames)
    targets: list = retrieve_targets(images_labels)

    neural_network = NeuralNetwork(
        np.prod(DEFAULT_IMAGE_SIZE),
        args.amount_hidden_layers,
        args.hidden_activation_function,
        args.output_activation_function,
        targets
    )

    start_time = datetime.now()

    if mode == 'training':
        neural_network.train(images_data, images_labels, args.learning_rate, args.training_iterations, args.batch_size)
        mode = 'prediction' # just for checking in

    if mode == 'prediction':
        for i, image_data in enumerate(images_data):
            prediction = neural_network.predict(image_data)
            print(f'Prediction result for file {images_filenames[i]} (label=\'{images_labels[i]}\'): {prediction}')

    print(f'Finished {mode} in {(datetime.now() - start_time).total_seconds()} seconds.')
