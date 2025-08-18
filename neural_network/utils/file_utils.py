import glob
import os
import pickle

def save(something, filename):
    with open(filename, 'wb') as file:
        pickle.dump(something, file)

def load(filename, create_if_not_exists=False):
    if create_if_not_exists and not os.path.exists(filename):
        save([], filename)
    with open(filename, 'rb') as file:
        return pickle.load(file)

def delete(filename):
    for file in glob.glob(filename):
        os.remove(file)
