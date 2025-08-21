import glob
import os
import pickle

def save(something, filename):
    with open(filename, 'wb') as file:
        pickle.dump(something, file)

def load(filename, create_if_not_exists=False) -> list | None:
    if not os.path.exists(filename):
        if not create_if_not_exists:
            return None
        save([], filename)
    with open(filename, 'rb') as file:
        return pickle.load(file)

def delete(filename):
    for file in glob.glob(filename):
        os.remove(file)
