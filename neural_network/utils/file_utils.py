import glob
import os
import pickle

def save_to_file(filename, something):
    with open(filename, 'wb') as file:
        pickle.dump(something, file)

def load_from_file(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

def delete_files(filename_pattern):
    for file in glob.glob(filename_pattern):
        os.remove(file)
