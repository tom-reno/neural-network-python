import pickle

def save_to_file(filename, something):
    with open(filename, 'wb') as file:
        pickle.dump(something, file)
