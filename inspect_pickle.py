import pickle

file_path = 'data/en-fr/prepared/prepared/epoch_40/train.en'

with open(file_path, 'rb') as file:
    data = pickle.load(file)

print(data)
