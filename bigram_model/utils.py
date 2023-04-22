import os
from sklearn.model_selection import train_test_split

def read_file(path):
    if not os.path.exists(path):
        raise FileExistsError(f"{path} doesn't exists")
    with open(path) as f:
        return f.read()

def generate_character_vocab(text):
    vocab = "".join(sorted(set(text)))
    ctoi = {ch: idx for idx, ch in enumerate(vocab)}
    itoc = {idx: ch for idx, ch in enumerate(vocab)}
    return vocab, ctoi, itoc

def sequential_data_split(data, train_size):
    train, test = train_test_split(data, train_size=train_size, shuffle=False)
    return train, test


encoder = lambda x, ctoi: [ctoi[i] for i in x ]
decoder = lambda x, itoc: [itoc[i] for i in x]
