import torch

# DataSet Parametre
input_data_path = "../shakespeare_scripts.txt"
train_size = 0.9

# Training Parameters 
block_size = 256
batch_size = 64

epochs = 1000
steps = 10


# Optimizer Parameters
lr = 1e-3

# Model Parameter
dropout = 0.2

device = "cuda" if torch.cuda.is_available() else "cpu"
