from utils import read_file, generate_character_vocab, sequential_data_split, encoder, decoder
from config import input_data_path, train_size, block_size, batch_size, device
import torch

class TextDataset:

    def __init__(self):
        # Input Data
        input_data = read_file(input_data_path)
        self.vocab, self.ctoi, self.itoc = generate_character_vocab(input_data)
        self.vocab_size = len(self.vocab)

        train_text, test_text = sequential_data_split(input_data, train_size=train_size)
        self.train_data = torch.tensor(encoder(train_text, self.ctoi), dtype=torch.long)
        self.test_data = torch.tensor(encoder(test_text, self.ctoi), dtype=torch.long)

    def generate_batch(self, split):
        data = self.train_data if split == "train" else self.test_data
        rand_idx = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[idx: idx + block_size] for idx in rand_idx]).to(device)
        y = torch.stack([data[idx + 1: idx + block_size + 1] for idx in rand_idx]).to(device)
        return {"x": x, "y": y}
    
    def decode(self, tensor_data):
        first_row = tensor_data[0].numpy()
        return "".join(decoder(first_row, self.itoc))
