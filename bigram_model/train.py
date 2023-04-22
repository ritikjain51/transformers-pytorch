from dataset import TextDataset
from config import steps
import torch
import numpy as np
from tqdm import tqdm

def train_fn(model, dataset: TextDataset, optim: torch.optim.Optimizer):
    model.train()
    losses = []
    for _ in tqdm(range(steps), leave=False, desc = "Train Steps "):
        data = dataset.generate_batch("train")
        _, loss = model(**data)
        losses.append(loss.item())
        optim.zero_grad()
        loss.backward()
        optim.step()     
    return np.mean(losses)

def eval_fn(model, dataset: TextDataset):
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in tqdm(range(steps), leave=False, desc = "Valid Steps "):
            data = dataset.generate_batch("val")
            _, loss = model(**data)
            losses.append(loss.item())
    return np.mean(losses)
