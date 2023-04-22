import torch
from torch import optim

from dataset import TextDataset
from model import BiGramModel, BiGramModelV2
from attention_model import SelfAttentionModel
from train import train_fn, eval_fn
from tqdm import tqdm
import config
import numpy as np
import mlflow

job_type = input("Enter the Run Name: ")

    
# with mlflow.start_run(run_name=job_type) as run:
# Initialize the dataset
dataset = TextDataset()

#Model initialization

model_name = "v2"
# model = (BiGramModel(dataset.vocab_size) if model_name == "v1" else BiGramModelV2(vocab_size=dataset.vocab_size, block_size=config.block_size)).to(config.device)
model = SelfAttentionModel(dataset.vocab_size, block_size=config.block_size, head_size=32)
# optimizer
optimizer = optim.AdamW(model.parameters(), lr = config.lr)

with mlflow.start_run(run_name=job_type) as run:

    # Parameters Logging
    mlflow.log_params({
        "lr": config.lr,
        "block_size": config.block_size,
        "batch_size": config.batch_size,
        "n_epochs": config.epochs,
        "n_step": config.steps,
        "train_size": config.train_size,
        "vocab_size": dataset.vocab_size,
        "model_name": model_name
    })

    final_train_loss, final_val_loss = [], []
    train_loss, eval_loss = 0, 0
    for epoch in tqdm(range(config.epochs), desc="Execution Loop"):

        # Training
        train_loss = train_fn(model, dataset, optimizer)

        # Evaluation 
        eval_loss = eval_fn(model, dataset=dataset)
        
        mlflow.log_metrics({"training_loss": train_loss,"validation_loss": eval_loss})
        final_train_loss.append(train_loss)
        final_val_loss.append(eval_loss)
        
    print(f"\nTrain Loss: {np.mean(final_train_loss)}, Validation loss: {np.mean(final_val_loss)}")
    # run.finish()

    mlflow.pytorch.log_model(model, "models")

print("\n\n")
print("="*40)
print("Text Generation")
print("=" * 40)
initial = torch.zeros((1,1), dtype=torch.long).to(config.device)
print(dataset.decode(model.generate(initial, 1000)))