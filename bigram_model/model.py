from torch import nn
import torch
from torch.nn import functional as F


class BiGramModel(nn.Module):
    
    def __init__(self, vocab_size):
        super(BiGramModel, self).__init__()
        self.emb = nn.Embedding(vocab_size, vocab_size)
    
    def forward(self, x, y = None):
        logits = self.emb(x) # B x T x C
        if y != None:
            # Converting the Tensors as pytorch requires BT x C tensor
            B, T, C = logits.shape
            log = logits.view(B*T, C)
            target = y.view(B*T)
            loss = F.cross_entropy(log, target)
            return logits, loss
        return logits, None
       
    def generate(self, x, max_new_tokens):
        
        for _ in range(max_new_tokens):
            
            logits, _ = self(x)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            
            # Sampling from sample distribution 
            idx_next = torch.multinomial(probs, num_samples = 1)
            
            x = torch.cat((x, idx_next), dim=1)
        return x
    
class BiGramModelV2(nn.Module):
    """
    This model is with Positional Embeddings
    """
    def __init__(self, vocab_size, block_size) -> None:
        super().__init__()

        self.token_embedding_table = nn.Embedding(vocab_size, 32)
        self.pos_embedding_table = nn.Embedding(block_size, 32)
        self.block_size = block_size
        self.lm_head = nn.Linear(32, vocab_size)
    
    def forward(self, x, y = None):
        B, T = x.shape
        token_emb = self.token_embedding_table(x) # B x T x C
        pos_emb = self.pos_embedding_table(torch.arange(T))
        x = token_emb + pos_emb
        logits = self.lm_head(x)

        if y != None:
            # Converting the Tensors as pytorch requires BT x C tensor
            B, T, C = logits.shape
            log = logits.view(B*T, C)
            target = y.view(B*T)
            loss = F.cross_entropy(log, target)
            return logits, loss
        return logits, None
       
    def generate(self, x, max_new_tokens):
        
        for _ in range(max_new_tokens):
            
            logits, _ = self(x[:, -self.block_size: ])
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            
            # Sampling from sample distribution 
            idx_next = torch.multinomial(probs, num_samples = 1)
            
            x = torch.cat((x, idx_next), dim=1)
        return x