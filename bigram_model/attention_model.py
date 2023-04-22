import torch 
from torch import nn
from torch.nn import functional as F
import config

class AttentionHead(nn.Module):
    def __init__(self, n_emb: int,  n_heads: int, block_size: int, masked_attention: bool = True) -> None:
        super(AttentionHead, self).__init__()
        # Attenion Head Values
        self.key = nn.Linear(n_emb, n_heads, bias=False)
        self.query = nn.Linear(n_emb, n_heads, bias=False)
        self.value = nn.Linear(n_emb, n_heads, bias=False)
        self.masked_attention = masked_attention
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x)
        q = self.query(x)

        weights = q @ k.transpose(-2, -1) # B,T,C x B,C,T --> B, T, T
        if self.masked_attention:
            weights = weights.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        weights = F.softmax(weights, dim = -1)
        weights = self.dropout(weights)
        val = self.value(x)
        out = weights @ val # B,T,T x B,T,C -> B,T,C
        return out

class FeedForward(nn.Module):
    def __init__(self, in_emb) -> None:
        super().__init__()
        self.sequence = nn.Sequential(
            nn.Linear(in_emb, in_emb * 4),
            nn.ReLU(),
            nn.Linear(in_emb * 4, in_emb), # Projection Layer
            nn.Dropout(config.dropout)
        )
    
    def forward(self, x):
        return self.sequence(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, n_emb, head_size, block_size) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.n_embs = n_emb

        self.heads = nn.ModuleList(
            [
                AttentionHead(n_emb=n_emb, n_heads=head_size, block_size=block_size)
                for _ in range(n_heads)
            ]
        )
        self.project = nn.Linear(n_emb, n_emb)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, X):
        out =  torch.cat([h(X) for h in self.heads], dim=-1)
        out = self.project(out)
        out = self.dropout(out)
        return out


class Block(nn.Module):
    def __init__(self, n_emb, n_heads, head_size, block_size, is_single_head=False) -> None:
        super().__init__()
        self.n_embs = n_emb
        self.n_heads = n_heads
        self.head_size = head_size
        self.block_size = block_size
        self.is_single_head = is_single_head

        if self.is_single_head:
            self.heads = AttentionHead(n_emb=n_emb, n_heads=head_size, block_size=block_size)
        else:
            self.heads = MultiHeadAttention(n_emb=n_emb, n_heads=n_heads, head_size = head_size//n_heads, block_size=block_size)

        self.ffwd = FeedForward(n_emb)
        self.ln1 = nn.LayerNorm(n_emb)
        self.ln2 = nn.LayerNorm(n_emb)

    def forward(self, x):
        x = x + self.heads(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
    

class SelfAttentionModel(nn.Module):

    def __init__(self, vocab_size, block_size, head_size, n_heads=4) -> None:
        super().__init__()

        self.token_embedding_table = nn.Embedding(vocab_size, 32)
        self.pos_embedding_table = nn.Embedding(block_size, 32)
        self.block_size = block_size
        self.attention_head = nn.Sequential(
            Block(n_emb=32, n_heads=n_heads, head_size=head_size, block_size=block_size),
            Block(n_emb=32, n_heads=n_heads, head_size=head_size, block_size=block_size),
            Block(n_emb=32, n_heads=n_heads, head_size=head_size, block_size=block_size),
            Block(n_emb=32, n_heads=n_heads, head_size=head_size, block_size=block_size),
            nn.LayerNorm(32)
        )
        # self.attention_head = MultiHeadAtstention(n_emb=32, n_heads=n_heads, head_size=head_size//n_heads, block_size=block_size)
        self.lm_head = nn.Linear(32, vocab_size)
    
    def forward(self, x, y = None):
        B, T = x.shape
        token_emb = self.token_embedding_table(x) # B x T x C
        pos_emb = self.pos_embedding_table(torch.arange(T))
        x = token_emb + pos_emb
        x = self.attention_head(x)
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