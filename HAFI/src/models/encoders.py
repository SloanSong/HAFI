import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GINConv, global_mean_pool
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class DrugGraphEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim=384, out_dim=768, heads=4, dropout=0.3):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden_dim, heads=heads, dropout=dropout, concat=True)
        self.conv2 = GATConv(hidden_dim * heads, out_dim, heads=1, dropout=dropout, concat=False)
        self.norm1 = nn.LayerNorm(hidden_dim * heads)
        self.norm2 = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = F.gelu(x)
        x = self.norm1(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = F.gelu(x)
        x = self.norm2(x)
        x = global_mean_pool(x, batch)
        return x


class ProteinContactGraphEncoder(nn.Module):
    def __init__(self, in_dim=6, hidden_dim=384, out_dim=768, dropout=0.3):
        super().__init__()
        nn1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim)
        )
        self.gin = GINConv(nn1)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, data):
        x = self.gin(data.x, data.edge_index)
        x = F.gelu(x)
        x = self.norm(x)
        x = global_mean_pool(x, data.batch)
        return x


class SequenceEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=384, nhead=8, num_layers=6, hidden_dim=1536, dropout=0.3, max_len=1024):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_encoder = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        encoder_layer = TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, dim_feedforward=hidden_dim,
            dropout=dropout, batch_first=True
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, 768)

    def forward(self, seq):
        x = self.embedding(seq)
        x = x + self.pos_encoder[:, :x.size(1), :]
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.fc(x)
        return x