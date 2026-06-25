import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureIterativeInteraction(nn.Module):
    def __init__(self, feat_dim=768, hidden_dim=512, num_steps=5, dropout=0.3):
        super().__init__()
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim
        self.num_steps = num_steps

        self.init_proj = nn.Linear(feat_dim * 2, hidden_dim)
        self.gru = nn.GRU(
            input_size=feat_dim * 2,
            hidden_size=hidden_dim,
            num_layers=2,
            dropout=dropout,
            batch_first=True
        )
        self.refine_drug = nn.Linear(hidden_dim, feat_dim)
        self.refine_protein = nn.Linear(hidden_dim, feat_dim)
        self.gate_drug = nn.Sequential(nn.Linear(feat_dim * 2, feat_dim), nn.Sigmoid())
        self.gate_protein = nn.Sequential(nn.Linear(feat_dim * 2, feat_dim), nn.Sigmoid())

    def forward(self, drug_feat, protein_feat):
        batch_size = drug_feat.size(0)
        init_concat = torch.cat([drug_feat, protein_feat], dim=1)
        h0 = self.init_proj(init_concat).unsqueeze(0).repeat(2, 1, 1)

        current_drug = drug_feat
        current_protein = protein_feat
        for t in range(self.num_steps):
            step_input = torch.cat([current_drug, current_protein], dim=1).unsqueeze(1)
            out, h0 = self.gru(step_input, h0)
            h = out[:, -1, :]
            refined_drug = self.refine_drug(h)
            refined_protein = self.refine_protein(h)

            combined_drug = torch.cat([current_drug, refined_drug], dim=1)
            combined_protein = torch.cat([current_protein, refined_protein], dim=1)
            gate_drug = self.gate_drug(combined_drug)
            gate_protein = self.gate_protein(combined_protein)
            current_drug = gate_drug * refined_drug + (1 - gate_drug) * current_drug
            current_protein = gate_protein * refined_protein + (1 - gate_protein) * current_protein

        return current_drug, current_protein