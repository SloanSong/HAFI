import torch
import torch.nn as nn

from .vae import VAE
from .encoders import DrugGraphEncoder, ProteinContactGraphEncoder, SequenceEncoder
from .interaction import FeatureIterativeInteraction
from .fusion import HierarchicalAttentionFusion


class HAFIDTA(nn.Module):
    def __init__(self,
                 drug_smiles_vocab_size, protein_aa_vocab_size,
                 drug_graph_in_dim=10,
                 protein_graph_in_dim=6,
                 fingerprint_size=2048,
                 protein_phys_dim=47,
                 hidden_dim=768,
                 num_heads=8,
                 dropout=0.3,
                 drug_max_len=100,
                 protein_max_len=1024):
        super().__init__()

        self.drug_graph_encoder = DrugGraphEncoder(in_dim=drug_graph_in_dim, hidden_dim=384, out_dim=hidden_dim, heads=4, dropout=dropout)
        self.drug_seq_encoder = SequenceEncoder(vocab_size=drug_smiles_vocab_size, embed_dim=384, nhead=8, num_layers=6, hidden_dim=1536, dropout=dropout, max_len=drug_max_len)
        self.drug_fp_vae = VAE(input_dim=fingerprint_size, latent_dim=256, hidden_dim=512)
        self.drug_fp_proj = nn.Linear(256, hidden_dim)

        self.protein_graph_encoder = ProteinContactGraphEncoder(in_dim=protein_graph_in_dim, hidden_dim=384, out_dim=hidden_dim, dropout=dropout)
        self.protein_seq_encoder = SequenceEncoder(vocab_size=protein_aa_vocab_size, embed_dim=384, nhead=8, num_layers=6, hidden_dim=1536, dropout=dropout, max_len=protein_max_len)
        self.protein_phys_vae = VAE(input_dim=protein_phys_dim, latent_dim=128, hidden_dim=256)
        self.protein_phys_proj = nn.Linear(128, hidden_dim)

        self.iterative_interaction = FeatureIterativeInteraction(feat_dim=hidden_dim, hidden_dim=512, num_steps=5, dropout=dropout)
        self.attention_fusion = HierarchicalAttentionFusion(feat_dim=hidden_dim, num_heads=num_heads, dropout=dropout)

        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )

    def forward(self, drug_data, protein_data, drug_seq, protein_seq, fingerprints, protein_features):
        drug_graph_feat = self.drug_graph_encoder(drug_data)
        drug_seq_feat = self.drug_seq_encoder(drug_seq)
        drug_fp_latent, _, _, _ = self.drug_fp_vae(fingerprints, compute_loss=False)
        drug_fp_feat = self.drug_fp_proj(drug_fp_latent)

        protein_graph_feat = self.protein_graph_encoder(protein_data)
        protein_seq_feat = self.protein_seq_encoder(protein_seq)
        protein_phys_latent, _, _, _ = self.protein_phys_vae(protein_features, compute_loss=False)
        protein_phys_feat = self.protein_phys_proj(protein_phys_latent)

        refined_drug_graph, refined_protein_graph = self.iterative_interaction(drug_graph_feat, protein_graph_feat)

        fused_feat = self.attention_fusion(
            refined_drug_graph, refined_protein_graph,
            drug_seq_feat, protein_seq_feat,
            drug_fp_feat, protein_phys_feat
        )

        return self.predictor(fused_feat)