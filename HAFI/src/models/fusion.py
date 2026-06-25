import torch
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        attn_out, _ = self.attn(x, x, x)
        return attn_out.squeeze(1)


class CrossModalAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, query_feat, key_value_feat):
        q = query_feat.unsqueeze(1)
        kv = key_value_feat.unsqueeze(1)
        attn_out, _ = self.attn(q, kv, kv)
        out = query_feat + attn_out.squeeze(1)
        out = self.norm(out)
        return out


class HierarchicalAttentionFusion(nn.Module):
    def __init__(self, feat_dim=768, num_heads=8, dropout=0.1):
        super().__init__()
        self.feat_dim = feat_dim
        self.self_attn_drug_graph = MultiHeadSelfAttention(feat_dim, num_heads, dropout)
        self.self_attn_protein_graph = MultiHeadSelfAttention(feat_dim, num_heads, dropout)
        self.self_attn_drug_seq = MultiHeadSelfAttention(feat_dim, num_heads, dropout)
        self.self_attn_protein_seq = MultiHeadSelfAttention(feat_dim, num_heads, dropout)
        self.self_attn_drug_fp = MultiHeadSelfAttention(feat_dim, num_heads, dropout)
        self.self_attn_protein_phys = MultiHeadSelfAttention(feat_dim, num_heads, dropout)

        self.cross_druggraph_proteingraph = CrossModalAttention(feat_dim, num_heads, dropout)
        self.cross_proteingraph_druggraph = CrossModalAttention(feat_dim, num_heads, dropout)
        self.cross_druggraph_drugseq = CrossModalAttention(feat_dim, num_heads, dropout)
        self.cross_drugseq_druggraph = CrossModalAttention(feat_dim, num_heads, dropout)
        self.cross_proteingraph_proteinseq = CrossModalAttention(feat_dim, num_heads, dropout)
        self.cross_proteinseq_proteingraph = CrossModalAttention(feat_dim, num_heads, dropout)

        self.fusion_proj = nn.Sequential(
            nn.Linear(feat_dim * 6, feat_dim * 2),
            nn.LayerNorm(feat_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim * 2, feat_dim)
        )

    def forward(self, drug_graph, protein_graph, drug_seq, protein_seq, drug_fp, protein_phys):
        drug_graph = self.self_attn_drug_graph(drug_graph)
        protein_graph = self.self_attn_protein_graph(protein_graph)
        drug_seq = self.self_attn_drug_seq(drug_seq)
        protein_seq = self.self_attn_protein_seq(protein_seq)
        drug_fp = self.self_attn_drug_fp(drug_fp)
        protein_phys = self.self_attn_protein_phys(protein_phys)

        drug_graph = self.cross_druggraph_proteingraph(drug_graph, protein_graph)
        protein_graph = self.cross_proteingraph_druggraph(protein_graph, drug_graph)
        drug_graph = self.cross_druggraph_drugseq(drug_graph, drug_seq)
        drug_seq = self.cross_drugseq_druggraph(drug_seq, drug_graph)
        protein_graph = self.cross_proteingraph_proteinseq(protein_graph, protein_seq)
        protein_seq = self.cross_proteinseq_proteingraph(protein_seq, protein_graph)

        fused = torch.cat([drug_graph, protein_graph, drug_seq, protein_seq, drug_fp, protein_phys], dim=1)
        fused = self.fusion_proj(fused)
        return fused