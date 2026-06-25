import torch
import os
from tqdm import tqdm
import esm
from torch_geometric.data import Data


class EnhancedESMProcessor:
    def __init__(self, model_dir="esm"):
        model_path = os.path.join(model_dir, "esm2_t33_650M_UR50D.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ESM model not found at {model_path}")
        self.model, self.alphabet = esm.pretrained.load_model_and_alphabet(model_path)
        self.model = self.model.eval().to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.batch_converter = self.alphabet.get_batch_converter()
        self.model.requires_grad_(False)

    @torch.no_grad()
    def get_contact_graph(self, sequences, cache_dir=None, batch_size=1, enhanced_node_features=True):
        results = []
        os.makedirs(cache_dir, exist_ok=True) if cache_dir else None
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        for i in tqdm(range(0, len(sequences), batch_size), desc="Processing proteins"):
            batch = sequences[i:i+batch_size]
            need_process = []
            for idx, (seq_hash, seq) in enumerate(batch):
                cache_path = os.path.join(cache_dir, f"{seq_hash}.pt") if cache_dir else None
                if cache_path and os.path.exists(cache_path):
                    try:
                        graph = torch.load(cache_path)
                        results.append(graph)
                        continue
                    except:
                        pass
                need_process.append((idx, seq_hash, seq))

            if not need_process:
                continue

            try:
                _, _, tokens = self.batch_converter([(h, s) for _, h, s in need_process])
                tokens = tokens.to(device)
                with torch.amp.autocast('cuda', enabled=False):
                    outputs = self.model(tokens, repr_layers=[33], return_contacts=True)
                    contacts = outputs["contacts"].float().cpu()
                del tokens, outputs
                torch.cuda.empty_cache()

                for idx_in_batch, (orig_idx, seq_hash, seq) in enumerate(need_process):
                    cmap = contacts[idx_in_batch]
                    cmap = torch.nan_to_num(cmap, 0.0, 1.0, -1.0).clamp(0, 1)
                    edge_index, edge_attr = self._contact_to_graph(cmap)
                    num_nodes = cmap.size(0)
                    mask = (edge_index[0] < num_nodes) & (edge_index[1] < num_nodes)
                    edge_index = edge_index[:, mask]
                    edge_attr = edge_attr[mask]

                    if enhanced_node_features:
                        node_feat = self._compute_node_features(seq, num_nodes)
                    else:
                        node_feat = torch.ones(num_nodes, 2)

                    graph = Data(x=node_feat, edge_index=edge_index, edge_attr=edge_attr)
                    if cache_dir:
                        torch.save(graph, os.path.join(cache_dir, f"{seq_hash}.pt"))
                    results.append(graph)
                del contacts
                torch.cuda.empty_cache()
            except RuntimeError as e:
                for _, _, seq in need_process:
                    num_nodes = len(seq)
                    if enhanced_node_features:
                        node_feat = torch.zeros(num_nodes, 6)
                    else:
                        node_feat = torch.zeros(num_nodes, 2)
                    results.append(Data(x=node_feat, edge_index=torch.empty((2,0), dtype=torch.long), edge_attr=torch.empty((0,1))))
                torch.cuda.empty_cache()
        return results

    def _compute_node_features(self, sequence, num_nodes):
        hydrophobicity = {'A':0.62, 'R':-2.53, 'N':-0.78, 'D':-0.90, 'C':0.29, 'E':-0.74, 'Q':-0.85,
                          'G':0.48, 'H':-0.40, 'I':1.38, 'L':1.06, 'K':-1.50, 'M':0.64, 'F':1.19,
                          'P':0.12, 'S':-0.18, 'T':-0.05, 'W':0.81, 'Y':0.26, 'V':1.08}
        charge = {'K':1, 'R':1, 'H':0.5, 'D':-1, 'E':-1}
        polarity = {'S':1, 'T':1, 'N':1, 'Q':1, 'C':0.5, 'Y':0.5}
        accessibility = {'A':1.28, 'R':2.34, 'N':1.60, 'D':1.59, 'C':1.43, 'E':1.87, 'Q':1.93,
                         'G':0.91, 'H':1.86, 'I':1.81, 'L':1.81, 'K':2.01, 'M':1.94, 'F':2.02,
                         'P':1.35, 'S':1.31, 'T':1.48, 'W':2.25, 'Y':2.15, 'V':1.60}
        helix_tendency = {'A':1.45, 'R':0.96, 'N':0.76, 'D':1.04, 'C':0.78, 'E':1.59, 'Q':1.27,
                          'G':0.43, 'H':1.05, 'I':1.09, 'L':1.34, 'K':1.07, 'M':1.30, 'F':1.12,
                          'P':0.34, 'S':0.82, 'T':0.82, 'W':1.02, 'Y':0.80, 'V':1.06}

        features = []
        seq_upper = sequence.upper()
        for i, aa in enumerate(seq_upper[:num_nodes]):
            pos_norm = i / max(1, num_nodes-1)
            hyd = hydrophobicity.get(aa, 0.0)
            chg = charge.get(aa, 0.0)
            pol = polarity.get(aa, 0.0)
            acc = accessibility.get(aa, 1.0) / 2.5
            helix = helix_tendency.get(aa, 1.0) / 1.6
            features.append([pos_norm, hyd, chg, pol, acc, helix])
        if len(features) < num_nodes:
            features += [[0]*6] * (num_nodes - len(features))
        return torch.tensor(features, dtype=torch.float32)

    def _contact_to_graph(self, contact_map, threshold=0.2, top_k=32):
        seq_len = contact_map.size(0)
        if seq_len < 2:
            return torch.empty((2,0), dtype=torch.long), torch.empty(0,1)
        k = min(top_k, seq_len-1)
        values, indices = torch.topk(contact_map, k=k, dim=1)
        rows = torch.arange(seq_len).unsqueeze(1).repeat(1, k)
        edge_index = torch.stack([rows.flatten(), indices.flatten()])
        edge_attr = values.flatten()
        mask = edge_attr > threshold
        edge_index = edge_index[:, mask]
        edge_attr = edge_attr[mask]
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        edge_attr = torch.cat([edge_attr, edge_attr])
        return edge_index.long(), edge_attr.unsqueeze(1)