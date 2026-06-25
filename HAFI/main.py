import torch
import numpy as np

from src.data import EnhancedDTADataset
from src.models import HAFIDTA
from src.training import NestedCVTrainer
from src.utils import device


def main():
    print(f"Using device: {device}")
    print("Loading dataset...")
    dataset = EnhancedDTADataset(data_path='data/davis/')
    print(f"Loaded {len(dataset.drug_graphs)} samples for nested cross validation.")

    nested_cv_loaders = dataset.get_nested_cv_loaders(
        batch_size=32,
        outer_splits=6,
        inner_splits=5,
        n_repeats=5,
        seed=42
    )

    model_args = {
        'drug_smiles_vocab_size': len(dataset.smiles_vocab) + 1,
        'protein_aa_vocab_size': len(dataset.aa_vocab) + 1,
        'drug_graph_in_dim': 10,
        'protein_graph_in_dim': 6,
        'fingerprint_size': dataset.fingerprint_size,
        'protein_phys_dim': 47,
        'hidden_dim': 768,
        'num_heads': 8,
        'dropout': 0.3,
        'drug_max_len': dataset.max_smiles_len,
        'protein_max_len': dataset.max_protein_len
    }

    model = HAFIDTA(**model_args)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    trainer = NestedCVTrainer(
        model_class=HAFIDTA,
        model_args=model_args,
        nested_cv_loaders=nested_cv_loaders,
        scaler=dataset.scaler,
        device=device,
        epochs=500,
        seed=42
    )

    trainer.train()
    print("\nNested cross validation completed.")


if __name__ == "__main__":
    main()