import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
# from rdkit import Chem  # Skip RDKit for now
from combined_model import CombinedDTAModel
from scipy.stats import pearsonr


class DTADataset(Dataset):
    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        return row['compound_iso_smiles'], row['target_sequence'], float(row['affinity'])


def smiles_to_graph(smiles):
    # Simple dummy graph for testing without RDKit
    # In real implementation, use RDKit to parse SMILES
    num_atoms = min(len(smiles), 10)  # Simple approximation

    # Create dummy node features (78 features per node)
    node_features = [[1.0] * 78 for _ in range(max(1, num_atoms))]

    # Create simple edge indices (chain structure)
    edge_indices = []
    for i in range(max(1, num_atoms - 1)):
        edge_indices.extend([[i, i+1], [i+1, i]])

    if not edge_indices: edge_indices = [[0, 0]]

    return Data(x=torch.tensor(node_features, dtype=torch.float),
               edge_index=torch.tensor(edge_indices, dtype=torch.long).t().contiguous())


def collate_fn(batch):
    valid_batch = [(smiles_to_graph(smiles), protein, affinity)
                   for smiles, protein, affinity in batch
                   if smiles_to_graph(smiles) is not None]

    if not valid_batch: return None

    graphs, proteins, affinities = zip(*valid_batch)
    return {
        'drug_data': Batch.from_data_list(graphs),
        'protein_sequences': list(proteins),
        'affinities': torch.tensor(affinities, dtype=torch.float)
    }


def train_model(model, train_loader, val_loader, epochs, lr):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for batch in train_loader:
            if batch is None: continue
            batch['drug_data'] = batch['drug_data'].to(device)
            batch['affinities'] = batch['affinities'].to(device)

            optimizer.zero_grad()
            predictions = model(batch['drug_data'], batch['protein_sequences'])
            loss = criterion(predictions.squeeze(), batch['affinities'])
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                if batch is None: continue
                batch['drug_data'] = batch['drug_data'].to(device)
                batch['affinities'] = batch['affinities'].to(device)

                predictions = model(batch['drug_data'], batch['protein_sequences'])
                val_preds.extend(predictions.cpu().numpy())
                val_targets.extend(batch['affinities'].cpu().numpy())

        if val_preds:
            pearson_corr, _ = pearsonr(val_preds, val_targets)
            print(f"Epoch {epoch+1}: Loss: {train_loss/len(train_loader):.4f}, Pearson: {pearson_corr:.4f}")


def main():
    train_dataset = DTADataset("DoubleSG-DTA/data/kiba_train.csv")
    val_dataset = DTADataset("DoubleSG-DTA/data/kiba_test.csv")

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)  # Much smaller batch
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

    model = CombinedDTAModel()

    print("Phase 1: Frozen ESM")
    model.set_training_phase(1)
    train_model(model, train_loader, val_loader, 20, 1e-3)

    print("Phase 2: ESM Fine-tuning")
    model.set_training_phase(2)
    train_model(model, train_loader, val_loader, 10, 1e-4)

    torch.save(model.state_dict(), 'combined_model.pth')
    print("Model saved")


if __name__ == "__main__":
    main()
