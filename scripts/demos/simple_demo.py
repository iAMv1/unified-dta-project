"""
Simple demo showing the combined architecture working
"""
import torch
import torch.nn as nn
from torch_geometric.nn import GINConv, global_mean_pool
from torch_geometric.data import Data, Batch
import pandas as pd


class SimpleProteinEncoder(nn.Module):
    """Simple protein encoder without ESM for demo"""
    def __init__(self, output_dim=128):
        super().__init__()
        # Simple character-level encoding
        self.embedding = nn.Embedding(26, 64)  # 26 amino acids
        self.conv = nn.Conv1d(64, output_dim, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, protein_sequences):
        # Convert sequences to indices (A=0, C=1, etc.)
        batch_size = len(protein_sequences)
        max_len = min(max(len(seq) for seq in protein_sequences), 200)
        
        indices = torch.zeros(batch_size, max_len, dtype=torch.long)
        for i, seq in enumerate(protein_sequences):
            for j, aa in enumerate(seq[:max_len]):
                indices[i, j] = max(0, ord(aa.upper()) - ord('A'))
        
        # Embed and process
        embedded = self.embedding(indices)  # [batch, seq_len, 64]
        embedded = embedded.transpose(1, 2)  # [batch, 64, seq_len]
        conv_out = torch.relu(self.conv(embedded))  # [batch, 128, seq_len]
        pooled = self.pool(conv_out).squeeze(-1)  # [batch, 128]
        
        return pooled


class SimpleDrugEncoder(nn.Module):
    """Simple GIN-based drug encoder"""
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.gin = GINConv(nn.Sequential(
            nn.Linear(78, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        ))
        
    def forward(self, data):
        x = self.gin(data.x, data.edge_index)
        return global_mean_pool(x, data.batch)


class SimpleCombinedModel(nn.Module):
    """Simple combined model for demo"""
    def __init__(self):
        super().__init__()
        self.protein_encoder = SimpleProteinEncoder()
        self.drug_encoder = SimpleDrugEncoder()
        self.predictor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, drug_data, protein_sequences):
        drug_features = self.drug_encoder(drug_data)
        protein_features = self.protein_encoder(protein_sequences)
        combined = torch.cat([drug_features, protein_features], dim=1)
        return self.predictor(combined)


def create_dummy_data():
    """Create dummy data for demo"""
    # Dummy graph
    x = torch.randn(5, 78)  # 5 atoms, 78 features each
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    graph = Data(x=x, edge_index=edge_index)
    
    # Dummy protein sequence
    protein = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
    
    return graph, protein


def main():
    print("=== Simple Combined DTA Model Demo ===")
    
    # Create model
    model = SimpleCombinedModel()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create dummy data
    graph, protein = create_dummy_data()
    batch_graph = Batch.from_data_list([graph])
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        prediction = model(batch_graph, [protein])
    
    print(f"Predicted affinity: {prediction.item():.4f}")
    
    # Show architecture components
    print("\n=== Architecture Components ===")
    print("✓ Simple Protein Encoder (character-level)")
    print("✓ GIN Drug Encoder")
    print("✓ Combined Predictor")
    print("✓ End-to-end forward pass working")
    
    print("\n=== Next Steps ===")
    print("1. Replace SimpleProteinEncoder with ESM-2 (when you have more memory)")
    print("2. Add real SMILES-to-graph conversion with RDKit")
    print("3. Load actual KIBA/Davis datasets")
    print("4. Add cross-attention between drug and protein features")
    print("5. Implement 2-phase training (frozen ESM → fine-tuned ESM)")
    
    print("\n✅ Demo completed successfully!")


if __name__ == "__main__":
    main()
