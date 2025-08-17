"""
Quick Training Test
==================

Test the training pipeline with a small subset of data.
"""

import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from simple_demo import SimpleCombinedModel
from scipy.stats import pearsonr


class QuickDataset(Dataset):
    """Quick dataset for testing"""
    def __init__(self, num_samples=20):
        # Create dummy data
        self.data = []
        for i in range(num_samples):
            smiles = f"C{'C' * (i % 5)}O"  # Simple molecules
            protein = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"[:50 + (i % 20)]
            affinity = 5.0 + (i % 10) * 0.5  # Affinity between 5.0 and 9.5
            
            self.data.append((smiles, protein, affinity))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


def create_dummy_graph(smiles):
    """Create dummy graph from SMILES"""
    num_atoms = min(len(smiles), 8)
    node_features = torch.randn(max(1, num_atoms), 78)
    
    # Create chain edges
    edges = []
    for i in range(max(1, num_atoms - 1)):
        edges.extend([[i, i+1], [i+1, i]])
    
    if not edges:
        edges = [[0, 0]]
    
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return Data(x=node_features, edge_index=edge_index)


def collate_fn(batch):
    """Collate function for batching"""
    graphs = []
    proteins = []
    affinities = []
    
    for smiles, protein, affinity in batch:
        graph = create_dummy_graph(smiles)
        graphs.append(graph)
        proteins.append(protein)
        affinities.append(affinity)
    
    return {
        'drug_data': Batch.from_data_list(graphs),
        'protein_sequences': proteins,
        'affinities': torch.tensor(affinities, dtype=torch.float)
    }


def quick_training_test():
    """Run a quick training test"""
    print("🚀 Quick Training Test")
    print("=" * 30)
    
    # Create dataset
    print("📊 Creating test dataset...")
    dataset = QuickDataset(num_samples=40)
    
    # Split into train/val
    train_size = 30
    val_size = 10
    
    train_dataset = torch.utils.data.Subset(dataset, range(train_size))
    val_dataset = torch.utils.data.Subset(dataset, range(train_size, train_size + val_size))
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=5, shuffle=False, collate_fn=collate_fn)
    
    print(f"✅ Train samples: {len(train_dataset)}")
    print(f"✅ Val samples: {len(val_dataset)}")
    
    # Create model
    print("\n🧠 Creating model...")
    model = SimpleCombinedModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    print(f"✅ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training loop
    print("\n🎯 Training...")
    epochs = 5
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_count = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            predictions = model(batch['drug_data'], batch['protein_sequences'])
            loss = criterion(predictions.squeeze(), batch['affinities'])
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_count += 1
        
        # Validation
        model.eval()
        val_predictions = []
        val_targets = []
        val_loss = 0
        val_count = 0
        
        with torch.no_grad():
            for batch in val_loader:
                predictions = model(batch['drug_data'], batch['protein_sequences'])
                loss = criterion(predictions.squeeze(), batch['affinities'])
                
                val_predictions.extend(predictions.cpu().numpy())
                val_targets.extend(batch['affinities'].cpu().numpy())
                
                val_loss += loss.item()
                val_count += 1
        
        # Calculate metrics
        avg_train_loss = train_loss / train_count if train_count > 0 else 0
        avg_val_loss = val_loss / val_count if val_count > 0 else 0

        if len(val_predictions) > 1 and len(val_predictions) == len(val_targets):
            try:
                pearson_corr, _ = pearsonr(val_predictions, val_targets)
                print(f"Epoch {epoch+1}/{epochs}: Train Loss: {avg_train_loss:.4f}, "
                      f"Val Loss: {avg_val_loss:.4f}, Pearson: {pearson_corr:.4f}")
            except:
                print(f"Epoch {epoch+1}/{epochs}: Train Loss: {avg_train_loss:.4f}, "
                      f"Val Loss: {avg_val_loss:.4f}")
        else:
            print(f"Epoch {epoch+1}/{epochs}: Train Loss: {avg_train_loss:.4f}, "
                  f"Val Loss: {avg_val_loss:.4f}")
    
    print("\n✅ Training test completed successfully!")
    
    # Test prediction on new data
    print("\n🔮 Testing prediction...")
    test_smiles = "CCO"
    test_protein = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
    
    test_graph = create_dummy_graph(test_smiles)
    test_batch = Batch.from_data_list([test_graph])
    
    model.eval()
    with torch.no_grad():
        prediction = model(test_batch, [test_protein])
    
    print(f"✅ Prediction for '{test_smiles}': {prediction.item():.4f}")
    
    return True


def main():
    """Main function"""
    try:
        success = quick_training_test()
        
        if success:
            print("\n🎉 All training tests passed!")
            print("\nNext steps:")
            print("1. ✅ Basic model works")
            print("2. ✅ Training pipeline works")
            print("3. ✅ Data loading works")
            print("4. 🔄 Ready for full training with real data")
            print("5. 🔄 Ready for ESM-2 integration")
        else:
            print("\n❌ Training test failed")
            
    except Exception as e:
        print(f"\n❌ Training test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
