import torch
from combined_model import CombinedDTAModel
from train_combined import smiles_to_graph
from torch_geometric.data import Batch

# Load trained model
model = CombinedDTAModel()
model.load_state_dict(torch.load('combined_model.pth'))
model.eval()

# Example prediction
smiles = "CCO"  # Ethanol
protein = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"

graph = smiles_to_graph(smiles)
if graph:
    batch_graph = Batch.from_data_list([graph])
    with torch.no_grad():
        affinity = model(batch_graph, [protein])
    print(f"Predicted affinity: {affinity.item():.4f}")
else:
    print("Invalid SMILES")
