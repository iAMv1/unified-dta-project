import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool, global_max_pool
from transformers import EsmModel, EsmTokenizer


class ESMProteinEncoder(nn.Module):
    def __init__(self, output_dim=128):
        super().__init__()
        self.esm_model = EsmModel.from_pretrained("facebook/esm2_t6_8M_UR50D")
        self.esm_tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
        self.projection = nn.Linear(320, output_dim)

        # Freeze ESM initially
        for param in self.esm_model.parameters():
            param.requires_grad = False

    def forward(self, protein_sequences):
        # Truncate sequences to avoid memory issues
        truncated_seqs = [seq[:200] for seq in protein_sequences]  # Much shorter

        inputs = self.esm_tokenizer(truncated_seqs, return_tensors="pt", padding=True,
                                   truncation=True, max_length=200)
        device = next(self.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.esm_model(**inputs)
            esm_features = outputs.last_hidden_state[:, 0, :]

        return self.projection(esm_features)

    def unfreeze_esm(self):
        for param in self.esm_model.encoder.layer[-4:]:
            for p in param.parameters():
                p.requires_grad = True


class GINDrugEncoder(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.gin_layers = nn.ModuleList([
            GINConv(nn.Sequential(nn.Linear(78 if i==0 else hidden_dim, hidden_dim),
                                 nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)))
            for i in range(3)
        ])
        self.final_proj = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for gin_layer in self.gin_layers:
            x = F.relu(gin_layer(x, edge_index))

        mean_pool = global_mean_pool(x, batch)
        max_pool = global_max_pool(x, batch)
        return self.final_proj(torch.cat([mean_pool, max_pool], dim=1))


class CombinedDTAModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.protein_encoder = ESMProteinEncoder()
        self.drug_encoder = GINDrugEncoder()
        self.predictor = nn.Sequential(
            nn.Linear(256, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 1)
        )

    def forward(self, drug_data, protein_sequences):
        drug_features = self.drug_encoder(drug_data)
        protein_features = self.protein_encoder(protein_sequences)
        combined = torch.cat([drug_features, protein_features], dim=1)
        return self.predictor(combined)

    def set_training_phase(self, phase):
        if phase == 2:
            self.protein_encoder.unfreeze_esm()
