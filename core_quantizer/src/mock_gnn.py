import torch
import torch.nn as nn
import os

class BioInteractionGNN(nn.Module):
    """
    Mock Graph Neural Network representing a computationally heavy
    Protein-Protein Interaction (PPI) model for oncology.
    In a real scenario, this would include PyTorch Geometric layers (GCNConv).
    """
    def __init__(self, input_dim=256, hidden_dim=512, output_dim=1):
        super(BioInteractionGNN, self).__init__()
        # Simulating heavy dense layers that bloat model size
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.out(x)
        return self.sigmoid(x)

if __name__ == "__main__":
    # When this script is run, it generates our "heavy" baseline model
    model = BioInteractionGNN()
    
    # Ensure the models directory exists
    os.makedirs("../models", exist_ok=True)
    
    save_path = "../models/heavy_bio_model.pt"
    torch.save(model.state_dict(), save_path)
    
    size_mb = os.path.getsize(save_path) / (1024 * 1024)
    print(f"Generated unoptimized FP32 baseline model.")
    print(f"Saved to: {save_path} ({size_mb:.2f} MB)")