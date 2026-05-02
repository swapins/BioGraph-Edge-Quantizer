import pandas as pd
import torch
import numpy as np
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder

def load_biological_network(file_path):
    """
    Ingests raw STRING interaction data and transforms it into 
    a PyTorch Geometric Data object for Edge-GNN processing.
    """
    # 1. Load interaction data (expected: protein1, protein2, combined_score)
    df = pd.read_csv(file_path, sep=' ')
    
    # 2. Encode protein IDs into integer indices for node mapping
    all_proteins = pd.concat([df['protein1'], df['protein2']]).unique()
    le = LabelEncoder()
    le.fit(all_proteins)
    
    source = le.transform(df['protein1'])
    target = le.transform(df['protein2'])
    
    # 3. Create the Edge Index (COO format)
    # edge_index = torch.tensor([source, target], dtype=torch.long)

    edge_index = torch.from_numpy(np.stack([source, target])).long()
    # 4. Generate Node Features (In research, these would be GO terms/embeddings)
    # Here we use a standard 128-dim identity/random initialization for SBC profiling
    num_nodes = len(all_proteins)
    x = torch.randn(num_nodes, 128)
    
    # 5. Extract Edge Attributes (Normalization of confidence scores)
    edge_attr = torch.tensor(df['combined_score'].values / 1000, dtype=torch.float)
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr), le

if __name__ == "__main__":
    import os
    import sys

    data_dir = "data"
    file_name = "9606.protein.links.v12.0.txt"
    file_path = os.path.join(data_dir, file_name)

    # 1. Force directory creation
    os.makedirs(data_dir, exist_ok=True)

    # 2. Check for the generation flag OR missing file
    if "--generate-sample" in sys.argv or not os.path.exists(file_path):
        print(f"🛠️ Generating synthetic research slice at {file_path}...")
        with open(file_path, "w") as f:
            # Header must match load_biological_network logic
            f.write("protein1 protein2 combined_score\n")
            f.write("9606.ENSP00000269305 9606.ENSP00000398698 999\n") 
            f.write("9606.ENSP00000269305 9606.ENSP00000263025 850\n")
            f.write("9606.ENSP00000398698 9606.ENSP00000263025 700\n")
        print("Sample data generated successfully.")

    # 3. Safe Loading
    if os.path.exists(file_path):
        data, encoder = load_biological_network(file_path)
        print(f"Success: Loaded graph with {data.num_nodes} nodes and {data.num_edges} edges.")
    else:
        print(f"Critical Error: Could not find or create {file_path}")