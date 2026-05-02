import sys
import json
import torch
import os
from src.model import BioGraphSAGE
from src.data_loader import load_biological_network

def run_inference(target_id):
    # 1. Load the real-world dataset and encoder
    # Using the STRING dataset slice for Homo sapiens
    data, encoder = load_biological_network("data/9606.protein.links.v12.0.txt")
    
    try:
        # 2. Map the ID to the graph node index
        node_idx = encoder.transform([target_id])[0]
    except ValueError:
        return {"error": "ID not found in biological network"}

    # 3. Initialize and load the INT8 Quantized Model
    model = BioGraphSAGE(in_channels=128, hidden_channels=64, out_channels=1)
    
    # Target Linear layers for dynamic quantization to match the quantizer.py logic
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    
    model_path = "models/edge_optimized_model.pt"
    if os.path.exists(model_path):
        quantized_model.load_state_dict(torch.load(model_path))
    
    quantized_model.eval()

    # 4. Perform localized inference
    with torch.no_grad():
        output = quantized_model(data.x, data.edge_index)
        # Extract the score for the specific node
        score = torch.sigmoid(output[node_idx]).item()

    return {
        "id": target_id,
        "score": round(score, 4),
        "nodes_processed": data.num_nodes,
        "edges_processed": data.num_edges
    }

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Accept ID from Laravel shell_exec
        print(json.dumps(run_inference(sys.argv[1])))