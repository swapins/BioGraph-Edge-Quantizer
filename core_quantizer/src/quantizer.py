import torch
import os
from mock_gnn import BioInteractionGNN

def quantize_model(model_path, output_path):
    """
    Applies dynamic INT8 quantization to a heavy PyTorch model.
    This reduces the memory footprint significantly for SBC/Edge deployment.
    """
    # 1. Load the heavy FP32 model
    model = BioInteractionGNN()
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
    else:
        print(f"Warning: {model_path} not found. Using untrained weights.")

    model.eval()

    # 2. Apply dynamic quantization to Linear layers
    # Converts weights from Float32 to Int8
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )

    # 3. Save the optimized model
    torch.save(quantized_model.state_dict(), output_path)
    
    # Calculate savings
    original_size = os.path.getsize(model_path) if os.path.exists(model_path) else 0
    new_size = os.path.getsize(output_path)
    
    return original_size, new_size, quantized_model

if __name__ == "__main__":
    os.makedirs("../models", exist_ok=True)
    heavy_path = "../models/heavy_bio_model.pt"
    edge_path = "../models/edge_optimized_model.pt"
    
    orig_size, new_size, _ = quantize_model(heavy_path, edge_path)
    
    print("Quantization Complete.")
    if orig_size > 0:
        print(f"Model compressed from {orig_size/1024/1024:.2f} MB to {new_size/1024/1024:.2f} MB")