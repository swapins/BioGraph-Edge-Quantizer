import torch
import os
import sys
import pytest
sys.path.append(os.path.abspath("src"))  # Ensure src is in path for imports

from src.quantizer import quantize_model
from src.mock_gnn import BioInteractionGNN

def test_quantization_reduces_size():
    os.makedirs("models", exist_ok=True)
    heavy_path = "models/test_heavy.pt"
    edge_path = "models/test_edge.pt"
    
    # Generate temporary heavy model
    model = BioInteractionGNN()
    torch.save(model.state_dict(), heavy_path)
    
    # Run quantizer
    orig_size, new_size, _ = quantize_model(heavy_path, edge_path)
    
    # Assertions
    assert os.path.exists(edge_path), "Quantized model was not saved."
    assert new_size < orig_size, "Quantization failed to reduce model size!"
    
    # Cleanup
    os.remove(heavy_path)
    os.remove(edge_path)

def test_inference_shape_after_quantization():
    model = BioInteractionGNN()
    q_model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    
    # Input dim is 256 based on mock_gnn.py
    dummy_input = torch.randn(1, 256) 
    output = q_model(dummy_input)
    
    assert output.shape == (1, 1), "Inference shape mismatch!"