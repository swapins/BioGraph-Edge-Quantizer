import torch
import os
import torch.jit
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from src.model import BioGraphSAGE
from src.utils import seed_everything


# -------------------------
# INT8 PACKING 
# -------------------------

def quantize_tensor(tensor):
    """
    Manual INT8 quantization
    """
    scale = tensor.abs().max() / 127
    scale = scale if scale != 0 else 1e-8

    q_tensor = (tensor / scale).round().clamp(-128, 127).to(torch.int8)
    return q_tensor, scale


def pack_model_weights(model):
    """
    Convert ALL Linear weights to INT8 packed format
    """
    packed_state = {}

    for name, param in model.state_dict().items():
        if "weight" in name:
            q, scale = quantize_tensor(param)
            packed_state[name] = q
            packed_state[name + "_scale"] = torch.tensor(scale)
        else:
            packed_state[name] = param

    return packed_state


# -------------------------
# SIZE UTILS
# -------------------------

def get_size_mb(path):
    return os.path.getsize(path) / 1024 / 1024


# -------------------------
# MAIN PIPELINE
# -------------------------

def optimize_model(model_path, output_path):
    seed_everything(42)

    model = BioGraphSAGE(
        in_channels=4096,
        hidden_channels=2048,
        out_channels=1
    )

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
    else:
        torch.save(model.state_dict(), model_path)

    model.eval()

    # --- Baseline ---
    baseline_size = get_size_mb(model_path)

    # --- PACK WEIGHTS (REAL INT8) ---
    packed_state = pack_model_weights(model)

    packed_path = model_path.replace(".pt", "_int8packed.pt")
    torch.save(packed_state, packed_path)

    packed_size = get_size_mb(packed_path)

    # --- TorchScript (unchanged, for runtime) ---
    scripted_model = torch.jit.script(model)
    torch.jit.save(scripted_model, output_path)
    scripted_size = get_size_mb(output_path)

    return baseline_size, packed_size, scripted_size


# -------------------------
# ENTRY
# -------------------------

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)

    heavy = "models/heavy_bio_model.pt"
    edge = "models/edge_optimized_model.pt"

    base, packed, script = optimize_model(heavy, edge)

    print("\n--- Edge Optimization Summary ---")
    print(f"FP32 Baseline:        {base:.2f} MB")
    print(f"INT8 Packed Weights:  {packed:.2f} MB")
    print(f"TorchScript Package:  {script:.2f} MB")

    reduction = ((base - packed) / base) * 100
    print(f"\nTotal Reduction:      {reduction:.2f}%")

    print("\nStatus: 🚀 TRUE Compression (INT8 Packed)")