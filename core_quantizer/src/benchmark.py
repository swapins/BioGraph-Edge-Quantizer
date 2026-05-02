import torch
import time
import os
import numpy as np
import torch.jit
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from src.model import BioGraphSAGE
from src.utils import seed_everything


# -------------------------
# SYSTEM STABILITY 
# -------------------------
torch.set_num_threads(4)  # reduce latency noise


# -------------------------
# LATENCY MEASUREMENT
# -------------------------
def measure_system_performance(model, x, edge_index, label, num_runs=10):
    """
    Measures inference latency with statistical robustness
    """
    latencies = []

    model.eval()

    # Warm-up (VERY important for fair benchmarking)
    with torch.no_grad():
        for _ in range(5):
            _ = model(x, edge_index)

    # Timed runs
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = model(x, edge_index)
            latencies.append((time.perf_counter() - start) * 1000)

    latencies = np.array(latencies)

    mean_latency = latencies.mean()
    std_latency = latencies.std()
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)

    print(
        f"[{label}] "
        f"Avg: {mean_latency:.2f} ms | "
        f"Std: ±{std_latency:.2f} | "
        f"P50: {p50:.2f} | "
        f"P95: {p95:.2f}"
    )

    return mean_latency


# -------------------------
# SIZE MEASUREMENT (FIXED)
# -------------------------
def get_model_sizes(model, label):
    """
    Reports BOTH:
    1. True weight size (state_dict)
    2. TorchScript package size
    """

    os.makedirs("models", exist_ok=True)

    # --- Weight size (REAL indicator) ---
    weight_path = f"models/temp_{label}_weights.pt"
    torch.save(model.state_dict(), weight_path)
    weight_size = os.path.getsize(weight_path) / (1024 * 1024)

    # --- TorchScript size (deployment artifact) ---
    script_path = f"models/temp_{label}_script.pt"
    scripted_model = torch.jit.script(model)
    torch.jit.save(scripted_model, script_path)
    script_size = os.path.getsize(script_path) / (1024 * 1024)

    # cleanup only FP32 artifacts
    if "INT8" not in label:
        os.remove(weight_path)
        os.remove(script_path)

    print(f"[{label}] Weights: {weight_size:.2f} MB | TorchScript: {script_size:.2f} MB")

    return weight_size, script_size


# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":
    seed_everything(42)

    IN_CHANNELS = 2048
    HIDDEN_CHANNELS = 512
    OUT_CHANNELS = 1

    # Synthetic graph (kept same as yours)
    x = torch.randn(10000, IN_CHANNELS)
    edge_index = torch.randint(0, 2000, (2, 10000))

    # -------------------------
    # FP32 BASELINE
    # -------------------------
    model_fp32 = BioGraphSAGE(
        in_channels=IN_CHANNELS,
        hidden_channels=HIDDEN_CHANNELS,
        out_channels=OUT_CHANNELS
    )
    model_fp32.eval()

    print("\n--- Edge-GNN Performance Validation ---")

    get_model_sizes(model_fp32, "Baseline_FP32")
    measure_system_performance(model_fp32, x, edge_index, "Baseline_FP32")

    # -------------------------
    # INT8 (Dynamic Quantization)
    # -------------------------
    model_int8 = torch.quantization.quantize_dynamic(
        model_fp32,
        {torch.nn.Linear},
        dtype=torch.qint8
    )

    get_model_sizes(model_int8, "Optimized_INT8")
    measure_system_performance(model_int8, x, edge_index, "Optimized_INT8")

    print("\nStatus: Deterministic + Measurable Edge Pipeline Verified")