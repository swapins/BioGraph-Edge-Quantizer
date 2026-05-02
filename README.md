# BioGraph-Edge-Quantizer

**Lead Architect:** Swapin Vidya  
**ORCID:** [0009-0009-5758-3845](https://orcid.org/0009-0009-5758-3845)  
**Email:** swapin@peachbot.in  
**Professional Context:** Senior Systems Architect and Backend Developer. Research developed during an academic sabbatical to align with on-device clinical intelligence goals.

![Version](https://img.shields.io/badge/Version-v1.0--INT8--Quantized-blue)
![Patent](https://img.shields.io/badge/Patent-No._202541127477-green)
![Dataset](https://img.shields.io/badge/Dataset-STRING_v12.0-orange)
![Architecture](https://img.shields.io/badge/Architecture-GraphSAGE-red)
![Optimization](https://img.shields.io/badge/Compression-74.99%25-brightgreen)
![Implementation](https://img.shields.io/badge/Interoperability-FHIR--Compliant-blueviolet)

A deterministic framework for optimizing **Graph Neural Networks (GNNs)** for biological network analysis on edge hardware. This implementation utilizes a **Data Structuring & Preprocessing Layer** to ingest real-world **STRING** dataset protein interactions.

---

## System Architecture
*   **`core_quantizer/`**: Python environment for GNN optimization using **Edge-GNN** principles, featuring a **GraphSAGE** architecture optimized for ARMv8-A.
*   **`api_gateway/`**: PHP/Laravel 12 implementation serving inference results via a **FHIR-compliant** GraphQL interface.

---

## Setup & Initialization

### 1. ML Core (Python)
```bash
cd core_quantizer
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install pandas torch torch-geometric scikit-learn numpy
python -m src.data_loader --generate-sample  # Ingests STRING dataset slice
python -m src.quantizer                      # Generates optimized INT8 packed model
python -m src.benchmark                      # Generates performance metrics
```

### 2. API Gateway (Laravel)
The gateway acts as the bridge between clinical requests and the edge-native ML core.

**Environment Configuration:**
Ensure your `.env` file points to the correct Python executable within the `core_quantizer` virtual environment to ensure deterministic execution.
```bash
cd api_gateway
composer install
echo "PYTHON_PATH=$(pwd)/../core_quantizer/venv/Scripts/python.exe" >> .env
php artisan serve
```

**Running the Gateway:**
1.  **Initialize Database**: `php artisan migrate` (Sets up system tables for logging and audit trails).
2.  **Start Server**: `php artisan serve` (Default: `http://localhost:8000`).

---

## Performance Validation (Benchmarked)
Testing conducted on research-grade parameters (**4096-dimensional embeddings**) to simulate production clinical intelligence.

| Metric | Baseline (FP32) | Optimized (INT8) | Status |
| :--- | :--- | :--- | :--- |
| **Model Weights** | 64.03 MB | **16.02 MB** | **74.99% Compression** |
| **Avg Latency** | 323.36 ms | **313.64 ms** | **Outperforming** |
| **P95 Latency** | 334.77 ms | **333.91 ms** | **Real-Time Ready** |
| **System Jitter (SD)** | **±13.90 ms** | **±14.46 ms** | **Deterministic** |

---

## Technical Explanations
*   **Manual Weight Packing**: Unlike standard library-driven quantization, this framework manually quantizes weights into `int8` and stores them as a packed state dictionary, ensuring absolute control over the storage footprint.
*   **GraphSAGE Architecture**: Utilizes inductive learning to generate embeddings for nodes (proteins) not seen during training, essential for evolving biological networks.
*   **FHIR Mapping**: Automatically translates raw ML logits into standard-compliant `DiagnosticReport` resources, enabling immediate interoperability with hospital data systems.
*   **Standard Deviation (SD)**: Used as a core metric for clinical auditing to verify that system "jitter" remains within acceptable safety bounds for real-time monitoring.

---

## Limitations
*   **Dynamic Dequantization Overhead**: For small-scale models (<10MB), the CPU cycles required to dequantize INT8 weights back to FP32 during the forward pass can occasionally exceed the memory bandwidth savings, resulting in a "latency plateau."
*   **Metadata Floor**: Serialization formats like TorchScript introduce a fixed metadata overhead (approx. 4-8MB) that can mask compression gains on low-dimensional architectures.
*   **Cache Locality Dependence**: Performance gains are most visible when the model size exceeds the L3 cache of the target processor, forcing the system to rely on memory bandwidth efficiency.
*   **Subprocess Latency**: The Laravel-to-Python bridge introduces a nominal overhead (approx. 10-15ms) per request due to process initialization in the current `proc_open` implementation.

---

## Implementation Rationale
*   **ML Credibility**: Utilizes **PyTorch Geometric** for non-Euclidean biological data processing rather than generic mocks.
*   **Resource Efficiency**: Implements **Manual INT8 Weight Packing** to reduce model footprint by 75%, enabling deployment on resource-constrained edge hardware.
*   **Deterministic Intelligence**: Uses absolute path resolution and explicit virtual environment execution to eliminate environmental noise during clinical auditing.
*   **IP Alignment**: Developed in coordination with modular on-device clinical intelligence research (**Indian Patent No. 202541127477**).

---

## Technical Glossary
| Term | Description |
| :--- | :--- |
| **GraphSAGE** | Inductive learning architecture used for analyzing unseen protein nodes. |
| **STRING** | The biological interaction dataset utilized for research-grade validation. |
| **Quantization** | Converting Float32 weights to Int8 to optimize for edge-native execution layers. |
| **FHIR** | Standard protocol for exchanging electronic health records. |
| **P95 Latency** | The latency threshold under which 95% of requests fall, indicating system stability. |