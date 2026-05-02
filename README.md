# BioGraph-Edge-Quantizer

**Lead Architect:** Swapin Vidya <br>
**ORCID:** [0009-0009-5758-3845](https://orcid.org/0009-0009-5758-3845)<br>
**Email:** [swapin@peachbot.in](mailto:swapin@peachbot.in)

![Version](https://img.shields.io/badge/Version-v1.0--INT8--Quantized-blue)
![Dataset](https://img.shields.io/badge/Dataset-STRING_v12.0-orange)
![Architecture](https://img.shields.io/badge/Architecture-GraphSAGE-red)
![Optimization](https://img.shields.io/badge/Compression-74.99%25-brightgreen)

---
## Overview

BioGraph-Edge-Quantizer is a **resource-aware Graph Neural Network pipeline** designed for:

* edge-constrained inference
* large-scale biological graphs
* reproducible performance evaluation

The system focuses on:

* **bounded-variance inference latency**
* **reduced model footprint via INT8 weight packing**
* **deployable execution using TorchScript**

---

## Problem Definition

This project models protein–protein interaction graphs derived from the
STRING database.

**Task (Current Prototype):**

* Node-level inference (binary classification placeholder)

**Input Characteristics:**

* Node features: 4096-dimensional embeddings
* Graph size: ~10,000 nodes / ~50,000 edges

**Objective:**
Enable **practical inference under CPU-only, edge-constrained environments**.

---

## System Architecture

* **`core_quantizer/`**
  Python-based GNN pipeline using GraphSAGE and PyTorch Geometric

* **`api_gateway/`**
  Laravel-based interface exposing inference through a structured API

---

## ⚙️ Setup & Initialization

### 1. ML Core (Python)

```bash
cd core_quantizer
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install pandas torch torch-geometric scikit-learn numpy

python -m src.data_loader --generate-sample
python -m src.quantizer
python -m src.benchmark
```

---

### 2. API Gateway (Laravel)

```bash
cd api_gateway
composer install
echo "PYTHON_PATH=$(pwd)/../core_quantizer/venv/Scripts/python.exe" >> .env
php artisan migrate
php artisan serve
```

---

## Benchmark Configuration

**Hardware:**
- CPU: Intel Core i5-10210U (4C/8T, 1.60 GHz)
- RAM: 8 GB (7.88 GB usable)
- OS: Windows 11 Home Single Language (Build 22600, x64)
- System: AVITA NS14A8

**Execution Settings:**

* Runs: 100
* Threads: 1 (controlled variance mode)
* Input: full graph

---

## Performance Results

| Metric               | FP32 Baseline | INT8 Packed  | Observation        |
| -------------------- | ------------- | ------------ | ------------------ |
| **Model Weights**    | 64.03 MB      | **16.02 MB** | **~75% reduction** |
| **Avg Latency**      | 323.36 ms     | 313.64 ms    | ~3% improvement    |
| **P95 Latency**      | 334.77 ms     | 333.91 ms    | negligible change  |
| **Std Dev (Jitter)** | ±13.90 ms     | ±14.46 ms    | bounded variance   |

---

## Key Insight

Quantization does **not significantly improve latency** in this pipeline because:

* graph aggregation dominates compute
* high-dimensional feature movement is memory-bound
* Linear layers are not the primary bottleneck

👉 **Conclusion:**
Optimization primarily reduces **storage footprint**, not raw compute time.

---

## Quantization Strategy

This implementation uses **manual INT8 weight packing**:

* Weights converted → `int8`
* Scale factors stored separately
* Dequantization occurs during inference

**Trade-offs:**

* ✔ ~70–75% model size reduction
* ❗ Dequantization overhead
* ❗ Limited latency gain under current architecture

---

## 🔌 System Integration

Current pipeline:

```
Laravel → subprocess → Python → GNN → Response
```

**Measured Overhead:**

* ~10–15 ms per request

**Limitation:**

* Not scalable for high-throughput systems

**Future Direction:**

* Replace subprocess with persistent inference service (FastAPI / gRPC)

---

## Clinical Alignment (Experimental)

The system includes structured output compatible with FHIR-style schemas
to simulate integration into clinical workflows.

**Note:**
This is a research prototype and **not validated for medical use**.

---

## ⚠️ Limitations

* No formal accuracy benchmarking yet
* Quantization does not significantly reduce latency
* TorchScript size does not reflect compression gains
* Subprocess-based execution adds overhead
* No ARM / edge hardware validation yet

---

## Intellectual Property

Indian Patent Application: **202541127477**

---

## Roadmap

* [ ] Accuracy validation (FP32 vs INT8)
* [ ] ARM / edge hardware benchmarking
* [ ] Persistent inference service
* [ ] Sparse GNN optimization
* [ ] ONNX INT8 deployment pipeline

---

## Technical Glossary

| Term         | Description                    |
| ------------ | ------------------------------ |
| GraphSAGE    | Inductive GNN for unseen nodes |
| STRING       | Protein interaction dataset    |
| Quantization | FP32 → INT8 weight conversion  |
| P95 Latency  | 95th percentile latency        |

---
