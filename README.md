# BioGraph-Edge-Quantizer

**Lead Architect:** Swapin Vidya  
**ORCID:** [0009-0009-5758-3845](https://orcid.org/0009-0009-5758-3845)  
**Email:** swapin@peachbot.in  
**Professional Context:** Senior Systems Architect and Backend Developer; research developed during academic sabbatical.

A deterministic framework for optimizing **Graph Neural Networks (GNNs)** for biological network analysis on edge hardware.

## System Architecture
*   **`core_quantizer/`**: Python environment for GNN optimization using **Edge-GNN** principles.
*   **`api_gateway/`**: PHP/Laravel 12 implementation for serving inference results via a **FHIR-compliant** GraphQL interface.

## Setup & Initialization

### 1. ML Core (Python)
```bash
cd core_quantizer
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m src.quantizer  # Generates optimized INT8 model
```

### 2. API Gateway (Laravel)
```bash
cd api_gateway
composer install
touch database/database.sqlite
php artisan migrate      # Initializes system tables (cache/sessions)
php artisan serve
```

## API Testing Example
Once the server is running at `http://localhost:8000`, you can verify the FHIR integration by sending a **POST** request to `http://localhost:8000/graphql` with the following payload:

**Query:**
```graphql
query {
  diagnosticReport(patientId: "SW-1985") {
    id
    status
    category
    subject {
      reference
    }
    aiInferenceScore
    edgeModelVersion
  }
}
```

**Expected Response:**
```json
{
  "data": {
    "diagnosticReport": {
      "id": "fhir-report-...",
      "status": "final",
      "category": "Oncology-PPI-Analysis",
      "subject": { "reference": "Patient/SW-1985" },
      "aiInferenceScore": 0.874,
      "edgeModelVersion": "INT8-Quantized-v1.0"
    }
  }
}
```

## ⚡ Quick Use (Inference Demo)

If you have already initialized the environments, run the full pipeline with these two commands:

**Step 1: Quantize & Validate Model**
```bash
# From root
cd core_quantizer && python -m src.quantizer && python -m pytest
```
**Step 2: Query FHIR API**
```bash
# Start server in one terminal
cd api_gateway && php artisan serve

# Run this curl in another to get the FHIR DiagnosticReport
curl -X POST http://localhost:8000/graphql \
     -H "Content-Type: application/json" \
     -d '{"query": "query { diagnosticReport(patientId: \"SW-1985\") { aiInferenceScore edgeModelVersion } }"}'
```

## Implementation Rationale
*   **Resource Efficiency:** Implements INT8 quantization to bridge the gap between heavy GNN research and edge-native execution.
*   **Interoperability:** Maps raw AI scores to **FHIR** `DiagnosticReport` resources for seamless clinical integration.
*   **IP Alignment:** Developed in coordination with modular on-device clinical intelligence research (Patent No. 202541127477).

---

### Documentation
*   **`CITATION.cff`**: Academic and professional attribution.
*   **`CONTRIBUTING.md`**: Architectural and coding standards for open-source contributors.


