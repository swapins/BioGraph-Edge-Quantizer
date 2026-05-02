# Contributing to BioGraph-Edge-Quantizer

Thank you for your interest in contributing to this project. We welcome contributions that improve model quantization efficiency, clinical data interoperability, or edge deployment strategies.

## Architectural Standards
This project maintains a strict separation between the **ML Core** and the **API Gateway**. 
*   **Core (Python):** Must follow the `src/` layout and include `pytest` suites for any new optimization logic.
*   **Gateway (Laravel):** Must adhere to PSR-12 coding standards and maintain FHIR-compliant data structures.

## How to Contribute

### 1. Reporting Bugs
*   Ensure the bug is reproducible in the current `venv` or `composer` environment.
*   Open an issue with a clear description and steps to reproduce.

### 2. Feature Requests
*   Features should align with the goal of **deterministic, edge-native intelligence**.
*   Please open an issue to discuss major architectural changes before starting work.

### 3. Pull Requests
1.  Fork the repository and create your branch from `main`.
2.  **ML Changes:** Ensure `python -m pytest` passes with 100% success.
3.  **API Changes:** Ensure `php artisan test` (if applicable) and GraphQL resolvers are validated.
4.  Update the `README.md` if your change introduces new dependencies or configuration requirements.
5.  Maintain a clean commit history.

## ⚖️ Code of Conduct
Contributors are expected to maintain professional and technical excellence. We prioritize clarity, efficiency, and data sovereignty in all technical discussions.

## License
By contributing, you agree that your contributions will be licensed under the project's MIT License.