# ShieldAI: Zero-Day Attack Prevention (Model-Agnostic ML Security System)

## Overview

ShieldAI is a model-agnostic machine learning security framework designed to detect and mitigate zero-day attacks using a hybrid detection architecture.

The system integrates:

- Supervised attack detection
- Unsupervised anomaly detection
- Risk fusion engine
- Victim model protection layer
- Attack simulation framework
- Evaluation and visualization engine
- API + Frontend dashboard

The goal is to protect ML-based systems against previously unseen adversarial behaviors.

---

## System Architecture

The framework consists of three main layers:

### 1. Victim Layer
- Fraud detection model (ML & Neural Network variants)
- Inference pipeline
- Baseline prediction system

### 2. Shield Layer
- Supervised Detector
- Unsupervised Anomaly Detector
- Risk Fusion Engine
- Decision Aggregation Mechanism

### 3. Evaluation & Monitoring Layer
- Attack Simulator
- Metrics Engine
- ROC Curve & Precision-Recall Analysis
- Confusion Matrix Visualization
- Dashboard Interface

---

## Key Components

### Core Modules

- `core/anomaly_detector.py`
- `core/supervised_detector.py`
- `core/risk_fusion.py`
- `core/base_detector.py`

### Training Pipeline

- `training/train_pipeline.py`
- `training/config.json`

### Attack Simulation

- `utils/attack_simulator.py`

### Evaluation Engine

- `evaluation/evaluator.py`

### Victim Models

- `victim_models/fraud_model.py`
- `victim_models/nn_fraud_model.py`
- `victim_pipeline/train_victim_model.py`
- `victim_pipeline/train_victim_nn_model.py`

### API Layer

- `api/api.py`

### Frontend Dashboard

- `frontend/dashboard.html`

---

## Detection Strategy

ShieldAI follows a hybrid strategy:

1. Supervised detection for known attack patterns
2. Unsupervised anomaly detection for zero-day threats
3. Risk score fusion to combine model outputs
4. Threshold-based final security decision

This architecture allows the system to detect:
- Known adversarial patterns
- Distributional shifts
- Unseen attack behaviors
- Model exploitation attempts

---

## Experimental Setup

- Dataset: Credit Card Fraud Dataset
- Multiple attack simulation strategies
- Timestamped model checkpoints
- Performance logged per experiment

Evaluation Metrics:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Log Loss

---

## Results

The system demonstrates:

- Strong anomaly detection capability
- High recall for adversarial samples
- Effective risk aggregation across detectors
- Improved robustness over standalone models

Detailed evaluation outputs are stored in:
backend/evaluation_results/


---

## How To Run

### Install Dependencies
pip install -r backend/requirements.txt
### Train Shield System
python backend/src/training/train_pipeline.py
### Train Victim Model
python backend/src/victim_pipeline/train_victim_model.py
### Run Demo
python backend/demo.py

## Project Structure


SheildAi/
│
├── backend/
│ ├── src/
│ │ ├── api/
│ │ ├── core/
│ │ ├── detectors/
│ │ ├── evaluation/
│ │ ├── training/
│ │ ├── utils/
│ │ ├── victim_models/
│ │ └── victim_pipeline/
│ │
│ ├── data/
│ ├── evaluation_results/
│ ├── frontend/
│ ├── demo.py
│ └── requirements.txt

---

## Research Motivation

Traditional security systems rely on signature-based detection and struggle against zero-day attacks.

ShieldAI explores a model-agnostic ML-based defense layer that:

- Works independently of victim model architecture
- Detects unseen adversarial behavior
- Adapts to distribution shifts
- Provides modular integration capability

## Future Improvements
- Real-time streaming attack detection
- Adversarial training integration
- Cloud deployment (Docker/Kubernetes)
- Auto-threshold calibration
- Model explainability integration (SHAP/LIME)

## Author
Ananya Shah  
Machine Learning & AI Security Engineering
