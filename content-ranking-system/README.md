# Content Recommendation Ranking System (Offline + Online)

**Production-style Learning-to-Rank system that orders streaming content candidates using offline training and real-time online scoring.**

---

## Problem it solves
Modern recommender systems are two-stage:
- **Retrieval** finds possible items
- **Ranking** decides what users actually see

This project focuses on **ranking**, optimizing list quality under latency constraints using **Learning-to-Rank (LTR)** rather than simple recommendation scores.

---

## Key results (offline)
| Metric | Validation | Test |
|------|------------|------|
| **NDCG@10** | ~0.6–0.7* | ~0.6–0.7* |
| **MAP@10**  | ~0.4–0.5* | ~0.4–0.5* |

\*Synthetic dataset; fully reproducible via config.

---

## Tech stack
**Python · LightGBM (LambdaMART) · FastAPI · Pandas · NumPy · Pytest · GitHub Actions**

---

## Quickstart (2 minutes)

```bash
# 1. Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Generate data
python -m src.ranking.data.generate_interactions --config configs/ranker.yaml

# 3. Train & evaluate
python -m src.ranking.models.train_ltr --config configs/ranker.yaml

# 4. Run API
uvicorn src.ranking.api.app:app --reload --port 8000
```
---
## Architecture
OFFLINE
logs → negative sampling → features → LTR training → evaluation → model registry

ONLINE
request(user + context + candidates)
        ↓
feature lookup/build → score → rank → response

## Repository Structure

```text
configs/     → pipeline & model parameters
src/         → modular data, features, models, inference, API
artifacts/   → trained models & evaluation reports
tests/       → unit tests
.github/     → CI workflows (GitHub Actions)
```

## Metrics & Evaluation

Ranking quality is evaluated using **listwise metrics**, computed per `(user_id, session_id)` group.

- **Metrics**
  - **NDCG@K (K = 10):** Measures ranking quality with higher weight on correctly ordered top results
  - **MAP@K (K = 10):** Captures precision across ranked positions for relevant items

- **Data Splits**
  - **Time-based split** (train → validation → test)
  - Ensures the model never trains on future user behavior

- **Negative Sampling**
  - Fixed **K negatives per positive interaction**
  - Simulates real-world ranking lists where relevant items compete against many irrelevant ones

- **Leakage Prevention**
  - No future interactions or aggregates used during training
  - Rolling user/item features computed only from past data
  - Validation and test sets strictly follow temporal order

**This project demonstrates a production-aligned approach to content ranking, combining offline Learning-to-Rank training with online inference and industry-standard evaluation.**
