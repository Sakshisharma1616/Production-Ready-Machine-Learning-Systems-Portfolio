# User Engagement Prediction

Production-style machine learning system that predicts user engagement probability from historical interaction data, designed as a reusable signal for ranking, personalization, and decision systems.

---

## Dataset
MovieLens Latest Small from GroupLens.

## Problem it Solves

User engagement signals (clicks, views, interactions) determine:
- What content is ranked higher
- Which users receive interventions
- How systems optimize retention and growth

This project focuses on **predicting engagement as a calibrated probability**, not a heuristic score, enabling consistent downstream usage in ranking, filtering, and optimization pipelines.

## Approach
- Time-aware split: last interaction per user is held out for testing
- Feature engineering:
  - User historical behavior (count, mean rating, engagement rate)
  - Movie historical popularity/quality (count, mean rating, engagement rate)
  - Temporal features (hour, day-of-week)
  - Genre multi-hot features
- Model: Logistic Regression pipeline (scaler + classifier)
- Explainability:
  - Permutation feature importance
  - SHAP summary plot
---

## Architecture

### OFFLINE (Training & Evaluation)

raw interaction logs  
→ dataset construction (time-aware)  
→ feature engineering (behavioral, temporal, aggregate)  
→ model training  
→ offline evaluation  
→ explainability artifacts  

The offline pipeline mirrors production ML workflows and enforces strict separation between data preparation, modeling, and evaluation.

---

### ONLINE (Future Extension)

request (user + context)  
→ feature lookup / build  
→ engagement probability scoring  
→ downstream ranking / personalization systems  

The engagement score is designed to act as an **input signal**, not a final decision, allowing flexible use across multiple systems.

---

## Metrics & Evaluation

### Metrics

| Metric   | Why it matters in this system |
|--------|--------------------------------|
| ROC-AUC | Validates that engaged users are consistently scored higher than non-engaged users, making the output suitable as a ranking or prioritization signal |
| Accuracy | Evaluates decision quality at a fixed threshold when the score is used for binary actions (e.g., trigger, filter, alert) |
| RMSE | Measures calibration error of engagement probabilities, ensuring scores are numerically meaningful for weighting and optimization |

Metrics are selected based on **how engagement scores are consumed**, not just model correctness.

---

## Key results (offline)

Metric        Validation        Test
ROC-AUC      ~0.75–0.85*        ~0.75–0.85*
Accuracy     ~0.70–0.80*        ~0.70–0.80*
RMSE         ~0.35–0.45*        ~0.35–0.45*

*Synthetic / sample dataset; fully reproducible via configuration.

---

## Tech stack

Python · Pandas · NumPy · Scikit-learn · SHAP · Pytest

---

## Quickstart (2 minutes)

```bash
# 1. Setup environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Download / load data
python src/download_data.py

# 3. Build dataset
python src/make_dataset.py

# 4. Train model
python src/train.py

# 5. Evaluate performance
python src/evaluate.py

# 6. Explain predictions
python src/explain.py
```

## Repository Structure

```text
User-Engagement-Prediction/
├── src/
│   ├── config.py           # global configuration & reproducibility
│   ├── download_data.py    # data ingestion
│   ├── make_dataset.py     # dataset creation & time-aware splits
│   ├── features.py         # feature engineering
│   ├── train.py            # model training
│   ├── evaluate.py         # offline metrics & reports
│   └── explain.py          # explainability (SHAP / feature importance)
├── requirements.txt        # dependencies
├── .gitignore              # ignore rules
└── README.md               # documentation
```

This project demonstrates a production-aligned approach to user engagement modeling, emphasizing reproducibility, leakage-safe evaluation, calibrated predictions, and explainability which are core requirements for real-world machine learning systems.


