# Bias & Fairness Auditing for ML Decision Systems

**Production-style pipeline to detect, quantify, and mitigate bias in ML models using industry-standard fairness metrics and reproducible artifacts.**

---

## 🔍 What problem this solves
Modern ML systems optimize accuracy but can unintentionally produce **systematic disparities** across user groups.  
This project adds a **fairness layer** to the ML lifecycle:

- Audits model outcomes across sensitive attributes (e.g., sex, race)
- Quantifies bias using standard fairness metrics
- Applies mitigation while tracking accuracy–fairness trade-offs

**Scope**: decision models used in personalization, eligibility, and ranking pipelines  
**Focus**: correctness, transparency, and reproducibility (not just accuracy)

---

## 📊 Key results (example)
- Accuracy / F1 preserved within **<2–3%** after mitigation  
- **Demographic Parity Difference ↓** significantly post-mitigation  
- **Equalized Odds gap ↓** with explicit trade-off reporting  

(Exact values logged in `reports/eval_baseline.json` and `reports/mitigation_report.json`)

---

## 🛠 Tech stack
- **Python**, scikit-learn  
- **Fairlearn** (fairness metrics)  
- pandas, numpy, matplotlib  
- joblib (model artifacts)  

---

## ⚡ Can I run this in 2 minutes?

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python src/download_data.py
python src/make_dataset.py
python src/train.py
python src/evaluate.py
python src/mitigate.py --sensitive sex
```
---

## Architecture

The system is designed with a clear **offline training** and **online decisioning** separation, mirroring production ML systems.

```text
OFFLINE (Training & Evaluation)
┌────────┐   ┌────────────┐   ┌────────┐   ┌──────────┐   ┌─────────────┐
│  Logs  │ → │ Preprocess │ → │ Train  │ → │ Evaluate │ → │ Model Registry│
└────────┘   └────────────┘   └────────┘   └──────────┘   └─────────────┘
                                                        ↓
                                            Fairness Metrics & Thresholds
                                                        ↓
ONLINE (Inference & Decisioning)
┌─────────┐   ┌──────────────┐   ┌────────┐   ┌──────────┐
│ Request │ → │ Feature Build│ → │ Score  │ → │ Decision │
└─────────┘   └──────────────┘   └────────┘   └──────────┘
                         ↑
               Fairness-aware Thresholds

```
---
## Metrics & Evaluation

### Performance metrics
- **Accuracy**
- **F1 score**

### Fairness metrics (Fairlearn)
- **Demographic Parity Difference**
- **Demographic Parity Ratio**
- **Equalized Odds Difference**
- **Selection Rate** (per group)
- **True Positive Rate (TPR)** and **False Positive Rate (FPR)** (per group)

### Evaluation methodology
- Stratified train/test split to preserve class balance
- Sensitive attributes **excluded from training** and used **only for auditing**
- Group-wise metrics reported alongside global performance
- All metrics saved as versioned JSON artifacts for reproducibility

### Leakage prevention
- No sensitive attributes used as predictive features
- No post-test tuning of thresholds
- Evaluation performed strictly on held-out data

---

## Project Structure

```text
netflix-bias-fairness-ml/
├── README.md                 # project documentation
├── requirements.txt          # python dependencies
├── .gitignore
├── data/                     # auto-created; do not commit raw data
├── models/                   # saved model artifacts
├── reports/                  # metrics (JSON) + plots
└── src/
    ├── config.py             # configuration & hyperparameters
    ├── utils.py              # shared utility functions
    ├── download_data.py      # dataset download (OpenML)
    ├── make_dataset.py       # preprocessing & train/test split
    ├── train.py              # baseline model training
    ├── evaluate.py           # performance + fairness evaluation
    └── mitigate.py           # bias mitigation strategy



