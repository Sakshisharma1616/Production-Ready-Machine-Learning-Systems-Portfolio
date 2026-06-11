# Explainable & Trustworthy Recommendation System
Hybrid recommender that ranks content and explains *why* items are recommended using feature attribution and counterfactual analysis.

---

## Problem it solves  
Traditional recommender systems optimize ranking quality but lack transparency and trust.  
This project adds **explainability and robustness checks** to answer:
- Why was this item recommended?
- Which past interaction influenced it most?
- What would change if that interaction never happened?

## Key results (offline)
- Precision@10: ~0.25–0.35*
- AUC: ~0.80–0.90*
- Catalog coverage: ~60–70%
- Stable recommendations under perturbation  
*MovieLens 1M; fully reproducible*

## Tech stack  
Python · LightFM · NumPy · Pandas · SciPy · Scikit-learn

## Quickstart (2 minutes)
```bash
pip install -r requirements.txt
python -m src.run_all
```

## Repository structure (visual)

```
explainability-trust-recs/
├── README.md
├── requirements.txt
├── data/                  # auto-created; do not commit raw data
├── models/                # saved models (checkpoints, serialized artifacts)
├── reports/               # explanations & trust metrics outputs (CSV/JSON/plots)
└── src/
    ├── config.py          # default configuration / hyperparameters
    ├── download_data.py   # script to download MovieLens or other datasets
    ├── make_dataset.py    # process raw data → implicit interactions + feature matrices
    ├── train.py           # train LightFM hybrid model and save artifacts to models/
    ├── recommend.py       # generate recommendations from a saved model
    ├── explain.py         # compute feature attributions and counterfactuals
    ├── trust_metrics.py   # compute coverage / novelty / diversity / stability
    └── run_all.py         # convenience script to run the full pipeline end-to-end
```

## Architecture

**OFFLINE**  
interactions → hybrid training → evaluation → explainability → trust metrics  

**ONLINE (conceptual)**  
user → score → rank → explain → respond  

---

## Explainability & Trust

- **Feature attribution**: identifies top contributing genres per recommendation  
- **Counterfactual explanations**:  
  *“If the user hadn’t liked X, the model would recommend Y”*  
- **Bias & robustness checks**:
  - Novelty (popularity bias)
  - Diversity
  - Catalog coverage
  - Stability under perturbation

---

## Outputs

- `reports/metrics.json`
- `reports/feature_explanations.json`
- `reports/counterfactual_explanations.json`
- `reports/trust_report.json`

---

This project demonstrates production-aligned recommender system design with a focus on interpretability, trust, and robustness, not just accuracy.




