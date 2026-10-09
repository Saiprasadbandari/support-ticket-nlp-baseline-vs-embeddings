# Support Ticket Classification: TF-IDF vs. Transformer Embeddings

A controlled comparison of a classical NLP baseline (TF-IDF + Logistic Regression) against a pretrained transformer feature extractor (`all-MiniLM-L6-v2` sentence embeddings + the same Logistic Regression) on e-commerce customer support complaints.

The question is not only which scores higher, but whether the transformer's accuracy gain justifies its extra compute cost, especially on rare and semantically overlapping categories.

## Dataset

1,334 complaints, 16 `Category` labels (classification target) and 4 `Urgency` levels. Based on [`Ataur77/ecommerce-customer-support`](https://huggingface.co/datasets/Ataur77/ecommerce-customer-support) on Hugging Face. Class sizes range from 20 ("Refunds") to 137 ("Promotions"), a 6.8x imbalance.

## Method

- One shared 80/20 stratified train/test split (`random_state=42`), stored in `data/shared_split/` so both approaches see identical rows.
- Same downstream classifier for both approaches, so the feature representation is the only variable.
- `class_weight='balanced'` evaluated against the default to measure the effect on rare classes.
- Macro F1 as the primary metric; compute cost (extraction time, training time, latency, parameters, memory) tracked alongside.

## Results (balanced Logistic Regression, held-out test set)

| Approach | Accuracy | Macro F1 | Feature extraction | Model size |
|---|---|---|---|---|
| TF-IDF + LogReg | 0.861 | 0.866 | ~0.02 s for 1,334 texts | 1,445-word vocabulary |
| MiniLM embeddings + LogReg | ~0.88 | ~0.89 | ~1.2 s (+ ~3.5 s model load) | 22.7M parameters |

The transformer gives a real but modest gain (about +0.02 macro F1) at a much higher compute cost. Class weighting helped rare categories far more than common ones in both approaches. Exact figures from the last run are in `artifacts/*.json`; the full discussion is in [`docs/PROJECT_DOCUMENTATION.md`](docs/PROJECT_DOCUMENTATION.md).

## Repository layout

```
├── data/            cleaned dataset and canonical train/test split
├── notebooks/       00 end-to-end, 01 EDA, 02 baseline, 03 transformer
├── artifacts/       metrics JSON (model pickles are git-ignored)
├── docs/            full project write-up
└── reports/         final presentation
```

## Reproducing

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cd notebooks
jupyter lab
```

Run notebooks from the `notebooks/` folder (paths are relative to it). Start with `00_end_to_end_combined.ipynb`. The first run downloads the `all-MiniLM-L6-v2` weights.

## Stack

Python, pandas, NumPy, SciPy, scikit-learn, sentence-transformers, matplotlib, seaborn.
