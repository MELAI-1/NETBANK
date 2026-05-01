# Nedbank Transaction Volume Forecasting Challenge

This repository contains a structured, reproducible solution for the [Nedbank Transaction Volume Forecasting Challenge](https://zindi.africa/competitions/nedbank-transaction-volume-forecasting-challenge) on Zindi.

## Project Structure

```
nedbank_challenge/
├── main.py              # Main entry point to run the pipeline
├── requirements.txt     # Dependencies
├── README.md            # Instructions
└── src/
    ├── data.py          # Data loading and feature engineering (Polars based)
    └── model.py         # Model training (Stacking Ensemble: LGBM, XGBoost, CatBoost)
```

## Setup

1. Clone this repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Place the challenge data in a `data/` folder (or specify the path using `--data_path`).

## Usage

Run the entire pipeline (data processing + training + inference):

```bash
python main.py --data_path ./data/ --output_path submission.csv
```

## Reproducibility

The code uses a fixed random seed (`42` by default) for all data splitting and model training to ensure consistent results across runs.

## Methodology

- **Data Processing:** Uses `Polars` for fast handling of large transaction datasets (18M+ rows).
- **Feature Engineering:** Includes transaction history statistics, recency, velocity, and demographic features.
- **Model:** A Stacking Ensemble of:
    - LightGBM
    - XGBoost
    - CatBoost
- **Meta-Model:** Ridge Regression combines the predictions of the three base models.
- **Metric Optimization:** Target is log-transformed (`log1p`) to align with the RMSLE metric.
