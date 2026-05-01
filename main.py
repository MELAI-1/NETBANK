import os
import pandas as pd
import numpy as np
import argparse
import logging
from src.data import load_and_preprocess
from src.model import train_and_predict

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Nedbank Transaction Volume Forecasting')
    parser.add_argument('--data_path', type=str, default='./data', help='Path to data directory')
    parser.add_argument('--output_path', type=str, default='submission_final.csv', help='Path to output submission file')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    data_path = args.data_path
    if not os.path.exists(data_path):
        logger.error(f"Data path {data_path} not found!")
        return

    # 1. Load and Preprocess
    logger.info("--- Stage 1: Data Preparation ---")
    X, y, X_test, test_ids = load_and_preprocess(data_path, seed=args.seed)

    # 2. Train and Predict
    logger.info("--- Stage 2: Model Training & Inference ---")
    final_preds_log = train_and_predict(X, y, X_test, seed=args.seed)

    # 3. Save Submission
    logger.info("--- Stage 3: Submission Generation ---")
    final_preds_real = np.expm1(final_preds_log)
    final_preds_real = np.maximum(final_preds_real, 0)

    submission = pd.DataFrame({
        'UniqueID': test_ids,
        'next_3m_txn_count': final_preds_real
    })

    # Align with SampleSubmission.csv
    sample_sub_path = os.path.join(data_path, 'SampleSubmission .csv')
    if not os.path.exists(sample_sub_path):
         # Try without space if needed
         sample_sub_path = os.path.join(data_path, 'SampleSubmission.csv')

    if os.path.exists(sample_sub_path):
        logger.info(f"Aligning with {sample_sub_path}...")
        sample_sub = pd.read_csv(sample_sub_path)
        mapping = dict(zip(submission['UniqueID'], submission['next_3m_txn_count']))
        sample_sub['next_3m_txn_count'] = sample_sub['UniqueID'].map(mapping).fillna(0)
        sample_sub.to_csv(args.output_path, index=False)
    else:
        logger.warning("SampleSubmission.csv not found. Saving raw predictions.")
        submission.to_csv(args.output_path, index=False)
    
    logger.info(f"✅ Success! Submission saved to {args.output_path}")

if __name__ == "__main__":
    main()
