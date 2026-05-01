import os
import pandas as pd
import numpy as np
import argparse
import logging
from src.data import load_and_preprocess
from src.model import train_and_predict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Nedbank Challenge Optimized Pipeline')
    parser.add_argument('--data_path', type=str, default='./data', help='Path to data directory')
    parser.add_argument('--output_path', type=str, default='submission_optimized.csv', help='Output file')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Kaggle Compatibility: If default path doesn't exist, check kaggle input
    data_path = args.data_path
    if not os.path.exists(data_path):
        kaggle_path = "/kaggle/input/nedbank-transaction-volume-forecasting-challenge" # Update with actual kaggle slug
        if os.path.exists(kaggle_path):
            data_path = kaggle_path
            logger.info(f"Using Kaggle path: {data_path}")

    # 1. Pipeline
    X, y, X_test, test_ids = load_and_preprocess(data_path, seed=args.seed)
    
    # 2. Train Ensemble
    final_preds_log = train_and_predict(X, y, X_test, seed=args.seed)
    
    # 3. Transform and Save
    final_preds_real = np.expm1(final_preds_log)
    final_preds_real = np.maximum(final_preds_real, 0)
    
    submission = pd.DataFrame({'UniqueID': test_ids, 'next_3m_txn_count': final_preds_real})
    
    # Align with SampleSubmission
    sample_sub_path = os.path.join(data_path, 'SampleSubmission .csv')
    if not os.path.exists(sample_sub_path):
        sample_sub_path = os.path.join(data_path, 'SampleSubmission.csv')
        
    if os.path.exists(sample_sub_path):
        sample_sub = pd.read_csv(sample_sub_path)
        mapping = dict(zip(submission['UniqueID'], submission['next_3m_txn_count']))
        sample_sub['next_3m_txn_count'] = sample_sub['UniqueID'].map(mapping).fillna(0)
        sample_sub.to_csv(args.output_path, index=False)
        logger.info(f"✅ Submission saved and aligned: {args.output_path}")
    else:
        submission.to_csv(args.output_path, index=False)
        logger.info(f"✅ Raw submission saved: {args.output_path}")

if __name__ == "__main__":
    main()
