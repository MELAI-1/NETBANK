import os
import pandas as pd
import numpy as np
import argparse
import logging
import torch
from src.data import load_and_preprocess
from src.model import train_and_predict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Nedbank Quad Ensemble (LGBM, XGB, Cat, TabNet)')
    parser.add_argument('--data_path', type=str, default='./data', help='Path to data directory')
    parser.add_argument('--output_path', type=str, default='submission_quad_ensemble.csv', help='Output file')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"🖥️ Using device: {device}")

    # 1. Pipeline
    X, y, X_test, test_ids = load_and_preprocess(args.data_path, seed=args.seed)
    
    # 2. Train Ensemble (Returns a dictionary of all models and blends)
    preds_dict = train_and_predict(X, y, X_test, seed=args.seed)
    
    # 3. Save Multiple Submissions
    logger.info(f"📂 Saving {len(preds_dict)} different submission versions...")
    
    sample_sub_path = os.path.join(args.data_path, 'SampleSubmission .csv')
    if not os.path.exists(sample_sub_path):
        sample_sub_path = os.path.join(args.data_path, 'SampleSubmission.csv')
    
    base_output = args.output_path.replace('.csv', '')
    
    for name, preds in preds_dict.items():
        file_path = f"{base_output}_{name}.csv"
        
        # Prepare submission
        submission = pd.DataFrame({'UniqueID': test_ids, 'next_3m_txn_count': preds})
        
        if os.path.exists(sample_sub_path):
            sample_sub = pd.read_csv(sample_sub_path)
            mapping = dict(zip(submission['UniqueID'], submission['next_3m_txn_count']))
            sample_sub['next_3m_txn_count'] = sample_sub['UniqueID'].map(mapping).fillna(0)
            sample_sub.to_csv(file_path, index=False)
        else:
            submission.to_csv(file_path, index=False)
            
        logger.info(f"💾 Saved: {file_path}")

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
