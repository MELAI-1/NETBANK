import os
import pandas as pd
import numpy as np
import argparse
import logging
from src.data import load_and_preprocess
from src.model import train_and_predict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SEED = 42

def main():
    parser = argparse.ArgumentParser(description='Nedbank Kaggle Grandmaster MoE Pipeline')
    parser.add_argument('--data_path', type=str, default='./data', help='Path to data directory')
    parser.add_argument('--output_path', type=str, default='submission_moe.csv', help='Output file')
    
    args = parser.parse_args()
    
    # 1. Pipeline Research & Feature Engineering
    X, y, X_test, test_ids = load_and_preprocess(args.data_path, seed=SEED)
    
    # 2. Train Mixture of Experts
    preds_dict = train_and_predict(X, y, X_test, seed=SEED)
    
    # 3. Save Expert Results
    sample_sub_path = os.path.join(args.data_path, 'SampleSubmission.csv')
    if not os.path.exists(sample_sub_path):
        sample_sub_path = os.path.join(args.data_path, 'SampleSubmission .csv')
    
    base_output = args.output_path.replace('.csv', '')
    
    for name, raw_preds in preds_dict.items():
        if name == 'gater_confidence': continue # skip confidence scores
        
        file_path = f"{base_output}_{name}.csv"
        
        # Format alignment
        if os.path.exists(sample_sub_path):
            sample_sub = pd.read_csv(sample_sub_path)
            mapping = dict(zip(test_ids, raw_preds))
            sample_sub['next_3m_txn_count'] = sample_sub['UniqueID'].map(mapping).fillna(np.mean(raw_preds))
            # Clip at minimum 1 transaction in raw space
            sample_sub['next_3m_txn_count'] = sample_sub['next_3m_txn_count'].clip(lower=0)
            sample_sub.to_csv(file_path, index=False)
        else:
            sub = pd.DataFrame({'UniqueID': test_ids, 'next_3m_txn_count': raw_preds})
            sub.to_csv(file_path, index=False)
            
        logger.info(f"💾 Saved MoE Expert: {file_path}")

if __name__ == "__main__":
    main()
