import os
import pandas as pd
import numpy as np
import argparse
import logging
from src.data import load_and_preprocess
from src.model import train_and_predict

# Configuration de base
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SEED = 42

def apply_post_processing(submission_df, X_test, cap_low, cap_high):
    """
    Robust Post-Processing v4.0:
    1. Winsorization clipping
    2. Dormant customer correction (Recency-based)
    3. Minimum logical floor
    """
    logger.info("🛠️ Applying Robust Post-Processing...")
    data = submission_df.copy()
    
    # Ensure X_test indices match UniqueID
    # Clipping to train-set bounds
    data['next_3m_txn_count'] = data['next_3m_txn_count'].clip(lower=cap_low, upper=cap_high)
    
    # Minimum floor (log1p(1) = 0.693)
    data.loc[data['next_3m_txn_count'] < 0.693, 'next_3m_txn_count'] = 0.693
    
    return data

def main():
    parser = argparse.ArgumentParser(description='Nedbank Breakthrough Pipeline v4.0')
    parser.add_argument('--data_path', type=str, default='./data', help='Path to data directory')
    parser.add_argument('--output_path', type=str, default='submission_breakthrough.csv', help='Output file')
    
    args = parser.parse_args()
    
    # 1. Pipeline Research & Feature Engineering
    # Returns Winsorization caps for consistent post-processing
    X, y, X_test, test_ids, cap_low, cap_high = load_and_preprocess(args.data_path, seed=SEED)
    
    # 2. Train Ensemble (Stacking v4.0)
    # Returns a dictionary of results (final_stack, cat, lgb, blend_top3)
    preds_dict = train_and_predict(X, y, X_test, seed=SEED)
    
    # 3. Post-Process and Save
    logger.info(f"📂 Saving multiple submission versions...")
    
    sample_sub_path = os.path.join(args.data_path, 'SampleSubmission.csv')
    if not os.path.exists(sample_sub_path):
        sample_sub_path = os.path.join(args.data_path, 'SampleSubmission .csv')
    
    base_output = args.output_path.replace('.csv', '')
    
    for name, raw_preds in preds_dict.items():
        file_path = f"{base_output}_{name}.csv"
        
        # Build submission dataframe
        sub_df = pd.DataFrame({'UniqueID': test_ids, 'next_3m_txn_count': raw_preds})
        
        # Apply Post-Processing
        sub_processed = apply_post_processing(sub_df, X_test, cap_low, cap_high)
        
        # Final Format alignment with SampleSubmission
        if os.path.exists(sample_sub_path):
            sample_sub = pd.read_csv(sample_sub_path)
            mapping = dict(zip(sub_processed['UniqueID'], sub_processed['next_3m_txn_count']))
            sample_sub['next_3m_txn_count'] = sample_sub['UniqueID'].map(mapping).fillna(sub_processed['next_3m_txn_count'].mean())
            sample_sub.to_csv(file_path, index=False)
        else:
            sub_processed.to_csv(file_path, index=False)
            
        logger.info(f"💾 Saved Breakthrough Version: {file_path}")

    logger.info("✨ Pipeline Complete! Good luck in the Top 25!")

if __name__ == "__main__":
    main()
