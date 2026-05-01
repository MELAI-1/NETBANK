import os
import pandas as pd
import numpy as np
import argparse
from src.data import load_and_preprocess
from src.model import train_stacking

def main():
    parser = argparse.ArgumentParser(description='Nedbank Transaction Volume Forecasting')
    parser.add_argument('--data_path', type=str, default='/kaggle/input/datasets/melvinfokam/nedbank/', help='Path to data directory')
    parser.add_argument('--output_path', type=str, default='submission_final.csv', help='Path to output submission file')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Check if data path exists locally (for user's local run)
    data_path = args.data_path
    if not os.path.exists(data_path):
        # Try a local path relative to the script if the kaggle path doesn't exist
        data_path = './data/'
        print(f"⚠️ Warning: {args.data_path} not found. Trying {data_path}")

    # 1. Load and Preprocess
    X, y, X_test, test_ids = load_and_preprocess(data_path, seed=args.seed)

    # 2. Train and Predict
    final_preds_log = train_stacking(X, y, X_test, seed=args.seed)

    # 3. Save Submission
    # Back-transform from log (expm1 is the inverse of log1p)
    # The notebook sometimes submitted log directly or expm1. 
    # Usually Zindi expects the actual value. 
    # Looking at the notebook, in some places it says:
    # "preds_log = model.predict(X_test_final)" 
    # then "submission_log = pd.DataFrame({'UniqueID': test_df['UniqueID'], 'next_3m_txn_count': preds_log_final})"
    # Wait, let me double check the notebook's final submission logic.
    
    # In In[44]: mes_vraies_preds = np.expm1(test_preds)
    # So it uses expm1.
    
    final_preds_real = np.expm1(final_preds_log)
    
    # Ensure no negatives
    final_preds_real = np.maximum(final_preds_real, 0)

    # Create submission file
    submission = pd.DataFrame({
        'UniqueID': test_ids,
        'next_3m_txn_count': final_preds_real
    })

    # If SampleSubmission.csv exists, align with its order
    sample_sub_path = os.path.join(data_path, 'SampleSubmission.csv')
    if os.path.exists(sample_sub_path):
        sample_sub = pd.read_csv(sample_sub_path)
        mapping = dict(zip(submission['UniqueID'], submission['next_3m_txn_count']))
        sample_sub['next_3m_txn_count'] = sample_sub['UniqueID'].map(mapping).fillna(0)
        sample_sub.to_csv(args.output_path, index=False)
    else:
        submission.to_csv(args.output_path, index=False)
    
    print(f"✅ Submission saved to {args.output_path}")

if __name__ == "__main__":
    main()
