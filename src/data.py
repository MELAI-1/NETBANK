import polars as pl
import pandas as pd
import numpy as np
import datetime
import zipfile
import os
from sklearn.preprocessing import LabelEncoder
import gc
import logging
from typing import Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def unzip_data(data_dir: str):
    """Auto-unzip files for Colab environments."""
    for item in os.listdir(data_dir):
        if item.endswith(".zip"):
            file_name = os.path.join(data_dir, item)
            output_dir = os.path.join(data_dir, item.replace(".zip", ""))
            if not os.path.exists(output_dir):
                logger.info(f"📦 Unzipping {item}...")
                with zipfile.ZipFile(file_name, 'r') as zip_ref:
                    zip_ref.extractall(output_dir)

def create_advanced_features(data_dir: str) -> pl.DataFrame:
    """Creates behavioral features optimized for memory."""
    unzip_data(data_dir)
    
    # Path handling (search for parquet in unzipped folders)
    def find_parquet(folder_name):
        base_path = os.path.join(data_dir, folder_name)
        for root, dirs, files in os.walk(base_path):
            for file in files:
                if file.endswith(".parquet"):
                    return os.path.join(root, file)
        return None

    txn_path = find_parquet("transactions_features")
    if not txn_path: raise FileNotFoundError("Transaction parquet not found!")
    
    txn = pl.scan_parquet(txn_path).select(['UniqueID', 'TransactionDate', 'TransactionAmount', 'AccountID'])
    txn = txn.with_columns(pl.col("TransactionDate").cast(pl.Date))
    
    ref_date = datetime.date(2015, 10, 31)
    
    # 1. RFM Aggregations
    behavior = txn.group_by("UniqueID").agg([
        pl.len().alias("frequency"),
        pl.col("TransactionAmount").sum().alias("monetary"),
        pl.col("TransactionAmount").mean().alias("avg_spend"),
        pl.col("TransactionAmount").std().alias("std_spend"),
        ((pl.lit(ref_date) - pl.col("TransactionDate").max()).dt.total_days()).alias("recency"),
        pl.col("AccountID").n_unique().alias("account_diversity"),
        # Holiday Season behavior (Transactions in Oct 2015 vs total)
        (pl.col("TransactionDate").filter(pl.col("TransactionDate") >= datetime.date(2015, 10, 1)).len()).alias("oct_2015_count"),
        # Average monthly transaction count (historical average)
        (pl.len() / (pl.col("TransactionDate").max() - pl.col("TransactionDate").min()).dt.total_days() * 30).alias("monthly_avg_freq")
    ]).collect()
    
    # 2. Cross-Features (Interactions)
    behavior_df = behavior.to_pandas()
    behavior_df['spend_per_account'] = behavior_df['monetary'] / (behavior_df['account_diversity'] + 1)
    behavior_df['oct_ratio'] = behavior_df['oct_2015_count'] / (behavior_df['frequency'] + 1)
    
    # 3. Handle Infinity and Large Values
    # Replace Inf with NaN then fill with 0
    behavior_df = behavior_df.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Clip extreme values to prevent 'Value too large' errors
    # Using 99.9th percentile clipping for spend and monetary features
    cols_to_clip = ['monetary', 'avg_spend', 'std_spend', 'spend_per_account']
    for col in cols_to_clip:
        if col in behavior_df.columns:
            upper_limit = behavior_df[col].quantile(0.999)
            behavior_df[col] = behavior_df[col].clip(upper=upper_limit)
    
    return pl.from_pandas(behavior_df)

def load_and_preprocess(data_dir: str, seed: int = 42) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Pipeline ready for Colab/Kaggle."""
    unzip_data(data_dir)
    
    train = pl.read_csv(os.path.join(data_dir, "Train.csv"))
    test = pl.read_csv(os.path.join(data_dir, "Test.csv"))
    
    # Behavioral Features
    behavior = create_advanced_features(data_dir)
    
    logger.info("Merging datasets...")
    train_pd = train.join(behavior, on="UniqueID", how="left").fill_null(0).to_pandas()
    test_pd = test.join(behavior, on="UniqueID", how="left").fill_null(0).to_pandas()
    
    # Target
    y = np.log1p(train_pd["next_3m_txn_count"])
    
    # Simple Categorical Encoding
    cat_cols = train_pd.select_dtypes(include=['object']).columns.tolist()
    for col in ["UniqueID", "BirthDate", "RunDate"]:
        if col in cat_cols: cat_cols.remove(col)
        
    for col in cat_cols:
        le = LabelEncoder()
        combined = pd.concat([train_pd[col].astype(str), test_pd[col].astype(str)])
        le.fit(combined)
        train_pd[col] = le.transform(train_pd[col].astype(str))
        test_pd[col] = le.transform(test_pd[col].astype(str))

    features = [c for c in train_pd.columns if c not in ["UniqueID", "next_3m_txn_count", "BirthDate", "RunDate"]]
    
    return train_pd[features], y, test_pd[features], test_pd["UniqueID"]
