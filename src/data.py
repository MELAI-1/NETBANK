import polars as pl
import pandas as pd
import numpy as np
import datetime
import os
import logging
import zipfile
from sklearn.preprocessing import LabelEncoder
from typing import Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SEED = 42
CUTOFF_DATE = datetime.datetime(2015, 11, 1)

def unzip_data(data_dir: str):
    for item in os.listdir(data_dir):
        if item.endswith(".zip"):
            file_path = os.path.join(data_dir, item)
            if not os.path.exists(os.path.join(data_dir, item.replace(".zip", ""))):
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(data_dir)

def find_file(data_dir: str, pattern: str) -> str:
    full_path = os.path.join(data_dir, pattern)
    if os.path.exists(full_path): return full_path
    name, ext = os.path.splitext(pattern)
    alt_path = os.path.join(data_dir, f"{name} {ext}")
    if os.path.exists(alt_path): return alt_path
    for root, _, files in os.walk(data_dir):
        for f in files:
            if pattern.lower() in f.lower().replace(" ", ""):
                return os.path.join(root, f)
    raise FileNotFoundError(f"Could not find {pattern}")

def build_advanced_features(data_dir: str) -> pd.DataFrame:
    """
    Robust Temporal & Behavioral Features:
    - Activity Velocity (Recent vs. Historical)
    - Dormancy Status
    - NIR/Income Stability
    """
    logger.info("🚀 Building Temporal Tweedie Features...")
    unzip_data(data_dir)
    
    txn_path = find_file(data_dir, "transactions_features.parquet")
    demo_path = find_file(data_dir, "demographics_clean.parquet")
    fin_path = find_file(data_dir, "financials_features.parquet")

    # 1. Transactions - Temporal Analysis
    txn = pl.scan_parquet(txn_path).with_columns(
        pl.col("TransactionDate").cast(pl.Date)
    )
    
    # Define time windows
    recent_start = datetime.date(2015, 10, 1)
    mid_start = datetime.date(2015, 8, 1)
    
    txn_agg = txn.group_by("UniqueID").agg([
        pl.len().alias("freq_total"),
        pl.col("TransactionAmount").sum().alias("monetary_total"),
        pl.col("TransactionAmount").mean().alias("avg_spend"),
        # Temporal Slices
        (pl.col("TransactionDate").filter(pl.col("TransactionDate") >= recent_start).len()).alias("freq_last_month"),
        (pl.col("TransactionDate").filter((pl.col("TransactionDate") >= mid_start) & (pl.col("TransactionDate") < recent_start)).len()).alias("freq_mid_term"),
        # Recency & Diversity
        ((pl.lit(CUTOFF_DATE.date()) - pl.col("TransactionDate").max()).dt.total_days()).alias("recency_days"),
        pl.col("AccountID").n_unique().alias("n_accounts"),
        # Credit/Debit Ratio
        (pl.col("TransactionAmount").filter(pl.col("TransactionAmount") > 0).sum()).alias("total_credit"),
        (pl.col("TransactionAmount").filter(pl.col("TransactionAmount") < 0).abs().sum()).alias("total_debit")
    ]).collect().to_pandas()

    # Feature Engineering: Temporal Velocity
    txn_agg['activity_velocity'] = txn_agg['freq_last_month'] / (txn_agg['freq_mid_term'] / 2 + 1)
    txn_agg['is_dormant_recent'] = (txn_agg['freq_last_month'] == 0).astype(int)
    txn_agg['credit_debit_ratio'] = txn_agg['total_credit'] / (txn_agg['total_debit'] + 1)

    # 2. Financials
    fin_agg = pl.scan_parquet(fin_path).group_by("UniqueID").agg([
        pl.col("NetInterestRevenue").sum().alias("total_nir"),
        pl.col("NetInterestIncome").mean().alias("avg_nii")
    ]).collect().to_pandas()

    # 3. Merge & Demographics
    demo = pl.scan_parquet(demo_path).collect().to_pandas()
    df = demo.merge(txn_agg, on="UniqueID", how="left").merge(fin_agg, on="UniqueID", how="left")
    
    # Final Clean Up
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(0)
    
    # Clip extreme monetary values
    for col in ['monetary_total', 'total_credit', 'total_debit', 'total_nir']:
        upper = df[col].quantile(0.99)
        df[col] = df[col].clip(upper=upper)

    return df

def load_and_preprocess(data_dir: str, seed: int = SEED) -> Tuple:
    train_path = find_file(data_dir, "Train.csv")
    test_path = find_file(data_dir, "Test.csv")
    
    train_base = pd.read_csv(train_path)
    test_base = pd.read_csv(test_path)
    
    all_features = build_advanced_features(data_dir)
    
    train_df = train_base.merge(all_features, on="UniqueID", how="left")
    test_df = test_base.merge(all_features, on="UniqueID", how="left")
    
    # Target transformation (RMSLE focus)
    y = np.log1p(train_df["next_3m_txn_count"])
    
    cat_cols = all_features.select_dtypes(include=['object']).columns.tolist()
    if 'UniqueID' in cat_cols: cat_cols.remove('UniqueID')
    
    for col in cat_cols:
        le = LabelEncoder()
        combined = pd.concat([train_df[col].astype(str), test_df[col].astype(str)])
        le.fit(combined)
        train_df[col] = le.transform(train_df[col].astype(str))
        test_df[col] = le.transform(test_df[col].astype(str))

    features = [c for c in train_df.columns if c not in ["UniqueID", "next_3m_txn_count", "BirthDate", "RunDate", "TransactionDate"]]
    
    return train_df[features].fillna(0), y, test_df[features].fillna(0), test_df["UniqueID"]
