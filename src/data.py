import polars as pl
import pandas as pd
import numpy as np
import datetime
import os
import logging
import zipfile
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from typing import Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SEED = 42
CUTOFF_DATE = datetime.datetime(2015, 11, 1)

def unzip_data(data_dir: str):
    for item in os.listdir(data_dir):
        if item.endswith(".zip"):
            file_path = os.path.join(data_dir, item)
            folder_name = item.replace(".zip", "")
            output_dir = os.path.join(data_dir, folder_name)
            if not os.path.exists(output_dir):
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
    Kaggle Grandmaster 'Golden Features':
    1. NIR per Transaction (Financial Efficiency)
    2. Balance Stability Index (Coefficient of Variation)
    3. Income Utilization Ratio (Total Debit / Total Credit)
    """
    logger.info("🚀 Building Breakthrough Golden Features...")
    unzip_data(data_dir)
    
    txn_path = find_file(data_dir, "transactions_features.parquet")
    demo_path = find_file(data_dir, "demographics_clean.parquet")
    fin_path = find_file(data_dir, "financials_features.parquet")

    # 1. Transactions - Deep Dive
    txn = pl.scan_parquet(txn_path).with_columns(
        pl.col("TransactionDate").cast(pl.Datetime)
    )
    
    txn_agg = txn.group_by("UniqueID").agg([
        pl.len().alias("freq_total"),
        pl.col("TransactionAmount").sum().alias("monetary_total"),
        pl.col("TransactionAmount").mean().alias("avg_spend"),
        pl.col("TransactionAmount").std().alias("std_spend"),
        # Golden Feature: Balance Volatility (Stability)
        (pl.col("StatementBalance").std() / (pl.col("StatementBalance").mean().abs() + 1)).alias("balance_volatility"),
        ((pl.lit(CUTOFF_DATE) - pl.col("TransactionDate").max()).dt.total_days()).alias("recency_days"),
        (pl.col("TransactionAmount").filter(pl.col("TransactionAmount") > 0).sum()).alias("total_credit"),
        (pl.col("TransactionAmount").filter(pl.col("TransactionAmount") < 0).abs().sum()).alias("total_debit")
    ]).collect().to_pandas()

    # 2. Financials - Profitability Features
    fin = pl.scan_parquet(fin_path)
    fin_agg = fin.group_by("UniqueID").agg([
        pl.col("NetInterestRevenue").sum().alias("total_nir"),
        pl.col("NetInterestIncome").mean().alias("avg_nii")
    ]).collect().to_pandas()

    # 3. Merging & Golden Ratios
    demo = pl.scan_parquet(demo_path).collect().to_pandas()
    df = demo.merge(txn_agg, on="UniqueID", how="left").merge(fin_agg, on="UniqueID", how="left")
    
    # Golden Feature 1: Income Utilization Ratio
    df['income_utilization'] = df['total_debit'] / (df['total_credit'] + 1)
    
    # Golden Feature 2: NIR Efficiency (Revenue per Activity)
    df['nir_per_txn'] = df['total_nir'] / (df['freq_total'] + 1)
    
    # Golden Feature 3: Spending Volatility Index
    df['spending_volatility'] = df['std_spend'] / (df['avg_spend'].abs() + 1)

    # Clean Up
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(0)
    
    # Log-transform skewed features for stability
    skewed_cols = ['monetary_total', 'total_credit', 'total_debit', 'total_nir']
    for col in skewed_cols:
        df[col] = np.log1p(df[col].clip(0))

    return df

def load_and_preprocess(data_dir: str, seed: int = SEED) -> Tuple:
    train_path = find_file(data_dir, "Train.csv")
    test_path = find_file(data_dir, "Test.csv")
    
    train_base = pd.read_csv(train_path)
    test_base = pd.read_csv(test_path)
    
    all_features = build_advanced_features(data_dir)
    
    train_df = train_base.merge(all_features, on="UniqueID", how="left")
    test_df = test_base.merge(all_features, on="UniqueID", how="left")
    
    y = np.log1p(train_df["next_3m_txn_count"])
    
    # Keep it simple: Label Encoding for Categoricals
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
