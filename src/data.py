import polars as pl
import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import LabelEncoder
import gc
import logging
from typing import Tuple, List

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_transaction_features(txn_path: str) -> pl.DataFrame:
    """Creates advanced transaction features using Polars."""
    logger.info("Processing transaction features...")
    
    # Use scan_parquet for memory efficiency
    txn = pl.scan_parquet(txn_path)
    
    # Convert dates and basic cleaning
    txn = txn.with_columns(pl.col("TransactionDate").cast(pl.Date))
    
    # Reference date (end of train period)
    ref_date = datetime.date(2015, 10, 31)
    
    # Basic Aggregations
    agg_features = txn.group_by("UniqueID").agg([
        pl.len().alias("txn_count_total"),
        pl.col("TransactionAmount").sum().alias("txn_amt_total"),
        pl.col("TransactionAmount").mean().alias("txn_amt_avg"),
        pl.col("TransactionAmount").std().alias("txn_amt_std"),
        pl.col("TransactionAmount").max().alias("txn_amt_max"),
        pl.col("TransactionDate").max().alias("last_txn_date"),
        pl.col("TransactionDate").min().alias("first_txn_date"),
        # Number of unique accounts
        pl.col("AccountID").n_unique().alias("num_accounts")
    ])
    
    # Recency and Tenure
    agg_features = agg_features.with_columns([
        ((pl.lit(ref_date) - pl.col("last_txn_date")).dt.total_days()).alias("days_since_last_txn"),
        ((pl.lit(ref_date) - pl.col("first_txn_date")).dt.total_days()).alias("customer_tenure"),
    ])
    
    # Velocity features (last 1 month vs last 3 months)
    last_1m = txn.filter(pl.col("TransactionDate") >= datetime.date(2015, 10, 1)).group_by("UniqueID").agg([
        pl.len().alias("txn_count_1m"),
        pl.col("TransactionAmount").sum().alias("txn_amt_1m")
    ])
    
    last_3m = txn.filter(pl.col("TransactionDate") >= datetime.date(2015, 8, 1)).group_by("UniqueID").agg([
        pl.len().alias("txn_count_3m"),
        pl.col("TransactionAmount").sum().alias("txn_amt_3m")
    ])
    
    # Combine all
    txn_final = agg_features.join(last_1m, on="UniqueID", how="left").join(last_3m, on="UniqueID", how="left")
    
    # Fill nulls for those with no recent transactions
    txn_final = txn_final.fill_null(0)
    
    # Derived features
    txn_final = txn_final.with_columns([
        (pl.col("txn_count_1m") / (pl.col("txn_count_3m") / 3).replace(0, 1)).alias("count_velocity"),
        (pl.col("txn_amt_1m") / (pl.col("txn_amt_3m") / 3).replace(0, 1)).alias("amt_velocity"),
    ])
    
    return txn_final.collect()

def load_and_preprocess(data_dir: str, seed: int = 42) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Main data pipeline with best practices."""
    
    # Define paths
    train_path = f"{data_dir}/Train.csv"
    test_path = f"{data_dir}/Test.csv"
    demo_path = f"{data_dir}/demographics_clean/demographics_clean.parquet"
    txn_path = f"{data_dir}/transactions_features/transactions_features.parquet"
    fin_path = f"{data_dir}/financials_features/financials_features.parquet"

    # 1. Load Transaction Features
    txn_features = create_transaction_features(txn_path)
    
    # 2. Load Demographics
    logger.info("Loading demographics...")
    demo = pl.read_parquet(demo_path)
    
    # 3. Load Financials (simplified aggregation)
    logger.info("Loading financials...")
    fin = pl.read_parquet(fin_path).group_by("UniqueID").agg([
        pl.col("NetInterestIncome").sum().alias("total_net_interest_income"),
        pl.col("NetInterestRevenue").sum().alias("total_net_interest_revenue"),
    ])
    
    # 4. Load Train/Test
    logger.info("Merging datasets...")
    train = pl.read_csv(train_path)
    test = pl.read_csv(test_path)
    
    def merge_all(df):
        return (df.join(demo, on="UniqueID", how="left")
                .join(txn_features, on="UniqueID", how="left")
                .join(fin, on="UniqueID", how="left")
                .fill_null(0))

    train_full = merge_all(train)
    test_full = merge_all(test)
    
    # Convert to pandas for model compatibility
    train_df = train_full.to_pandas()
    test_df = test_full.to_pandas()
    
    # Cleanup to save memory
    del train, test, demo, txn_features, fin, train_full, test_full
    gc.collect()

    # 5. Feature Engineering (Pandas side)
    # Age from BirthDate
    if "BirthDate" in train_df.columns:
        for df in [train_df, test_df]:
            df["BirthDate"] = pd.to_datetime(df["BirthDate"], errors="coerce")
            ref_date = pd.to_datetime("2015-10-31")
            df["Age"] = (ref_date - df["BirthDate"]).dt.days // 365
            df["Age"] = df["Age"].fillna(df["Age"].median())

    # Label Encoding for categorical columns
    cat_cols = train_df.select_dtypes(include=['object']).columns.tolist()
    for col in ["UniqueID", "BirthDate", "RunDate"]:
        if col in cat_cols: cat_cols.remove(col)
        
    logger.info(f"Encoding categorical features: {cat_cols}")
    for col in cat_cols:
        le = LabelEncoder()
        # Handle unknown categories in test
        combined = pd.concat([train_df[col].astype(str), test_df[col].astype(str)])
        le.fit(combined)
        train_df[col] = le.transform(train_df[col].astype(str))
        test_df[col] = le.transform(test_df[col].astype(str))

    # 6. Final Prep
    drop_cols = ["UniqueID", "next_3m_txn_count", "BirthDate", "RunDate", "last_txn_date", "first_txn_date"]
    features = [c for c in train_df.columns if c not in drop_cols]
    
    X = train_df[features]
    y = np.log1p(train_df["next_3m_txn_count"])
    X_test = test_df[features]
    test_ids = test_df["UniqueID"]
    
    logger.info(f"Final feature set size: {len(features)}")
    return X, y, X_test, test_ids
