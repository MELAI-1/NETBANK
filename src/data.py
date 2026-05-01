import polars as pl
import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import LabelEncoder
import gc
import logging
from typing import Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_advanced_features(data_dir: str) -> pl.DataFrame:
    """Creates high-signal behavioral features using Polars."""
    logger.info("🛠️ Extracting advanced behavioral features...")
    
    txn_path = f"{data_dir}/transactions_features/transactions_features.parquet"
    txn = pl.scan_parquet(txn_path)
    txn = txn.with_columns(pl.col("TransactionDate").cast(pl.Date))
    
    ref_date = datetime.date(2015, 10, 31)
    
    # 1. RFM & Basic Aggs
    rfm = txn.group_by("UniqueID").agg([
        pl.len().alias("frequency"),
        pl.col("TransactionAmount").sum().alias("monetary"),
        pl.col("TransactionAmount").mean().alias("avg_spend"),
        pl.col("TransactionAmount").std().alias("std_spend"),
        ((pl.lit(ref_date) - pl.col("TransactionDate").max()).dt.total_days()).alias("recency"),
        pl.col("AccountID").n_unique().alias("account_diversity"),
        # Weekend behavior
        pl.col("TransactionDate").dt.weekday().is_in([6, 7]).sum().alias("weekend_txn_count")
    ])
    
    # 2. Diversity Score (Account concentration)
    rfm = rfm.with_columns([
        (pl.col("frequency") / pl.col("account_diversity")).alias("txn_per_account"),
        (pl.col("weekend_txn_count") / pl.col("frequency")).alias("weekend_ratio")
    ])
    
    # 3. Monthly Trends (Last 3 months)
    m1 = txn.filter(pl.col("TransactionDate") >= datetime.date(2015, 10, 1)).group_by("UniqueID").agg(pl.len().alias("m1_count"))
    m2 = txn.filter((pl.col("TransactionDate") >= datetime.date(2015, 9, 1)) & (pl.col("TransactionDate") < datetime.date(2015, 10, 1))).group_by("UniqueID").agg(pl.len().alias("m2_count"))
    m3 = txn.filter((pl.col("TransactionDate") >= datetime.date(2015, 8, 1)) & (pl.col("TransactionDate") < datetime.date(2015, 9, 1))).group_by("UniqueID").agg(pl.len().alias("m3_count"))
    
    trends = rfm.join(m1, on="UniqueID", how="left").join(m2, on="UniqueID", how="left").join(m3, on="UniqueID", how="left").fill_null(0)
    
    # 4. Momentum (Is activity increasing or decreasing?)
    trends = trends.with_columns([
        (pl.col("m1_count") / (pl.col("m2_count") + 1)).alias("momentum_1m"),
        (pl.col("m1_count") / ((pl.col("m1_count") + pl.col("m2_count") + pl.col("m3_count")) / 3 + 1)).alias("velocity_3m")
    ])
    
    return trends.collect()

def load_and_preprocess(data_dir: str, seed: int = 42) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Robust pipeline with best practices for Zindi/Kaggle."""
    
    train = pl.read_csv(f"{data_dir}/Train.csv")
    test = pl.read_csv(f"{data_dir}/Test.csv")
    demo = pl.read_parquet(f"{data_dir}/demographics_clean/demographics_clean.parquet")
    fin = pl.read_parquet(f"{data_dir}/financials_features/financials_features.parquet").group_by("UniqueID").agg([
        pl.col("NetInterestIncome").sum().alias("fin_income"),
        pl.col("NetInterestRevenue").sum().alias("fin_revenue")
    ])
    
    # Advanced features
    behavior = create_advanced_features(data_dir)
    
    logger.info("Merging and cleaning...")
    def pipeline(df):
        return (df.join(demo, on="UniqueID", how="left")
                .join(behavior, on="UniqueID", how="left")
                .join(fin, on="UniqueID", how="left")
                .fill_null(0))

    train_pd = pipeline(train).to_pandas()
    test_pd = pipeline(test).to_pandas()
    
    # Target transformation
    y = np.log1p(train_pd["next_3m_txn_count"])
    
    # Date handling
    for df in [train_pd, test_pd]:
        if "BirthDate" in df.columns:
            df["BirthDate"] = pd.to_datetime(df["BirthDate"], errors="coerce")
            df["Age"] = (pd.to_datetime("2015-10-31") - df["BirthDate"]).dt.days // 365
            df["Age"] = df["Age"].fillna(df["Age"].median())
            
    # Categorical Encoding
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
