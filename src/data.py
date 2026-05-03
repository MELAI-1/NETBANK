import polars as pl
import pandas as pd
import numpy as np
import datetime
import os
import logging
import gc
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from typing import Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants for Reproducibility
SEED = 42
CUTOFF_DATE = datetime.datetime(2015, 11, 1)

def build_advanced_features(data_dir: str) -> pd.DataFrame:
    """
    Comprehensive Feature Engineering:
    1. Multi-source Merging (Transactions + Demographics + Financials)
    2. Behavioral Clustering (Unsupervised K-Means)
    3. Transaction Velocity & Stability
    """
    logger.info("🚀 Starting Advanced Feature Engineering...")
    
    # 1. Load Parquet Files
    txn = pl.read_parquet(os.path.join(data_dir, "transactions_features/transactions_features.parquet")).with_columns(
        pl.col("TransactionDate").cast(pl.Datetime)
    )
    demo = pl.read_parquet(os.path.join(data_dir, "demographics_clean/demographics_clean.parquet"))
    fin = pl.read_parquet(os.path.join(data_dir, "financials_features/financials_features.parquet")).with_columns(
        pl.col("RunDate").cast(pl.Datetime)
    )

    # 2. Transactional Features (RFM + Velocity)
    logger.info("  [FE] Processing Transactions...")
    
    # 30-day window for velocity
    recent_date = CUTOFF_DATE - datetime.timedelta(days=30)
    
    txn_agg = txn.group_by("UniqueID").agg([
        pl.len().alias("freq_total"),
        pl.col("TransactionAmount").sum().alias("monetary_total"),
        pl.col("TransactionAmount").mean().alias("avg_spend"),
        pl.col("TransactionAmount").std().alias("std_spend"),
        ((pl.lit(CUTOFF_DATE) - pl.col("TransactionDate").max()).dt.total_days()).alias("recency_days"),
        pl.col("AccountID").n_unique().alias("n_accounts"),
        # Velocity: Activity in the last 30 days vs Total
        (pl.col("TransactionDate").filter(pl.col("TransactionDate") >= recent_date).len()).alias("freq_30d"),
        # Credit/Debit Signal (assuming positive is credit, negative is debit)
        (pl.col("TransactionAmount").filter(pl.col("TransactionAmount") > 0).sum()).alias("total_credit"),
        (pl.col("TransactionAmount").filter(pl.col("TransactionAmount") < 0).abs().sum()).alias("total_debit")
    ]).collect().to_pandas()

    # Derived Transactional Features
    txn_agg['velocity_index'] = txn_agg['freq_30d'] / (txn_agg['freq_total'] / 6 + 1) # Normalized by ~6 months of data
    txn_agg['stability_index'] = txn_agg['std_spend'] / (txn_agg['avg_spend'].abs() + 1)
    txn_agg['credit_debit_ratio'] = txn_agg['total_credit'] / (txn_agg['total_debit'] + 1)

    # 3. Financial Snapshot Features
    logger.info("  [FE] Processing Financials...")
    fin_agg = fin.group_by("UniqueID").agg([
        pl.col("NetInterestIncome").mean().alias("avg_nii"),
        pl.col("NetInterestRevenue").sum().alias("total_nir"),
        pl.col("Product").n_unique().alias("product_diversity")
    ]).collect().to_pandas()

    # 4. Merge All Sources
    logger.info("  [FE] Merging Sources...")
    demo_pd = demo.to_pandas()
    df = demo_pd.merge(txn_agg, on="UniqueID", how="left")
    df = df.merge(fin_agg, on="UniqueID", how="left")
    
    # Fill missing values for numericals
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(0)

    # 5. Unsupervised Learning: Behavioral Clustering
    logger.info("  [FE] Running Unsupervised Clustering (K-Means)...")
    cluster_features = ['freq_total', 'monetary_total', 'recency_days', 'avg_spend']
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[cluster_features])
    
    kmeans = KMeans(n_clusters=5, random_state=SEED, n_init=10)
    df['behavioral_cluster'] = kmeans.fit_predict(scaled_data)
    
    # PCA Components (Dimensionality Reduction for noise reduction)
    pca = PCA(n_components=3, random_state=SEED)
    pca_comps = pca.fit_transform(scaled_data)
    for i in range(3):
        df[f'pca_comp_{i+1}'] = pca_comps[:, i]

    return df

def load_and_preprocess(data_dir: str, seed: int = SEED) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, float, float]:
    """
    Main Pipeline Implementation
    """
    # 1. Load Core Tables
    train_base = pd.read_csv(os.path.join(data_dir, "Train.csv"))
    test_base = pd.read_csv(os.path.join(data_dir, "Test.csv"))
    
    # 2. Build Advanced Features
    all_features = build_advanced_features(data_dir)
    
    # 3. Merge with Target
    train_df = train_base.merge(all_features, on="UniqueID", how="left")
    test_df = test_base.merge(all_features, on="UniqueID", how="left")
    
    # 4. Target Transformation: Log1p + Winsorization
    logger.info("  [FE] Applying Target Transformation (Log1p + Winsorization)...")
    y_raw = train_df["next_3m_txn_count"]
    y_log = np.log1p(y_raw)
    
    # Winsorization (P1 to P99)
    cap_low = y_log.quantile(0.01)
    cap_high = y_log.quantile(0.99)
    y_final = y_log.clip(lower=cap_low, upper=cap_high)
    
    # 5. Categorical Encoding
    logger.info("  [FE] Encoding Categorical Features...")
    cat_cols = all_features.select_dtypes(include=['object']).columns.tolist()
    if 'UniqueID' in cat_cols: cat_cols.remove('UniqueID')
    
    # Remove dates from features
    for col in ['BirthDate', 'RunDate', 'TransactionDate']:
        if col in cat_cols: cat_cols.remove(col)
        if col in train_df.columns: train_df.drop(columns=[col], inplace=True)
        if col in test_df.columns: test_df.drop(columns=[col], inplace=True)

    for col in cat_cols:
        le = LabelEncoder()
        # Combine train/test for consistent encoding
        combined = pd.concat([train_df[col].astype(str), test_df[col].astype(str)])
        le.fit(combined)
        train_df[col] = le.transform(train_df[col].astype(str))
        test_df[col] = le.transform(test_df[col].astype(str))

    # 6. Final Feature Selection
    exclude = ["UniqueID", "next_3m_txn_count"]
    features = [c for c in train_df.columns if c not in exclude]
    
    X_train = train_df[features].fillna(0)
    X_test = test_df[features].fillna(0)
    
    logger.info(f"✅ Preprocessing Complete. Features: {len(features)}")
    return X_train, y_final, X_test, test_df["UniqueID"], cap_low, cap_high
