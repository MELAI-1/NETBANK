import polars as pl
import pandas as pd
import numpy as np
import datetime
import os
import logging
import zipfile
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

def unzip_data(data_dir: str):
    """Auto-unzip files if they haven't been unzipped yet."""
    for item in os.listdir(data_dir):
        if item.endswith(".zip"):
            file_path = os.path.join(data_dir, item)
            # Create a folder name based on the zip file name
            folder_name = item.replace(".zip", "")
            output_dir = os.path.join(data_dir, folder_name)
            
            # Unzip if the folder doesn't exist
            if not os.path.exists(output_dir):
                logger.info(f"📦 Unzipping {item}...")
                try:
                    with zipfile.ZipFile(file_path, 'r') as zip_ref:
                        zip_ref.extractall(data_dir) # Extract to data_dir directly
                except Exception as e:
                    logger.error(f"❌ Error unzipping {item}: {e}")

def find_file(data_dir: str, pattern: str) -> str:
    """Robustly find a file even with spaces or different extensions."""
    # First check direct matches
    full_path = os.path.join(data_dir, pattern)
    if os.path.exists(full_path):
        return full_path
    
    # Check for versions with spaces (e.g., "Train .csv")
    name, ext = os.path.splitext(pattern)
    alt_pattern = f"{name} {ext}"
    alt_path = os.path.join(data_dir, alt_pattern)
    if os.path.exists(alt_path):
        return alt_path
    
    # Recursive search as fallback
    for root, dirs, files in os.walk(data_dir):
        for f in files:
            if pattern.lower() in f.lower().replace(" ", ""):
                return os.path.join(root, f)
    
    raise FileNotFoundError(f"Could not find file matching: {pattern} in {data_dir}")

def build_advanced_features(data_dir: str) -> pd.DataFrame:
    """
    Comprehensive Feature Engineering with Robust Path Handling
    """
    logger.info("🚀 Starting Advanced Feature Engineering...")
    unzip_data(data_dir)
    
    # Robust Parquet Loading
    txn_path = find_file(data_dir, "transactions_features.parquet")
    demo_path = find_file(data_dir, "demographics_clean.parquet")
    fin_path = find_file(data_dir, "financials_features.parquet")

    logger.info(f"  📂 Loading: {os.path.basename(txn_path)}, {os.path.basename(demo_path)}, {os.path.basename(fin_path)}")

    txn = pl.read_parquet(txn_path).with_columns(
        pl.col("TransactionDate").cast(pl.Datetime)
    )
    demo = pl.read_parquet(demo_path)
    fin = pl.read_parquet(fin_path).with_columns(
        pl.col("RunDate").cast(pl.Datetime)
    )

    # 2. Transactional Features (RFM + Velocity)
    logger.info("  [FE] Processing Transactions...")
    recent_date = CUTOFF_DATE - datetime.timedelta(days=30)
    
    txn_agg = txn.group_by("UniqueID").agg([
        pl.len().alias("freq_total"),
        pl.col("TransactionAmount").sum().alias("monetary_total"),
        pl.col("TransactionAmount").mean().alias("avg_spend"),
        pl.col("TransactionAmount").std().alias("std_spend"),
        ((pl.lit(CUTOFF_DATE) - pl.col("TransactionDate").max()).dt.total_days()).alias("recency_days"),
        pl.col("AccountID").n_unique().alias("n_accounts"),
        (pl.col("TransactionDate").filter(pl.col("TransactionDate") >= recent_date).len()).alias("freq_30d"),
        (pl.col("TransactionAmount").filter(pl.col("TransactionAmount") > 0).sum()).alias("total_credit"),
        (pl.col("TransactionAmount").filter(pl.col("TransactionAmount") < 0).abs().sum()).alias("total_debit")
    ]).collect().to_pandas()

    txn_agg['velocity_index'] = txn_agg['freq_30d'] / (txn_agg['freq_total'] / 6 + 1)
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
    
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(0)

    # 5. Unsupervised Learning: Behavioral Clustering
    logger.info("  [FE] Running Unsupervised Clustering (K-Means)...")
    cluster_features = ['freq_total', 'monetary_total', 'recency_days', 'avg_spend']
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[cluster_features])
    
    kmeans = KMeans(n_clusters=5, random_state=SEED, n_init=10)
    df['behavioral_cluster'] = kmeans.fit_predict(scaled_data)
    
    pca = PCA(n_components=3, random_state=SEED)
    pca_comps = pca.fit_transform(scaled_data)
    for i in range(3):
        df[f'pca_comp_{i+1}'] = pca_comps[:, i]

    return df

def load_and_preprocess(data_dir: str, seed: int = SEED) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, float, float]:
    """Main Pipeline Implementation with robust path finding."""
    unzip_data(data_dir)
    
    # Find CSVs (Handling "Train .csv" vs "Train.csv")
    train_path = find_file(data_dir, "Train.csv")
    test_path = find_file(data_dir, "Test.csv")
    
    train_base = pd.read_csv(train_path)
    test_base = pd.read_csv(test_path)
    
    all_features = build_advanced_features(data_dir)
    
    train_df = train_base.merge(all_features, on="UniqueID", how="left")
    test_df = test_base.merge(all_features, on="UniqueID", how="left")
    
    logger.info("  [FE] Applying Target Transformation...")
    y_raw = train_df["next_3m_txn_count"]
    y_log = np.log1p(y_raw)
    
    cap_low = y_log.quantile(0.01)
    cap_high = y_log.quantile(0.99)
    y_final = y_log.clip(lower=cap_low, upper=cap_high)
    
    logger.info("  [FE] Encoding Categorical Features...")
    cat_cols = all_features.select_dtypes(include=['object']).columns.tolist()
    if 'UniqueID' in cat_cols: cat_cols.remove('UniqueID')
    
    for col in ['BirthDate', 'RunDate', 'TransactionDate']:
        if col in cat_cols: cat_cols.remove(col)
        if col in train_df.columns: train_df.drop(columns=[col], inplace=True)
        if col in test_df.columns: test_df.drop(columns=[col], inplace=True)

    for col in cat_cols:
        le = LabelEncoder()
        combined = pd.concat([train_df[col].astype(str), test_df[col].astype(str)])
        le.fit(combined)
        train_df[col] = le.transform(train_df[col].astype(str))
        test_df[col] = le.transform(test_df[col].astype(str))

    features = [c for c in train_df.columns if c not in ["UniqueID", "next_3m_txn_count"]]
    
    X_train = train_df[features].fillna(0)
    X_test = test_df[features].fillna(0)
    
    logger.info(f"✅ Preprocessing Complete. Features: {len(features)}")
    return X_train, y_final, X_test, test_df["UniqueID"], cap_low, cap_high
