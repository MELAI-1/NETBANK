import polars as pl
import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import LabelEncoder
import gc

def load_and_preprocess(path, seed=42):
    print("🚀 Loading data...")
    train_pl = pl.read_csv(path + 'Train.csv')
    test_pl = pl.read_csv(path + 'Test.csv')
    demo = pl.read_parquet(path + 'demographics_clean.parquet')
    txn = pl.read_parquet(path + 'transactions_features.parquet').select(['UniqueID', 'TransactionDate', 'TransactionAmount'])

    # Date cleaning
    txn = txn.with_columns(pl.col("TransactionDate").cast(pl.Date))
    last_date = datetime.date(2015, 10, 31)

    print("Etching transaction features...")
    txn_features = txn.group_by('UniqueID').agg([
        pl.count('TransactionAmount').alias('hist_txn_count'),
        pl.mean('TransactionAmount').alias('hist_avg_amt'),
        pl.col('TransactionDate').max().alias('max_date'),
        pl.col('TransactionAmount').filter((pl.col('TransactionDate') >= datetime.date(2014, 11, 1)) & (pl.col('TransactionDate') <= datetime.date(2015, 1, 31))).count().alias('same_period_last_year_count'),
        pl.col('TransactionAmount').filter(pl.col('TransactionDate') >= datetime.date(2015, 10, 1)).count().alias('last_month_count'),
    ])

    txn_features = txn_features.with_columns([
        (last_date - pl.col('max_date')).dt.total_days().alias('days_since_last_txn'),
        (pl.col('last_month_count') / (pl.col('hist_txn_count') / 24).replace(0, 1)).alias('txn_velocity')
    ]).drop('max_date')

    train = train_pl.join(demo, on='UniqueID', how='left').join(txn_features, on='UniqueID', how='left')
    test = test_pl.join(demo, on='UniqueID', how='left').join(txn_features, on='UniqueID', how='left')

    del txn, txn_features, demo
    gc.collect()

    train_df = train.to_pandas()
    test_df = test.to_pandas()

    # Age feature
    if 'BirthDate' in train_df.columns:
        train_df['BirthDate'] = pd.to_datetime(train_df['BirthDate'], errors='coerce')
        test_df['BirthDate'] = pd.to_datetime(test_df['BirthDate'], errors='coerce')
        ref_date = pd.to_datetime('2015-10-31')
        train_df['Age'] = (ref_date - train_df['BirthDate']).dt.days // 365
        test_df['Age'] = (ref_date - test_df['BirthDate']).dt.days // 365
        train_df['Age'] = train_df['Age'].fillna(train_df['Age'].median())
        test_df['Age'] = test_df['Age'].fillna(test_df['Age'].median())

    # Categorical encoding
    obj_cols = train_df.select_dtypes(include=['object']).columns.tolist()
    if 'UniqueID' in obj_cols: obj_cols.remove('UniqueID')

    print(f"Encoding {len(obj_cols)} categorical columns...")
    for col in obj_cols:
        le = LabelEncoder()
        train_df[col] = train_df[col].fillna("Missing").astype(str)
        test_df[col] = test_df[col].fillna("Missing").astype(str)
        le.fit(pd.concat([train_df[col], test_df[col]]))
        train_df[col] = le.transform(train_df[col])
        test_df[col] = le.transform(test_df[col])

    features = [c for c in train_df.columns if c not in ['UniqueID', 'next_3m_txn_count', 'BirthDate', 'RunDate']]
    X = train_df[features].fillna(0)
    y = np.log1p(train_df['next_3m_txn_count']) if 'next_3m_txn_count' in train_df.columns else None
    X_test = test_df[features].fillna(0)
    
    return X, y, X_test, test_df['UniqueID']
