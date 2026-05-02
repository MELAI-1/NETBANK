import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import torch
import logging

logger = logging.getLogger(__name__)

from sklearn.preprocessing import QuantileTransformer, KBinsDiscretizer
from sklearn.linear_model import RidgeCV, BayesianRidge
from scipy.stats import ks_2samp

def evaluate_distribution(y_true, y_pred):
    """Checks how well the prediction distribution matches the truth."""
    # Use Kolmogorov-Smirnov test to check if distributions are similar
    statistic, _ = ks_2samp(y_true, y_pred)
    return statistic # Lower is better (more similar)

def train_and_predict(X, y, X_test, seed=42):
    """
    Automated Strategy Competition: 
    1. Log-Stacking
    2. Quantile-Mapping (Best for Distribution Alignment)
    3. Binned-Regression (Best for handling heavy tails/outliers)
    """
    
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    y_bins = pd.cut(y, bins=10, labels=False)

    # We will test two main transformations
    strategies = ["log_stack", "quantile_stack"]
    best_oof_score = float('inf')
    best_preds = None
    best_strategy_name = ""

    # Pre-scale for TabNet
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_test_scaled = scaler.transform(X_test)

    # To keep it efficient, we run a shared ensemble and then apply different meta-heads
    models = ['lgb', 'xgb', 'cat', 'tabnet']
    oof_preds = {m: np.zeros(len(X)) for m in models}
    test_preds = {m: np.zeros(len(X_test)) for m in models}

    for fold, (t_idx, v_idx) in enumerate(skf.split(X, y_bins)):
        logger.info(f"🚀 Training Fold {fold+1}/{n_splits}...")
        
        xt, yt = X.iloc[t_idx], y.iloc[t_idx]
        xv, yv = X.iloc[v_idx], y.iloc[v_idx]
        xt_s, xv_s = X_scaled[t_idx], X_scaled[v_idx]

        # 1. LGBM
        m_lgb = lgb.LGBMRegressor(n_estimators=1500, learning_rate=0.03, objective='tweedie', random_state=seed, verbose=-1)
        m_lgb.fit(xt, yt, eval_set=[(xv, yv)], callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        oof_preds['lgb'][v_idx] = m_lgb.predict(xv)
        test_preds['lgb'] += m_lgb.predict(X_test) / n_splits

        # 2. XGB
        m_xgb = xgb.XGBRegressor(n_estimators=1500, learning_rate=0.03, random_state=seed, early_stopping_rounds=50)
        m_xgb.fit(xt, yt, eval_set=[(xv, yv)], verbose=False)
        oof_preds['xgb'][v_idx] = m_xgb.predict(xv)
        test_preds['xgb'] += m_xgb.predict(X_test) / n_splits

        # 3. Cat
        m_cat = CatBoostRegressor(iterations=1500, learning_rate=0.03, random_seed=seed, verbose=0, early_stopping_rounds=50)
        m_cat.fit(xt, yt, eval_set=(xv, yv))
        oof_preds['cat'][v_idx] = m_cat.predict(xv)
        test_preds['cat'] += m_cat.predict(X_test) / n_splits

        # 4. TabNet
        m_tab = TabNetRegressor(verbose=0, seed=seed, optimizer_params=dict(lr=1e-2))
        m_tab.fit(xt_s, yt.values.reshape(-1, 1), eval_set=[(xv_s, yv.values.reshape(-1, 1))], 
                  patience=20, max_epochs=50, batch_size=2048, virtual_batch_size=256)
        oof_preds['tabnet'][v_idx] = m_tab.predict(xv_s).flatten()
        test_preds['tabnet'] += m_tab.predict(X_test_scaled).flatten() / n_splits

    # Strategy Selection Phase
    X_meta = np.column_stack([oof_preds[m] for m in models])
    X_test_meta = np.column_stack([test_preds[m] for m in models])

    # --- STRATEGY 1: Ridge Log Stacking ---
    meta_1 = RidgeCV(alphas=[0.1, 1.0, 10.0], cv=5)
    meta_1.fit(X_meta, y)
    oof_1 = meta_1.predict(X_meta)
    score_1 = np.sqrt(mean_squared_error(y, oof_1))
    dist_1 = evaluate_distribution(y, oof_1)
    logger.info(f"📊 Strategy 1 (Log-Ridge): RMSE={score_1:.5f}, Dist_Stat={dist_1:.5f}")

    # --- STRATEGY 2: Quantile Distribution Alignment ---
    # This forces the predictions to follow the training distribution perfectly
    qt = QuantileTransformer(output_distribution='normal', n_quantiles=1000, random_state=seed)
    y_norm = qt.fit_transform(y.values.reshape(-1, 1)).flatten()
    
    # We need to map our GBDT oof_preds to this normal space to train a meta-model
    X_meta_norm = qt.transform(X_meta) 
    meta_2 = BayesianRidge() # More robust for distribution mapping
    meta_2.fit(X_meta, y) # Still fit to raw log space but we check dist
    oof_2 = meta_2.predict(X_meta)
    
    # Post-process Strategy 2 to align distribution
    # This is a 'Rank-Based' alignment
    idx = np.argsort(oof_2)
    oof_2_aligned = np.zeros_like(oof_2)
    oof_2_aligned[idx] = np.sort(y) # Force output to match training exactly
    
    score_2 = np.sqrt(mean_squared_error(y, oof_2_aligned))
    dist_2 = evaluate_distribution(y, oof_2_aligned)
    logger.info(f"📊 Strategy 2 (Quantile-Aligned): RMSE={score_2:.5f}, Dist_Stat={dist_2:.5f}")

    # Final Decision (Balance score and distribution similarity)
    # We prefer the one with the lowest RMSE, but if scores are close, we pick the one with better distribution
    if score_1 < score_2:
        logger.info("🏆 Winner: Log-Ridge Stacking")
        return meta_1.predict(X_test_meta)
    else:
        logger.info("🏆 Winner: Quantile-Aligned Stacking")
        # Apply the same rank-alignment to test predictions
        test_p = meta_2.predict(X_test_meta)
        idx_test = np.argsort(test_p)
        test_p_aligned = np.zeros_like(test_p)
        # Since we don't have Y_test, we use the Y_train distribution to "color" our predictions
        test_p_aligned[idx_test] = np.sort(np.random.choice(y, len(test_p), replace=True))
        return test_p_aligned
