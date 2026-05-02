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

from sklearn.preprocessing import RobustScaler, PowerTransformer

def train_and_predict(X, y, X_test, seed=42):
    """
    SOTA Frontier Pipeline:
    1. Seed Averaging (Stability)
    2. Rank-Based Blending (Robustness to Outliers)
    3. PowerTransform (Normalization)
    """
    n_splits = 5
    seeds = [seed, seed + 1, seed + 2] # Average across 3 seeds for max stability
    
    all_test_preds = []
    
    for current_seed in seeds:
        logger.info(f"✨ Running Seed {current_seed}...")
        y_bins = pd.cut(y, bins=10, labels=False)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=current_seed)
        
        # Scaling
        pt = PowerTransformer()
        X_pt = pt.fit_transform(X)
        X_test_pt = pt.transform(X_test)
        
        models = ['lgb', 'xgb', 'cat', 'tabnet']
        test_preds = {m: np.zeros(len(X_test)) for m in models}

        for fold, (t_idx, v_idx) in enumerate(skf.split(X, y_bins)):
            xt, yt = X.iloc[t_idx], y.iloc[t_idx]
            xv, yv = X.iloc[v_idx], y.iloc[v_idx]
            xt_s, xv_s = X_pt[t_idx], X_pt[v_idx]

            # LGBM (Tweedie is SOTA for count data)
            m_lgb = lgb.LGBMRegressor(n_estimators=2000, learning_rate=0.015, objective='tweedie', 
                                    max_depth=9, num_leaves=63, random_state=current_seed, verbose=-1)
            m_lgb.fit(xt, yt, eval_set=[(xv, yv)], callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
            test_preds['lgb'] += m_lgb.predict(X_test) / n_splits

            # XGB (Dart can be more robust)
            m_xgb = xgb.XGBRegressor(n_estimators=2000, learning_rate=0.015, max_depth=7, 
                                   tree_method='hist', random_state=current_seed, early_stopping_rounds=100)
            m_xgb.fit(xt, yt, eval_set=[(xv, yv)], verbose=False)
            test_preds['xgb'] += m_xgb.predict(X_test) / n_splits

            # Cat (Symmetry trees are excellent for generalization)
            m_cat = CatBoostRegressor(iterations=2000, learning_rate=0.015, depth=7, 
                                    random_seed=current_seed, verbose=0, early_stopping_rounds=100)
            m_cat.fit(xt, yt, eval_set=(xv, yv))
            test_preds['cat'] += m_cat.predict(X_test) / n_splits

            # TabNet
            m_tab = TabNetRegressor(verbose=0, seed=current_seed)
            m_tab.fit(xt_s, yt.values.reshape(-1, 1), eval_set=[(xv_s, yv.values.reshape(-1, 1))], 
                      patience=30, max_epochs=100, batch_size=4096, virtual_batch_size=256)
            test_preds['tabnet'] += m_tab.predict(X_test_pt).flatten() / n_splits

        # Rank Blending for the current seed
        # This converts predictions to ranks to avoid one model pulling the average too far
        seed_preds = (
            pd.Series(test_preds['lgb']).rank(pct=True) * 0.4 +
            pd.Series(test_preds['cat']).rank(pct=True) * 0.3 +
            pd.Series(test_preds['xgb']).rank(pct=True) * 0.2 +
            pd.Series(test_preds['tabnet']).rank(pct=True) * 0.1
        )
        
        # Map rank back to Log-Distribution of Y_train
        # This is the "secret sauce" of top Zindi competitors
        final_seed_preds = np.percentile(y, seed_preds * 100)
        all_test_preds.append(final_seed_preds)

    # Average the results of all seeds
    return np.mean(all_test_preds, axis=0)
