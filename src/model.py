import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
import torch
import logging

# Module-level logger
logger = logging.getLogger(__name__)

def train_and_predict(X, y, X_test, seed=42):
    """
    Optimized Model Factory: 
    Prioritizes CatBoost and HGBR (Top leaderboard performers)
    """
    n_splits = 5
    y_bins = pd.cut(y, bins=10, labels=False)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    # Pre-scaling for TabNet and distance-based logic
    pt = PowerTransformer()
    X_pt = pt.fit_transform(X)
    X_test_pt = pt.transform(X_test)
    
    # Model Storage
    models_list = ['cat', 'hgbr', 'lgb', 'xgb', 'tabnet']
    final_preds_dict = {m: np.zeros(len(X_test)) for m in models_list}

    for fold, (t_idx, v_idx) in enumerate(skf.split(X, y_bins)):
        logger.info(f"🏆 Training Fold {fold+1}/{n_splits}...")
        
        xt, yt = X.iloc[t_idx], y.iloc[t_idx]
        xv, yv = X.iloc[v_idx], y.iloc[v_idx]
        xt_s, xv_s = X_pt[t_idx], X_pt[v_idx]

        # 1. CatBoost (Leaderboard Winner #1)
        # Using more iterations and slightly deeper trees for better capture of bank behavior
        m_cat = CatBoostRegressor(iterations=3000, learning_rate=0.02, depth=8, 
                                    l2_leaf_reg=3, random_seed=seed, verbose=0, 
                                    early_stopping_rounds=100)
        m_cat.fit(xt, yt, eval_set=(xv, yv))
        final_preds_dict['cat'] += m_cat.predict(X_test) / n_splits

        # 2. HistGradientBoosting (Leaderboard Winner #2)
        # Excellent generalization on non-linear distributions
        m_hgbr = HistGradientBoostingRegressor(max_iter=1500, learning_rate=0.02, max_depth=10,
                                               l2_regularization=1.0, random_state=seed, early_stopping=True)
        m_hgbr.fit(xt, yt)
        final_preds_dict['hgbr'] += m_hgbr.predict(X_test) / n_splits

        # 3. LightGBM (Tweedie) - Supporting role
        m_lgb = lgb.LGBMRegressor(n_estimators=2000, learning_rate=0.02, objective='tweedie', 
                                  max_depth=8, num_leaves=31, random_state=seed, verbose=-1)
        m_lgb.fit(xt, yt, eval_set=[(xv, yv)], callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
        final_preds_dict['lgb'] += m_lgb.predict(X_test) / n_splits

        # 4. XGBoost - Supporting role
        m_xgb = xgb.XGBRegressor(n_estimators=1500, learning_rate=0.02, max_depth=7, 
                                 subsample=0.8, colsample_bytree=0.8, random_state=seed)
        m_xgb.fit(xt, yt, eval_set=[(xv, yv)], verbose=False, early_stopping_rounds=100)
        final_preds_dict['xgb'] += m_xgb.predict(X_test) / n_splits

        # 5. TabNet - Diversity role
        m_tab = TabNetRegressor(verbose=0, seed=seed)
        m_tab.fit(xt_s, yt.values.reshape(-1, 1), eval_set=[(xv_s, yv.values.reshape(-1, 1))], 
                  patience=30, max_epochs=100, batch_size=4096, virtual_batch_size=256)
        final_preds_dict['tabnet'] += m_tab.predict(X_test_pt).flatten() / n_splits

    # --- Champion-Centric Blending ---
    
    # NEW Blend: Focus heavily on the winners (Cat + HGBR)
    # We give 80% of the weight to the top performers
    final_preds_dict['blend_champion'] = (
        final_preds_dict['cat'] * 0.45 +
        final_preds_dict['hgbr'] * 0.35 +
        final_preds_dict['lgb'] * 0.10 +
        final_preds_dict['xgb'] * 0.05 +
        final_preds_dict['tabnet'] * 0.05
    )

    # Rank-Based Blend (Always keep for private LB stability)
    ranks = pd.DataFrame({m: pd.Series(final_preds_dict[m]).rank(pct=True) for m in models_list})
    avg_rank = ranks.mean(axis=1)
    final_preds_dict['blend_rank_robust'] = np.percentile(y, avg_rank * 100)

    return final_preds_dict
