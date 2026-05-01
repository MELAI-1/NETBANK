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

def train_and_predict(X, y, X_test, seed=42):
    """Quad Stacking (LGBM, XGB, CatBoost, TabNet) with Stratified K-Fold."""
    
    n_splits = 5
    y_bins = pd.cut(y, bins=10, labels=False)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    # Storage for OOF and Test predictions
    models = ['lgb', 'xgb', 'cat', 'tabnet']
    oof_preds = {m: np.zeros(len(X)) for m in models}
    test_preds = {m: np.zeros(len(X_test)) for m in models}

    # Scaling for TabNet
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_test_scaled = scaler.transform(X_test)

    for fold, (t_idx, v_idx) in enumerate(skf.split(X, y_bins)):
        logger.info(f"🚀 Training Fold {fold+1}/{n_splits}...")
        
        # GBDT Split
        xt, yt = X.iloc[t_idx], y.iloc[t_idx]
        xv, yv = X.iloc[v_idx], y.iloc[v_idx]
        
        # TabNet Split (scaled)
        xt_s, xv_s = X_scaled[t_idx], X_scaled[v_idx]

        # 1. LightGBM
        m_lgb = lgb.LGBMRegressor(n_estimators=2000, learning_rate=0.02, objective='tweedie', random_state=seed, verbose=-1)
        m_lgb.fit(xt, yt, eval_set=[(xv, yv)], callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
        oof_preds['lgb'][v_idx] = m_lgb.predict(xv)
        test_preds['lgb'] += m_lgb.predict(X_test) / n_splits

        # 2. XGBoost
        m_xgb = xgb.XGBRegressor(n_estimators=2000, learning_rate=0.02, random_state=seed, early_stopping_rounds=100)
        m_xgb.fit(xt, yt, eval_set=[(xv, yv)], verbose=False)
        oof_preds['xgb'][v_idx] = m_xgb.predict(xv)
        test_preds['xgb'] += m_xgb.predict(X_test) / n_splits

        # 3. CatBoost
        m_cat = CatBoostRegressor(iterations=2000, learning_rate=0.02, random_seed=seed, verbose=0, early_stopping_rounds=100)
        m_cat.fit(xt, yt, eval_set=(xv, yv))
        oof_preds['cat'][v_idx] = m_cat.predict(xv)
        test_preds['cat'] += m_cat.predict(X_test) / n_splits

        # 4. TabNet (Deep Learning)
        m_tab = TabNetRegressor(verbose=0, seed=seed, optimizer_params=dict(lr=2e-2))
        m_tab.fit(xt_s, yt.values.reshape(-1, 1), eval_set=[(xv_s, yv.values.reshape(-1, 1))], 
                  patience=30, max_epochs=100, batch_size=1024, virtual_batch_size=128)
        oof_preds['tabnet'][v_idx] = m_tab.predict(xv_s).flatten()
        test_preds['tabnet'] += m_tab.predict(X_test_scaled).flatten() / n_splits

    # Meta-Model Stacking
    X_meta = np.column_stack([oof_preds[m] for m in models])
    X_test_meta = np.column_stack([test_preds[m] for m in models])
    
    meta_model = Ridge(alpha=1.0)
    meta_model.fit(X_meta, y)
    
    final_preds = meta_model.predict(X_test_meta)
    
    # Report
    rmse_stack = np.sqrt(mean_squared_error(y, meta_model.predict(X_meta)))
    logger.info(f"📊 Final Stacked RMSE: {rmse_stack:.5f}")
    
    return final_preds
