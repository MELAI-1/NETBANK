import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
import logging

logger = logging.getLogger(__name__)

def train_and_predict(X, y, X_test, seed=42):
    """Triple Stacking (LGBM, XGB, CatBoost) with Stratified K-Fold."""
    
    # Stratified K-Fold on Binned Target (for regression stability)
    n_splits = 5
    y_bins = pd.cut(y, bins=10, labels=False)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    oof_lgb = np.zeros(len(X))
    oof_xgb = np.zeros(len(X))
    oof_cat = np.zeros(len(X))
    
    test_lgb = np.zeros(len(X_test))
    test_xgb = np.zeros(len(X_test))
    test_cat = np.zeros(len(X_test))

    lgb_params = {'n_estimators': 3000, 'learning_rate': 0.015, 'num_leaves': 127, 'max_depth': -1, 'random_state': seed, 'verbose': -1}
    xgb_params = {'n_estimators': 3000, 'learning_rate': 0.015, 'max_depth': 8, 'subsample': 0.8, 'random_state': seed}
    cat_params = {'iterations': 3000, 'learning_rate': 0.015, 'depth': 8, 'random_seed': seed, 'verbose': 0}

    for fold, (t_idx, v_idx) in enumerate(skf.split(X, y_bins)):
        logger.info(f"🚀 Training Fold {fold+1}/{n_splits}...")
        xt, yt = X.iloc[t_idx], y.iloc[t_idx]
        xv, yv = X.iloc[v_idx], y.iloc[v_idx]

        # 1. LightGBM (Tweedie objective for counts)
        m_lgb = lgb.LGBMRegressor(**lgb_params, objective='tweedie')
        m_lgb.fit(xt, yt, eval_set=[(xv, yv)], callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)])
        oof_lgb[v_idx] = m_lgb.predict(xv)
        test_lgb += m_lgb.predict(X_test) / n_splits

        # 2. XGBoost
        m_xgb = xgb.XGBRegressor(**xgb_params, early_stopping_rounds=150)
        m_xgb.fit(xt, yt, eval_set=[(xv, yv)], verbose=False)
        oof_xgb[v_idx] = m_xgb.predict(xv)
        test_xgb += m_xgb.predict(X_test) / n_splits

        # 3. CatBoost
        m_cat = CatBoostRegressor(**cat_params, early_stopping_rounds=150)
        m_cat.fit(xt, yt, eval_set=(xv, yv))
        oof_cat[v_idx] = m_cat.predict(xv)
        test_cat += m_cat.predict(X_test) / n_splits

    # Stacking (Level 1)
    logger.info("🧠 Training Stacking Meta-Model (Ridge)...")
    X_meta = np.column_stack((oof_lgb, oof_xgb, oof_cat))
    X_test_meta = np.column_stack((test_lgb, test_xgb, test_cat))
    
    meta_model = Ridge(alpha=1.0)
    meta_model.fit(X_meta, y)
    
    final_preds = meta_model.predict(X_test_meta)
    
    # Performance Report
    rmse_lgb = np.sqrt(mean_squared_error(y, oof_lgb))
    rmse_stack = np.sqrt(mean_squared_error(y, meta_model.predict(X_meta)))
    logger.info(f"📊 OOF LGBM RMSE: {rmse_lgb:.5f}")
    logger.info(f"📊 OOF Stacked RMSE: {rmse_stack:.5f}")
    
    return final_preds
