import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
import logging

logger = logging.getLogger(__name__)
SEED = 42

def train_and_predict(X, y, X_test, seed=SEED):
    """
    Robust Temporal Tweedie Stacking (Level 1):
    - Models: LGBM, XGB, CatBoost (all with Tweedie objective)
    - Meta-Learner: RidgeCV (Regularized Linear Stacking)
    """
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    # Model Storage
    models = ['lgb', 'xgb', 'cat']
    oof_preds = {m: np.zeros(len(X)) for m in models}
    test_preds = {m: np.zeros(len(X_test)) for m in models}
    
    # Tweedie Power 1.2 is generally robust for zero-inflated bank transaction counts
    TWEEDIE_POWER = 1.2
    
    for fold, (t_idx, v_idx) in enumerate(kf.split(X, y)):
        logger.info(f"🏆 Training Fold {fold+1}/{n_splits}...")
        
        xt, yt = X.iloc[t_idx], y.iloc[t_idx]
        xv, yv = X.iloc[v_idx], y.iloc[v_idx]
        
        # Raw space for Tweedie models
        yt_raw, yv_raw = np.expm1(yt), np.expm1(yv)

        # 1. LightGBM (Tweedie)
        m_lgb = lgb.LGBMRegressor(
            objective='tweedie',
            tweedie_variance_power=TWEEDIE_POWER,
            n_estimators=2000,
            learning_rate=0.015,
            num_leaves=32,
            feature_fraction=0.8,
            random_state=seed,
            verbose=-1
        )
        m_lgb.fit(xt, yt_raw, eval_set=[(xv, yv_raw)], callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
        oof_preds['lgb'][v_idx] = np.log1p(np.maximum(0, m_lgb.predict(xv)))
        test_preds['lgb'] += np.log1p(np.maximum(0, m_lgb.predict(X_test))) / n_splits

        # 2. XGBoost (Tweedie)
        m_xgb = xgb.XGBRegressor(
            objective='reg:tweedie',
            tweedie_variance_power=TWEEDIE_POWER,
            n_estimators=1500,
            learning_rate=0.015,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed
        )
        m_xgb.fit(xt, yt_raw, eval_set=[(xv, yv_raw)], early_stopping_rounds=100, verbose=False)
        oof_preds['xgb'][v_idx] = np.log1p(np.maximum(0, m_xgb.predict(xv)))
        test_preds['xgb'] += np.log1p(np.maximum(0, m_xgb.predict(X_test))) / n_splits

        # 3. CatBoost (Tweedie)
        m_cat = CatBoostRegressor(
            loss_function=f'Tweedie:variance_power={TWEEDIE_POWER}',
            iterations=1500,
            learning_rate=0.015,
            depth=6,
            random_seed=seed,
            verbose=0,
            early_stopping_rounds=100
        )
        m_cat.fit(xt, yt_raw, eval_set=(xv, yv_raw))
        oof_preds['cat'][v_idx] = np.log1p(np.maximum(0, m_cat.predict(xv)))
        test_preds['cat'] += np.log1p(np.maximum(0, m_cat.predict(X_test))) / n_splits

    # --- LEVEL 1: RidgeCV Stacking ---
    logger.info("🧠 Level 1: RidgeCV Stacking...")
    X_meta = np.column_stack([oof_preds[m] for m in models])
    X_meta_test = np.column_stack([test_preds[m] for m in models])
    
    # Meta-learner is trained in log-space (RMSLE focus)
    meta_model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0], cv=5)
    meta_model.fit(X_meta, y)
    
    final_stack = meta_model.predict(X_meta_test)
    
    logger.info(f"✅ Stacking Complete. Ridge Alpha: {meta_model.alpha_}")
    
    return {
        'final_stack': final_stack,
        'lgb': test_preds['lgb'],
        'xgb': test_preds['xgb'],
        'cat': test_preds['cat']
    }
