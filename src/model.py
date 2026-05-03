import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import logging

logger = logging.getLogger(__name__)
SEED = 42

def train_and_predict(X, y, X_test, seed=SEED):
    """
    Robust Pipeline v4.0:
    - Quadruple Ensemble: CatBoost, LGBM, XGBoost, RandomForest
    - RidgeCV Stacking as meta-learner
    - Tweedie Objective for Boosters (handles zero-inflation)
    """
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    # Model Storage for OOF predictions
    models = ['cat', 'lgb', 'xgb', 'rf']
    oof_preds = {m: np.zeros(len(X)) for m in models}
    test_preds = {m: np.zeros(len(X_test)) for m in models}
    
    # 1. Hyperparameters (Optimized for robustness)
    cat_params = {
        'loss_function': 'Tweedie:variance_power=1.5',
        'iterations': 3000,
        'learning_rate': 0.02,
        'depth': 6,
        'random_seed': seed,
        'verbose': 0,
        'early_stopping_rounds': 100
    }
    
    lgb_params = {
        'objective': 'tweedie',
        'tweedie_variance_power': 1.5,
        'learning_rate': 0.02,
        'num_leaves': 31,
        'feature_fraction': 0.8,
        'n_estimators': 3000,
        'random_state': seed,
        'verbose': -1
    }
    
    xgb_params = {
        'objective': 'reg:tweedie',
        'tweedie_variance_power': 1.5,
        'learning_rate': 0.02,
        'max_depth': 6,
        'n_estimators': 3000,
        'random_state': seed,
        'tree_method': 'hist'
    }

    # 2. Cross-Validation Loop
    for fold, (t_idx, v_idx) in enumerate(kf.split(X, y)):
        logger.info(f"🏆 Training Fold {fold+1}/{n_splits}...")
        
        xt, yt = X.iloc[t_idx], y.iloc[t_idx]
        xv, yv = X.iloc[v_idx], y.iloc[v_idx]
        
        # Raw space for Tweedie models
        yt_raw, yv_raw = np.expm1(yt), np.expm1(yv)

        # --- Model 1: CatBoost ---
        m_cat = CatBoostRegressor(**cat_params)
        m_cat.fit(xt, yt_raw, eval_set=(xv, yv_raw))
        oof_preds['cat'][v_idx] = np.log1p(np.maximum(0, m_cat.predict(xv)))
        test_preds['cat'] += np.log1p(np.maximum(0, m_cat.predict(X_test))) / n_splits

        # --- Model 2: LightGBM ---
        m_lgb = lgb.LGBMRegressor(**lgb_params)
        m_lgb.fit(xt, yt_raw, eval_set=[(xv, yv_raw)], 
                  callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
        oof_preds['lgb'][v_idx] = np.log1p(np.maximum(0, m_lgb.predict(xv)))
        test_preds['lgb'] += np.log1p(np.maximum(0, m_lgb.predict(X_test))) / n_splits

        # --- Model 3: XGBoost ---
        m_xgb = xgb.XGBRegressor(**xgb_params, early_stopping_rounds=100)
        m_xgb.fit(xt, yt_raw, eval_set=[(xv, yv_raw)], verbose=False)
        oof_preds['xgb'][v_idx] = np.log1p(np.maximum(0, m_xgb.predict(xv)))
        test_preds['xgb'] += np.log1p(np.maximum(0, m_xgb.predict(X_test))) / n_splits

        # --- Model 4: Random Forest (Log Space) ---
        m_rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=seed, n_jobs=-1)
        m_rf.fit(xt, yt)
        oof_preds['rf'][v_idx] = m_rf.predict(xv)
        test_preds['rf'] += m_rf.predict(X_test) / n_splits

    # 3. Meta-Learner (RidgeCV Stacking)
    logger.info("🧠 Running RidgeCV Stacking...")
    X_meta = np.column_stack([oof_preds[m] for m in models])
    X_meta_test = np.column_stack([test_preds[m] for m in models])
    
    meta_model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0], cv=5)
    meta_model.fit(X_meta, y)
    
    final_stack_preds = meta_model.predict(X_meta_test)
    
    # Return all predictions for ensemble options
    results = {
        'final_stack': final_stack_preds,
        'cat': test_preds['cat'],
        'lgb': test_preds['lgb'],
        'blend_top3': (test_preds['cat'] * 0.4 + test_preds['lgb'] * 0.4 + test_preds['xgb'] * 0.2)
    }
    
    logger.info(f"✅ Stacking Complete. Ridge Alpha: {meta_model.alpha_}")
    return results
