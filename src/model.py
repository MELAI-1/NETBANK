import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import optuna
import logging

logger = logging.getLogger(__name__)

def tune_hyperparameters(X, y, model_type="lgb", n_trials=20, seed=42):
    """Tunes hyperparameters using Optuna."""
    logger.info(f"Tuning {model_type} hyperparameters for {n_trials} trials...")
    
    def objective(trial):
        kf = KFold(n_splits=3, shuffle=True, random_state=seed)
        scores = []
        
        if model_type == "lgb":
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'verbosity': -1,
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'random_state': seed
            }
            model_class = lgb.LGBMRegressor
        elif model_type == "xgb":
            params = {
                'objective': 'reg:squarederror',
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'random_state': seed
            }
            model_class = xgb.XGBRegressor
        
        for tr_idx, val_idx in kf.split(X, y):
            xt, yt = X.iloc[tr_idx], y.iloc[tr_idx]
            xv, yv = X.iloc[val_idx], y.iloc[val_idx]
            
            model = model_class(**params, n_estimators=500)
            model.fit(xt, yt, eval_set=[(xv, yv)], callbacks=[lgb.early_stopping(50)] if model_type=="lgb" else None)
            preds = model.predict(xv)
            scores.append(np.sqrt(mean_squared_error(yv, preds)))
            
        return np.mean(scores)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params

def train_and_predict(X, y, X_test, seed=42):
    """Trains a weighted ensemble of LGBM and XGBoost."""
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    # Optional: Tune hyperparameters (comment out if you want to use defaults)
    # lgb_params = tune_hyperparameters(X, y, "lgb", n_trials=10, seed=seed)
    # xgb_params = tune_hyperparameters(X, y, "xgb", n_trials=10, seed=seed)
    
    # Better default parameters based on competition experience
    lgb_params = {
        'n_estimators': 3000,
        'learning_rate': 0.02,
        'num_leaves': 63,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.7,
        'random_state': seed,
        'verbose': -1
    }
    
    xgb_params = {
        'n_estimators': 3000,
        'learning_rate': 0.02,
        'max_depth': 7,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': seed
    }

    oof_lgb = np.zeros(len(X))
    oof_xgb = np.zeros(len(X))
    test_preds_lgb = np.zeros(len(X_test))
    test_preds_xgb = np.zeros(len(X_test))

    for fold, (t_idx, v_idx) in enumerate(kf.split(X, y)):
        logger.info(f"Training Fold {fold+1}...")
        xt, yt = X.iloc[t_idx], y.iloc[t_idx]
        xv, yv = X.iloc[v_idx], y.iloc[v_idx]

        # LightGBM
        m_lgb = lgb.LGBMRegressor(**lgb_params)
        m_lgb.fit(xt, yt, eval_set=[(xv, yv)], callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)])
        oof_lgb[v_idx] = m_lgb.predict(xv)
        test_preds_lgb += m_lgb.predict(X_test) / n_splits

        # XGBoost
        m_xgb = xgb.XGBRegressor(**xgb_params, early_stopping_rounds=150)
        m_xgb.fit(xt, yt, eval_set=[(xv, yv)], verbose=False)
        oof_xgb[v_idx] = m_xgb.predict(xv)
        test_preds_xgb += m_xgb.predict(X_test) / n_splits

    # Simple Blending Weight Optimization (could use Optuna here too)
    # For now, let's use 50/50 or check OOF scores
    rmse_lgb = np.sqrt(mean_squared_error(y, oof_lgb))
    rmse_xgb = np.sqrt(mean_squared_error(y, oof_xgb))
    logger.info(f"OOF LGBM RMSE: {rmse_lgb:.5f}")
    logger.info(f"OOF XGBoost RMSE: {rmse_xgb:.5f}")
    
    # Blend based on performance
    total_rmse = rmse_lgb + rmse_xgb
    w_lgb = 1 - (rmse_lgb / total_rmse)
    w_xgb = 1 - (rmse_xgb / total_rmse)
    
    # Normalize weights
    total_w = w_lgb + w_xgb
    w_lgb /= total_w
    w_xgb /= total_w
    
    logger.info(f"Blending weights: LGBM={w_lgb:.4f}, XGBoost={w_xgb:.4f}")
    
    final_preds = (w_lgb * test_preds_lgb) + (w_xgb * test_preds_xgb)
    return final_preds
