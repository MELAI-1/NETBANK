import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.ensemble import HistGradientBoostingRegressor

def train_and_predict(X, y, X_test, seed=42):
    """
    Model Factory: Returns a dictionary of ALL model predictions
    Allows user to test individual performance and various blends.
    """
    n_splits = 5
    y_bins = pd.cut(y, bins=10, labels=False)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    # Pre-scaling
    pt = PowerTransformer()
    X_pt = pt.fit_transform(X)
    X_test_pt = pt.transform(X_test)
    
    # Individual Model Storage
    models_list = ['lgb', 'xgb', 'cat', 'tabnet', 'hgbr']
    final_preds_dict = {m: np.zeros(len(X_test)) for m in models_list}

    for fold, (t_idx, v_idx) in enumerate(skf.split(X, y_bins)):
        logger.info(f"🚀 Training Fold {fold+1}/{n_splits}...")
        
        xt, yt = X.iloc[t_idx], y.iloc[t_idx]
        xv, yv = X.iloc[v_idx], y.iloc[v_idx]
        xt_s, xv_s = X_pt[t_idx], X_pt[v_idx]

        # 1. LightGBM (Tweedie)
        m_lgb = lgb.LGBMRegressor(n_estimators=2000, learning_rate=0.02, objective='tweedie', random_state=seed, verbose=-1)
        m_lgb.fit(xt, yt, eval_set=[(xv, yv)], callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
        final_preds_dict['lgb'] += m_lgb.predict(X_test) / n_splits

        # 2. XGBoost
        m_xgb = xgb.XGBRegressor(n_estimators=2000, learning_rate=0.02, random_state=seed, early_stopping_rounds=100)
        m_xgb.fit(xt, yt, eval_set=[(xv, yv)], verbose=False)
        final_preds_dict['xgb'] += m_xgb.predict(X_test) / n_splits

        # 3. CatBoost
        m_cat = CatBoostRegressor(iterations=2000, learning_rate=0.02, random_seed=seed, verbose=0, early_stopping_rounds=100)
        m_cat.fit(xt, yt, eval_set=(xv, yv))
        final_preds_dict['cat'] += m_cat.predict(X_test) / n_splits

        # 4. TabNet
        m_tab = TabNetRegressor(verbose=0, seed=seed)
        m_tab.fit(xt_s, yt.values.reshape(-1, 1), eval_set=[(xv_s, yv.values.reshape(-1, 1))], 
                  patience=30, max_epochs=100, batch_size=4096, virtual_batch_size=256)
        final_preds_dict['tabnet'] += m_tab.predict(X_test_pt).flatten() / n_splits

        # 5. HistGradientBoosting (Scikit-Learn's version of LGBM, sometimes generalizes better)
        m_hgbr = HistGradientBoostingRegressor(max_iter=1000, learning_rate=0.02, random_state=seed, early_stopping=True)
        m_hgbr.fit(xt, yt)
        final_preds_dict['hgbr'] += m_hgbr.predict(X_test) / n_splits

    # Create Blends
    # Blend A: Simple Weighted Average (GBDTs focus)
    final_preds_dict['blend_weighted'] = (
        final_preds_dict['lgb'] * 0.35 +
        final_preds_dict['cat'] * 0.35 +
        final_preds_dict['xgb'] * 0.15 +
        final_preds_dict['hgbr'] * 0.15
    )

    # Blend B: Rank-Based (Most robust for RMSLE)
    ranks = pd.DataFrame({m: pd.Series(final_preds_dict[m]).rank(pct=True) for m in models_list})
    avg_rank = ranks.mean(axis=1)
    # Map back to training distribution
    final_preds_dict['blend_rank'] = np.percentile(y, avg_rank * 100)

    return final_preds_dict
