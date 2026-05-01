import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

def train_stacking(X, y, X_test, seed=42):
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    oof_lgb = np.zeros(len(X))
    oof_xgb = np.zeros(len(X))
    oof_cat = np.zeros(len(X))

    test_preds_lgb = np.zeros(len(X_test))
    test_preds_xgb = np.zeros(len(X_test))
    test_preds_cat = np.zeros(len(X_test))

    print(f"🚀 Training Triple Stacking ({n_splits} folds)...")

    for fold, (t_idx, v_idx) in enumerate(kf.split(X, y)):
        X_trn, y_trn = X.iloc[t_idx], y.iloc[t_idx]
        X_val, y_val = X.iloc[v_idx], y.iloc[v_idx]

        # 1. LIGHTGBM
        m_lgb = lgb.LGBMRegressor(
            objective='regression', 
            n_estimators=2500, 
            learning_rate=0.02, 
            num_leaves=80, 
            feature_fraction=0.8, 
            random_state=seed,
            verbose=-1
        )
        m_lgb.fit(X_trn, y_trn, eval_set=[(X_val, y_val)], 
                  callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)])
        oof_lgb[v_idx] = m_lgb.predict(X_val)
        test_preds_lgb += m_lgb.predict(X_test) / n_splits

        # 2. XGBOOST
        m_xgb = xgb.XGBRegressor(
            objective='reg:squarederror', 
            n_estimators=2500, 
            learning_rate=0.02, 
            max_depth=8, 
            subsample=0.8, 
            colsample_bytree=0.8, 
            random_state=seed, 
            early_stopping_rounds=150
        )
        m_xgb.fit(X_trn, y_trn, eval_set=[(X_val, y_val)], verbose=False)
        oof_xgb[v_idx] = m_xgb.predict(X_val)
        test_preds_xgb += m_xgb.predict(X_test) / n_splits

        # 3. CATBOOST
        m_cat = CatBoostRegressor(
            iterations=2500, 
            learning_rate=0.02, 
            depth=8, 
            random_seed=seed, 
            verbose=0, 
            early_stopping_rounds=150
        )
        m_cat.fit(X_trn, y_trn, eval_set=(X_val, y_val))
        oof_cat[v_idx] = m_cat.predict(X_val)
        test_preds_cat += m_cat.predict(X_test) / n_splits

        print(f"✅ Fold {fold+1} complete.")

    # Meta-model
    print("\n🧠 Training Meta-Model (Ridge)...")
    X_meta = np.column_stack((oof_lgb, oof_xgb, oof_cat))
    X_test_meta = np.column_stack((test_preds_lgb, test_preds_xgb, test_preds_cat))

    meta_model = Ridge(alpha=1.0)
    meta_model.fit(X_meta, y)

    print("Meta-model coefficients:")
    model_names = ['LightGBM', 'XGBoost', 'CatBoost']
    for name, coef in zip(model_names, meta_model.coef_):
        print(f"  {name} weight: {coef:.4f}")

    final_preds = meta_model.predict(X_test_meta)
    
    # Calculate CV score
    oof_final = meta_model.predict(X_meta)
    rmse_log = np.sqrt(mean_squared_error(y, oof_final))
    print(f"\nFinal CV RMSLE: {rmse_log:.5f}")

    return final_preds
