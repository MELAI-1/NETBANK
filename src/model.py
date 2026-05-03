import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
import logging

logger = logging.getLogger(__name__)
SEED = 42

def asymmetric_mse_loss(y_true, y_pred):
    """
    Asymmetric Loss: Penalize under-predictions more heavily.
    Optimized for RMSLE in log-space.
    """
    residual = y_pred - y_true
    # k=1.5 penalty for under-prediction (residual < 0)
    grad = np.where(residual < 0, 2.0 * residual * 2.0, 2.0 * residual)
    hess = np.where(residual < 0, 2.0 * 2.0, 2.0)
    return grad, hess

def train_and_predict(X, y, X_test, seed=SEED):
    """
    Expert Mixture Architecture (MoE):
    1. Gater: Identifies High-Variance/Hard customers.
    2. Expert A: Optimized for Bulk (Central Distribution).
    3. Expert B: Specialized in High-Variance (Asymmetric Loss).
    """
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    oof_base = np.zeros(len(X))
    gater_probs = np.zeros(len(X_test))
    
    # --- PHASE 1: Detection (Identify the Problem Cluster) ---
    logger.info("🔍 Phase 1: Detecting High-Error Cluster via OOF Residuals...")
    for t_idx, v_idx in kf.split(X, y):
        m_base = CatBoostRegressor(iterations=1000, learning_rate=0.05, depth=6, random_seed=seed, verbose=0)
        m_base.fit(X.iloc[t_idx], y.iloc[t_idx])
        oof_base[v_idx] = m_base.predict(X.iloc[v_idx])
    
    # Calculate Residuals and label 'Hard' cases (Top 15% error)
    residuals = np.abs(y - oof_base)
    threshold = np.percentile(residuals, 85)
    is_hard = (residuals > threshold).astype(int)
    
    # --- PHASE 2: Gater Model (Binary Classifier) ---
    logger.info("🤖 Phase 2: Training Gater (Meta-Learner)...")
    gater = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=seed)
    gater.fit(X, is_hard)
    gater_probs = gater.predict_proba(X_test)[:, 1]
    
    # --- PHASE 3: Specialized Experts ---
    logger.info("⚔️ Phase 3: Training Specialized Experts...")
    
    # Expert A: Bulk (Balanced)
    m_bulk = lgb.LGBMRegressor(n_estimators=2000, learning_rate=0.02, num_leaves=31, random_state=seed, verbose=-1)
    m_bulk.fit(X, y)
    pred_bulk = m_bulk.predict(X_test)
    
    # Expert B: High-Variance (Asymmetric Loss)
    # Using XGBoost with Custom Objective for Expert B
    m_hv = xgb.XGBRegressor(n_estimators=2000, learning_rate=0.02, max_depth=8, random_state=seed)
    m_hv.set_params(objective=asymmetric_mse_loss)
    m_hv.fit(X, y)
    pred_hv = m_hv.predict(X_test)
    
    # --- PHASE 4: Expert Mixture (Soft Gating) ---
    # Combine based on Gater's probability of being a 'Hard' customer
    final_preds = (1 - gater_probs) * pred_bulk + gater_probs * pred_hv
    
    results = {
        'moe_final': final_preds,
        'expert_bulk': pred_bulk,
        'expert_hv': pred_hv,
        'gater_confidence': gater_probs
    }
    
    logger.info("✅ Mixture of Experts training complete.")
    return results
