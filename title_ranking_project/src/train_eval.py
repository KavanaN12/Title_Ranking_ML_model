# src/train_eval.py
import numpy as np
from sklearn.model_selection import KFold
from scipy.stats import spearmanr
import joblib
import os
from sklearn.base import clone

def safe_spearman(y_true, y_pred):
    try:
        r = spearmanr(y_true, y_pred).correlation
        if np.isnan(r):
            return 0.0
        return float(r)
    except:
        return 0.0

def kfold_evaluate(models_dict, X, y, n_splits=5, out_dir="outputs/models"):
    os.makedirs(out_dir, exist_ok=True)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = {name: [] for name in models_dict.keys()}
    fold = 0
    for train_idx, val_idx in kf.split(X):
        fold += 1
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        for name, model in models_dict.items():
            m = clone(model)
            m.fit(X_tr, y_tr)
            preds = m.predict(X_val)
            r = safe_spearman(y_val, preds)
            results[name].append(r)
            print(f"Fold {fold} - {name} spearman: {r:.4f}")
    # compute mean/std
    summary = {}
    for name, scores in results.items():
        arr = np.array(scores)
        summary[name] = {"mean_spearman": float(arr.mean()), "std_spearman": float(arr.std())}
    # After CV, train on full data and save final models
    for name, model in models_dict.items():
        print("Training final model on full data:", name)
        final_model = clone(model)
        final_model.fit(X, y)
        joblib.dump(final_model, os.path.join(out_dir, f"{name}.joblib"))
    return summary
