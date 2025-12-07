# run_pipeline_unified.py
"""
Unified training pipeline (updated)
- Loads ArXiv / ICML / S2ORC sources (if present)
- Automatically loads new CrossRef training dataset:
    D:/aimlTextPr/datasets/train_real_world_dataset_10000.csv
- Ensures NO overlap with evaluation set:
    D:/aimlTextPr/datasets/real_world_dataset_2000_cleaned.csv
- Cleans text, deduplicates, builds SBERT + fusion features (FeatureFusionBuilder)
- Trains LightGBM with K-Fold and saves artifacts:
    outputs/target_stats.json
    outputs/scaler.joblib
    outputs/models/lgbm.joblib
    outputs/feature_builder.joblib
    outputs/predictions_lgbm.csv
- Usage: python run_pipeline_unified.py
"""

import os
import json
import time
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr
from sentence_transformers import SentenceTransformer
from lightgbm import LGBMRegressor

from src.preprocess import simple_clean
from src.features_fusion import FeatureFusionBuilder

# -------------------------
# CONFIG
# -------------------------
DATASET_FOLDER = "D:/aimlTextPr/datasets"
CROSSREF_TRAIN_PATH = os.path.join(DATASET_FOLDER, "train_real_world_dataset_10000.csv")
EVAL_TEST_PATH = os.path.join(DATASET_FOLDER, "real_world_dataset_2000_cleaned.csv")
OUT_DIR = "outputs"
MODEL_DIR = os.path.join(OUT_DIR, "models")
FEATURE_BUILDER_PATH = os.path.join(OUT_DIR, "feature_builder.joblib")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

SBERT_MODEL = "sentence-transformers/paraphrase-MiniLM-L6-v2"
SEED = 42
N_SPLITS = 5
SBATCH = 64

RANDOM_STATE = SEED
np.random.seed(RANDOM_STATE)

# -------------------------
# Utility helpers
# -------------------------
def load_all_datasets(folder):
    dfs = []
    print("Loading datasets from:", folder)

    # ArXiv (csv)
    arxiv_path = os.path.join(folder, "arxiv_data.csv")
    if os.path.exists(arxiv_path):
        print("→ Loading ArXiv:", arxiv_path)
        dfs.append(pd.read_csv(arxiv_path))

    # ICML parquet files
    for fn in os.listdir(folder):
        if fn.lower().startswith("icml") and fn.lower().endswith(".parquet"):
            p = os.path.join(folder, fn)
            print("→ Loading ICML parquet:", p)
            dfs.append(pd.read_parquet(p))

    # S2ORC metadata files (jsonl)
    meta_dir = os.path.join(folder, "metadata")
    if os.path.exists(meta_dir):
        for fn in os.listdir(meta_dir):
            if fn.endswith(".jsonl"):
                p = os.path.join(meta_dir, fn)
                print("→ Loading S2ORC jsonl:", p)
                dfs.append(pd.read_json(p, lines=True))

    print("Total datasets discovered:", len(dfs))
    return dfs

def read_crossref_train(path):
    if os.path.exists(path):
        print("→ Loading CrossRef train dataset:", path)
        return pd.read_csv(path)
    else:
        print("→ CrossRef train dataset not found at:", path)
        return pd.DataFrame(columns=["title","abstract","expected","expected_label","sim_score","venue"])

def remove_test_overlap(df_train, eval_path):
    if not os.path.exists(eval_path):
        print("Evaluation file not found, skipping overlap check:", eval_path)
        return df_train
    eval_df = pd.read_csv(eval_path)
    eval_titles = set(eval_df['title'].astype(str).str.strip().str.lower())
    eval_abstracts = set(eval_df['abstract'].astype(str).str.strip().str.lower())
    initial = len(df_train)
    mask_title = df_train['title'].astype(str).str.strip().str.lower().isin(eval_titles)
    mask_abs = df_train['abstract'].astype(str).str.strip().str.lower().isin(eval_abstracts)
    df_train_clean = df_train[~(mask_title | mask_abs)].reset_index(drop=True)
    removed = initial - len(df_train_clean)
    print(f"Removed {removed} rows that overlapped with evaluation set.")
    return df_train_clean

def ensure_columns(df, title_cols=("title","paper_title"), abstract_cols=("abstract","paper_abstract")):
    # Normalize column names to 'title' and 'abstract'
    for alt in title_cols:
        if alt in df.columns and 'title' not in df.columns:
            df = df.rename(columns={alt:"title"})
    for alt in abstract_cols:
        if alt in df.columns and 'abstract' not in df.columns:
            df = df.rename(columns={alt:"abstract"})
    # keep only title & abstract and drop rows missing them
    if 'title' in df.columns and 'abstract' in df.columns:
        df = df[['title','abstract']].dropna().reset_index(drop=True)
    else:
        # create empty frame with two cols
        df = df.copy()
    return df

def batch_encode(model, texts, batch_size=SBATCH):
    embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding", unit="batch"):
        batch = texts[i:i+batch_size]
        embs.append(model.encode(batch, convert_to_numpy=True))
    return np.vstack(embs) if len(embs) else np.zeros((0, model.get_sentence_embedding_dimension()))

# -------------------------
# 1. Load & merge datasets
# -------------------------
print("\n=== STEP 1: LOAD & MERGE DATASETS ===")
datasets = load_all_datasets(DATASET_FOLDER)
if datasets:
    df_all = pd.concat(datasets, ignore_index=True, sort=False)
else:
    df_all = pd.DataFrame(columns=["title","abstract"])

# load crossref training and add
df_crossref = read_crossref_train(CROSSREF_TRAIN_PATH)

# Normalize column names & pick title/abstract
df_all = ensure_columns(df_all)
df_crossref = ensure_columns(df_crossref)

# Merge and remove overlap vs evaluation set
df_merged = pd.concat([df_all, df_crossref], ignore_index=True, sort=False)
print("Total merged rows (before cleaning):", len(df_merged))

# -------------------------
# 2. Clean, drop NA, dedupe
# -------------------------
print("\n=== STEP 2: CLEANING & DEDUPLICATION ===")
# rename possible paper_title/paper_abstract if present
df_merged = ensure_columns(df_merged)

# Drop rows with empty title/abstract
df_merged['title'] = df_merged['title'].astype(str)
df_merged['abstract'] = df_merged['abstract'].astype(str)
df_merged = df_merged[(df_merged['title'].str.strip() != "") & (df_merged['abstract'].str.strip() != "")]
initial_count = len(df_merged)

# Remove duplicates by exact title+abstract
df_merged['__sig'] = (df_merged['title'].str.strip().str.lower() + " ||| " + df_merged['abstract'].str.strip().str.lower())
df_merged = df_merged.drop_duplicates(subset='__sig').drop(columns='__sig').reset_index(drop=True)

# Remove any samples that overlap with evaluation test set
df_merged = remove_test_overlap(df_merged, EVAL_TEST_PATH)

print(f"Rows after cleaning/dedup/overlap removal: {len(df_merged)} (was {initial_count})")

# Optionally shuffle
df_merged = df_merged.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)

# -------------------------
# 3. Take subset for training if extremely large (safety)
# -------------------------
TARGET_MIN_ROWS = 10000
if len(df_merged) < TARGET_MIN_ROWS:
    print(f"Warning: merged dataset has only {len(df_merged)} rows (< {TARGET_MIN_ROWS}). Proceeding with available data.")
else:
    # If merged is larger, keep up to 15000 to limit memory, prefer newest samples first (already shuffled)
    df_merged = df_merged.iloc[:15000].reset_index(drop=True)
    print("Trimmed merged dataset to top 15000 for processing.")

print("Final training candidate size:", len(df_merged))

# -------------------------
# 4. Clean text fields using your preprocess.simple_clean
# -------------------------
print("\n=== STEP 3: TEXT CLEANING ===")
df_merged['title'] = df_merged['title'].astype(str).apply(simple_clean)
df_merged['abstract'] = df_merged['abstract'].astype(str).apply(simple_clean)

# -------------------------
# 5. SBERT encoding + target build
# -------------------------
print("\n=== STEP 4: SBERT ENCODING & TARGET BUILDING ===")
sbert = SentenceTransformer(SBERT_MODEL)

titles = df_merged['title'].tolist()
abstracts = df_merged['abstract'].tolist()

title_emb = batch_encode(sbert, titles, batch_size=SBATCH)
abs_emb = batch_encode(sbert, abstracts, batch_size=SBATCH)

# Safe cosine
num = np.sum(title_emb * abs_emb, axis=1)
den = np.linalg.norm(title_emb, axis=1) * np.linalg.norm(abs_emb, axis=1)
den = np.where(den == 0, 1e-9, den)
sbert_cos = num / den

# Map to your expected scale
target_raw = 0.10 + np.clip(sbert_cos, 0, 1) * 0.85
df_merged['target_raw'] = target_raw
print("Target (raw) stats:", float(df_merged['target_raw'].mean()), float(df_merged['target_raw'].std()))

# -------------------------
# 6. Feature fusion (TF-IDF, SVD, SBERT features, etc.)
# -------------------------
print("\n=== STEP 5: BUILD FEATURE MATRIX WITH FeatureFusionBuilder ===")
fb = FeatureFusionBuilder(
    use_sbert=True,
    sbert_model_name=SBERT_MODEL,
    batch_size=SBATCH
)

# Fit TF-IDF/SVD on training merged set (fit_tfidf=True)
X, feat_names, _ = fb.build_feature_matrix(df_merged, fit_tfidf=True, return_vectors=False)
X = np.array(X)
print("Feature matrix shape:", X.shape)

# Save feature builder components for reuse
fb.save(OUT_DIR)
joblib.dump(fb, FEATURE_BUILDER_PATH)  # convenience; fb.save already saves components in OUT_DIR

# -------------------------
# 7. Target normalization & scaler
# -------------------------
print("\n=== STEP 6: TARGET NORMALIZATION & SCALING ===")
t_mean = float(df_merged['target_raw'].mean())
t_std = float(df_merged['target_raw'].std(ddof=0) if df_merged['target_raw'].std(ddof=0) > 0 else 1.0)
y_norm = (df_merged['target_raw'].values - t_mean) / t_std

json.dump({"mean": t_mean, "std": t_std}, open(os.path.join(OUT_DIR, "target_stats.json"), "w"))

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, os.path.join(OUT_DIR, "scaler.joblib"))
print("Saved scaler to outputs/scaler.joblib")

# -------------------------
# 8. Train LightGBM with K-Fold
# -------------------------
print("\n=== STEP 7: TRAIN LIGHTGBM (K-FOLD) ===")
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
rmse_list, spearman_list = [], []

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_scaled), 1):
    print(f"\n-- Fold {fold}")
    model = LGBMRegressor(
        learning_rate=0.05,
        n_estimators=1500,
        num_leaves=32,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    model.fit(X_scaled[tr_idx], y_norm[tr_idx])
    pred_val = model.predict(X_scaled[val_idx]) * t_std + t_mean
    rmse = np.sqrt(mean_squared_error(df_merged['target_raw'].values[val_idx], pred_val))
    sp = float(spearmanr(df_merged['target_raw'].values[val_idx], pred_val)[0])
    print(f"Fold {fold} RMSE: {rmse:.4f}  Spearman: {sp:.4f}")
    rmse_list.append(rmse)
    spearman_list.append(sp)

print("\nLightGBM CV mean RMSE:", np.mean(rmse_list))
print("LightGBM CV mean Spearman:", np.mean(spearman_list))

# -------------------------
# 9. Train final model on all data & save
# -------------------------
print("\n=== STEP 8: TRAIN FINAL MODEL & SAVE ARTIFACTS ===")
final_model = LGBMRegressor(
    learning_rate=0.03,
    n_estimators=2000,
    num_leaves=32,
    random_state=RANDOM_STATE,
    n_jobs=-1
)
final_model.fit(X_scaled, y_norm)

joblib.dump(final_model, os.path.join(MODEL_DIR, "lgbm.joblib"))
print("Saved LightGBM model to:", os.path.join(MODEL_DIR, "lgbm.joblib"))

# -------------------------
# 10. Save predictions & metadata
# -------------------------
print("\n=== STEP 9: SAVE PREDICTIONS & METADATA ===")
df_merged['pred_lgbm'] = final_model.predict(X_scaled) * t_std + t_mean
pred_out = os.path.join(OUT_DIR, "predictions_lgbm.csv")
df_merged.to_csv(pred_out, index=False)
print("Saved predictions to:", pred_out)

meta = {
    "rows_used": len(df_merged),
    "feature_count": X.shape[1],
    "sbert_model": SBERT_MODEL,
    "target_mean": t_mean,
    "target_std": t_std
}
with open(os.path.join(OUT_DIR, "pipeline_meta.json"), "w") as f:
    json.dump(meta, f, indent=2)

print("\n==============================")
print("Unified pipeline complete — ALL ARTIFACTS SAVED")
print("You can now run:")
print("   python model_test_lgb.py")
print("   python bulk_test.py")
print("==============================")
