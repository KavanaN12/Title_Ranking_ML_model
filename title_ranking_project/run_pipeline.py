# run_pipeline.py
import argparse
import numpy as np
import pandas as pd
from src.data_loader import build_combined_dataset
from src.preprocess import simple_clean
from src.features_fusion import FeatureFusionBuilder
from src.models import get_default_models
from src.train_eval import kfold_evaluate
from src.utils import save_predictions_df, save_joblib, compute_spearman
import os
import joblib

DEFAULT_DATA_FOLDER = r"D:\aimlTextPr\datasets"

def generate_pseudo_labels_sbert(df, sbert_model_name=None, batch_size=64):
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    model_name = sbert_model_name if sbert_model_name else "sentence-transformers/all-mpnet-base-v2"
    print("Generating pseudo labels with SBERT model:", model_name)
    model = SentenceTransformer(model_name)
    titles = df['title'].tolist()
    abstracts = df['abstract'].tolist()
    # batch safe
    def batch_encode(texts):
        embs = []
        for i in range(0, len(texts), batch_size):
            embs.append(model.encode(texts[i:i+batch_size], show_progress_bar=False))
        return np.vstack(embs)
    t_emb = batch_encode(titles)
    a_emb = batch_encode(abstracts)
    sims = cosine_similarity(t_emb, a_emb).diagonal()
    # scale 0..1
    from sklearn.preprocessing import MinMaxScaler
    sims_scaled = MinMaxScaler().fit_transform(sims.reshape(-1,1)).reshape(-1)
    return sims_scaled

def main(args):
    print("Loading dataset from:", args.datasets_folder)
    df = build_combined_dataset(args.datasets_folder)
    print("Loaded rows:", len(df))
    if len(df) == 0:
        raise SystemExit("No data found. Check datasets folder.")

    # Preprocess
    df['title'] = df['title'].apply(simple_clean)
    df['abstract'] = df['abstract'].apply(simple_clean)

    # SAVE CLEANED DATASET HERE
    import os
    os.makedirs("outputs", exist_ok=True)
    clean_path = os.path.join("outputs", "cleaned_dataset.csv")
    df.to_csv(clean_path, index=False)
    print("Cleaned dataset saved to:", clean_path)


    # Labels: use real label column if provided, otherwise pseudo-labels
    if args.label_column and args.label_column in df.columns:
        y = df[args.label_column].astype(float).values
        print("Using provided label column:", args.label_column)
    else:
        y = generate_pseudo_labels_sbert(df, sbert_model_name=args.sbert_model)
        print("Pseudo-labels generated. sample:", y[:5])

    # feature builder
    fb = FeatureFusionBuilder(use_sbert=args.use_sbert,
                              sbert_model_name=args.sbert_model,
                              tfidf_title_max=args.tfidf_title_max,
                              tfidf_abs_max=args.tfidf_abs_max,
                              svd_components=args.svd_components,
                              batch_size=args.batch_size)

    print("Building fused features (this may take time)...")
    X, feat_names, vectors, scaler = fb.build_feature_matrix(df, fit_tfidf=True, use_cache_path=args.embedding_cache, return_vectors=True)
    print("Feature matrix shape:", X.shape)
    # save feature builder
    fb.save(args.outputs_dir)

    # models
    models = get_default_models(random_state=42)
    # optionally reduce to only xgb and rf
    if args.models_list:
        keep = args.models_list.split(",")
        models = {k:v for k,v in models.items() if k in keep}

    # K-Fold evaluation + final training
    print("Starting K-Fold CV (n_splits=%d)..." % args.n_splits)
    summary = kfold_evaluate(models, X, y, n_splits=args.n_splits, out_dir=os.path.join(args.outputs_dir, "models"))
    print("CV Summary:")
    for k,v in summary.items():
        print(k, v)

    # final predictions
    preds = {}
    for name in models.keys():
        model_path = os.path.join(args.outputs_dir, "models", f"{name}.joblib")
        if os.path.exists(model_path):
            m = joblib.load(model_path)
            preds[name] = m.predict(X)
            r = compute_spearman(y, preds[name])
            print(f"{name} final spearman on full data: {r:.4f}")

    # save predictions
    out = df.copy()
    out['label'] = y
    for k,v in preds.items():
        out[f'pred_{k}'] = v
        out[f'rank_{k}'] = (-v).argsort().argsort()  # 0-based rank (lower = better)
    os.makedirs("outputs", exist_ok=True)
    save_predictions_df(out, os.path.join("outputs", "predictions_all.csv"))

    print("Pipeline complete. Models and artifacts saved to:", args.outputs_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets_folder", type=str, default=DEFAULT_DATA_FOLDER)
    parser.add_argument("--label_column", type=str, default=None)
    parser.add_argument("--use_sbert", action="store_true", help="Include SBERT features (cosine + diffs)")
    parser.add_argument("--sbert_model", type=str, default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--tfidf_title_max", type=int, default=5000)
    parser.add_argument("--tfidf_abs_max", type=int, default=20000)
    parser.add_argument("--svd_components", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--outputs_dir", type=str, default="outputs")
    parser.add_argument("--embedding_cache", type=str, default=None, help="Path to cache embeddings (joblib) to speed reruns")
    parser.add_argument("--models_list", type=str, default=None, help="Comma separated models to run (xgb,rf,ridge,svr)")
    args = parser.parse_args()
    # by default enable SBERT features for best results
    if not args.use_sbert:
        args.use_sbert = True
    main(args)
