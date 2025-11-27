import argparse
import os
import joblib
import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.data_loader import load_datasets
from src.preprocess import simple_clean
from src.features_fusion import FeatureFusionBuilder
from src.models import get_default_models
from src.utils import save_predictions_df
from src.train_eval import train_and_evaluate


# ================================
# EXTRA METRICS
# ================================
def compute_metrics(y_true, y_pred):
    """Compute all regression metrics."""
    spearman = pd.Series(y_true).corr(pd.Series(y_pred), method="spearman")
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # Avoid division-by-zero for MAPE
    y_true_safe = np.where(y_true == 0, 1e-8, y_true)
    mape = np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100

    return {
        "spearman": float(spearman),
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
        "mape": float(mape),
    }


def main(args):
    print("Loading dataset from:", args.datasets_folder)
    df = load_datasets(args.datasets_folder)
    print("Loaded rows:", len(df))

    # Clean text
    df["title"] = df["title"].apply(simple_clean)
    df["abstract"] = df["abstract"].apply(simple_clean)

    # Save cleaned dataset
    os.makedirs(args.outputs_dir, exist_ok=True)
    cleaned_path = os.path.join(args.outputs_dir, "cleaned_dataset.csv")
    df.to_csv(cleaned_path, index=False)
    print("Cleaned dataset saved to:", cleaned_path)

    # ------------------------------------------------
    # PSEUDO LABELS (SBERT Cosine Similarity)
    # ------------------------------------------------
    from sentence_transformers import SentenceTransformer
    print(f"Generating pseudo labels with SBERT model: {args.sbert_model}")

    sbert = SentenceTransformer(args.sbert_model)
    title_emb = sbert.encode(df["title"].tolist(), convert_to_numpy=True)
    abs_emb = sbert.encode(df["abstract"].tolist(), convert_to_numpy=True)

    # Cosine similarity manually
    y = (title_emb * abs_emb).sum(axis=1) / (
        (title_emb**2).sum(axis=1)**0.5 * (abs_emb**2).sum(axis=1)**0.5
    )

    print("Pseudo-labels generated. Sample:", y[:5])

    # ------------------------------------------------
    # FEATURE FUSION
    # ------------------------------------------------
    print("Building fused features (this may take time)...")
    fb = FeatureFusionBuilder(
        use_sbert=True,
        sbert_model_name=args.sbert_model,
        tfidf_title_max=5000,
        tfidf_abs_max=20000,
        svd_components=300,
        batch_size=32
    )

    X, title_vecs, abs_vecs = fb.build_feature_matrix(
        df,
        fit_tfidf=True,
        return_vectors=True
    )

    print("Feature matrix shape:", X.shape)

    # Save TF-IDF + SVD + Scaler
    feat_components = {
        "tfidf_title": fb.tfidf_title,
        "tfidf_abs": fb.tfidf_abs,
        "svd": fb.svd,
        "scaler": fb.scaler
    }
    feat_path = os.path.join(args.outputs_dir, "feature_builder.joblib")
    joblib.dump(feat_components, feat_path)
    print("Saved feature components to:", feat_path)

    # ------------------------------------------------
    # TRAINING + K-FOLD CV
    # ------------------------------------------------
    print(f"Starting K-Fold CV (n_splits={args.n_splits})...")

    models = get_default_models(random_state=42)
    results, trained_models = train_and_evaluate(
        X, y,
        models=models,
        n_splits=args.n_splits
    )

    # -----------------------------
    # PRINT CV SUMMARY
    # -----------------------------
    print("\n====================")
    print("CROSS-VALIDATION SUMMARY")
    print("====================")

    for name, score_data in results.items():
        print(f"\nModel: {name}")
        print(f" Spearman (mean): {score_data['spearman_mean']:.4f}")
        print(f" Spearman (std):  {score_data['spearman_std']:.4f}")

    # ------------------------------------------------
    # FULL DATA METRICS (Train on 100% X, evaluate same X)
    # ------------------------------------------------
    print("\n====================")
    print("FULL DATA EVALUATION (Train on 100% of data)")
    print("====================\n")

    full_metrics = {}

    for model_name, model in trained_models.items():
        print(f"Model: {model_name}")

        y_pred = model.predict(X)
        metrics = compute_metrics(y, y_pred)
        full_metrics[model_name] = metrics

        print(f" Spearman: {metrics['spearman']:.4f}")
        print(f" RMSE:     {metrics['rmse']:.4f}")
        print(f" MAE:      {metrics['mae']:.4f}")
        print(f" R²:       {metrics['r2']:.4f}")
        print(f" MAPE:     {metrics['mape']:.2f}%\n")

    # ------------------------------------------------
    # SAVE TRAINED MODELS
    # ------------------------------------------------
    for model_name, model in trained_models.items():
        model_path = os.path.join(args.outputs_dir, "models", f"{model_name}.joblib")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model, model_path)
        print(f"Saved model: {model_path}")

    # ------------------------------------------------
    # SAVE FINAL PREDICTIONS
    # ------------------------------------------------
    pred_df = df.copy()
    for model_name, model in trained_models.items():
        pred_df[f"pred_{model_name}"] = model.predict(X)

    save_predictions_df(pred_df, os.path.join(args.outputs_dir, "predictions_all.csv"))

    print("\nPipeline complete. Models and artifacts saved to:", args.outputs_dir)


# ================================
# MAIN SCRIPT
# ================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--datasets_folder", type=str, required=True)
    parser.add_argument("--use_sbert", action="store_true")
    parser.add_argument("--sbert_model", type=str, default="sentence-transformers/paraphrase-MiniLM-L6-v2")
    parser.add_argument("--embedding_cache", type=str, default=None)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--outputs_dir", type=str, default="outputs")

    args = parser.parse_args()
    main(args)
