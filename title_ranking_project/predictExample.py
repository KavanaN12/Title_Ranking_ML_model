import joblib
import pandas as pd
from src.features_fusion import FeatureFusionBuilder
from src.preprocess import simple_clean

# -----------------------------
# 1. Load your trained model
# -----------------------------
model = joblib.load("outputs/models/xgb.joblib")

# -----------------------------
# 2. Recreate the FeatureFusionBuilder
# -----------------------------
# Make sure model names & params match your training config
fb = FeatureFusionBuilder(
    use_sbert=True,
    sbert_model_name="sentence-transformers/paraphrase-MiniLM-L6-v2",
    tfidf_title_max=5000,
    tfidf_abs_max=20000,
    svd_components=300,
    batch_size=32
)

# -----------------------------
# 3. Load feature components saved during training
# -----------------------------
components = joblib.load("outputs/feature_builder.joblib")

fb.tfidf_title = components["tfidf_title"]
fb.tfidf_abs = components["tfidf_abs"]
fb.svd = components["svd"]
fb.scaler = components["scaler"]

# -----------------------------
# 4. Input example (title + abstract)
# -----------------------------
title = "Neural Networks for Image Recognition"
abstract = "This paper proposes deep learning-based architectures for visual classification tasks."

df = pd.DataFrame([{
    "title": simple_clean(title),
    "abstract": simple_clean(abstract)
}])

# -----------------------------
# 5. Build feature matrix for NEW DATA
#    (fit_tfidf=False because we must use the saved TF-IDF)
# -----------------------------
X, _, _ = fb.build_feature_matrix(
    df, 
    fit_tfidf=False,
    use_cache_path=None,
    return_vectors=False
)

# -----------------------------
# 6. Predict score
# -----------------------------
score = model.predict(X)[0]

print("\nPredicted Score:", score)
