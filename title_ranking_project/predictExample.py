import joblib
import pandas as pd
from src.features_fusion import FeatureFusionBuilder
from src.preprocess import simple_clean

model = joblib.load("outputs/models/xgb.joblib")

fb = FeatureFusionBuilder(
    use_sbert=True,
    sbert_model_name="sentence-transformers/paraphrase-MiniLM-L6-v2",
    tfidf_title_max=5000,
    tfidf_abs_max=20000,
    svd_components=300,
    batch_size=32
)

components = joblib.load("outputs/feature_builder.joblib")
fb.tfidf_title = components["tfidf_title"]
fb.tfidf_abs = components["tfidf_abs"]
fb.svd = components["svd"]
fb.scaler = components["scaler"]

title = "Neural Networks for Image Recognition"
abstract = "This paper proposes deep learning-based architectures for visual classification tasks."

df = pd.DataFrame([{
    "title": simple_clean(title),
    "abstract": simple_clean(abstract)
}])

X, _, _ = fb.build_feature_matrix(df, fit_tfidf=False, return_vectors=False)
score = float(model.predict(X)[0])

print("Predicted Score:", score)
