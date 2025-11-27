import tkinter as tk
from tkinter import ttk, scrolledtext
import joblib
import pandas as pd
from src.features_fusion import FeatureFusionBuilder
from src.preprocess import simple_clean

# --------------------------
# Load Model
# --------------------------
print("Loading ML Model...")
model = joblib.load("outputs/models/xgb.joblib")

# --------------------------
# Recreate FeatureFusionBuilder EXACTLY like training
# --------------------------
print("Rebuilding FeatureFusionBuilder with training config...")
fb = FeatureFusionBuilder(
    use_sbert=True,
    sbert_model_name="sentence-transformers/paraphrase-MiniLM-L6-v2",
    tfidf_title_max=5000,
    tfidf_abs_max=20000,
    svd_components=300,
    batch_size=32
)

# --------------------------
# Load saved TF-IDF, SVD, scaler
# --------------------------
components = joblib.load("outputs/feature_builder.joblib")
fb.tfidf_title = components["tfidf_title"]
fb.tfidf_abs = components["tfidf_abs"]
fb.svd = components["svd"]
fb.scaler = components["scaler"]

# --------------------------
# Prediction Function
# --------------------------
def predict_score():
    title = title_input.get("1.0", tk.END).strip()
    abstract = abstract_input.get("1.0", tk.END).strip()

    if not title or not abstract:
        result_label.config(text="❌ Please enter both Title and Abstract")
        return

    df = pd.DataFrame([{
        "title": simple_clean(title),
        "abstract": simple_clean(abstract)
    }])

    # Build feature matrix using EXACT SAME pipeline
    X, _, _ = fb.build_feature_matrix(
        df,
        fit_tfidf=False,
        return_vectors=False
    )

    score = float(model.predict(X)[0])

    # Interpretation
    if score < 0.2:
        level = "Very Weak Match"
    elif score < 0.4:
        level = "Weak Match"
    elif score < 0.6:
        level = "Moderate Match"
    elif score < 0.8:
        level = "Strong Match"
    else:
        level = "Excellent Match"

    result_label.config(
        text=f"🔢 Score: {score:.4f}\n📊 Relevance: {level}"
    )

# --------------------------
# GUI Layout
# --------------------------
root = tk.Tk()
root.title("Title–Abstract Relevance Predictor")
root.geometry("700x600")

title_lbl = ttk.Label(root, text="Enter Title:", font=("Arial", 12))
title_lbl.pack(pady=5)

title_input = scrolledtext.ScrolledText(root, height=4, width=80)
title_input.pack()

abstract_lbl = ttk.Label(root, text="Enter Abstract:", font=("Arial", 12))
abstract_lbl.pack(pady=5)

abstract_input = scrolledtext.ScrolledText(root, height=10, width=80)
abstract_input.pack()

predict_btn = ttk.Button(root, text="Predict Score", command=predict_score)
predict_btn.pack(pady=20)

result_label = ttk.Label(root, text="", font=("Arial", 14))
result_label.pack(pady=10)

root.mainloop()
