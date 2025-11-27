import pandas as pd
import joblib
from src.features_fusion import FeatureFusionBuilder
from src.preprocess import simple_clean

# --------------------------
# Load model and feature builder
# --------------------------
model = joblib.load("outputs/models/xgb.joblib")
components = joblib.load("outputs/feature_builder.joblib")

# Rebuild feature fusion builder EXACTLY like training
fb = FeatureFusionBuilder(
    use_sbert=True,
    sbert_model_name="sentence-transformers/paraphrase-MiniLM-L6-v2",
    tfidf_title_max=5000,
    tfidf_abs_max=20000,
    svd_components=300,
    batch_size=32
)

fb.tfidf_title = components["tfidf_title"]
fb.tfidf_abs = components["tfidf_abs"]
fb.svd = components["svd"]
fb.scaler = components["scaler"]

# --------------------------
# Load evaluation dataset
# --------------------------
data = [
    [
        "A Deep Learning Framework for Medical Image Classification",
        "Deep learning architectures such as convolutional neural networks have shown excellent performance in medical image classification. This paper proposes an optimized CNN for CT/MRI analysis, outperforming earlier baselines.",
        0.88, 0.95
    ],
    [
        "Transformers for Text Summarization",
        "We explore BART and T5 models for abstractive summarization and compare them with RNN-based approaches. Experiments show strong improvements in quality and factual accuracy.",
        0.75, 0.85
    ],
    [
        "Graph Neural Networks for Molecular Prediction",
        "We study supervised learning methods for predicting molecular properties such as toxicity. Traditional fingerprint methods are compared with lightweight neural architectures.",
        0.55, 0.65
    ],
    [
        "Optimization Algorithms for Sparse Linear Systems",
        "This paper presents a new dialogue generation model using transformer-based attention for context-aware response generation.",
        0.30, 0.45
    ],
    [
        "Quantum Entanglement in Multi-Photon Systems",
        "We introduce a new sentiment analysis dataset of Twitter posts annotated for emotion classification using LSTM and transformer models.",
        0.10, 0.22
    ],
    [
        "Climate Change Impacts on Ocean Temperatures",
        "A fast integer factorization algorithm based on number theory is introduced, offering improvements in computational complexity for cryptography.",
        0.01, 0.08
    ],
    [
        "Neural Networks for Traffic Flow Prediction",
        "This work proposes a hybrid CNN-GRU architecture for forecasting highway traffic patterns using temporal-spatial sensor data. Experiments demonstrate state-of-the-art performance.",
        0.70, 0.85
    ],
    [
        "Reinforcement Learning for Autonomous Drone Navigation",
        "We propose a DRL algorithm using actor-critic networks trained in simulation to perform collision-free drone navigation in uncertain environments.",
        0.80, 0.92
    ]
]

df = pd.DataFrame(data, columns=["title", "abstract", "min_exp", "max_exp"])
df["title"] = df["title"].apply(simple_clean)
df["abstract"] = df["abstract"].apply(simple_clean)

# --------------------------
# Predict using your trained model
# --------------------------
preds = []

for idx, row in df.iterrows():
    sample_df = pd.DataFrame([{
        "title": row["title"],
        "abstract": row["abstract"]
    }])
    
    X, _, _ = fb.build_feature_matrix(sample_df, fit_tfidf=False, return_vectors=False)
    score = float(model.predict(X)[0])

    preds.append(score)

df["pred_score"] = preds

# --------------------------
# Compute accuracy
# --------------------------
correct = 0
for idx, row in df.iterrows():
    if row["pred_score"] >= row["min_exp"] and row["pred_score"] <= row["max_exp"]:
        correct += 1

accuracy = (correct / len(df)) * 100

# --------------------------
# Print Results
# --------------------------
print("\n===== MODEL ACCURACY TEST =====\n")
print(df[["title", "pred_score", "min_exp", "max_exp"]])
print("\nAccuracy:", round(accuracy, 2), "%")
print("\n===============================\n")
