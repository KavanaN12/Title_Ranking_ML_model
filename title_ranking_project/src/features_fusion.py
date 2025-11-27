# src/features_fusion.py

import numpy as np
import joblib
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler
from sentence_transformers import SentenceTransformer, util

class FeatureFusionBuilder:
    def __init__(self,
                 use_sbert=True,
                 sbert_model_name="sentence-transformers/paraphrase-MiniLM-L6-v2",
                 tfidf_title_max=5000,
                 tfidf_abs_max=20000,
                 svd_components=300,
                 batch_size=32):

        self.use_sbert = use_sbert
        self.sbert_model_name = sbert_model_name
        self.batch_size = batch_size

        # TF-IDF models
        self.tfidf_title = None
        self.tfidf_abs = None

        # Dimensionality reducer
        self.svd = None

        # Scaler (IMPORTANT)
        self.scaler = None

        self.sbert = None  # lazy-loaded

    # -------------------------
    # SBERT embedding
    # -------------------------
    def load_sbert(self):
        if self.sbert is None:
            print(f"Loading SBERT: {self.sbert_model_name}")
            self.sbert = SentenceTransformer(self.sbert_model_name)
        return self.sbert

    def compute_embeddings(self, texts, use_cache_path=None):
        """
        Compute SBERT embeddings with caching support.
        """
        model = self.load_sbert()

        if use_cache_path is not None:
            try:
                cached = joblib.load(use_cache_path)
                if "title_emb" in cached and "abstract_emb" in cached:
                    print(f"Loading cached embeddings from {use_cache_path}")
                    return cached["title_emb"], cached["abstract_emb"]
            except:
                pass  # no cache; continue

        print("SBERT batches:")
        emb = []
        for i in tqdm(range(0, len(texts), self.batch_size)):
            batch = texts[i:i + self.batch_size].tolist()
            emb.append(model.encode(batch, convert_to_numpy=True))

        embeddings = np.vstack(emb)
        return embeddings

    # -------------------------
    # Main Feature Builder
    # -------------------------
    def build_feature_matrix(self, df, fit_tfidf=True, use_cache_path=None, return_vectors=False):
        """
        IMPORTANT FIX:
        - During training: fit_tfidf=True → fit TF-IDF, SVD, SCALER.
        - During inference: fit_tfidf=False → use ONLY transform(), NEVER fit().
        """

        titles = df["title"].tolist()
        abstracts = df["abstract"].tolist()

        # ============= SBERT FEATURES =============
        if self.use_sbert:
            # embeddings recomputed for inference
            title_emb = self.compute_embeddings(df["title"])
            abs_emb = self.compute_embeddings(df["abstract"])

            sbert_cos = np.sum(title_emb * abs_emb, axis=1) / (
                np.linalg.norm(title_emb, axis=1) * np.linalg.norm(abs_emb, axis=1)
            )

            sbert_absdiff = np.mean(np.abs(title_emb - abs_emb), axis=1)
            sbert_dot = np.sum(title_emb * abs_emb, axis=1)
        else:
            sbert_cos = np.zeros(len(df))
            sbert_absdiff = np.zeros(len(df))
            sbert_dot = np.zeros(len(df))

        # ============= TF-IDF FEATURES =============
        if fit_tfidf:
            self.tfidf_title = TfidfVectorizer(max_features=5000)
            title_vecs = self.tfidf_title.fit_transform(titles)

            self.tfidf_abs = TfidfVectorizer(max_features=20000)
            abs_vecs = self.tfidf_abs.fit_transform(abstracts)

            self.svd = TruncatedSVD(n_components=300)
            abs_svd = self.svd.fit_transform(abs_vecs)
        else:
            title_vecs = self.tfidf_title.transform(titles)
            abs_vecs = self.tfidf_abs.transform(abstracts)
            abs_svd = self.svd.transform(abs_vecs)

        from sklearn.metrics.pairwise import cosine_similarity
        tfidf_cosine = np.zeros(len(df))



        # ============= STRUCTURAL FEATURES =============
        title_len = np.array([len(t.split()) for t in titles])
        abs_len = np.array([len(a.split()) for a in abstracts])

        # Simple overlap
        def overlap(t, a):
            t_s = set(t.split())
            a_s = set(a.split())
            if len(t_s) == 0: 
                return 0
            return len(t_s & a_s) / len(t_s)

        word_overlap = np.array([overlap(t, a) for t, a in zip(titles, abstracts)])

        # ============= MERGE ALL FEATURES =============
        X_numeric = np.vstack([
            sbert_cos,
            tfidf_cosine,
            word_overlap,
            sbert_absdiff,
            sbert_dot,
            title_len,
            abs_len
        ]).T

        # ============= FINAL SCALING FIX =============
        if fit_tfidf:
            self.scaler = MinMaxScaler()
            X_scaled = self.scaler.fit_transform(X_numeric)
        else:
            X_scaled = self.scaler.transform(X_numeric)

        if return_vectors:
            return X_scaled, title_vecs, abs_vecs

        return X_scaled, None, None
