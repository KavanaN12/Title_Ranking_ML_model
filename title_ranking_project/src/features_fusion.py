# src/features_fusion.py
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
import joblib
import os

DEFAULT_SBERT = "sentence-transformers/all-mpnet-base-v2"

class FeatureFusionBuilder:
    def __init__(self,
                 use_sbert=True,
                 sbert_model_name=DEFAULT_SBERT,
                 tfidf_title_max=5000,
                 tfidf_abs_max=20000,
                 svd_components=300,
                 batch_size=64,
                 sbert_pooling=True):
        self.use_sbert = use_sbert
        self.sbert_model_name = sbert_model_name
        self.batch_size = batch_size
        self.tfidf_title_max = tfidf_title_max
        self.tfidf_abs_max = tfidf_abs_max
        self.svd_components = svd_components
        self.tfidf_title = None
        self.tfidf_abs = None
        self.svd = None
        self.scaler = None
        if self.use_sbert:
            print("Loading SBERT:", sbert_model_name)
            self.sbert = SentenceTransformer(sbert_model_name)
        else:
            self.sbert = None

    def fit_tfidf(self, titles, abstracts):
        print("Fitting TF-IDF (title) max_features=", self.tfidf_title_max)
        self.tfidf_title = TfidfVectorizer(max_features=self.tfidf_title_max, stop_words='english', ngram_range=(1,2))
        print("Fitting TF-IDF (abstract) max_features=", self.tfidf_abs_max)
        self.tfidf_abs = TfidfVectorizer(max_features=self.tfidf_abs_max, stop_words='english', ngram_range=(1,2))

        T_title = self.tfidf_title.fit_transform(titles)
        T_abs = self.tfidf_abs.fit_transform(abstracts)

        if self.svd_components and self.svd_components < T_abs.shape[1]:
            print("Fitting SVD to reduce abstract TF-IDF to", self.svd_components)
            self.svd = TruncatedSVD(n_components=min(self.svd_components, T_abs.shape[1]-1), random_state=42)
            T_abs_reduced = self.svd.fit_transform(T_abs)
        else:
            T_abs_reduced = T_abs

        return T_title, T_abs_reduced

    def transform_tfidf(self, titles, abstracts):
        T_title = self.tfidf_title.transform(titles)
        T_abs = self.tfidf_abs.transform(abstracts)
        if self.svd is not None:
            T_abs_reduced = self.svd.transform(T_abs)
        else:
            T_abs_reduced = T_abs
        return T_title, T_abs_reduced

    def batch_encode(self, texts):
        embs = []
        for i in tqdm(range(0, len(texts), self.batch_size), desc="SBERT batches"):
            batch = texts[i:i+self.batch_size]
            embs.append(self.sbert.encode(batch, show_progress_bar=False))
        return np.vstack(embs)

    def compute_embeddings(self, titles, abstracts, use_cache_path=None):
        vectors = {}
        if use_cache_path and os.path.exists(use_cache_path):
            print("Loading cached embeddings from", use_cache_path)
            data = joblib.load(use_cache_path)
            return data['title_emb'], data['abstract_emb']

        if not self.use_sbert:
            return None, None

        title_emb = self.batch_encode(titles)
        abstract_emb = self.batch_encode(abstracts)
        if use_cache_path:
            joblib.dump({'title_emb': title_emb, 'abstract_emb': abstract_emb}, use_cache_path)
        return title_emb, abstract_emb

    def build_feature_matrix(self, df, fit_tfidf=True, use_cache_path=None, return_vectors=False):
        titles = df['title'].tolist()
        abstracts = df['abstract'].tolist()

        if fit_tfidf:
            T_title, T_abs = self.fit_tfidf(titles, abstracts)
        else:
            T_title, T_abs = self.transform_tfidf(titles, abstracts)

        # dense arrays (careful with memory - will convert to float32)
        try:
            T_title_arr = T_title.toarray().astype(np.float32)
        except:
            T_title_arr = T_title

        if hasattr(T_abs, "toarray"):
            T_abs_arr = T_abs.toarray().astype(np.float32)
        else:
            T_abs_arr = T_abs.astype(np.float32)

        # embeddings
        title_emb, abstract_emb = self.compute_embeddings(titles, abstracts, use_cache_path=use_cache_path)

        features = []
        feat_names = []

        # SBERT cosine similarity
        if title_emb is not None and abstract_emb is not None:
            cos = cosine_similarity(title_emb, abstract_emb).diagonal().reshape(-1,1)
            features.append(cos.astype(np.float32))
            feat_names.append("sbert_cos")

        # TF-IDF cosine (title vs abstract)
        try:
            tfidf_cos = cosine_similarity(T_title_arr, T_abs_arr).diagonal().reshape(-1,1)
            features.append(tfidf_cos.astype(np.float32))
            feat_names.append("tfidf_cos")
        except Exception:
            pass

        # length features
        from src.preprocess import title_length, abstract_length, word_overlap_ratio, rouge_l_score
        len_title = np.array([title_length(t) for t in titles]).reshape(-1,1).astype(np.float32)
        len_abs = np.array([abstract_length(a) for a in abstracts]).reshape(-1,1).astype(np.float32)
        features.append(len_title); feat_names.append("title_len")
        features.append(len_abs); feat_names.append("abstract_len")

        # overlap & rouge_l
        overlap = np.array([word_overlap_ratio(t,a) for t,a in zip(titles, abstracts)]).reshape(-1,1).astype(np.float32)
        features.append(overlap); feat_names.append("word_overlap")
        rouge_scores = np.array([rouge_l_score(t,a) for t,a in zip(titles, abstracts)]).reshape(-1,1).astype(np.float32)
        features.append(rouge_scores); feat_names.append("rouge_l")

        # Optionally include raw SBERT embeddings aggregated stats (mean, max) if using tree model
        if title_emb is not None and abstract_emb is not None:
            # elementwise absolute diff and dot product
            abs_diff = np.abs(title_emb - abstract_emb).astype(np.float32)
            dot = (title_emb * abstract_emb).sum(axis=1).reshape(-1,1).astype(np.float32)
            # reduce abs_diff by mean to keep feature size small
            abs_diff_mean = abs_diff.mean(axis=1).reshape(-1,1).astype(np.float32)
            features.append(abs_diff_mean); feat_names.append("sbert_absdiff_mean")
            features.append(dot); feat_names.append("sbert_dot")

        # concat features
        X_numeric = np.hstack(features)
        # scale
        self.scaler = MinMaxScaler()
        X_scaled = self.scaler.fit_transform(X_numeric)

        if return_vectors:
            vectors = {'title_emb': title_emb, 'abstract_emb': abstract_emb, 'T_title_arr': T_title_arr, 'T_abs_arr': T_abs_arr}
            return X_scaled, feat_names, vectors, self.scaler
        return X_scaled, feat_names, self.scaler

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        joblib.dump({
            'tfidf_title': self.tfidf_title,
            'tfidf_abs': self.tfidf_abs,
            'svd': self.svd,
            'scaler': self.scaler
        }, os.path.join(path, "feature_builder.joblib"))
