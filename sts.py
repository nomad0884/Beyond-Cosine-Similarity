"""STS 모델 래퍼 — 코사인 유사도 기반 거시적 필터링"""

import numpy as np
from sentence_transformers import SentenceTransformer


class STSModel:
    def __init__(self, model_name="jhgan/ko-sbert-sts"):
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        print(f"[STS] 모델 로드: {model_name}")

    def encode(self, text):
        return self.model.encode(text, convert_to_numpy=True)

    def encode_batch(self, texts):
        return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

    def cosine_similarity(self, vec1, vec2):
        n1 = np.linalg.norm(vec1)
        n2 = np.linalg.norm(vec2)
        if n1 == 0 or n2 == 0:
            return 0.0
        return float(np.dot(vec1, vec2) / (n1 * n2))

    def cosine_similarity_batch(self, query_vec, doc_matrix):
        query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
        doc_norms = doc_matrix / (np.linalg.norm(doc_matrix, axis=1, keepdims=True) + 1e-10)
        return np.dot(doc_norms, query_norm)
