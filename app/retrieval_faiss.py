import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class FaissStore:
    def __init__(self, index_dir=None, embedding_model_id="sentence-transformers/all-MiniLM-L6-v2"):
        # Store index 
        self.index_dir = index_dir or "data/index"
        os.makedirs(self.index_dir, exist_ok=True)

        # embedding model
        self.embedder = SentenceTransformer(embedding_model_id)

        # Faiss index 
        self.index = None
        self.metas = []

    def add(self, texts, metas=None):
        """Add new documents to the FAISS index."""
        if metas is None:
            metas = [{} for _ in texts]

        # Generate embeddings
        embeddings = self.embedder.encode(texts, convert_to_numpy=True, show_progress_bar=True)

        
        if self.index is None:
            dim = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dim)

        
        self.index.add(embeddings)
        self.metas.extend(metas)

    def search(self, query, k=5):
        """Search for top-k similar documents."""
        if self.index is None:
            raise RuntimeError("FAISS index is empty. Build or load it first.")

        q_emb = self.embedder.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(q_emb, k)

        results = []
        for dist_row, idx_row in zip(distances, indices):
            for dist, idx in zip(dist_row, idx_row):
                if 0 <= idx < len(self.metas):
                    results.append((self.metas[idx], float(dist)))
        return results

    def save(self):
        """Save FAISS index and metadata to index_dir."""
        index_path = os.path.join(self.index_dir, "index.faiss")
        meta_path = os.path.join(self.index_dir, "metas.npy")
        faiss.write_index(self.index, index_path)
        np.save(meta_path, self.metas)
        print(f"[FAISS] Saved index to {index_path}")

    def load(self):
        """Load FAISS index and metadata from index_dir."""
        index_path = os.path.join(self.index_dir, "index.faiss")
        meta_path = os.path.join(self.index_dir, "metas.npy")

        if not os.path.exists(index_path):
            raise FileNotFoundError(f"No FAISS index found in {self.index_dir}")

        self.index = faiss.read_index(index_path)
        self.metas = np.load(meta_path, allow_pickle=True).tolist()
        print(f"[FAISS] Loaded index from {index_path}")
