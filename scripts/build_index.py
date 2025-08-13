import os
import numpy as np
import faiss

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.retrieval_faiss import FaissStore

# Faiss directories
INDEX_DIRS = [
    "data/index_pubmed",
    "data/index_eupmc",
    "data/index_guidelines"
]


OUT_DIR = "data/index"

def load_faiss_and_meta(index_dir):
    """Load FAISS index and metadata from a directory."""
    index_path = os.path.join(index_dir, "index.faiss")
    meta_path = os.path.join(index_dir, "metas.npy")

    if not os.path.exists(index_path) or not os.path.exists(meta_path):
        print(f"⚠ Missing FAISS or meta file in {index_dir}, skipping.")
        return None, None

    index = faiss.read_index(index_path)
    metas = np.load(meta_path, allow_pickle=True).tolist()
    return index, metas

def merge_faiss_indexes(index_dirs, out_dir):
    all_vectors = []
    all_metas = []
    faiss_index = None

    for d in index_dirs:
        index, metas = load_faiss_and_meta(d)
        if index is not None:
            # vectors DB
            vectors = index.reconstruct_n(0, index.ntotal)
            all_vectors.append(vectors)
            all_metas.extend(metas)
            print(f"✅ Loaded {len(metas)} docs from {d}")

            
            if faiss_index is None:
                faiss_index = faiss.IndexFlatIP(vectors.shape[1])

    if not all_vectors:
        print(" No FAISS indexes found to merge.")
        return

    #  all vectors 
    merged_vectors = np.vstack(all_vectors)

    # add merged vectors
    faiss_index.add(merged_vectors)

    
    os.makedirs(out_dir, exist_ok=True)
    faiss.write_index(faiss_index, os.path.join(out_dir, "index.faiss"))
    np.save(os.path.join(out_dir, "metas.npy"), np.array(all_metas, dtype=object))
    print(f"[FAISS] Saved merged index with {len(all_metas)} docs → {out_dir}")

if __name__ == "__main__":
    merge_faiss_indexes(INDEX_DIRS, OUT_DIR)
