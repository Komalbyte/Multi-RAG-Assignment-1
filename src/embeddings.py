"""
embeddings.py
Converts text chunks into vectors using sentence-transformers, 
then stores them in a FAISS index so we can search quickly.

Model used: all-MiniLM-L6-v2
  - gives 384-dimensional vectors
  - pretty small (~80MB download), runs fine on CPU
  - good enough for semantic similarity

FAISS index: IndexFlatL2
  - does exact search using L2 distance
  - complexity is O(n * d) per query which is fine for small datasets
  - for bigger stuff (100k+ chunks) you'd want something approximate like IVF or HNSW
"""

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"

# cache so we dont reload the model every time
_model = None


def get_model(name=MODEL_NAME):
    """Load the embedding model (downloads first time, cached after)."""
    global _model
    if _model is not None:
        return _model

    print(f"Loading model: {name}")
    _model = SentenceTransformer(name)
    dim = _model.get_sentence_embedding_dimension()
    print(f"Model ready. Embedding dim: {dim}")
    return _model


def make_embeddings(chunk_list, model):
    """
    Turn a list of chunk dicts into a numpy array of embeddings.
    Each chunk needs a 'text' key.
    """
    texts = [c["text"] for c in chunk_list]
    # batch encode is way faster than doing one at a time
    vecs = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    print(f"Created {vecs.shape[0]} embeddings (dim={vecs.shape[1]})")
    return vecs


def build_index(vecs):
    """
    Build a FAISS index from our embedding vectors.
    Using IndexFlatL2 - its brute force but works fine for our scale.
    """
    dim = vecs.shape[1]
    idx = faiss.IndexFlatL2(dim)
    idx.add(vecs.astype(np.float32))
    print(f"FAISS index ready: {idx.ntotal} vectors, dim={dim}")
    return idx


def save_index(idx, path):
    faiss.write_index(idx, path)
    print(f"Saved index to {path}")


def load_index(path):
    idx = faiss.read_index(path)
    print(f"Loaded index from {path} ({idx.ntotal} vectors)")
    return idx


if __name__ == "__main__":
    model = get_model()
    test_chunks = [
        {"text": "Machine learning is a part of AI."},
        {"text": "Deep learning uses neural networks."},
        {"text": "NLP deals with understanding text."},
    ]
    vecs = make_embeddings(test_chunks, model)
    index = build_index(vecs)
    print(f"Test done. Index has {index.ntotal} vectors.")
