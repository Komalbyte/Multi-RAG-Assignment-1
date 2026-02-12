"""
retrieval.py
Searches the FAISS index to find chunks similar to the user's question.

This is the core of RAG - if we get bad chunks here, the answer will be bad too.
We also check if the similarity score is too low and warn about it.
"""

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# if L2 distance is above this, the match probably isnt great
DISTANCE_THRESHOLD = 1.5


def find_top_chunks(query, index, chunk_list, model, top_k=3):
    """
    Find the top_k closest chunks to the query.
    
    Returns a list of dicts with the chunk, its L2 distance score,
    and its rank. Lower score = more similar.
    """
    # embed the query with same model we used for chunks
    q_vec = model.encode([query], convert_to_numpy=True).astype(np.float32)

    # search - faiss returns distances and indices
    dists, idxs = index.search(q_vec, min(top_k, len(chunk_list)))

    found = []
    for rank, (dist, i) in enumerate(zip(dists[0], idxs[0])):
        if i == -1:
            continue  # faiss returns -1 when theres not enough vectors

        item = {
            "chunk": chunk_list[i],
            "score": float(dist),
            "rank": rank + 1,
        }

        # flag low quality matches
        if dist > DISTANCE_THRESHOLD:
            item["warning"] = "Low similarity - might not be relevant"

        found.append(item)

    return found


def build_context(found_chunks):
    """
    Combine the retrieved chunks into one string that we'll feed to the LLM.
    This is basically the 'knowledge' the model gets to work with.
    """
    parts = []
    for r in found_chunks:
        header = f"[Chunk {r['rank']}, dist={r['score']:.4f}]"
        parts.append(f"{header}\n{r['chunk']['text']}")
    return "\n\n".join(parts)


def show_results(found_chunks):
    """Print what we found in a readable way."""
    print(f"\n{'='*50}")
    print(f"Found {len(found_chunks)} chunks:")
    print(f"{'='*50}")

    for r in found_chunks:
        print(f"\n--- Rank {r['rank']} (dist: {r['score']:.4f}) ---")
        if "warning" in r:
            print(f"  ⚠ {r['warning']}")
        print(f"  {r['chunk']['text'][:200]}...")


if __name__ == "__main__":
    from embeddings import get_model, make_embeddings, build_index

    model = get_model()
    test_chunks = [
        {"text": "The methodology involves training a neural net on labeled data."},
        {"text": "Limitations include high compute cost and data requirements."},
        {"text": "Results show 95% accuracy on the test set."},
        {"text": "Related work covers transformers and attention."},
    ]
    vecs = make_embeddings(test_chunks, model)
    idx = build_index(vecs)

    results = find_top_chunks("What is the methodology?", idx, test_chunks, model, top_k=2)
    show_results(results)
