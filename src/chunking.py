"""
chunking.py
Splits the extracted text into smaller pieces (chunks) for embedding.

We use overlap between chunks to avoid breaking important sentences
right at the boundary. Without overlap you'd lose context at the edges
and retrieval quality drops. I tested with and without overlap and the
difference was noticeable.
"""


def chunk_text(text, chunk_size=600, overlap=100):
    """
    Splits text into overlapping chunks.
    
    chunk_size = how many characters per chunk (600 works well for most papers)
    overlap = how many chars to repeat between consecutive chunks
    
    Returns a list of dicts with the chunk text and some position info.
    """
    if not text or not text.strip():
        return []

    result = []
    pos = 0
    idx = 0

    while pos < len(text):
        end = pos + chunk_size

        # try to break at a sentence boundary instead of cutting mid-word
        if end < len(text):
            # look for a period, newline, or at least a space
            bp = text.rfind('. ', pos, end)
            if bp == -1:
                bp = text.rfind('\n', pos, end)
            if bp == -1:
                bp = text.rfind(' ', pos, end)
            if bp != -1 and bp > pos:
                end = bp + 1

        piece = text[pos:end].strip()

        if piece:
            result.append({
                "text": piece,
                "chunk_id": idx,
                "start": pos,
                "end": end,
            })
            idx += 1

        # slide the window forward, but keep some overlap
        pos += chunk_size - overlap

    return result


def show_chunk_stats(chunk_list):
    """Print some basic stats about the chunks we made."""
    print(f"Total chunks: {len(chunk_list)}")
    if chunk_list:
        avg = sum(len(c['text']) for c in chunk_list) / len(chunk_list)
        print(f"Avg chunk length: {avg:.0f} chars")
        print(f"\nFirst chunk:\n{chunk_list[0]['text'][:200]}...")


if __name__ == "__main__":
    test = "This is a test sentence. " * 100
    chunks = chunk_text(test)
    show_chunk_stats(chunks)
