import re


def chunk_text(text, chunk_size=400, overlap=80):
    """
    Splits text into overlapping word-based chunks.

    Args:
        text (str): Input text
        chunk_size (int): Number of words per chunk
        overlap (int): Overlapping words between chunks

    Returns:
        list[str]: List of text chunks
    """

    if not text or not text.strip():
        return []

    if overlap >= chunk_size:
        raise ValueError("Overlap must be smaller than chunk_size.")

    words = text.split()
    chunks = []

    step = chunk_size - overlap
    for start in range(0, len(words), step):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)

        if end >= len(words):
            break

    return chunks
