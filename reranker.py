import re


STOPWORDS = {
    "the", "is", "in", "at", "of", "a", "an",
    "and", "to", "for", "on", "with", "by"
}


def tokenize(text):
    words = re.findall(r"\b\w+\b", text.lower())
    return [w for w in words if w not in STOPWORDS]


def simple_rerank(query, docs):
    """
    Lightweight keyword-based reranker.
    """

    query_tokens = set(tokenize(query))
    scored = []

    for doc in docs:
        doc_tokens = set(tokenize(doc))
        overlap = query_tokens.intersection(doc_tokens)
        score = len(overlap)

        scored.append((score, doc))

    scored.sort(key=lambda x: x[0], reverse=True)

    return [d for _, d in scored]
