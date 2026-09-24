"""Explore topics without sending corpus content to an API."""

import numpy as np
import pandas as pd


def corpus_overview(index, clusters: int = 4) -> pd.DataFrame:
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer

    if not 1 <= clusters <= len(index.documents):
        raise ValueError("Cluster count must be between 1 and the number of chunks.")
    vectors = index.index.reconstruct_n(0, index.index.ntotal)
    labels = KMeans(n_clusters=clusters, random_state=42, n_init=10).fit_predict(vectors)
    texts = [d.page_content for d in index.documents]
    vectorizer = TfidfVectorizer(stop_words="english", max_features=2000)
    try:
        terms = vectorizer.fit_transform(texts)
        names = vectorizer.get_feature_names_out()
    except ValueError:
        terms, names = None, []
    rows = []
    for label in sorted(set(labels)):
        positions = np.flatnonzero(labels == label)
        keywords = ""
        if terms is not None:
            weights = np.asarray(terms[positions].mean(axis=0)).ravel()
            keywords = ", ".join(names[i] for i in weights.argsort()[-6:][::-1] if weights[i] > 0)
        rows.append(
            {
                "cluster": int(label),
                "chunks": len(positions),
                "keywords": keywords,
                "sources": ", ".join(
                    sorted({index.documents[i].metadata["source"] for i in positions})
                ),
            }
        )
    return pd.DataFrame(rows)
