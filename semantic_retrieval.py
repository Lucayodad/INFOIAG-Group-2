"""Deterministic in-memory candidate retrieval, never an equivalence verdict."""

from revision_retrieval import cosine


def shortlist_equivalence(candidates, embedder, *, cache=None, model="all-minilm",
                          top_k=2, min_similarity=0.8):
    """Retrieve at most two earlier same-stance candidates per proposition.

    Only Qwen may approve a merge. Retrieval is approximate: a low score or an
    unselected neighbor is not a semantic proof of difference. Reports identify
    this scope, and coverage is conditional on the resulting unreviewed partition.
    """
    cache = {} if cache is None else cache
    keys = sorted(candidates)
    texts = {key: candidates[key]["claim"] + "\nTarget: " + candidates[key]["target"] +
             "\nStance: " + candidates[key]["stance"] for key in keys}
    missing = sorted({text for text in texts.values() if (model, text) not in cache})
    calls = 0
    for start in range(0, len(missing), 32):
        batch = missing[start:start + 32]
        calls += 1
        vectors = embedder(batch)
        if not isinstance(vectors, list) or len(vectors) != len(batch):
            raise ValueError("Embedding response has the wrong number of vectors.")
        for text, vector in zip(batch, vectors):
            cosine(vector, vector)  # Check finite, nonzero vectors before caching.
            cache[model, text] = vector
    proposals, scores = {}, {}
    for index, key in enumerate(keys):
        ranked = []
        for old in keys[:index]:
            if candidates[key]["stance"].casefold() != candidates[old]["stance"].casefold():
                continue
            score = cosine(cache[model, texts[key]], cache[model, texts[old]])
            if score >= min_similarity:
                ranked.append((old, score))
        ranked.sort(key=lambda pair: (-pair[1], pair[0]))
        proposals[key] = [old for old, _ in ranked[:top_k]]
        scores[key] = [{"candidate_id": old, "cosine": score} for old, score in ranked[:top_k]]
    return proposals, {"embedding_call_count": calls, "embedding_model": model,
        "top_k": top_k, "min_similarity": min_similarity, "neighbors": scores,
        "equivalence_scope": "embedding_shortlist_only", "global_equivalence_exhaustive": False}
