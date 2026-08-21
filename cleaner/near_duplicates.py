import difflib
import pandas as pd


def detect_near_duplicates(df, threshold=0.9, max_rows=600):
    """
    Flags row pairs that are highly similar but not identical (e.g. typo
    variants, inconsistent casing/whitespace) — exact duplicates are
    handled separately by df.duplicated().

    Full pairwise fuzzy comparison is O(n^2), so it only runs when the
    dataset has <= max_rows. Larger datasets fall back to a normalized
    exact-match pass (catches case/whitespace-only variants) instead.

    Returns: {"pairs": [(row_i, row_j, similarity)], "mode": "fuzzy"|"normalized", "count": int}
    """
    normalized = df.astype(str).apply(lambda col: col.str.strip().str.lower())
    row_strings = normalized.apply(lambda row: "|".join(row.values), axis=1)

    if len(df) > max_rows:
        exact_dupe_idx = set(df.index[df.duplicated(keep=False)])
        seen = {}
        pairs = []
        for i in row_strings.index:
            if i in exact_dupe_idx:
                continue  # already counted as an exact duplicate elsewhere
            key = row_strings[i]
            if key in seen:
                pairs.append((seen[key], i, 1.0))
            else:
                seen[key] = i
        return {"pairs": pairs, "mode": "normalized", "count": len(pairs)}

    values = row_strings.tolist()
    idx = row_strings.index.tolist()
    pairs = []
    for a in range(len(values)):
        for b in range(a + 1, len(values)):
            if values[a] == values[b]:
                continue  # exact duplicate, not "near"
            ratio = difflib.SequenceMatcher(None, values[a], values[b]).ratio()
            if ratio >= threshold:
                pairs.append((idx[a], idx[b], round(ratio, 3)))
    return {"pairs": pairs, "mode": "fuzzy", "count": len(pairs)}