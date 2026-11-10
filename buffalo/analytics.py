"""Statistics behind the three extra-credit tabs. Pure pandas/sklearn, no Dash."""
import itertools
import math
import re
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


PLACE_TYPES = {"GPE", "LOC", "FAC"}


def unique_articles(df):
    """One row per article per snapshot (the same story can sit in two categories)."""
    return df.drop_duplicates("uid")


# EC 1: entities
def entity_rows(df):
    """Explode df['entities'] into one row per (article, entity)."""
    rows = []
    for rec in df.itertuples():
        for name, label in (rec.entities or []):
            rows.append({"uid": rec.uid, "snapshot": rec.snapshot, "category": rec.category,
                         "title": rec.title, "url": rec.url, "polarity": rec.polarity,
                         "entity": name, "label": label})
    return pd.DataFrame(rows, columns=["uid", "snapshot", "category", "title", "url",
                                       "polarity", "entity", "label"])


def filter_entities(E, snapshot, categories, types):
    if E.empty:
        return E
    out = E
    if snapshot and snapshot != "ALL":
        out = out[out["snapshot"] == snapshot]
    if categories:
        out = out[out["category"].isin(categories)]
    if types:
        wanted = set()
        for t in types:
            wanted |= PLACE_TYPES if t == "PLACE" else {t}
        out = out[out["label"].isin(wanted)]
    return out


def entity_summary(E):
    """Mentions = distinct articles naming the entity; polarity = mean TextBlob score."""
    if E.empty:
        return pd.DataFrame(columns=["entity", "mentions", "label", "mean_polarity", "snapshots"])
    s = (E.groupby("entity")
           .agg(mentions=("uid", "nunique"),
                label=("label", lambda x: x.mode().iloc[0]),
                mean_polarity=("polarity", "mean"),
                snapshots=("snapshot", "nunique"))
           .reset_index()
           .sort_values(["mentions", "snapshots"], ascending=False))
    return s


def cooccurrence(E, names):
    """How often two of the given entities appear in the same headline."""
    pairs = Counter()
    sub = E[E["entity"].isin(names)]
    for _, grp in sub.groupby("uid"):
        for a, b in itertools.combinations(sorted(set(grp["entity"])), 2):
            pairs[(a, b)] += 1
    return pairs


# EC 2: Echo Index
def _vectorizer(texts):
    return TfidfVectorizer(stop_words="english", sublinear_tf=True, min_df=1).fit(texts)


def echo_scores(df):
    """Echo = max TF-IDF cosine to any headline in an earlier snapshot (NaN for the first)."""
    arts = unique_articles(df).reset_index(drop=True)
    cols = ["uid", "snapshot", "category", "title", "url", "echo", "echo_of", "echo_of_snapshot"]
    if arts.empty or arts["snapshot"].nunique() < 2:
        return pd.DataFrame(columns=cols)
    order = sorted(arts["snapshot"].unique())
    X = _vectorizer(arts["title"]).transform(arts["title"])
    echo, echo_of, echo_snap = [np.nan] * len(arts), [""] * len(arts), [""] * len(arts)
    for k, snap in enumerate(order[1:], start=1):
        cur = arts.index[arts["snapshot"] == snap]
        prev = arts.index[arts["snapshot"].isin(order[:k])]
        sims = cosine_similarity(X[cur], X[prev])
        best = sims.argmax(axis=1)
        for row, i in enumerate(cur):
            j = prev[best[row]]
            echo[i] = float(sims[row, best[row]])
            echo_of[i], echo_snap[i] = arts.at[j, "title"], arts.at[j, "snapshot"]
    arts["echo"], arts["echo_of"], arts["echo_of_snapshot"] = echo, echo_of, echo_snap
    return arts[cols]


TOKEN_RE = re.compile(r"[a-z][a-z'\-]+")
EXTRA_STOP = {"new", "says", "say", "will", "just", "get", "may", "now", "one", "week", "day"}


def _doc_freq(titles):
    """Number of headlines containing each word (a word counts once per headline)."""
    stop = ENGLISH_STOP_WORDS | EXTRA_STOP
    counts = Counter()
    for t in titles:
        counts.update({w for w in TOKEN_RE.findall(t.lower()) if len(w) > 2 and w not in stop})
    return counts


def term_shift(df, focus, min_count=3):
    """Rising/fading words: log-odds with an informative Dirichlet prior (Monroe et al. 2008)."""
    arts = unique_articles(df)
    a = _doc_freq(arts.loc[arts["snapshot"] == focus, "title"])
    b = _doc_freq(arts.loc[arts["snapshot"] < focus, "title"])
    prior = a + b
    if not a or not b:
        return pd.DataFrame(columns=["word", "z", "focus_count", "earlier_count"])
    n_a, n_b, a0 = sum(a.values()), sum(b.values()), sum(prior.values())
    scale = 0.5                      # prior strength (pseudo-counts = 0.5 x pooled count)
    rows = []
    for w, pw in prior.items():
        if a[w] + b[w] < min_count:
            continue
        aw, A = pw * scale, a0 * scale
        delta = (math.log((a[w] + aw) / (n_a + A - a[w] - aw))
                 - math.log((b[w] + aw) / (n_b + A - b[w] - aw)))
        var = 1 / (a[w] + aw) + 1 / (b[w] + aw)
        rows.append({"word": w, "z": delta / math.sqrt(var),
                     "focus_count": a[w], "earlier_count": b[w]})
    return pd.DataFrame(rows).sort_values("z", ascending=False).reset_index(drop=True)


# EC 3: AI Summary Audit
# Split after . ! ? followed by a capital letter, but not after "U.S." / "U.K."
SENTENCE_END = re.compile(r"(?<=[.!?])(?<!U\.S\.)(?<!U\.K\.)\s+(?=[A-Z\"“'‘])")


def split_sentences(text):
    return [s.strip() for s in SENTENCE_END.split(text or "") if len(s.strip()) > 3]


def audit_summary(df_snap, summary):
    """Match headlines to summary clusters and summary sentences to headlines (TF-IDF cosine)."""
    arts = unique_articles(df_snap).reset_index(drop=True)
    clusters = summary.get("clusters", [])
    if arts.empty or not clusters:
        return {"heads": pd.DataFrame(), "sents": pd.DataFrame()}
    cluster_text = [f"{c.get('title', '')} {c.get('summary', '')}" for c in clusters]
    sentences = []
    for i, c in enumerate(clusters):
        sentences += [(c.get("title", f"Cluster {i + 1}"), s) for s in split_sentences(c.get("summary", ""))]
    sentences += [("Global summary", s) for s in split_sentences(summary.get("global_summary", ""))]

    vec = _vectorizer(list(arts["title"]) + cluster_text)
    H = vec.transform(arts["title"])
    S = cosine_similarity(H, vec.transform(cluster_text))
    heads = arts[["uid", "category", "title", "url"]].copy()
    heads["cluster"] = S.argmax(axis=1)
    heads["similarity"] = S.max(axis=1)

    sents = pd.DataFrame(columns=["where", "sentence", "support", "best_headline"])
    if sentences:
        SS = cosine_similarity(vec.transform([s for _, s in sentences]), H)
        sents = pd.DataFrame({
            "where": [w for w, _ in sentences],
            "sentence": [s for _, s in sentences],
            "support": SS.max(axis=1),
            "best_headline": [arts.at[int(j), "title"] for j in SS.argmax(axis=1)],
        }).sort_values("support").reset_index(drop=True)
    return {"heads": heads, "sents": sents}


def coverage_by_category(heads, threshold, categories):
    """Share of each category's headlines whose best cluster match >= threshold."""
    rows = []
    for c in categories:
        sub = heads[heads["category"] == c]
        k = int((sub["similarity"] >= threshold).sum())
        rows.append({"category": c, "represented": k, "total": len(sub),
                     "share": k / len(sub) if len(sub) else 0.0})
    return pd.DataFrame(rows)


# Briefing add-on: model agreement
def sentiment_agreement(df):
    """TextBlob vs zero-shot labels. Returns (confusion table, Cohen's kappa, n, rows)."""
    from sklearn.metrics import cohen_kappa_score
    from .nlp import SENTIMENTS, polarity_to_label
    arts = unique_articles(df)
    arts = arts[arts["sentiment_method"] == "zsc"].copy()
    if len(arts) < 5:
        return None, None, len(arts), arts
    arts["textblob_label"] = arts["polarity"].apply(polarity_to_label)
    conf = pd.crosstab(arts["textblob_label"], arts["zsc_sentiment"]).reindex(
        index=SENTIMENTS, columns=SENTIMENTS, fill_value=0)
    kappa = cohen_kappa_score(arts["textblob_label"], arts["zsc_sentiment"], labels=SENTIMENTS)
    return conf, float(kappa), len(arts), arts


# Echo add-on: sentiment by category/time
def sentiment_over_time(df):
    """Mean TextBlob polarity per (snapshot, category), on unique articles."""
    arts = unique_articles(df)
    return (arts.groupby(["snapshot", "category"])["polarity"].mean()
                .reset_index().sort_values("snapshot"))
