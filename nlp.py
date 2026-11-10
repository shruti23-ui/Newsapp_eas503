"""TextBlob sentiment, zero-shot sentiment and spaCy NER on headlines."""
import json
import re

import pandas as pd
from textblob import TextBlob

import config

SENTIMENTS = ["positive", "negative", "neutral"]
ZSC_CACHE_FILE = config.CACHE_DIR / "zsc_cache.json"


# TextBlob (P9)
def get_polarity(text):
    return TextBlob(text).sentiment.polarity


def get_subjectivity(text):
    return TextBlob(text).sentiment.subjectivity


def polarity_to_label(polarity, band=0.05):
    """Fallback label when the zero-shot model is not available."""
    if polarity > band:
        return "positive"
    if polarity < -band:
        return "negative"
    return "neutral"


# Zero-shot (P10)
def _load_zsc():
    if not config.ENABLE_ZSC:
        return None, "disabled (ENABLE_ZSC=false)"
    try:
        from transformers import pipeline
        return pipeline("zero-shot-classification",
                        model="facebook/bart-large-mnli"), "ok"
    except Exception as exc:  # not installed, no internet for first download, ...
        return None, f"unavailable ({type(exc).__name__})"


def add_zero_shot(df: pd.DataFrame):
    """Adds zsc_sentiment, zsc_scores, sentiment_method. Returns (df, method text)."""
    try:
        cache = json.loads(ZSC_CACHE_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        cache = {}

    titles = df["title"].unique().tolist()
    missing = [t for t in titles if t not in cache]
    zsc, state = (None, "ok") if not missing else _load_zsc()

    if missing and zsc is not None:
        print(f"[zsc] classifying {len(missing)} new headlines (cached afterwards)...")
        for title in missing:
            out = zsc(title, candidate_labels=SENTIMENTS)  # one call = label + score
            cache[title] = [out["labels"][0], float(out["scores"][0])]
        ZSC_CACHE_FILE.write_text(json.dumps(cache), encoding="utf-8")

    labels, scores, methods = [], [], []
    for title, pol in zip(df["title"], df["polarity"]):
        if title in cache:
            labels.append(cache[title][0]); scores.append(cache[title][1]); methods.append("zsc")
        else:
            labels.append(polarity_to_label(pol)); scores.append(None); methods.append("textblob")
    df["zsc_sentiment"], df["zsc_scores"], df["sentiment_method"] = labels, scores, methods

    share = (df["sentiment_method"] == "zsc").mean() if len(df) else 0
    if share == 1:
        text = "Zero-shot BART-large-MNLI"
    elif share == 0:
        text = f"TextBlob fallback - zero-shot model {state}"
    else:
        text = f"Zero-shot for {share:.0%} of headlines, TextBlob fallback for the rest"
    return df, text


# spaCy (P11)
_NLP = None

# Entity types we keep (dates, numbers and money are not "who/where" entities)
ENTITY_TYPES = {
    "PERSON": "People", "ORG": "Organizations", "GPE": "Countries and cities",
    "LOC": "Other locations", "NORP": "Nationalities and groups",
    "EVENT": "Events", "PRODUCT": "Products", "WORK_OF_ART": "Titles of works",
    "FAC": "Facilities", "LAW": "Laws",
}

# Small alias table so the same real-world entity is counted once
ENTITY_ALIASES = {
    "u.s.": "United States", "u.s": "United States", "us": "United States", "united states": "United States",
    "america": "United States", "the united states": "United States",
    "u.k.": "United Kingdom", "uk": "United Kingdom", "britain": "United Kingdom",
    "trump": "Donald Trump", "the u.s.": "United States", "nfl": "NFL", "nba": "NBA", "mlb": "MLB",
}


def normalise_entity(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip(" \"'“”‘’.,:;-")
    text = re.sub(r"^(the|The)\s+", "", text)
    text = re.sub(r"['’]s$", "", text)
    return ENTITY_ALIASES.get(text.lower(), text)


# en_core_web_sm was trained on lower-case news prose; on Title Case headlines it
# tagged "AI" as a country, "Donald Trump" as an organisation and "Nvidia" as a
# place. An EntityRuler runs BEFORE the statistical NER and fixes known names.
RULE_ORGS = ["Nvidia", "Netflix", "Google", "Microsoft", "Apple", "Amazon", "Meta",
             "Tesla", "Oracle", "Intel", "Broadcom", "Micron", "Walmart", "JPMorgan",
             "Xbox", "OpenAI", "Anthropic", "NASA", "SpaceX", "NFL", "NBA", "MLB",
             "NHL", "Jets", "Bills"]
RULE_PEOPLE = ["Donald Trump", "Trump", "Elon Musk", "Lamar Jackson", "Josh Allen"]
RULE_SKIP = ["AI", "Death Threats"]          # not named entities at all


def get_spacy():
    global _NLP
    if _NLP is None:
        import spacy
        _NLP = spacy.load("en_core_web_sm")
        ruler = _NLP.add_pipe("entity_ruler", before="ner")
        ruler.add_patterns([{"label": "ORG", "pattern": n} for n in RULE_ORGS]
                           + [{"label": "PERSON", "pattern": n} for n in RULE_PEOPLE]
                           + [{"label": "SKIP", "pattern": n} for n in RULE_SKIP])
    return _NLP


def add_entities(df: pd.DataFrame) -> pd.DataFrame:
    """Class P11 (df['entities'] = df['title'].apply(extract_ents)), batched with nlp.pipe."""
    nlp = get_spacy()
    unique_titles = df["title"].unique().tolist()
    ent_map = {}
    for title, doc in zip(unique_titles, nlp.pipe(unique_titles, batch_size=64)):
        ents, seen = [], set()
        for ent in doc.ents:
            if ent.label_ not in ENTITY_TYPES:
                continue
            name = normalise_entity(ent.text)
            if len(name) < 2 or name.isdigit() or (name, ent.label_) in seen:
                continue
            seen.add((name, ent.label_))
            ents.append((name, ent.label_))
        ent_map[title] = ents
    df["entities"] = df["title"].map(ent_map)
    return df


# run it all
def enrich(df: pd.DataFrame):
    """Run every NLP step. Returns (df, sentiment method description)."""
    if df.empty:
        for col in ["polarity", "subjectivity", "zsc_sentiment", "zsc_scores",
                    "sentiment_method", "entities"]:
            df[col] = pd.Series(dtype="object")
        return df, "no data"
    df["polarity"] = df["title"].apply(get_polarity)
    df["subjectivity"] = df["title"].apply(get_subjectivity)
    df, method = add_zero_shot(df)
    df = add_entities(df)
    return df, method
