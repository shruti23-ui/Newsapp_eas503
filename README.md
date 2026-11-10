# The Buffalo Times: News Intelligence Dashboard

**Assignment:** Extra credit, three missed opportunities in the class news app
**Student:** Shruti Priya
**Demo video:** [VIDEO_URL]

## Summary

The class news app fetched headlines, ran sentiment and NER, and asked Gemini for a summary, but it
did not use most of what it computed. This project turns three of those gaps into analytical features
in a Dash dashboard. It runs offline on the four real NewsAPI snapshots saved in the class notebooks.

## The three features

| # | Tab | Missed opportunity | What it computes |
|---|---|---|---|
| 1 | **Entity Explorer** | spaCy entities were extracted but never used | Ranked entities with mean sentiment, entity by category heatmap, co-mentions. An EntityRuler fixes spaCy's mislabels first. |
| 2 | **Echo Index** | Each notebook saved a different day's news, never compared | Echo score per headline (TF-IDF cosine to the closest earlier headline), repeated stories, rising and fading words (log-odds with Dirichlet prior) |
| 3 | **Newsroom Check** | Gemini's JSON summary was never parsed or checked | Coverage of the news by the summary, Tavily check of weak claims, Gemini rewrite re-scored with the same metric, ElevenLabs audio bulletin, ZIP report |

Also: opinion map (polarity vs subjectivity), Ask The Buffalo Times (cited answers with a grounding check),
TextBlob vs zero-shot agreement (Cohen's kappa), dark and light themes.

## How to run

Requires Python 3.12.

```bash
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python app.py                     # open http://127.0.0.1:8080
pytest -q                         # 10 offline tests
```

With no keys the app runs in demo mode on the saved snapshots. To use live APIs, create a `.env` file:

```
NEWS_API_KEY=...
GEMINI_API_KEY=...
TAVILY_API_KEY=...
ELEVENLABS_API_KEY=...
```

API results are cached in `data/cache/`. Optional zero-shot sentiment: `pip install -r requirements-zsc.txt`.

## Project structure

```
app.py                  Dash app: data loading, layout, callbacks (main submission)
streamlit_app.py        Streamlit version of the same app
buffalo/
  config.py             settings and paths, keys read from .env
  data_pipeline.py      NewsAPI fetch, snapshot archive, DataFrame
  nlp.py                TextBlob, zero-shot sentiment, spaCy NER
  analytics.py          statistics for the three features
  ai_summary.py         Gemini summary and JSON parsing
  newsroom.py           Gemini rewrite, Ask, ElevenLabs bulletin
  external.py           Yahoo Finance, Bluesky, Tavily
assets/style.css        dark and light themes
data/                   saved class snapshots, Gemini output, stock prices
docs/                   GitHub Pages showcase
scripts/build_pages.py  builds docs/data.js
tests/test_app.py       pytest tests
```

## Data sources

NewsAPI (4 snapshots from the class notebooks), Gemini (saved Nov 2 output), Yahoo Finance (saved prices),
Tavily and ElevenLabs (live, optional).

In live mode the app shows today and the past 7 days, one snapshot per day. Top headlines cannot be
fetched for past dates, so missing days are filled once from NewsAPI's article search: articles
published that day by each category's US sources. Offline demo mode uses only the class snapshots.

## Deployment

- **Streamlit Cloud:** New app, pick this repo and `streamlit_app.py`, Python 3.12, add keys under Secrets.
- **GitHub Pages:** run `python scripts/build_pages.py`, then Settings, Pages, branch `main`, folder `/docs`.
- **Render:** New Blueprint from this repo (`render.yaml`), add keys as environment variables.
