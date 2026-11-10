# The Buffalo Times: News Intelligence Dashboard (EAS 503)

A Dash app that extends the class news app with three new analytical features.
Without API keys it runs offline on the four real NewsAPI snapshots saved in class.

## Run it

```bash
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env            # optional: add your API keys here
python app.py                     # open http://127.0.0.1:8080
pytest -q                         # offline tests
```

Python 3.12 is required. Optional zero-shot sentiment: `pip install -r requirements-zsc.txt`.

## Features

| Tab | What it does | Methods |
|---|---|---|
| **Entity Explorer** | Ranks the people, places and organizations in the news, with sentiment, category heatmap and co-mentions | spaCy NER + EntityRuler, groupby, co-occurrence |
| **Echo Index** | Scores how much each headline repeats earlier days' news, and finds rising and fading words | TF-IDF, cosine similarity, log-odds with Dirichlet prior |
| **Newsroom Check** | Measures how much of the news the Gemini summary covers, checks weak claims with Tavily, rewrites and re-scores it, reads it aloud with ElevenLabs | TF-IDF coverage, threshold sweep, before/after evaluation |

Also included: an opinion map (polarity vs subjectivity), Ask The Buffalo Times (cited answers from headlines),
sentiment model agreement (Cohen's kappa), and dark/light themes.

## Modes

- **No `NEWS_API_KEY`:** demo mode on the saved snapshots (Oct 19, 21, 26, Nov 2).
- **With keys:** fetches live headlines and saves each fetch as a new snapshot.
- **If a call fails:** falls back to saved data and shows the reason on screen.

Gemini, Tavily and ElevenLabs results are cached in `data/cache/`.

## Files

```
app.py              Dash app (main submission)
analytics.py        statistics for the three features
nlp.py              TextBlob, zero-shot, spaCy
data_pipeline.py    NewsAPI fetch and snapshot archive
ai_summary.py       Gemini summary
newsroom.py         rewrite, Ask, radio bulletin
external.py         Yahoo Finance, Bluesky, Tavily
streamlit_app.py    Streamlit version for Streamlit Cloud
docs/               GitHub Pages showcase
data/               saved class snapshots, Gemini output, stock prices
tests/              pytest tests
```

## Deploy

- **Streamlit Cloud:** share.streamlit.io, New app, pick this repo and `streamlit_app.py`, Python 3.12. Add keys under Secrets.
- **GitHub Pages:** run `python scripts/build_pages.py`, then Settings, Pages, branch `main`, folder `/docs`.
- **Render:** New Blueprint from this repo (`render.yaml`), add keys as environment variables.
