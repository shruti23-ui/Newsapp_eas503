# The Buffalo Times — News Intelligence Dashboard (EAS 503 extra credit)

A Dash app that extends the class news app (NewsAPI, Gemini, Yahoo Finance, TextBlob,
zero-shot classification, spaCy) with three new analytical features, plus a few supporting additions.
It runs **offline in demo mode** on the four real NewsAPI snapshots saved in the class notebooks.

## Quick start (VS Code, about 5 minutes)

```bash
# 1. open this folder in VS Code, then in the terminal:
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

# 2. optional: API keys (without them the app runs in demo mode)
cp .env.example .env        # Windows: copy .env.example .env   then edit .env

# 3. run
python app.py               # open http://127.0.0.1:8080

# 4. tests (offline, no keys needed)
pytest -q
```

Optional zero-shot sentiment (about 2 GB): `pip install -r requirements-zsc.txt`.
Without it, sentiment labels fall back to TextBlob, and the app says so on screen.

## Modes

| Situation | What happens |
|---|---|
| No `NEWS_API_KEY` (or `DEMO_MODE=true`) | Demo mode: the 4 saved class snapshots (Oct 19, 21, 26, Nov 2, 2026) and the saved Gemini summary |
| `NEWS_API_KEY` set | Fetches live headlines for 7 categories, saves them as a new snapshot in `data/snapshots/`, and Gemini summarizes it if `GEMINI_API_KEY` is set |
| A live call fails | Falls back to the saved data and shows the reason in the header badge |

Tavily, Gemini and ElevenLabs results are cached in `data/cache/`. After one run with keys,
the demo replays without any network.

## The three extra-credit features

| Tab | Missed opportunity in the class app | What it adds | Techniques |
|---|---|---|---|
| **Entity Explorer (EC 1)** | spaCy entities were computed (Nov 3 notebook) but never used *(the one idea used from the professor's email)* | Ranked entities colored by sentiment, entity × category heatmap, co-mentions, drill-down. spaCy's mislabels on Title Case headlines are corrected with an `EntityRuler` first. | spaCy NER + rules, groupby aggregation, co-occurrence |
| **Echo Index (EC 2)** | Four real NewsAPI responses sat unused in notebook outputs, and the app forgot everything after one fetch | Every fetch is archived. Each headline gets an *echo* score (similarity to its closest earlier headline), plus déjà vu pairs, rising/fading words and sentiment by category over time | TF-IDF, cosine similarity, log-odds with an informative Dirichlet prior |
| **Newsroom Check (EC 3)** | Gemini's JSON summary was never parsed (the app showed a hardcoded string) | Audit: how much of the news does the AI summary cover? Verify weak claims with Tavily. Fix: Gemini rewrites it, and the same metric re-scores it. Listen: ElevenLabs multi-voice bulletin. Download the report as a ZIP. | TF-IDF coverage scoring, threshold sweep, before/after evaluation |

Supporting additions (not claimed as features): an **opinion map** (polarity × subjectivity; subjectivity was computed in class but never shown), **zero-shot label confidence** in the headline table (low-confidence labels faded), **Ask The Buffalo Times** (retrieval plus a cited Gemini answer
with a per-sentence grounding check), a **sentiment model agreement** panel (TextBlob vs zero-shot, Cohen's kappa),
dark/light themes, a fixed version of the class Bluesky code, and error handling everywhere.

## Project structure

```
app.py            Dash app: layout + all callbacks (sections 1-4)  <- main submission
streamlit_app.py  Streamlit front end reusing app.py (for Streamlit Cloud)
scripts/build_pages.py  builds docs/data.js for the GitHub Pages showcase
docs/             static showcase site (GitHub Pages)
config.py         env vars, paths, switches (no keys in code)
data_pipeline.py  NewsAPI fetch (class P1/P3), snapshot archive, DataFrame
nlp.py            TextBlob, zero-shot (cached, 1 call per headline), spaCy + EntityRuler
analytics.py      entity stats, echo scores, term shift, summary audit, kappa
ai_summary.py     Gemini summary prompt (class P7) + robust JSON parsing
newsroom.py       balanced rewrite, Ask (retrieval + citations + grounding), ElevenLabs
external.py       Yahoo Finance, Bluesky (fixed class code), Tavily
assets/style.css  dark + light themes (CSS variables)
data/snapshots/   4 real NewsAPI responses extracted from the class notebooks
data/summaries/   real Gemini JSON output (Nov 2) from News_Nov_3.ipynb
data/demo/        real Yahoo Finance prices from the Oct 27 and Nov 3 notebooks
tests/            pytest smoke tests
```

## Deployment: three ways to show it

| Target | What runs there | Steps |
|---|---|---|
| **Streamlit Community Cloud** (live demo link) | `streamlit_app.py`: the same data, analytics and charts as the Dash app, in Streamlit | 1. Push this folder to a public GitHub repo. 2. share.streamlit.io → **New app** → pick the repo, branch `main`, file `streamlit_app.py`. 3. In **Advanced settings** pick Python 3.12, and paste your keys under **Secrets** in TOML form (`GEMINI_API_KEY = "..."`). Without keys it runs in demo mode. |
| **GitHub Pages** (project showcase) | `docs/index.html`: a static page with interactive Plotly.js charts built from the real data (Pages cannot run Python) | 1. `python scripts/build_pages.py` (refreshes `docs/data.js`). 2. Replace `[STREAMLIT_APP_URL]`, `[GITHUB_REPO_URL]`, `[VIDEO_URL]` in `docs/index.html`. 3. Repo **Settings → Pages** → Deploy from branch → `main` / `/docs`. |
| **Render** (the real Dash app online) | `app.py` via gunicorn | render.com → New → Blueprint → pick the repo (`render.yaml`), then add keys as environment variables. |

Streamlit Cloud passes Secrets to the app as environment variables, so `config.py` reads them with no code change.
**Vercel** is not used: it runs Python as short-lived serverless functions with a bundle-size limit, which doesn't fit a
Dash server with spaCy. The Dash app (`app.py`) remains the main submission; the Streamlit app is a hosting convenience.

## Known limitations (be upfront about these)

- Only 4 snapshots, so the rising/fading word z-scores are all below 2: directional, not significant.
- The coverage audit uses word overlap (TF-IDF), so a correct but paraphrased claim can score low.
  The app says "weakly supported," never "false."
- spaCy `en_core_web_sm` still makes some errors after the rule fixes.
- Tavily's `published_date` is an estimate. ElevenLabs voice IDs can change; set `ELEVENLABS_VOICES` if the defaults fail.
- The Gemini, Tavily and ElevenLabs calls were written against their public REST APIs but could not be called
  from the build environment. Test them once with your keys.
