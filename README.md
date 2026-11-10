# The Buffalo Times: News Intelligence Dashboard

**Assignment:** Extra credit, EAS 503  
**Student:** Shruti Priya (50757311)  
**Demo video:** https://ubuffalo-my.sharepoint.com/:v:/r/personal/spriya3_buffalo_edu/_layouts/15/stream.aspx?id=%2Fpersonal%2Fspriya3_buffalo_edu%2FDocuments%2FAttachments%2FWorking%20Prototype.mp4&nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJTdHJlYW1XZWJBcHAiLCJyZWZlcnJhbFZpZXciOiJTaGFyZURpYWxvZy1MaW5rIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXcifX0%3D&share=cQpsy9gVStiTSIrgQJ9YZ0L2EgUBQf69Vp2zb_d446Dq5O5d2w
**Live app:** https://newsappeas503-ihbxry75s9tguguaepuerb.streamlit.app  
**Notebook:** [Buffalo_Times.ipynb](Buffalo_Times.ipynb) 

A Dash dashboard built on the class news app. The class app fetched headlines, ran sentiment and NER, and asked
Gemini for a summary, but used little of what it computed. This project turns three of those gaps into features.

## The three features

| Feature | Missed opportunity | What it computes |
|---|---|---|
| **Entity Explorer** | spaCy entities were extracted but never used | Most-mentioned people, organizations and places, their average sentiment, entity × category heatmap, co-mentions. An EntityRuler fixes spaCy's mislabels first. |
| **Echo Index** | Each class notebook saved a different day's news, never compared | Echo score per headline (TF-IDF cosine to the closest earlier headline), repeated and new stories, rising and fading words (log-odds with a Dirichlet prior) |
| **Newsroom Check** | Gemini's JSON summary was never parsed or checked | How much of the news the summary covers, weakest claims checked with Tavily, a Gemini rewrite re-scored with the same metric, an ElevenLabs audio bulletin, a ZIP report |

**Also added:** top stories with Read more / Listen, Ask The Buffalo Times (cited answers with a grounding check),
an opinion map, a **Gemini model comparison** (two models, same prompt, same audit, Wilcoxon signed-rank test),
**news framing** (zero-shot tags for conflict, economic, human interest, morality and responsibility frames,
Semetko & Valkenburg 2000), and **daily snapshots** saved automatically by GitHub Actions.

## Run it

Python 3.12 is required.

```bash
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python app.py                     # open http://127.0.0.1:8080
pytest -q                         # offline tests
```

Without API keys the app runs offline on the news snapshots saved in the repo. For live news (today and the
past 7 days) and the Newsroom Check steps, create a `.env` file:

```
NEWS_API_KEY=...
GEMINI_API_KEY=...
TAVILY_API_KEY=...
ELEVENLABS_API_KEY=...
```

## Project structure

```
app.py                       Dash app (main submission)
streamlit_app.py             Streamlit version (the live link above)
Buffalo_Times.ipynb          notebook walkthrough with results
buffalo/                     data, NLP, analytics and API code shared by both apps
scripts/compare_models.py    Gemini model comparison -> data/model_comparison.csv
scripts/tag_frames.py        zero-shot news frames -> data/framing/ (needs requirements-zsc.txt)
scripts/save_daily_snapshot.py  run daily by .github/workflows/daily-snapshot.yml
data/snapshots/              NewsAPI snapshots (class notebooks + live fetches)
data/summaries/              Gemini summaries
tests/                       pytest tests
```

## Data

Four NewsAPI snapshots come from the class notebooks. Live mode adds today and fills the past 7 days once from
NewsAPI's article search (articles published that day by each category's US sources). The daily GitHub Actions
job needs repository secrets `NEWS_API_KEY` and `GEMINI_API_KEY`.
