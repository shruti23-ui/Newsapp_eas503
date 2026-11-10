# Demo script (about 3 minutes)

Before recording, open the app once and, on **Nov 2**, click Verify, Rewrite and re-audit, Make the radio bulletin,
and ask "What is happening with Iran?". The results are cached, so nothing waits on the network during the recording.

| Time | Screen | Say |
|---|---|---|
| 0:00–0:15 | Header, **● Live news** | "The Buffalo Times is a dashboard built on our class news app. It shows live news for today and the past seven days, and turns things the class app computed but never used into analysis." |
| 0:15–0:40 | Daily Briefing: story cards, **Read more**, **Listen**, pick **Health** | "Gemini's JSON summary is parsed into story cards. The bar shows which categories each story draws on. Filtering to Health shows the AI stories that use health headlines, then the top health headlines." |
| 0:40–1:05 | Entity Explorer: click **NFL**, then a heatmap cell | "Feature one uses the spaCy entities from class. An EntityRuler fixes mislabels first: spaCy called 'AI' a country. The chart ranks who the news is about, colored by average mood." |
| 1:05–1:30 | Echo Index: trend line, repeated stories, click a word | "Feature two: each headline's echo score is its highest TF-IDF cosine similarity to an earlier headline. Rising words use log-odds with a Dirichlet prior; z is below 2, so they are directional." |
| 1:30–2:15 | Newsroom Check, **Nov 2**: 11%, 3 of 7 left out, Verify, Rewrite (28%), radio, then scroll to **Model comparison** | "Feature three audits the AI summary: it covered 11% of headlines and left out three categories. Tavily checks weak claims, and a Gemini rewrite raises coverage to 28% under the same metric. Comparing two Gemini models over 8 days, the larger model covers 7 points more, on every day, p = 0.008." |
| 2:15–2:40 | Framing tab | "A zero-shot model tags each headline with a news frame: conflict, economic, human interest, morality or responsibility. This shows how each category is told." |
| 2:40–3:00 | Ask: "What is happening with Iran?" | "Ask answers only from retrieved headlines, cites them, and checks every sentence. Code, a notebook with all results, and tests are on GitHub. Thank you." |

# Questions the professor might ask

**Which idea came from the email?** Only NER (Entity Explorer). The Echo Index uses titles, categories and dates;
the Newsroom Check uses the Gemini output.

**What is TF-IDF cosine similarity?** Each headline becomes a vector of word weights, with rare words weighted more.
Cosine similarity is 1 for the same words in the same proportions and 0 for no shared words.

**Why log-odds with a Dirichlet prior?** Raw frequency changes make rare words look huge. The prior
(Monroe, Colaresi & Quinn, 2008) shrinks them toward zero, and dividing by the standard error gives a z-score.

**How do you know the rewrite is better?** It is scored with the same audit function and threshold as the original.

**How were the Gemini models compared?** Same prompt, same headlines, same audit, two runs per model per day
(answers vary), 8 days. A Wilcoxon signed-rank test on the daily paired differences gives p = 0.008: the larger
model covered more on all 8 days.

**How reliable is the framing?** It is a zero-shot model, so the app shows how often it is unsure (confidence under
40%) and compares it with 50 hand-labelled headlines (accuracy and Cohen's kappa) once those labels are filled in.

**Is a "top headline" just the newest?** On the class snapshots, NewsAPI's order matches publish time exactly in all
28 category-snapshot groups (Spearman = −1.00).

**What if an API fails?** Every call has a timeout, returns a status message instead of raising, and falls back to
saved or cached data. `pytest -q` runs fully offline.

**Where are the API keys?** Only in `.env` (gitignored) locally, and in Streamlit Secrets online.
