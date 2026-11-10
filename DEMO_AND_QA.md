# 3-minute demo plan

Before recording: run once with all keys, and click Verify, Rewrite, Make the radio bulletin and
an Ask question so everything is cached. Then record. Nothing will wait on the network.

| Time | Screen | Click / show | Say (optional) |
|---|---|---|---|
| 0:00–0:25 | Daily Briefing | Point at the tickers and Gemini cluster cards; hover two dots on the **opinion map**; press **Light mode** then **Dark mode** | "The class app had no interactivity at all; this one does. Subjectivity and zero-shot confidence were computed in class but never shown; now they are." |
| 0:20–1:00 | Entity Explorer (EC 1) | Untick Places, set How many = 10, click **Donald Trump** in the bar chart, show the detail and co-mentions; click a heatmap cell | "spaCy entities were computed but never used. I fixed its mislabels first: it called 'AI' a country and Trump an organization." |
| 1:00–1:45 | Echo Index (EC 2) | Show the echo trend line rising; scroll to the déjà vu list; click **billion** in the rising-words chart | "Each notebook secretly saved a different day's news. Comparing them shows the news repeating itself more each day, from 0.12 to 0.19." |
| 1:45–2:40 | Newsroom Check (EC 3) | Show the 11% coverage and 3 of 7 categories left out; drag the threshold; click **Verify**, **Rewrite and re-audit** (before/after), **Make the radio bulletin** and play 5 seconds; click **Download report** | "The AI summary skipped Entertainment, Science and Technology. Tavily checks the weak claims, Gemini fixes it, the same metric proves the fix, and ElevenLabs reads it." |
| 2:40–3:00 | Ask The Buffalo Times | Ask "What is happening with Iran?", show the citations and the grounding check | "Answers cite headlines, and every sentence is checked against its sources." |

# Questions the professor might ask

**Why is this not just "retrieving text and putting it on the app"?**
Each tab computes something new: entity statistics, similarity scores across days, a coverage
metric for an AI output, and a before/after evaluation. The text shown is evidence for those numbers.

**Which idea came from the email?**
Only NER (EC 1). The Echo Index uses the saved snapshots (title, category, date). The Newsroom Check uses
the Gemini output. Neither relies on the unused NewsAPI fields or on Bluesky.

**What is TF-IDF cosine similarity?**
Each headline becomes a vector of word weights. Words that are rare across all headlines get more weight.
Cosine similarity is the cosine of the angle between two vectors: 1 means the same words in the
same proportions, and 0 means no words in common.

**How is the echo score defined?**
For a headline in snapshot k: the maximum cosine similarity to any headline in snapshots 1…k−1.
See `analytics.echo_scores`.

**Why log-odds with a Dirichlet prior for rising words?**
Raw frequency differences make words seen once or twice look huge. The prior (Monroe, Colaresi & Quinn,
2008) shrinks rare words toward zero, and dividing by the standard error gives a z-score.
All |z| are below 2 here, so the app labels the results as directional.

**What is Cohen's kappa?**
Agreement between two labelers, corrected for chance: κ = (observed − expected) / (1 − expected).
1 means perfect agreement, 0 means chance level. The test suite checks that identical labels give 1
and random labels give about 0.

**Talking point (data-quality note, not a claimed feature): is a "top headline" just the newest?**
In class we guessed yes. On the saved snapshots, NewsAPI's order matches publish time exactly in all 28
category-snapshot groups (Spearman = −1.00). So "top" means newest, not most important.

**What do the opinion map and label confidence show?**
The opinion map plots TextBlob polarity against subjectivity (0 = factual wording, 1 = opinion), using a column
the class computed but never displayed. Label confidence is the zero-shot model's probability for its chosen label;
under 60% the label is faded because the model was unsure.

**Why the EntityRuler?**
`en_core_web_sm` mislabels Title Case headlines. An EntityRuler placed before the statistical NER
pins known names to the right type. That changed the entities in 56 headlines.

**How do you know the Gemini rewrite is better, not just different?**
It is scored with exactly the same audit function and threshold as the original, and the report ZIP
contains both coverage tables.

**What if an API is down?**
Every external call has a timeout, returns a status message instead of raising, and falls back
to saved or cached data. `pytest -q` runs entirely offline.

**Where are the API keys?**
Only in `.env`, which `.gitignore` excludes. `config.py` reads them with python-dotenv.
