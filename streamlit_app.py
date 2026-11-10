"""Streamlit front end reusing the data, analytics and figures from app.py."""
import base64
import os
from html import escape

import pandas as pd
import streamlit as st

st.set_page_config(page_title="The Buffalo Times", page_icon="assets/favicon.ico", layout="wide")

# Streamlit Cloud keeps API keys in st.secrets, but config.py reads environment variables,
# so copy them over before app.py (and config.py) is imported
try:
    for name, value in st.secrets.items():
        if isinstance(value, str):
            os.environ.setdefault(name, value)
except Exception:   # running locally without a secrets file: keys come from .env instead
    pass


@st.cache_resource(show_spinner="Loading the news (first run takes a minute)...")
def load_core():
    import app as core          # loads data, NLP, entities, echo scores, audits once
    return core


A = load_core()
from buffalo import analytics  # noqa: E402  (after the cached import so paths/env are set)
from buffalo import newsroom  # noqa: E402

THEME = "dark"   # the page theme is set in .streamlit/config.toml, so charts always match it
TYPE_LABELS = {"PERSON": "People", "ORG": "Organizations", "PLACE": "Places"}

# small style sheet for the cards (colors match .streamlit/config.toml)
st.markdown("""<style>
.bt-kicker { font-size: 13px; letter-spacing: .14em; text-transform: uppercase; color: #A3B3CC; }
.bt-badges { display: flex; flex-wrap: wrap; gap: 8px; margin: 2px 0 6px; }
.bt-badge { height: 30px; display: inline-flex; align-items: center; padding: 0 12px; font-size: 13px;
            color: #A3B3CC; border: 1px solid #2A3D5E; border-radius: 999px; }
.bt-badge.live { color: #E8EEF6; }
.live-dot { width: 8px; height: 8px; margin-right: 7px; border-radius: 50%; background: #E5484D;
            animation: blink 1.2s ease-in-out infinite; }
@keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: .25; } }
.bt-lede { font-size: 17px; line-height: 1.6; padding-left: 14px; border-left: 3px solid #7CC4FA; margin: 4px 0 14px; }
.bt-num { display: inline-grid; place-items: center; width: 26px; height: 26px; border-radius: 50%;
          background: #7CC4FA; color: #0F1A2B; font-size: 13px; font-weight: 800; }
.bt-tag { display: inline-block; font-size: 11px; font-weight: 700; letter-spacing: .06em; text-transform: uppercase;
          padding: 3px 8px; border-radius: 4px; background: #7CC4FA; color: #0F1A2B; }
.bt-tag.plain { background: transparent; color: #A3B3CC; border: 1px solid #2A3D5E; }
.bt-title { font-size: 15px; font-weight: 700; line-height: 1.35; margin: 8px 0 6px; }
.bt-title a { color: #E8EEF6; text-decoration: none; }
.bt-text { font-size: 13px; line-height: 1.5; color: #A3B3CC; }
.bt-bar { display: flex; height: 8px; border-radius: 4px; overflow: hidden; background: #2A3D5E; margin: 10px 0 4px; }
.bt-meta { font-size: 12px; color: #A3B3CC; }
.bt-dot { display: inline-block; width: 9px; height: 9px; border-radius: 2px; margin-right: 5px; }
.bt-note { font-size: 13px; line-height: 1.5; padding: 8px 12px; margin-top: -6px; background: #1D2D47;
           border-left: 3px solid #7CC4FA; border-radius: 0 6px 6px 0; }
</style>""", unsafe_allow_html=True)


def chart(fig, key):
    st.plotly_chart(fig, width="stretch", key=key, config={"displayModeBar": False})


def note(text):
    if text:
        st.markdown(f'<div class="bt-note">{escape(text)}</div>', unsafe_allow_html=True)


def nice(s):
    return A.nice_date(s)


def short(text, limit=230):
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + "…"


def dot(category):
    return f'<span class="bt-dot" style="background:{A.CAT_COLORS.get(category, "#999")}"></span>'


def play(text, key):
    """Listen button: ElevenLabs reads the text aloud (cached after the first time)."""
    if st.button("Listen", key=key, icon=":material/volume_up:"):
        uri, status = newsroom.speak(text)
        if uri:
            st.audio(base64.b64decode(uri.split(",", 1)[1]), format="audio/mpeg", autoplay=True)
        else:
            st.caption(status)


def story_box(snap, i, cluster, matched, numbered):
    """One AI story: label, title, short summary, category bar, Read more, Listen."""
    counts = matched["category"].value_counts()
    total = int(counts.sum())
    label = f'<span class="bt-num">{i + 1}</span>' if numbered else '<span class="bt-tag">AI story</span>'
    bar = "".join(f'<span style="width:{n / total:.0%};background:{A.CAT_COLORS.get(c, "#999")}"></span>'
                  for c, n in counts.items()) if total else ""
    meta = (f"Based on {total} headline{'s' if total != 1 else ''} · " + ", ".join(c.title() for c in counts.index)
            if total else "No matching headlines")
    with st.container(border=True):
        st.markdown(f'{label}<div class="bt-title">{escape(cluster.get("title", ""))}</div>'
                    f'<div class="bt-text">{escape(short(cluster.get("summary", "")))}</div>'
                    f'<div class="bt-bar">{bar}</div><div class="bt-meta">{escape(meta)}</div>', unsafe_allow_html=True)
        with st.expander("Read more"):
            st.write(cluster.get("summary", ""))
            for r in matched.itertuples():
                st.markdown(f"- [{r.title}]({r.url})")
        play(f"{cluster.get('title', '')}. {cluster.get('summary', '')}", key=f"listen_{snap}_{i}")


def headline_box(r):
    """One top headline: label, category and source, title link, Read more."""
    with st.container(border=True):
        st.markdown(f'<span class="bt-tag plain">Top headline</span>'
                    f'<div class="bt-meta" style="margin-top:8px">{dot(r.category)}{r.category.title()} · {escape(r.source)}</div>'
                    f'<div class="bt-title"><a href="{escape(r.url)}" target="_blank">{escape(r.title)}</a></div>',
                    unsafe_allow_html=True)
        if r.description:
            with st.expander("Read more"):
                st.write(r.description)
        st.markdown(f"[Full article ↗]({r.url})")


def grid(items, per_row=4):
    """Lay out (kind, args) items in rows of per_row equal columns."""
    for start in range(0, len(items), per_row):
        cols = st.columns(per_row)
        for col, (kind, args) in zip(cols, items[start:start + per_row]):
            with col:
                kind(*args)


# header
live = A.IS_LIVE
badge = ('<span class="bt-badge live"><span class="live-dot"></span>Live news</span>' if live
         else '<span class="bt-badge">Saved news (offline)</span>')
st.markdown('<div class="bt-kicker">What\'s in the news, and how it\'s told</div>', unsafe_allow_html=True)
st.title("The Buffalo Times")
st.markdown(f'<div class="bt-badges">{badge}<span class="bt-badge">{A.DF["uid"].nunique()} headlines · '
            f'{len(A.SNAPS)} dates</span></div>', unsafe_allow_html=True)
if A.PRICES:
    st.caption("Markets: " + " · ".join(f"**{s}** {p:,.2f}" for s, p in list(A.PRICES.items())[:10]))

with st.sidebar:
    st.markdown("### How to use")
    st.markdown("1. Pick a tab\n2. Use the filters at the top of the tab\n3. Hover over charts for details\n"
                "4. Use **Read more** and **Listen** on story cards")
    st.markdown("---")
    st.caption("Live news from NewsAPI: today and the past 7 days." if live
               else "Offline: the four news snapshots saved in class. Add API keys for live news.")
    st.markdown("[Code on GitHub](https://github.com/shruti23-ui/Newsapp_eas503)")

tabs = st.tabs(["Daily Briefing", "Entity Explorer", "Echo Index", "Newsroom Check", "Ask"])

# Daily Briefing
with tabs[0]:
    c1, c2 = st.columns(2)
    snap = c1.selectbox("Date", A.SNAPS[::-1], format_func=nice, key="b_snap")
    cats = c2.multiselect("Categories", A.CATS, format_func=str.title, key="b_cats", placeholder="All categories")
    _, f_sent, f_pol, f_cat, f_op, _ = A.update_briefing(snap, cats or None, THEME)
    n_sent, n_pol, n_cat, n_op = A.briefing_notes(snap, cats or None)

    title = "Today's top stories" if snap == A.LATEST else f"Top stories on {nice(snap)}"
    st.subheader(title + (" · " + ", ".join(c.title() for c in cats) if cats else ""))
    summ = A.SUMMARIES.get(snap)
    if summ and not cats:
        st.caption("AI summary grouped into stories. The colored bar shows which categories each story's headlines come from.")
        st.markdown(f'<div class="bt-lede">{escape(summ.get("global_summary", ""))}</div>', unsafe_allow_html=True)
        stories = A.story_matches(snap, None)
        grid([(story_box, (snap, i, c, m, True)) for i, c, m in stories], per_row=len(stories) or 1)
    else:
        st.caption("No AI summary was made for this date, so these are the top headlines." if not summ else
                   "AI story cards summarise several headlines. Top headline cards are single articles, in NewsAPI's order.")
        stories = [(story_box, (snap, i, c, m, False)) for i, c, m in A.story_matches(snap, cats or None)]
        rows_key = f"rows_{snap}_{'-'.join(cats)}"          # "Load more" count, remembered per filter
        rows = st.session_state.get(rows_key, 1)
        heads = A.top_headline_rows(snap, cats or None)
        n_tiles = 4 * rows - len(stories) % 4                 # fill whole rows of 4
        grid(stories + [(headline_box, (r,)) for r in heads.head(n_tiles).itertuples()])
        if n_tiles < len(heads) and st.button("Load more", key=f"more_{rows_key}"):
            st.session_state[rows_key] = rows + 1
            st.rerun()

    a, b, c = st.columns(3)
    with a:
        st.markdown("**Headline mood**")
        chart(f_sent, "sent")
        note(n_sent)
    with b:
        st.markdown("**Mood scores** · from −1 (very negative) to +1 (very positive)")
        chart(f_pol, "pol")
        note(n_pol)
    with c:
        st.markdown("**Average mood by category**")
        chart(f_cat, "catpol")
        note(n_cat)
    st.markdown("**Opinion map** · each dot is a headline. Left to right: negative to positive. Bottom to top: factual to opinionated.")
    chart(f_op, "opinion")
    note(n_op)

    st.markdown("**All headlines** · click a column name to sort")
    sub = analytics.unique_articles(A.DF[A.DF["snapshot"] == snap])
    if cats:
        sub = sub[sub["category"].isin(cats)]
    table = sub[["title", "url", "category", "source", "polarity", "subjectivity", "zsc_sentiment"]].rename(
        columns={"title": "Headline", "category": "Category", "source": "Source", "polarity": "Mood",
                 "subjectivity": "Opinion", "zsc_sentiment": "Sentiment"})
    st.dataframe(table.round(2), column_config={"url": st.column_config.LinkColumn("Link", display_text="open")},
                 hide_index=True, width="stretch")

    if A.ZSC_OK:   # only meaningful when the zero-shot model ran
        st.markdown("**Do the two sentiment models agree?** · Cohen's kappa = agreement beyond chance")
        conf, kappa, n, _ = analytics.sentiment_agreement(A.DF)
        st.metric("Cohen's kappa", f"{kappa:.2f}", f"n = {n}")
        chart(A.update_agreement(THEME)[1], "agree")

# Entity Explorer
with tabs[1]:
    st.caption("Who is the news about? Names of people, organizations and places are found in every headline and counted.")
    c1, c2, c3, c4 = st.columns(4)
    e_snap = c1.selectbox("Date", ["ALL"] + A.SNAPS[::-1], format_func=lambda s: "All dates" if s == "ALL" else nice(s))
    e_cats = c2.multiselect("Categories", A.CATS, format_func=str.title, key="e_cats", placeholder="All categories")
    e_types = c3.multiselect("Entity types", list(TYPE_LABELS), default=list(TYPE_LABELS), format_func=TYPE_LABELS.get)
    topn = c4.slider("Show top", 5, 25, 15, 5)
    bar, heat, options = A.update_entities(e_snap, e_cats or None, e_types, topn, THEME)
    a, b = st.columns(2)
    with a:
        st.markdown("**Most mentioned** · bar length = headlines, color = their average mood")
        chart(bar, "ebar")
    with b:
        st.markdown("**Where they appear** · headlines per name and category")
        chart(heat, "eheat")
    if options:
        name = st.selectbox("Show details for", [o["value"] for o in options])
        E = analytics.filter_entities(A.E_ALL, e_snap, e_cats or None, e_types)
        hits = E[E["entity"] == name].drop_duplicates("uid")
        m1, m2, m3 = st.columns(3)
        m1.metric("Headlines", len(hits))
        m2.metric("Dates mentioned", f"{hits['snapshot'].nunique()} of {len(A.SNAPS)}")
        m3.metric("Average mood", f"{hits['polarity'].mean():+.2f}")
        pairs = analytics.cooccurrence(E, list(analytics.entity_summary(E).head(40)["entity"]) + [name])
        co = sorted([(y if x == name else x, k) for (x, y), k in pairs.items() if name in (x, y)], key=lambda t: -t[1])
        st.write(f"**{name} appears together with:** " + (", ".join(f"{o} ×{k}" for o, k in co[:10]) or "none"))
        st.dataframe(hits[["snapshot", "category", "title", "polarity"]].sort_values("snapshot", ascending=False)
                     .rename(columns={"snapshot": "Date", "category": "Category", "title": "Headline", "polarity": "Mood"}).round(2),
                     hide_index=True, width="stretch")

# Echo Index
with tabs[2]:
    st.caption("Is today's news actually new? Each headline is compared with every headline from earlier dates. "
               "Echo score: 0 = a brand new story, 1 = the same story again.")
    focus_opts = [s for s in A.SNAPS[::-1] if s != A.SNAPS[0]]
    c1, c2 = st.columns(2)
    focus = c1.selectbox("Date to check", focus_opts, format_func=nice)
    x_cats = c2.multiselect("Categories", A.CATS, format_func=str.title, key="x_cats", placeholder="All categories")
    _, trend, f_xcat, _, _, terms, senttime = A.update_echo(focus, x_cats or None, THEME)
    cur = A.ECHO.dropna(subset=["echo"])
    cur = cur[(cur["snapshot"] == focus) & (cur["category"].isin(x_cats) if x_cats else True)]
    m1, m2, m3 = st.columns(3)
    m1.metric("Average echo", f"{cur['echo'].mean():.2f}")
    m2.metric("Repeated stories", int((cur["echo"] >= 0.5).sum()), "echo 0.5 or higher", delta_color="off")
    m3.metric("New stories", int((cur["echo"] < 0.1).sum()), "echo under 0.1", delta_color="off")
    a, b = st.columns(2)
    with a:
        st.markdown("**Echo over time** · a rising line means more repeated stories")
        chart(trend, "trend")
    with b:
        st.markdown("**Echo by category** · on the selected date")
        chart(f_xcat, "xcat")
    a, b = st.columns(2)
    srt = cur.sort_values("echo", ascending=False)
    with a:
        st.markdown("**Repeated stories** · with the earlier headline each one resembles")
        for r in srt.head(5).itertuples():
            st.markdown(f"- **{r.title}** ({r.echo:.2f})  \n  closest earlier ({nice(r.echo_of_snapshot)}): {r.echo_of}")
    with b:
        st.markdown("**New stories** · nothing similar on any earlier date")
        for r in srt.tail(5).iloc[::-1].itertuples():
            st.markdown(f"- **{r.title}** ({r.echo:.2f})")
    st.markdown("**Rising and fading words** · used more (right) or less (left) on this date than before")
    chart(terms, "terms")
    shift = analytics.term_shift(A.DF, focus)
    if not shift.empty:
        word = st.selectbox("Headlines containing", list(pd.concat([shift.head(8), shift.tail(8)])["word"]))
        rows = A.term_headlines({"points": [{"y": word}]})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.markdown("**Mood by category over time**")
    chart(senttime, "senttime")

# Newsroom Check
with tabs[3]:
    st.caption("How well does the AI summary cover the news? Go through the four steps in order: "
               "measure it, check its weakest claims, fix it, then listen to it.")
    audit_snaps = [s for s in A.SNAPS[::-1] if s in A.SUMMARIES]
    if not audit_snaps:
        st.info("No AI summary to check yet.")
    else:
        c1, c2 = st.columns(2)
        n_snap = c1.selectbox("Summary date", audit_snaps, format_func=nice)
        th = c2.slider("Match strictness (higher = stricter)", 0.05, 0.30, 0.10, 0.01)
        heads, sents = A.AUDITS[n_snap]["heads"], A.AUDITS[n_snap]["sents"]
        fig, cov = A.coverage_figure(heads, th, THEME)
        k = int((heads["similarity"] >= th).sum())
        missing = list(cov.loc[cov["represented"] == 0, "category"])
        st.subheader("Step 1 · Measure coverage")
        m1, m2, m3 = st.columns(3)
        m1.metric("Headlines covered", f"{k / len(heads):.0%}", f"{k} of {len(heads)} headlines", delta_color="off")
        m2.metric("Categories left out", f"{len(missing)} of {len(A.CATS)}", ", ".join(missing) or "none", delta_color="off")
        m3.metric("Weak claims", int((sents["support"] < 0.25).sum()), f"of {len(sents)} sentences", delta_color="off")
        chart(fig, "cov")
        st.markdown("**Claim check** · each sentence of the summary, weakest first")
        st.dataframe(sents.sort_values("support")[["where", "sentence", "support", "best_headline"]]
                     .rename(columns={"where": "Section", "sentence": "Claim", "support": "Support",
                                      "best_headline": "Closest headline"}).round(2),
                     hide_index=True, width="stretch")

        st.subheader("Step 2 · Check weak claims")
        if st.button("Verify weakest claims"):
            st.session_state["verified"] = A.verify_claims(1, n_snap)[1]
        for v in st.session_state.get("verified", []) or []:
            st.markdown(f"**“{v['sentence']}”** · support {v['support']} · {v['status']}")
            for s in v["sources"]:
                st.markdown(f"- [{s['title']}]({s['url']}) · {s['domain']} · relevance {s['score']}")

        st.subheader("Step 3 · Fix the summary")
        if st.button("Rewrite and re-audit"):
            bulletin, status = newsroom.rewrite_balanced(n_snap, A.DF[A.DF["snapshot"] == n_snap])
            st.session_state["bulletin"], st.session_state["fix_status"] = bulletin, status
        bulletin = st.session_state.get("bulletin")
        if bulletin:
            after = analytics.audit_summary(A.DF[A.DF["snapshot"] == n_snap], newsroom.bulletin_as_summary(bulletin))["heads"]
            st.metric("Coverage before → after", f"{k / len(heads):.0%} → {(after['similarity'] >= th).mean():.0%}")
            chart(A.coverage_figure(heads, th, THEME, extra=after)[0], "fix")
        elif "fix_status" in st.session_state:
            st.warning(st.session_state["fix_status"])

        st.subheader("Step 4 · Listen")
        if st.button("Make the radio bulletin"):
            if not bulletin:
                st.warning("Do Step 3 first: the bulletin reads the rewritten summary.")
            else:
                uri, script, status = newsroom.radio_bulletin(bulletin)
                if uri:
                    st.audio(base64.b64decode(uri.split(",", 1)[1]), format="audio/mpeg")
                st.caption(status)
                for who, v, text in script:
                    st.markdown(f"**{who} (voice {v + 1}):** {text}")

        report = A.download_report(1, n_snap, th, st.session_state.get("verified"), bulletin)
        st.download_button("Download report (ZIP)", base64.b64decode(report["content"]),
                           file_name=report["filename"], mime="application/zip")

# Ask
with tabs[4]:
    st.caption("Ask a question about the news. The answer uses only the matching headlines, cites them by number, "
               "and each sentence is checked against its sources.")
    q = st.text_input("Your question", placeholder="e.g. What is happening with Iran?")
    scope = st.selectbox("Dates to search", ["ALL"] + A.SNAPS[::-1], format_func=lambda s: "All dates" if s == "ALL" else nice(s), key="a_snap")
    if st.button("Ask", type="primary") and q.strip():
        pool = A.DF if scope == "ALL" else A.DF[A.DF["snapshot"] == scope]
        hits = newsroom.retrieve(q, pool)
        if hits.empty:
            st.info("No headline shares words with that question. Try different wording.")
        else:
            answer, status = newsroom.ask(q, hits)
            if answer:
                st.markdown(answer)
                st.caption(status)
                st.markdown("**Grounding check** · each answer sentence vs the headlines it cites")
                st.dataframe(pd.DataFrame(newsroom.grounding(answer, hits)).round(2), hide_index=True, width="stretch")
            else:
                st.info(f"{status}. Showing the matching headlines only.")
            hits.index = range(1, len(hits) + 1)
            st.markdown("**Sources**")
            st.dataframe(hits[["score", "title", "snapshot", "category", "url"]]
                         .rename(columns={"score": "Match", "title": "Headline", "snapshot": "Date", "category": "Category"}).round(2),
                         column_config={"url": st.column_config.LinkColumn("Link", display_text="open")},
                         width="stretch")
