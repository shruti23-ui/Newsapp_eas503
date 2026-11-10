"""Streamlit front end reusing the data and figures from app.py."""
import base64

import pandas as pd
import streamlit as st

st.set_page_config(page_title="The Buffalo Times", layout="wide")


@st.cache_resource(show_spinner="Loading snapshots and running NLP (first run only)...")
def load_core():
    import app as core          # loads data, NLP, entities, echo scores, audits once
    return core


A = load_core()
import analytics  # noqa: E402  (after the cached import so paths/env are set)
import newsroom  # noqa: E402

THEME = "dark" if st.sidebar.radio("Chart theme", ["Dark", "Light"]) == "Dark" else "light"
st.sidebar.caption(A.STATUS["message"])
st.sidebar.caption(f"Sentiment: {A.SENTIMENT_METHOD}")


def chart(fig, key):
    st.plotly_chart(fig, use_container_width=True, key=key, config={"displayModeBar": False})


def nice(s):
    return A.nice_date(s)


st.markdown("##### EAS 503 · News Intelligence Dashboard")
st.title("The Buffalo Times")
st.caption(f"{A.DF['uid'].nunique()} headlines · {len(A.SNAPS)} snapshots · "
           f"Markets: {', '.join(f'{s} {p:,.2f}' for s, p in list(A.PRICES.items())[:8])} …")

tabs = st.tabs(["Daily Briefing", "Entity Explorer · EC 1", "Echo Index · EC 2",
                "Newsroom Check · EC 3", "Ask The Buffalo Times"])

# briefing
with tabs[0]:
    c1, c2 = st.columns(2)
    snap = c1.selectbox("Snapshot", A.SNAPS[::-1], format_func=nice, key="b_snap")
    cats = c2.multiselect("Categories", A.CATS, key="b_cats")
    _, f_sent, f_pol, f_cat, f_op, rows = A.update_briefing(snap, cats or None, THEME)

    summ = A.SUMMARIES.get(snap)
    if summ:
        st.subheader("Today's story clusters")
        st.write(summ["global_summary"])
        cols = st.columns(max(1, len(summ["clusters"])))
        for col, c in zip(cols, summ["clusters"]):
            col.markdown(f"**{c['title']}**")
            col.caption(c["summary"])
    else:
        st.info("No Gemini summary for this snapshot.")

    a, b, c = st.columns(3)
    with a:
        st.markdown("**Sentiment labels**")
        chart(f_sent, "sent")
    with b:
        st.markdown("**TextBlob polarity**")
        chart(f_pol, "pol")
    with c:
        st.markdown("**Mean polarity by category**")
        chart(f_cat, "catpol")
    st.markdown("**Opinion map** · polarity vs subjectivity (subjectivity was computed in class but never shown)")
    chart(f_op, "opinion")

    sub = analytics.unique_articles(A.DF[A.DF["snapshot"] == snap])
    if cats:
        sub = sub[sub["category"].isin(cats)]
    st.dataframe(sub[["title", "url", "category", "source", "polarity", "subjectivity", "zsc_sentiment", "zsc_scores"]]
                 .rename(columns={"zsc_sentiment": "label", "zsc_scores": "label confidence"}).round(2),
                 column_config={"url": st.column_config.LinkColumn("link", display_text="open")},
                 hide_index=True, use_container_width=True)

    st.markdown("**Do the two sentiment models agree?** (Cohen's kappa)")
    conf, kappa, n, _ = analytics.sentiment_agreement(A.DF)
    if conf is None:
        st.info("Zero-shot labels are not available here (the 2 GB model is not installed), so there is nothing to compare.")
    else:
        st.metric("Cohen's kappa", f"{kappa:.2f}", f"n = {n}")
        chart(A.update_agreement(THEME)[1], "agree")

# EC 1
with tabs[1]:
    st.caption("Who is the news about? spaCy NER, with an EntityRuler fixing names the small model mislabels.")
    c1, c2, c3, c4 = st.columns(4)
    e_snap = c1.selectbox("Snapshot", ["ALL"] + A.SNAPS[::-1], format_func=lambda s: "All snapshots" if s == "ALL" else nice(s))
    e_cats = c2.multiselect("Categories", A.CATS, key="e_cats")
    e_types = c3.multiselect("Types", ["PERSON", "ORG", "PLACE"], default=["PERSON", "ORG", "PLACE"])
    topn = c4.slider("How many", 5, 25, 15, 5)
    bar, heat, options = A.update_entities(e_snap, e_cats or None, e_types, topn, THEME)
    a, b = st.columns(2)
    with a:
        chart(bar, "ebar")
    with b:
        chart(heat, "eheat")
    if options:
        name = st.selectbox("Drill into an entity", [o["value"] for o in options])
        E = analytics.filter_entities(A.E_ALL, e_snap, e_cats or None, e_types)
        hits = E[E["entity"] == name].drop_duplicates("uid")
        m1, m2, m3 = st.columns(3)
        m1.metric("Headlines", len(hits))
        m2.metric("Snapshots present", f"{hits['snapshot'].nunique()} of {len(A.SNAPS)}")
        m3.metric("Mean polarity", f"{hits['polarity'].mean():+.2f}")
        pairs = analytics.cooccurrence(E, list(analytics.entity_summary(E).head(40)["entity"]) + [name])
        co = sorted([(y if x == name else x, k) for (x, y), k in pairs.items() if name in (x, y)], key=lambda t: -t[1])
        st.write("Appears together with: " + (", ".join(f"{o} ×{k}" for o, k in co[:10]) or "none"))
        st.dataframe(hits[["snapshot", "category", "title", "polarity"]].sort_values("snapshot", ascending=False).round(2),
                     hide_index=True, use_container_width=True)

# EC 2
with tabs[2]:
    st.caption("Echo = TF-IDF cosine similarity of a headline to its closest headline in earlier snapshots (0 new, 1 repeat).")
    focus_opts = [s for s in A.SNAPS[::-1] if s != A.SNAPS[0]]
    c1, c2 = st.columns(2)
    focus = c1.selectbox("Focus snapshot", focus_opts, format_func=nice)
    x_cats = c2.multiselect("Categories", A.CATS, key="x_cats")
    _, trend, f_xcat, _, _, terms, senttime = A.update_echo(focus, x_cats or None, THEME)
    cur = A.ECHO.dropna(subset=["echo"])
    cur = cur[(cur["snapshot"] == focus) & (cur["category"].isin(x_cats) if x_cats else True)]
    m1, m2, m3 = st.columns(3)
    m1.metric("Mean echo", f"{cur['echo'].mean():.2f}")
    m2.metric("Near-repeats (≥ 0.5)", int((cur["echo"] >= 0.5).sum()))
    m3.metric("Brand new (< 0.1)", int((cur["echo"] < 0.1).sum()))
    a, b = st.columns(2)
    with a:
        st.markdown("**Echo over time**")
        chart(trend, "trend")
    with b:
        st.markdown("**Echo by category**")
        chart(f_xcat, "xcat")
    a, b = st.columns(2)
    srt = cur.sort_values("echo", ascending=False)
    with a:
        st.markdown("**Déjà vu**")
        for r in srt.head(5).itertuples():
            st.markdown(f"- **{r.title}** ({r.echo:.2f})  \n  closest earlier ({nice(r.echo_of_snapshot)}): {r.echo_of}")
    with b:
        st.markdown("**Genuinely new**")
        for r in srt.tail(5).iloc[::-1].itertuples():
            st.markdown(f"- **{r.title}** ({r.echo:.2f})")
    st.markdown("**Rising and fading words** (log-odds, informative Dirichlet prior; |z| < 2 = directional)")
    chart(terms, "terms")
    shift = analytics.term_shift(A.DF, focus)
    if not shift.empty:
        word = st.selectbox("Headlines containing", list(pd.concat([shift.head(8), shift.tail(8)])["word"]))
        rows = A.term_headlines({"points": [{"y": word}]})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.markdown("**Sentiment by category over time**")
    chart(senttime, "senttime")

# EC 3
with tabs[3]:
    audit_snaps = [s for s in A.SNAPS[::-1] if s in A.SUMMARIES]
    if not audit_snaps:
        st.info("No Gemini summary to audit.")
    else:
        c1, c2 = st.columns(2)
        n_snap = c1.selectbox("Summary of snapshot", audit_snaps, format_func=nice)
        th = c2.slider("Match threshold (cosine similarity)", 0.05, 0.30, 0.10, 0.01)
        heads, sents = A.AUDITS[n_snap]["heads"], A.AUDITS[n_snap]["sents"]
        fig, cov = A.coverage_figure(heads, th, THEME)
        k = int((heads["similarity"] >= th).sum())
        missing = list(cov.loc[cov["represented"] == 0, "category"])
        st.subheader("Step 1 · Audit")
        m1, m2, m3 = st.columns(3)
        m1.metric("Headlines represented", f"{k / len(heads):.0%}", f"{k} of {len(heads)}")
        m2.metric("Categories left out", f"{len(missing)} of {len(A.CATS)}", ", ".join(missing) or "none", delta_color="off")
        m3.metric("Weak claims (< 0.25)", int((sents["support"] < 0.25).sum()))
        chart(fig, "cov")
        st.dataframe(sents.round(2), hide_index=True, use_container_width=True)

        st.subheader("Step 2 · Verify with Tavily")
        if st.button("Verify weakest claims"):
            st.session_state["verified"] = A.verify_claims(1, n_snap)[1]
        for v in st.session_state.get("verified", []) or []:
            st.markdown(f"**“{v['sentence']}”** · support {v['support']} · {v['status']}")
            for s in v["sources"]:
                st.markdown(f"- [{s['title']}]({s['url']}) · {s['domain']} · relevance {s['score']}")

        st.subheader("Step 3 · Fix with Gemini")
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
                st.warning("Run Step 3 first.")
            else:
                uri, script, status = newsroom.radio_bulletin(bulletin)
                if uri:
                    st.audio(base64.b64decode(uri.split(",", 1)[1]), format="audio/mpeg")
                st.caption(status)
                for who, v, text in script:
                    st.markdown(f"**{who} (voice {v + 1}):** {text}")

        report = A.download_report(1, n_snap, th, st.session_state.get("verified"), bulletin)
        st.download_button("Download audit report (ZIP)", base64.b64decode(report["content"]),
                           file_name=report["filename"], mime="application/zip")

# Ask
with tabs[4]:
    q = st.text_input("Your question", placeholder="e.g. What is happening with Iran?")
    scope = st.selectbox("Search in", ["ALL"] + A.SNAPS[::-1], format_func=lambda s: "All snapshots" if s == "ALL" else nice(s), key="a_snap")
    if st.button("Ask") and q.strip():
        pool = A.DF if scope == "ALL" else A.DF[A.DF["snapshot"] == scope]
        hits = newsroom.retrieve(q, pool)
        if hits.empty:
            st.info("No headline shares words with that question.")
        else:
            answer, status = newsroom.ask(q, hits)
            if answer:
                st.markdown(answer)
                st.caption(status)
                st.markdown("**Grounding check**")
                st.dataframe(pd.DataFrame(newsroom.grounding(answer, hits)).round(2), hide_index=True, use_container_width=True)
            else:
                st.info(f"{status}. Showing the retrieved headlines only.")
            hits.index = range(1, len(hits) + 1)
            st.dataframe(hits[["score", "title", "snapshot", "category", "url"]].round(2),
                         column_config={"url": st.column_config.LinkColumn("link", display_text="open")},
                         use_container_width=True)
