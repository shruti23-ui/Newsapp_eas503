"""The Buffalo Times - Dash news dashboard. Run: python app.py"""
import io
import json
import re
import zipfile

import pandas as pd
import plotly.graph_objects as go
from dash import MATCH, Dash, Input, Output, State, ctx, dash_table, dcc, html, no_update

from buffalo import ai_summary
from buffalo import analytics
from buffalo import config
from buffalo import data_pipeline
from buffalo import external
from buffalo import newsroom
from buffalo import nlp

# 1. DATA
DF, STATUS = data_pipeline.get_news_data()
DF, SENTIMENT_METHOD = nlp.enrich(DF)
SNAPS = sorted(DF["snapshot"].unique()) if not DF.empty else []
LATEST = SNAPS[-1] if SNAPS else None
IS_LIVE = STATUS["mode"] == "live"

SUMMARY_STATUS = "Saved Gemini output"
if IS_LIVE and LATEST and LATEST not in ai_summary.load_summaries():
    _heads = DF.loc[DF["snapshot"] == LATEST, "title"].drop_duplicates().tolist()
    _, SUMMARY_STATUS = ai_summary.summarise_snapshot(LATEST, _heads)
SUMMARIES = {k: v for k, v in ai_summary.load_summaries().items() if k in SNAPS}
AUDITS = {s: analytics.audit_summary(DF[DF["snapshot"] == s], summ) for s, summ in SUMMARIES.items()}

E_ALL = analytics.entity_rows(DF)
ECHO = analytics.echo_scores(DF)
PRICES, PRICE_STATUS = external.get_stock_prices(IS_LIVE)
BLUESKY, BLUESKY_STATUS = external.get_bluesky_posts()

CATS = config.CATEGORIES
CAT_COLORS = {"business": "#5AA9E6", "entertainment": "#F2A65A", "health": "#5FC9A8",
              "science": "#B39DDB", "sports": "#E8C547", "technology": "#F07F86", "general": "#A9B6CB"}
SENT_COLORS = {"positive": "#4C9BE8", "negative": "#F2994A", "neutral": "#8C9AB3"}
TYPE_NAMES = {"PERSON": "Person", "ORG": "Organization", "GPE": "Place", "LOC": "Place",
              "FAC": "Place", "NORP": "Group", "EVENT": "Event", "PRODUCT": "Product",
              "WORK_OF_ART": "Title", "LAW": "Law"}


def nice_date(label):
    try:
        return pd.Timestamp(label).strftime("%b %d").replace(" 0", " ")
    except (ValueError, TypeError):
        return str(label)


LOW_CONF = 0.6     # zero-shot labels below this probability are treated as uncertain
ZSC_OK = analytics.sentiment_agreement(DF)[0] is not None   # False when the zero-shot model is not installed
SNAP_OPTIONS = [{"label": nice_date(s), "value": s} for s in reversed(SNAPS)]
CAT_OPTIONS = [{"label": c.title(), "value": c} for c in CATS]

# 2. THEME + FIGURES
THEMES = {
    "dark": {"text": "#E8EEF6", "muted": "#A3B3CC", "grid": "#2A3D5E", "accent": "#7CC4FA", "warn": "#F2A65A"},
    "light": {"text": "#17202E", "muted": "#4F5E74", "grid": "#D5DCE6", "accent": "#1F6FB5", "warn": "#B45309"},
}


def styled(fig, theme, height=340):
    t = THEMES.get(theme, THEMES["dark"])
    fig.update_layout(template="plotly_dark" if theme == "dark" else "plotly_white",
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Archivo, system-ui, sans-serif", color=t["text"], size=13),
                      margin=dict(l=10, r=10, t=10, b=10), height=height,
                      legend=dict(orientation="h", y=-0.18, bgcolor="rgba(0,0,0,0)"),
                      hoverlabel=dict(font_family="Archivo, sans-serif"))
    fig.update_xaxes(gridcolor=t["grid"], zerolinecolor=t["grid"])
    fig.update_yaxes(gridcolor=t["grid"], zerolinecolor=t["grid"])
    return fig


def empty_fig(message, theme, height=260):
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font=dict(color=THEMES[theme]["muted"], size=14))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return styled(fig, theme, height)


# small components
def card(title, sub=None, *children, wide=False):
    head = [html.H3(title)] + ([html.P(sub, className="sub")] if sub else [])
    return html.Section(head + list(children), className="card wide" if wide else "card")


def kpi(label, value, note="", warn=False):
    return html.Div([html.Div(label, className="k-label"),
                     html.Div(value, className="k-value warn" if warn else "k-value"),
                     html.Div(note, className="k-note")], className="kpi")


def control(label, component):
    return html.Div([html.Label(label), component], className="control")


def note(id_):
    """A short 'what this shows' line under a chart, filled in by a callback."""
    return html.P(id=id_, className="note")


def graph(id_):
    return dcc.Loading(dcc.Graph(id=id_, config={"displayModeBar": False}), type="dot")


def table(id_, columns, page_size=8):
    """columns: list of (id, header, is_markdown)."""
    return dash_table.DataTable(
        id=id_, data=[], page_size=page_size, sort_action="native",
        columns=[{"id": c, "name": n, "presentation": "markdown" if md else "input"} for c, n, md in columns],
        markdown_options={"link_target": "_blank"}, style_as_list_view=True,
        style_table={"overflowX": "auto"},
        style_header={"backgroundColor": "var(--panel2)", "color": "var(--text)", "fontWeight": "600",
                      "border": "none", "fontFamily": "Archivo, sans-serif"},
        style_cell={"backgroundColor": "var(--panel)", "color": "var(--text)", "border": "none",
                    "borderBottom": "1px solid var(--border)", "fontFamily": "Archivo, sans-serif",
                    "fontSize": "13px", "textAlign": "left", "whiteSpace": "normal", "height": "auto",
                    "padding": "8px", "maxWidth": "520px"})


def link(title, url):
    safe = str(title).replace("[", "(").replace("]", ")")
    return f"[{safe}]({url})" if url else safe


def dot(category):
    return html.Span(className="dot", style={"background": CAT_COLORS.get(category, "#999")})


# 3. LAYOUT
def guide():
    steps = [("1", "Pick a tab"), ("2", "Use the filters at the top of the tab"),
             ("3", "Click any bar, cell or word to see the headlines behind it"), ("4", "Hover over charts for details")]
    return html.Div([html.Span([html.B(n + ". "), text]) for n, text in steps], className="guide")


def headline_columns():
    cols = [("title", "Headline", True), ("category", "Category", False), ("source", "Source", False),
            ("polarity", "Mood", False), ("subjectivity", "Opinion", False), ("sentiment", "Sentiment", False)]
    if ZSC_OK:
        cols.append(("confidence", "Confidence", False))
    return cols


def agreement_panel():
    # only shown when the zero-shot model ran, otherwise there is nothing to compare
    if not ZSC_OK:
        return None
    return html.Div([card("Do the two sentiment models agree?",
                          "TextBlob vs the zero-shot model. Kappa = agreement beyond chance (1 = perfect). Click a cell to read those headlines.",
                          html.Div(id="b-kappa"), graph("b-agree")),
                     card("Headlines in the selected cell", None, table("b-agree-table", [
                         ("title", "Headline", True), ("textblob", "TextBlob", False),
                         ("zsc", "Zero-shot", False), ("polarity", "Mood", False)], 6))], className="row")


def header():
    # live data gets a blinking red dot; saved data gets a plain label
    badge = [html.Span(className="live-dot"), "Live news"] if IS_LIVE else "Saved news (offline)"
    return html.Header([
        html.Div([html.Div("What's in the news, and how it's told", className="kicker"),
                  html.H1(["The Buffalo ", html.Span("Times")], className="title")]),
        html.Div([html.Span(badge, className="badge live" if IS_LIVE else "badge"),
                  html.Span(f"{DF['uid'].nunique() if not DF.empty else 0} headlines · {len(SNAPS)} dates", className="badge"),
                  html.Button(html.Span(className="icon-sun"), id="theme-btn", className="theme-btn", n_clicks=0,
                              title="Switch to light mode", **{"aria-label": "Switch between light and dark mode"})],
                 className="badges"),
    ], className="masthead")


def tickers():
    heads = DF.loc[DF["snapshot"] == LATEST, "title"].drop_duplicates().head(12).tolist() if LATEST else []
    head_items = [html.Span(h) for h in heads] or [html.Span("No headlines loaded")]
    price_items = [html.Span([html.B(s), f" {p:,.2f}"]) for s, p in PRICES.items()] or [html.Span(PRICE_STATUS)]
    return html.Div([
        html.Div([html.Span("HEADLINES", className="ticker-label"),
                  html.Div(html.Div(head_items + head_items, className="ticker-track"), className="ticker-window")],
                 className="ticker"),
        html.Div([html.Span("MARKETS", className="ticker-label"),
                  html.Div(html.Div(price_items + price_items, className="ticker-track fast"), className="ticker-window")],
                 className="ticker"),
    ])


def bluesky_panel():
    if not BLUESKY:
        body = html.Div("Bluesky posts could not be loaded right now.", className="notice")
    else:
        body = html.Div([html.Div([html.H4("@" + p["source"]), html.P(p["text"][:280]),
                                   html.A("Open on Bluesky", href=p["link"], target="_blank")], className="mini")
                         for p in BLUESKY[:6]], className="row")
    return card("News outlets on Bluesky", "Latest posts from major publishers", body, wide=True)


def tab_briefing():
    return html.Div([
        html.Div([control("Date", dcc.Dropdown(id="b-snap", options=SNAP_OPTIONS, value=LATEST, clearable=False)),
                  control("Categories", dcc.Dropdown(id="b-cats", options=CAT_OPTIONS, multi=True, placeholder="All categories"))],
                 className="controls"),
        html.Div(id="b-summary"),
        html.Div([card("Headline mood", "How many headlines read positive, negative or neutral", graph("b-sent"), note("b-sent-note")),
                  card("Mood scores", "From −1 (very negative) to +1 (very positive)", graph("b-pol"), note("b-pol-note")),
                  card("Average mood by category", None, graph("b-catpol"), note("b-cat-note"))], className="row"),
        card("Opinion map", "Each dot is a headline. Left to right: negative to positive. "
             "Bottom to top: factual to opinionated. Hover over a dot to read it.", graph("b-opinion"), note("b-op-note"), wide=True),
        card("All headlines", "Click a column name to sort. Click a headline to open the article.", table("b-table", headline_columns(), 10),
             html.P(f"Confidence is how sure the sentiment model was. Labels under {LOW_CONF:.0%} are faded.", className="sub")
             if ZSC_OK else None, wide=True),
        agreement_panel(),
        bluesky_panel(),
    ], className="section")


def tab_entities():
    return html.Div([
        html.P("Who is the news about? Names of people, organizations and places are found in every headline and counted. "
               "Click a bar or a heatmap cell to see the headlines behind it.", className="intro"),
        html.Div([control("Date", dcc.Dropdown(id="e-snap", clearable=False, value="ALL",
                                               options=[{"label": "All dates", "value": "ALL"}] + SNAP_OPTIONS)),
                  control("Categories", dcc.Dropdown(id="e-cats", options=CAT_OPTIONS, multi=True, placeholder="All categories")),
                  control("Entity types", dcc.Checklist(id="e-types", className="checklist", value=["PERSON", "ORG", "PLACE"],
                                                        options=[{"label": "People", "value": "PERSON"},
                                                                 {"label": "Organizations", "value": "ORG"},
                                                                 {"label": "Places", "value": "PLACE"}], inline=True)),
                  control("Show top", dcc.Slider(id="e-topn", min=5, max=25, step=5, value=15,
                                                 marks={i: str(i) for i in range(5, 30, 5)}))], className="controls"),
        html.Div([card("Most mentioned", "Bar length = number of headlines. Color = their average mood. Click a bar for details.", graph("e-bar")),
                  card("Where they appear", "Headlines per name and category. Click a cell for details.", graph("e-heat"))], className="row"),
        html.Div([control("Show details for", dcc.Dropdown(id="e-select", clearable=False))], className="controls"),
        html.Div(id="e-detail"),
    ], className="section")


def tab_echo():
    focus_opts = [o for o in SNAP_OPTIONS if o["value"] != (SNAPS[0] if SNAPS else None)]
    return html.Div([
        html.P("Is today's news actually new? Each headline is compared with every headline from earlier dates. "
               "Its echo score runs from 0 (a brand new story) to 1 (the same story again).", className="intro"),
        html.Div([control("Date to check", dcc.Dropdown(id="x-focus", options=focus_opts, clearable=False,
                                                         value=focus_opts[0]["value"] if focus_opts else None)),
                  control("Categories", dcc.Dropdown(id="x-cats", options=CAT_OPTIONS, multi=True, placeholder="All categories"))],
                 className="controls"),
        html.Div(id="x-kpis", className="row"),
        html.Div([card("Echo over time", "Average echo score per date. A rising line means more repeated stories.", graph("x-trend")),
                  card("Echo by category", "Average echo score on the selected date", graph("x-cat"))], className="row"),
        html.Div([card("Repeated stories", "Each shown with the earlier headline it most resembles", html.Div(id="x-deja")),
                  card("New stories", "Nothing similar on any earlier date", html.Div(id="x-fresh"))], className="row"),
        html.Div([card("Rising and fading words", "Words used more (right) or less (left) on this date than before. Click a word to see its headlines.", graph("x-terms")),
                  card("Headlines with the selected word", None, table("x-term-table", [
                      ("snapshot", "Date", False), ("category", "Category", False), ("title", "Headline", True)], 8))], className="row"),
        card("Mood by category over time", "Average mood score per category on each date", graph("x-senttime"), wide=True),
    ], className="section")


def tab_newsroom():
    audit_opts = [o for o in SNAP_OPTIONS if o["value"] in SUMMARIES]
    return html.Div([
        html.P("How well does the AI summary cover the news? Go through the four steps in order: "
               "measure it, check its weakest claims, fix it, then listen to it.", className="intro"),
        html.Div([control("Summary date", dcc.Dropdown(id="n-snap", options=audit_opts, clearable=False,
                                                              value=audit_opts[0]["value"] if audit_opts else None)),
                  control("Match strictness (higher = stricter)", dcc.Slider(id="n-th", min=0.05, max=0.30, step=0.01, value=0.10,
                                                                            marks={v: f"{v:.2f}" for v in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]}))],
                 className="controls"),
        html.H3("Step 1 · Measure coverage", style={"margin": "8px 0 0"}),
        html.Div(id="n-kpis", className="row"),
        html.Div([card("Coverage by category", "Share of each category's headlines that the summary mentions", graph("n-cov")),
                  card("Headlines behind each summary section", None, html.Div(id="n-clusters"))], className="row"),
        card("Left out of the summary", "Headlines from categories the summary skips entirely", html.Div(id="n-left"), wide=True),
        card("Claim check", "Each sentence of the summary, weakest first. Orange = little support in the headlines.",
             table("n-claims", [("where", "Section", False), ("sentence", "Claim", False),
                                ("support", "Support", False), ("best_headline", "Closest headline", False)], 8), wide=True),
        html.H3("Step 2 · Check weak claims", style={"margin": "8px 0 0"}),
        card("Do other news sites back them up?", "Searches recent news on the web for the three weakest sentences.",
             html.Button("Verify weakest claims", id="n-verify", className="btn", n_clicks=0),
             dcc.Loading(html.Div(id="n-verify-out")), wide=True),
        html.H3("Step 3 · Fix the summary", style={"margin": "8px 0 0"}),
        card("Balanced rewrite", "The AI rewrites the summary to cover every category, then it is scored again the same way.",
             html.Button("Rewrite and re-audit", id="n-fix", className="btn", n_clicks=0),
             dcc.Loading(html.Div(id="n-fix-out")), wide=True),
        html.H3("Step 4 · Listen", style={"margin": "8px 0 0"}),
        card("Radio bulletin", "The rewritten summary read aloud, one voice per section. Do Step 3 first.",
             html.Button("Make the radio bulletin", id="n-listen", className="btn", n_clicks=0),
             dcc.Loading(html.Div(id="n-listen-out")), wide=True),
        card("Download report", "Everything on this page as a ZIP of CSV and JSON files.",
             html.Button("Download report", id="n-dl-btn", className="btn", n_clicks=0), dcc.Download(id="n-dl"), wide=True),
        dcc.Store(id="n-verify-store"), dcc.Store(id="n-bulletin-store"),
    ], className="section")


def tab_ask():
    return html.Div([
        html.P("Ask a question about the news. The answer uses only the matching headlines, cites them by number, "
               "and each sentence is checked against its sources.", className="intro"),
        html.Div([control("Your question", dcc.Input(id="a-q", type="text", debounce=False, n_submit=0,
                                                     placeholder="e.g. What is happening with Iran?",
                                                     style={"minHeight": "40px", "padding": "0 12px", "borderRadius": "8px",
                                                            "border": "1px solid var(--border)", "background": "var(--panel2)",
                                                            "color": "var(--text)", "font": "inherit"})),
                  control("Dates to search", dcc.Dropdown(id="a-snap", clearable=False, value="ALL",
                                                          options=[{"label": "All dates", "value": "ALL"}] + SNAP_OPTIONS)),
                  html.Button("Ask", id="a-go", className="btn", n_clicks=0)], className="controls"),
        dcc.Loading(html.Div(id="a-out")),
    ], className="section")


app = Dash(__name__, title=config.APP_NAME, suppress_callback_exceptions=True)
server = app.server   # for gunicorn / Render

app.layout = html.Div(id="root", className="app theme-dark", children=[
    dcc.Store(id="theme", data="dark", storage_type="local"),
    header(), tickers(),
    html.Div([guide(), dcc.Tabs(id="tabs", value="briefing", children=[
        dcc.Tab(label="Daily Briefing", value="briefing", className="tab", selected_className="tab--selected", children=tab_briefing()),
        dcc.Tab(label="Entity Explorer", value="entities", className="tab", selected_className="tab--selected", children=tab_entities()),
        dcc.Tab(label="Echo Index", value="echo", className="tab", selected_className="tab--selected", children=tab_echo()),
        dcc.Tab(label="Newsroom Check", value="newsroom", className="tab", selected_className="tab--selected", children=tab_newsroom()),
        dcc.Tab(label="Ask", value="ask", className="tab", selected_className="tab--selected", children=tab_ask()),
    ])], className="tabs-wrap"),
])

# 4. CALLBACKS
# theme
def toggle_theme(n_clicks, current):
    if not n_clicks:
        return no_update
    return "light" if current == "dark" else "dark"


def apply_theme(theme):
    theme = theme or "dark"
    # the button shows the mode you would switch TO: a sun in dark mode, a moon in light mode
    icon = html.Span(className="icon-moon" if theme == "light" else "icon-sun")
    title = "Switch to dark mode" if theme == "light" else "Switch to light mode"
    return f"app theme-{theme}", icon, title


app.callback(Output("theme", "data"), Input("theme-btn", "n_clicks"), State("theme", "data"))(toggle_theme)
app.callback(Output("root", "className"), Output("theme-btn", "children"), Output("theme-btn", "title"),
             Input("theme", "data"))(apply_theme)


# briefing
STORY_MATCH = 0.10   # a headline counts toward a story when its similarity to the story is at least this


def story_card(i, cluster, matched):
    counts = matched["category"].value_counts()
    total = int(counts.sum())
    # stacked bar: one colored segment per category, width = its share of the story's headlines
    bar = html.Div([html.Span(style={"width": f"{n / total:.0%}", "background": CAT_COLORS.get(c, "#999")}, title=f"{c.title()}: {n}")
                    for c, n in counts.items()], className="story-bar") if total else None
    meta = f"{total} headline{'s' if total != 1 else ''} · " + ", ".join(c.title() for c in counts.index) if total else "No matching headlines"
    # all matched headlines are listed; CSS shows only the first two until "Read more" is clicked
    links = [html.Li(html.A(r.title, href=r.url, target="_blank")) for r in matched.itertuples()]
    return html.Div([
        html.Div([html.Span(str(i + 1), className="story-num"), html.H4(cluster.get("title", ""))], className="story-head"),
        html.P(cluster.get("summary", ""), className="story-text"),
        bar, html.Div(meta, className="story-meta"),
        html.Ul(links, className="story-links") if links else None,
        html.Div([html.Button("Read more", id={"type": "story-more", "index": i}, className="link-btn", n_clicks=0),
                  html.Button([html.Span(className="icon-speaker"), "Listen"], id={"type": "story-listen", "index": i},
                              className="link-btn", n_clicks=0, title="Read this story aloud")], className="story-actions"),
        dcc.Loading(html.Div(id={"type": "story-audio", "index": i}), type="dot"),
    ], id={"type": "story", "index": i}, className="story")


def top_headlines(snap, cats, per_category=3):
    """The first headlines NewsAPI returned for each category (NewsAPI's order is its 'top' order)."""
    sub = analytics.unique_articles(DF[DF["snapshot"] == snap])
    chosen = cats or CATS
    items = []
    for c in chosen:
        for r in sub[sub["category"] == c].head(per_category).itertuples():
            items.append(html.Div([html.A([dot(c), r.title], href=r.url, target="_blank"),
                                   html.Span(f"{r.source} · {c.title()}", className="meta"),
                                   html.Span(r.description[:220], className="meta") if r.description else None],
                                  className="list-item"))
    names = ", ".join(c.title() for c in chosen) if cats else "each category"
    return html.Div([html.H4(f"Top headlines in {names}", className="list-title")] + items) if items else None


def summary_cards(snap, cats=None):
    title = "Today's top stories" if snap == LATEST else f"Top stories on {nice_date(snap)}"
    if cats:
        title += " · " + ", ".join(c.title() for c in cats)
    summ = SUMMARIES.get(snap)
    if not summ:
        # no AI summary for this date: show the real top headlines instead
        return card(title, "No AI summary was made for this date, so these are the top headlines from NewsAPI.",
                    top_headlines(snap, cats), wide=True)

    heads = AUDITS[snap]["heads"]
    if cats:
        heads = heads[heads["category"].isin(cats)]   # keep only headlines from the selected categories
    stories = []
    for i, cluster in enumerate(summ["clusters"]):
        matched = heads[(heads["cluster"] == i) & (heads["similarity"] >= STORY_MATCH)].sort_values("similarity", ascending=False)
        if cats and matched.empty:
            continue   # with a category filter, skip stories that have no headline from it
        stories.append(story_card(i, cluster, matched))

    if not cats:
        return card(title, "AI summary grouped into stories. The colored bar shows which categories each story's headlines come from.",
                    html.P(summ.get("global_summary", ""), className="lede"),
                    html.Div(stories, className="stories"), wide=True)
    intro = ("AI stories that draw on these categories, then the top headlines from them." if stories
             else "No AI story draws on these categories today, so here are their top headlines.")
    return card(title, intro, html.Div(stories, className="stories") if stories else None,
                top_headlines(snap, cats, per_category=5), wide=True)


def update_briefing(snap, cats, theme):
    theme = theme or "dark"
    sub = analytics.unique_articles(DF[DF["snapshot"] == snap]) if snap else DF.head(0)
    if cats:
        sub = sub[sub["category"].isin(cats)]
    if sub.empty:
        e = empty_fig("No headlines for this selection", theme)
        return summary_cards(snap, cats), e, e, e, e, []

    counts = sub["zsc_sentiment"].value_counts().reindex(nlp.SENTIMENTS, fill_value=0)
    f_sent = styled(go.Figure(go.Bar(x=counts.index, y=counts.values, marker_color=[SENT_COLORS[s] for s in counts.index],
                                     text=counts.values, textposition="outside")), theme, 300)
    f_sent.update_yaxes(title="Headlines")

    f_pol = go.Figure(go.Histogram(x=sub["polarity"], xbins=dict(start=-1, end=1, size=0.1),
                                   marker_color=THEMES[theme]["accent"]))
    f_pol.add_vline(x=0, line_dash="dash", line_color=THEMES[theme]["muted"])
    f_pol = styled(f_pol, theme, 300)
    f_pol.update_xaxes(title="Mood score")
    f_pol.update_yaxes(title="Headlines")

    means = sub.groupby("category")["polarity"].mean().sort_values()
    f_cat = styled(go.Figure(go.Bar(x=means.values, y=means.index, orientation="h",
                                    marker_color=[CAT_COLORS.get(c, "#999") for c in means.index],
                                    text=[f"{v:+.2f}" for v in means.values], textposition="outside")), theme, 300)
    # symmetric axis of at least ±0.25, so tiny averages do not fill the whole chart
    reach = max(0.25, float(means.abs().max()) * 1.3)
    f_cat.update_xaxes(title="Average mood (−1 to +1)", range=[-reach, reach])

    f_op = go.Figure()
    for c in [c for c in CATS if c in set(sub["category"])]:
        g = sub[sub["category"] == c]
        f_op.add_trace(go.Scatter(x=g["polarity"], y=g["subjectivity"], mode="markers", name=c.title(),
                                  marker=dict(color=CAT_COLORS[c], size=9, opacity=0.8,
                                              line=dict(width=0.5, color=THEMES[theme]["grid"])),
                                  text=g["title"], hovertemplate="%{text}<br>mood %{x:.2f} · opinion %{y:.2f}<extra></extra>"))
    f_op.add_vline(x=0, line_dash="dot", line_color=THEMES[theme]["muted"])
    f_op.add_hline(y=0.5, line_dash="dot", line_color=THEMES[theme]["muted"])
    for x, y, label in [(-0.95, 0.97, "opinionated · negative"), (0.95, 0.97, "opinionated · positive"),
                        (-0.95, 0.03, "factual · negative"), (0.95, 0.03, "factual · positive")]:
        f_op.add_annotation(x=x, y=y, text=label, showarrow=False, xanchor="left" if x < 0 else "right",
                            font=dict(color=THEMES[theme]["muted"], size=12))
    f_op = styled(f_op, theme, 420)
    f_op.update_xaxes(title="Mood (−1 negative … +1 positive)", range=[-1.05, 1.05])
    f_op.update_yaxes(title="Opinion (0 factual … 1 opinionated)", range=[-0.05, 1.05])

    rows = [{"title": link(r.title, r.url), "category": r.category, "source": r.source,
             "polarity": round(r.polarity, 2), "subjectivity": round(r.subjectivity, 2),
             "sentiment": r.zsc_sentiment,
             "confidence": round(r.zsc_scores, 2) if pd.notna(r.zsc_scores) else ""}
            for r in sub.itertuples()]
    return summary_cards(snap, cats), f_sent, f_pol, f_cat, f_op, rows


app.callback(Output("b-summary", "children"), Output("b-sent", "figure"), Output("b-pol", "figure"),
             Output("b-catpol", "figure"), Output("b-opinion", "figure"), Output("b-table", "data"),
             Input("b-snap", "value"), Input("b-cats", "value"), Input("theme", "data"))(update_briefing)


def briefing_notes(snap, cats):
    """One or two plain sentences under each Briefing chart, computed from the same headlines."""
    sub = analytics.unique_articles(DF[DF["snapshot"] == snap]) if snap else DF.head(0)
    if cats:
        sub = sub[sub["category"].isin(cats)]
    if sub.empty:
        return "", "", "", ""
    n = len(sub)

    # Headline mood
    counts = sub["zsc_sentiment"].value_counts()
    pos, neg = int(counts.get("positive", 0)), int(counts.get("negative", 0))
    mood = f"What this shows: the largest group is {counts.idxmax()} ({counts.max()} of {n}, {counts.max() / n:.0%}). "
    if pos != neg:
        mood += f"{'Positive' if pos > neg else 'Negative'} headlines outnumber {'negative' if pos > neg else 'positive'} ones {max(pos, neg)} to {min(pos, neg)}."
    else:
        mood += f"Positive and negative are tied at {pos}."

    # Mood scores
    near_zero = (sub["polarity"].abs() < 0.1).mean()
    strong = int((sub["polarity"].abs() >= 0.5).sum())
    scores = (f"What this shows: {near_zero:.0%} of headlines score close to 0, because short headlines rarely use "
              f"emotional words. {strong} headline{'s are' if strong != 1 else ' is'} strongly worded (beyond ±0.5).")

    # Average mood by category
    means = sub.groupby("category")["polarity"].mean().sort_values()
    if len(means) > 1:
        by_cat = (f"What this shows: {means.index[-1].title()} reads most positive ({means.iloc[-1]:+.2f}) and "
                  f"{means.index[0].title()} most negative ({means.iloc[0]:+.2f}).")
        if means.abs().max() < 0.2:
            by_cat += " All averages are close to 0, so the differences are small."
    else:
        by_cat = f"What this shows: the average mood of {means.index[0].title()} headlines is {means.iloc[0]:+.2f}."

    # Opinion map
    factual = (sub["subjectivity"] < 0.5).mean()
    opinion = f"What this shows: {factual:.0%} of headlines use mostly factual wording (the lower half of the map)."
    subj = sub.groupby("category")["subjectivity"].mean()
    if len(subj) > 1:
        opinion += f" {subj.idxmax().title()} is the most opinionated category (average {subj.max():.2f})."
    return mood, scores, by_cat, opinion


app.callback(Output("b-sent-note", "children"), Output("b-pol-note", "children"), Output("b-cat-note", "children"),
             Output("b-op-note", "children"), Input("b-snap", "value"), Input("b-cats", "value"))(briefing_notes)


# story cards: each card has its own buttons, matched by index (MATCH = the card that was clicked)
def toggle_story(n_clicks, class_name):
    is_open = "open" in (class_name or "")
    return ("story" if is_open else "story open"), ("Read more" if is_open else "Show less")


app.callback(Output({"type": "story", "index": MATCH}, "className"), Output({"type": "story-more", "index": MATCH}, "children"),
             Input({"type": "story-more", "index": MATCH}, "n_clicks"), State({"type": "story", "index": MATCH}, "className"),
             prevent_initial_call=True)(toggle_story)


def listen_story(n_clicks, button_id, snap):
    clusters = SUMMARIES.get(snap, {}).get("clusters", [])
    if button_id["index"] >= len(clusters):
        return no_update
    story = clusters[button_id["index"]]
    uri, status = newsroom.speak(f"{story.get('title', '')}. {story.get('summary', '')}")
    if not uri:
        return html.Div(status, className="story-meta")
    return html.Audio(src=uri, controls=True, autoPlay=True, className="story-audio")


app.callback(Output({"type": "story-audio", "index": MATCH}, "children"),
             Input({"type": "story-listen", "index": MATCH}, "n_clicks"), State({"type": "story-listen", "index": MATCH}, "id"),
             State("b-snap", "value"), prevent_initial_call=True)(listen_story)


def update_agreement(theme):
    theme = theme or "dark"
    conf, kappa, n, _ = analytics.sentiment_agreement(DF)
    if conf is None:
        msg = ("Zero-shot labels are not available, so there is nothing to compare yet. Install transformers + torch "
               "(requirements-zsc.txt) and restart; results are cached after the first run.")
        return html.Div(msg, className="notice"), empty_fig("Waiting for zero-shot labels", theme)
    agree = conf.values.trace() / conf.values.sum()
    fig = go.Figure(go.Heatmap(z=conf.values, x=[f"ZSC {c}" for c in conf.columns], y=[f"TextBlob {r}" for r in conf.index],
                               text=conf.values, texttemplate="%{text}", colorscale="Blues", showscale=False))
    strength = "poor" if kappa < 0.2 else "fair" if kappa < 0.4 else "moderate" if kappa < 0.6 else "substantial"
    text = html.Div([html.Span(f"κ = {kappa:.2f} ({strength})", className="chip"),
                     html.Span(f"raw agreement {agree:.0%}", className="chip"),
                     html.Span(f"n = {n} headlines", className="chip")])
    return text, styled(fig, theme, 320)


app.callback(Output("b-table", "style_data_conditional"), Input("theme", "data"))(
    lambda theme: [{"if": {"filter_query": f"{{confidence}} < {LOW_CONF}"}, "opacity": 0.55, "fontStyle": "italic"}] if ZSC_OK else [])


app.callback(Output("b-kappa", "children"), Output("b-agree", "figure"), Input("theme", "data"))(update_agreement)


def agreement_cell(click):
    _, _, _, arts = analytics.sentiment_agreement(DF)
    if not click or arts is None or arts.empty:
        return []
    p = click["points"][0]
    tb, zs = p["y"].replace("TextBlob ", ""), p["x"].replace("ZSC ", "")
    hit = arts[(arts["textblob_label"] == tb) & (arts["zsc_sentiment"] == zs)]
    return [{"title": link(r.title, r.url), "textblob": tb, "zsc": zs, "polarity": round(r.polarity, 2)} for r in hit.itertuples()]


app.callback(Output("b-agree-table", "data"), Input("b-agree", "clickData"))(agreement_cell)


# EC 1 entities
def update_entities(snap, cats, types, topn, theme):
    theme = theme or "dark"
    E = analytics.filter_entities(E_ALL, snap, cats, types)
    summ = analytics.entity_summary(E).head(topn or 15)
    if summ.empty:
        e = empty_fig("No entities match these filters", theme)
        return e, e, []
    s = summ.iloc[::-1]
    bar = go.Figure(go.Bar(x=s["mentions"], y=s["entity"], orientation="h",
                           marker=dict(color=s["mean_polarity"], colorscale=[[0, "#F2994A"], [0.5, "#8C9AB3"], [1, "#4C9BE8"]],
                                       cmin=-0.5, cmax=0.5, colorbar=dict(title="Mood", thickness=10)),
                           customdata=list(zip(s["label"].map(TYPE_NAMES), s["mean_polarity"].round(2), s["snapshots"])),
                           hovertemplate="<b>%{y}</b><br>%{customdata[0]}<br>%{x} headlines<br>"
                                         "average mood %{customdata[1]}<br>on %{customdata[2]} dates<extra></extra>"))
    bar = styled(bar, theme, max(320, 26 * len(s) + 60))
    bar.update_xaxes(title="Headlines mentioning")

    names = list(summ["entity"])
    grid = (E[E["entity"].isin(names)].drop_duplicates(["uid", "entity"])
            .pivot_table(index="entity", columns="category", values="uid", aggfunc="count", fill_value=0)
            .reindex(index=names[::-1], columns=CATS, fill_value=0))
    heat = go.Figure(go.Heatmap(z=grid.values, x=[c.title() for c in grid.columns], y=grid.index,
                                colorscale=[[0, "#1D2D47" if theme == "dark" else "#EEF2F7"], [1, THEMES[theme]["accent"]]], text=grid.values, texttemplate="%{text}", showscale=False,
                                hovertemplate="%{y} in %{x}: %{z}<extra></extra>"))
    heat = styled(heat, theme, max(320, 26 * len(s) + 60))
    options = [{"label": f"{n}  ({m})", "value": n} for n, m in zip(summ["entity"], summ["mentions"])]
    return bar, heat, options


app.callback(Output("e-bar", "figure"), Output("e-heat", "figure"), Output("e-select", "options"),
             Input("e-snap", "value"), Input("e-cats", "value"), Input("e-types", "value"),
             Input("e-topn", "value"), Input("theme", "data"))(update_entities)


def pick_entity(bar_click, heat_click, options, current):
    valid = [o["value"] for o in options or []]
    trig = ctx.triggered_id
    click = bar_click if trig == "e-bar" else heat_click if trig == "e-heat" else None
    if click:
        name = click["points"][0]["y"]
        if name in valid:
            return name
    return current if current in valid else (valid[0] if valid else None)


app.callback(Output("e-select", "value"), Input("e-bar", "clickData"), Input("e-heat", "clickData"),
             Input("e-select", "options"), State("e-select", "value"))(pick_entity)


def entity_detail(name, snap, cats, types, theme):
    E = analytics.filter_entities(E_ALL, snap, cats, types)
    sub = E[E["entity"] == name].drop_duplicates("uid")
    if not name or sub.empty:
        return html.Div("Select an entity to see its details.", className="notice")
    summ = analytics.entity_summary(E)
    top = list(summ.head(40)["entity"])
    pairs = analytics.cooccurrence(E, top + ([name] if name not in top else []))
    co = sorted([(b if a == name else a, n) for (a, b), n in pairs.items() if name in (a, b)], key=lambda x: -x[1])
    pol = sub["polarity"].mean()
    rows = sub.sort_values("snapshot", ascending=False)
    return html.Div([
        html.Div([kpi("Headlines", str(len(sub))), kpi("Dates mentioned", f"{sub['snapshot'].nunique()} of {len(SNAPS)}"),
                  kpi("Average mood", f"{pol:+.2f}", "from −1 to +1", warn=pol < -0.05),
                  kpi("Categories", str(sub["category"].nunique()), ", ".join(sorted(sub["category"].unique())))], className="row"),
        html.Div([
            card(f"{name} appears together with", "Other top entities in the same headline",
                 html.Div([html.Span(f"{o}  ×{n}", className="chip") for o, n in co[:12]] or
                          [html.Span("No other top entity shares a headline with it.", className="sub")])),
            card(f"Headlines naming {name}", None,
                 html.Div([html.Div([html.Span([dot(r.category), r.title]),
                                     html.Span(f"{nice_date(r.snapshot)} · {r.category} · mood {r.polarity:+.2f}", className="meta")],
                                    className="list-item") for r in rows.head(8).itertuples()]))], className="row"),
    ], className="section")


app.callback(Output("e-detail", "children"), Input("e-select", "value"), Input("e-snap", "value"),
             Input("e-cats", "value"), Input("e-types", "value"), Input("theme", "data"))(entity_detail)


# EC 2 echo
def echo_item(r, color_var):
    return html.Div([html.Span([dot(r.category), r.title, html.B(f"  {r.echo:.2f}", style={"color": f"var({color_var})"})]),
                     html.Span(f"Closest earlier ({nice_date(r.echo_of_snapshot)}): {r.echo_of}", className="meta")],
                    className="list-item")


def update_echo(focus, cats, theme):
    theme = theme or "dark"
    ec = ECHO.dropna(subset=["echo"])
    if cats:
        ec = ec[ec["category"].isin(cats)]
    cur = ec[ec["snapshot"] == focus]
    if cur.empty:
        e = empty_fig("Echo needs at least two dates", theme)
        return [], e, e, [], [], e, e
    kpis = [kpi("Average echo", f"{cur['echo'].mean():.2f}", f"{len(cur)} headlines on {nice_date(focus)}"),
            kpi("Repeated stories", str(int((cur['echo'] >= 0.5).sum())), "echo score 0.5 or higher", warn=True),
            kpi("New stories", str(int((cur['echo'] < 0.1).sum())), "echo score under 0.1")]

    trend = go.Figure()
    for c in (cats or CATS):
        m = ec[ec["category"] == c].groupby("snapshot")["echo"].mean()
        trend.add_trace(go.Scatter(x=[nice_date(s) for s in m.index], y=m.values, name=c.title(), mode="lines+markers",
                                   line=dict(color=CAT_COLORS[c], width=2)))
    overall = ec.groupby("snapshot")["echo"].mean()
    trend.add_trace(go.Scatter(x=[nice_date(s) for s in overall.index], y=overall.values, name="All", mode="lines+markers",
                               line=dict(color=THEMES[theme]["text"], width=3, dash="dash")))
    trend = styled(trend, theme, 340)
    trend.update_yaxes(title="Average echo", rangemode="tozero")

    by_cat = cur.groupby("category")["echo"].mean().sort_values()
    f_cat = styled(go.Figure(go.Bar(x=by_cat.values, y=by_cat.index, orientation="h",
                                    marker_color=[CAT_COLORS[c] for c in by_cat.index],
                                    text=[f"{v:.2f}" for v in by_cat.values], textposition="outside")), theme, 340)
    f_cat.update_xaxes(title="Average echo")

    srt = cur.sort_values("echo", ascending=False)
    deja = [echo_item(r, "--warn") for r in srt.head(5).itertuples()]
    fresh = [echo_item(r, "--accent") for r in srt.tail(5).iloc[::-1].itertuples()]

    terms = analytics.term_shift(DF if not cats else DF[DF["category"].isin(cats)], focus)
    if terms.empty:
        f_terms = empty_fig("Not enough words to compare", theme)
    else:
        t = pd.concat([terms.head(8), terms.tail(8)]).drop_duplicates("word").sort_values("z")
        f_terms = go.Figure(go.Bar(x=t["z"], y=t["word"], orientation="h",
                                   marker_color=[THEMES[theme]["accent"] if z > 0 else THEMES[theme]["warn"] for z in t["z"]],
                                   customdata=list(zip(t["focus_count"], t["earlier_count"])),
                                   hovertemplate="<b>%{y}</b><br>z = %{x:.2f}<br>%{customdata[0]} now · %{customdata[1]} before<extra></extra>"))
        f_terms = styled(f_terms, theme, 420)
        f_terms.update_xaxes(title="Change score (right = rising, left = fading)")

    st = analytics.sentiment_over_time(DF)
    if cats:
        st = st[st["category"].isin(cats)]
    f_st = go.Figure()
    for c in (cats or CATS):
        m = st[st["category"] == c]
        f_st.add_trace(go.Scatter(x=[nice_date(s) for s in m["snapshot"]], y=m["polarity"], name=c.title(),
                                  mode="lines+markers", line=dict(color=CAT_COLORS[c], width=2)))
    f_st.add_hline(y=0, line_dash="dot", line_color=THEMES[theme]["muted"])
    f_st = styled(f_st, theme, 340)
    f_st.update_yaxes(title="Average mood")
    return kpis, trend, f_cat, deja, fresh, f_terms, f_st


app.callback(Output("x-kpis", "children"), Output("x-trend", "figure"), Output("x-cat", "figure"),
             Output("x-deja", "children"), Output("x-fresh", "children"), Output("x-terms", "figure"),
             Output("x-senttime", "figure"),
             Input("x-focus", "value"), Input("x-cats", "value"), Input("theme", "data"))(update_echo)


def term_headlines(click):
    if not click:
        return []
    word = click["points"][0]["y"]
    arts = analytics.unique_articles(DF)
    hit = arts[arts["title"].str.contains(rf"\b{re.escape(word)}\b", case=False, regex=True)]
    return [{"snapshot": nice_date(r.snapshot), "category": r.category, "title": link(r.title, r.url)}
            for r in hit.sort_values("snapshot", ascending=False).itertuples()]


app.callback(Output("x-term-table", "data"), Input("x-terms", "clickData"))(term_headlines)


# EC 3 newsroom
def coverage_figure(heads, th, theme, extra=None):
    cov = analytics.coverage_by_category(heads, th, CATS)
    fig = go.Figure(go.Bar(x=cov["share"] * 100, y=[c.title() for c in cov["category"]], orientation="h",
                           name="Original summary", marker_color=[CAT_COLORS[c] for c in cov["category"]],
                           text=[f"{k}/{n}" for k, n in zip(cov["represented"], cov["total"])], textposition="outside"))
    if extra is not None:
        cov2 = analytics.coverage_by_category(extra, th, CATS)
        fig.data[0].marker.color = THEMES[theme]["muted"]
        fig.add_trace(go.Bar(x=cov2["share"] * 100, y=[c.title() for c in cov2["category"]], orientation="h",
                             name="Balanced rewrite", marker_color=THEMES[theme]["accent"],
                             text=[f"{k}/{n}" for k, n in zip(cov2["represented"], cov2["total"])], textposition="outside"))
        fig.update_layout(barmode="group")
    fig = styled(fig, theme, 360)
    fig.update_xaxes(title="% of category headlines represented", range=[0, 110])
    return fig, cov


def update_audit(snap, th, theme):
    theme = theme or "dark"
    audit = AUDITS.get(snap)
    if not audit or audit["heads"].empty:
        e = empty_fig("No AI summary for this date", theme)
        return [], e, [], [], []
    heads, sents, summ = audit["heads"], audit["sents"], SUMMARIES[snap]
    fig, cov = coverage_figure(heads, th, theme)
    k, n = int((heads["similarity"] >= th).sum()), len(heads)
    missing = list(cov.loc[cov["represented"] == 0, "category"])
    kpis = [kpi("Headlines covered", f"{k / n:.0%}", f"{k} of {n} headlines"),
            kpi("Categories left out", f"{len(missing)} of {len(CATS)}", ", ".join(missing) or "none", warn=bool(missing)),
            kpi("Weak claims", str(int((sents['support'] < 0.25).sum())), f"of {len(sents)} sentences", warn=True)]
    clusters = []
    for i, c in enumerate(summ["clusters"]):
        m = heads[(heads["cluster"] == i) & (heads["similarity"] >= th)].sort_values("similarity", ascending=False)
        clusters.append(html.Div([html.H4(f"{c.get('title', '')} · {len(m)} headlines")] +
                                 [html.P([dot(r.category), f"{r.title} ({r.similarity:.2f})"]) for r in m.head(3).itertuples()],
                                 className="mini"))
    left = [html.Div([dot(r.category), r.title], className="list-item")
            for c in missing for r in heads[heads["category"] == c].head(2).itertuples()]
    claims = [{"where": r.where, "sentence": r.sentence, "support": round(r.support, 2), "best_headline": r.best_headline}
              for r in sents.itertuples()]
    return kpis, fig, html.Div(clusters, className="row"), left or [html.Div("Every category is represented.", className="sub")], claims


app.callback(Output("n-kpis", "children"), Output("n-cov", "figure"), Output("n-clusters", "children"),
             Output("n-left", "children"), Output("n-claims", "data"),
             Input("n-snap", "value"), Input("n-th", "value"), Input("theme", "data"))(update_audit)

# orange text for weak claims
app.callback(Output("n-claims", "style_data_conditional"), Input("theme", "data"))(
    lambda theme: [{"if": {"filter_query": "{support} < 0.25"}, "color": THEMES[theme or "dark"]["warn"], "fontWeight": "600"}])


def verify_claims(n, snap):
    if not n or snap not in AUDITS:
        return no_update, no_update
    sents = AUDITS[snap]["sents"].head(3)
    results, blocks = [], []
    for r in sents.itertuples():
        sources, status = external.corroborate_claim(r.sentence)
        results.append({"sentence": r.sentence, "support": round(r.support, 2), "status": status, "sources": sources})
        items = [html.Li([html.A(s["title"] or s["url"], href=s["url"], target="_blank"),
                          html.Span(f"  {s['domain']} · relevance {s['score']} · {s['published_date']}", className="sub")])
                 for s in sources]
        blocks.append(html.Div([html.H4(f"“{r.sentence}”"),
                                html.P(f"Headline support {r.support:.2f} · {len(sources)} outside sources · {status}"),
                                html.Ul(items) if items else html.P("No sources returned.")], className="mini"))
    return html.Div(blocks, className="row"), results


app.callback(Output("n-verify-out", "children"), Output("n-verify-store", "data"),
             Input("n-verify", "n_clicks"), State("n-snap", "value"), prevent_initial_call=True)(verify_claims)


def fix_summary(n, snap, th, theme):
    if not n or snap not in AUDITS:
        return no_update, no_update
    theme = theme or "dark"
    bulletin, status = newsroom.rewrite_balanced(snap, DF[DF["snapshot"] == snap])
    if not bulletin:
        return html.Div(f"Could not create the rewrite: {status}.", className="notice"), None
    after = analytics.audit_summary(DF[DF["snapshot"] == snap], newsroom.bulletin_as_summary(bulletin))
    before_h, after_h = AUDITS[snap]["heads"], after["heads"]
    fig, _ = coverage_figure(before_h, th, theme, extra=after_h)
    b_k, a_k, total = int((before_h["similarity"] >= th).sum()), int((after_h["similarity"] >= th).sum()), len(before_h)
    b_miss = int((analytics.coverage_by_category(before_h, th, CATS)["represented"] == 0).sum())
    a_miss = int((analytics.coverage_by_category(after_h, th, CATS)["represented"] == 0).sum())
    text = [html.P(bulletin.get("intro", ""))] + [html.P([html.B(s.get("category", "").title() + ": "), s.get("text", "")])
                                                  for s in bulletin.get("sections", [])] + [html.P(bulletin.get("outro", ""))]
    return html.Div([
        html.Div([kpi("Coverage before → after", f"{b_k / total:.0%} → {a_k / total:.0%}", "scored the same way"),
                  kpi("Categories left out", f"{b_miss} → {a_miss}", "same audit metric"),
                  kpi("Source", status.split(" (")[0], status)], className="row"),
        dcc.Graph(figure=fig, config={"displayModeBar": False}),
        html.Div(text, className="mini"),
    ], className="section"), bulletin


app.callback(Output("n-fix-out", "children"), Output("n-bulletin-store", "data"),
             Input("n-fix", "n_clicks"), State("n-snap", "value"), State("n-th", "value"), State("theme", "data"),
             prevent_initial_call=True)(fix_summary)


def make_radio(n, bulletin):
    if not n:
        return no_update
    if not bulletin:
        return html.Div("Run Step 3 first: the radio bulletin reads the balanced rewrite.", className="notice")
    uri, script, status = newsroom.radio_bulletin(bulletin)
    lines = [html.Div([html.B(f"{who} (voice {v + 1}): "), text], className="list-item") for who, v, text in script]
    player = html.Audio(src=uri, controls=True, style={"width": "100%"}) if uri else html.Div(status, className="notice")
    return html.Div([player, html.P(status, className="sub")] + lines)


app.callback(Output("n-listen-out", "children"), Input("n-listen", "n_clicks"), State("n-bulletin-store", "data"),
             prevent_initial_call=True)(make_radio)


def download_report(n, snap, th, verified, bulletin):
    if not n or snap not in AUDITS:
        return no_update
    sents = AUDITS[snap]["sents"].copy()
    ver = {v["sentence"]: v for v in (verified or [])}
    sents["tavily_sources"] = sents["sentence"].map(lambda s: len(ver[s]["sources"]) if s in ver else None)
    sents["tavily_domains"] = sents["sentence"].map(lambda s: "; ".join(x["domain"] for x in ver[s]["sources"]) if s in ver else "")
    cov = analytics.coverage_by_category(AUDITS[snap]["heads"], th, CATS).assign(version="original", threshold=th)
    if bulletin:
        after = analytics.audit_summary(DF[DF["snapshot"] == snap], newsroom.bulletin_as_summary(bulletin))
        cov = pd.concat([cov, analytics.coverage_by_category(after["heads"], th, CATS).assign(version="balanced", threshold=th)])
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("claims.csv", sents.round(3).to_csv(index=False))
        z.writestr("coverage.csv", cov.round(3).to_csv(index=False))
        z.writestr("summary_audited.json", json.dumps(SUMMARIES[snap], indent=1))
        if bulletin:
            z.writestr("balanced_bulletin.json", json.dumps(bulletin, indent=1))
    return dcc.send_bytes(buf.getvalue(), f"buffalo_times_audit_{snap.replace(' ', '_').replace(':', '')}.zip")


app.callback(Output("n-dl", "data"), Input("n-dl-btn", "n_clicks"), State("n-snap", "value"), State("n-th", "value"),
             State("n-verify-store", "data"), State("n-bulletin-store", "data"), prevent_initial_call=True)(download_report)


# Ask
def ask_question(n_clicks, n_submit, question, snap, theme):
    theme = theme or "dark"
    if not question or not question.strip():
        return html.Div("Type a question first, for example “What is happening with Iran?”", className="notice")
    pool = DF if snap == "ALL" else DF[DF["snapshot"] == snap]
    hits = newsroom.retrieve(question, pool)
    if hits.empty:
        return html.Div("No headline shares words with that question. Try different wording.", className="notice")
    labels = [f"[{i + 1}] {t[:60]}{'…' if len(t) > 60 else ''}" for i, t in enumerate(hits["title"])]
    fig = go.Figure(go.Bar(x=hits["score"][::-1], y=labels[::-1], orientation="h",
                           marker_color=[CAT_COLORS.get(c, "#999") for c in hits["category"][::-1]],
                           hovertemplate="%{y}<br>similarity %{x:.2f}<extra></extra>"))
    fig = styled(fig, theme, 40 * len(hits) + 60)
    fig.update_xaxes(title="How closely it matches your question")
    answer, status = newsroom.ask(question, hits)
    sources = html.Ol([html.Li([html.A(r.title, href=r.url, target="_blank"),
                                html.Span(f"  {nice_date(r.snapshot)} · {r.category}", className="sub")]) for r in hits.itertuples()])
    if not answer:
        return html.Div([html.Div(f"{status}. Showing the retrieved headlines only.", className="notice"),
                         html.Div([card("Retrieved headlines", None, dcc.Graph(figure=fig, config={"displayModeBar": False})),
                                   card("Sources", None, sources)], className="row")], className="section")
    checks = newsroom.grounding(answer, hits)
    check_rows = [html.Div([html.Span(c["sentence"]),
                            html.Span(f"cites {c['cites'] or 'nothing'} · support {c['support']:.2f}",
                                      className="meta", style={"color": "var(--warn)"} if c["support"] < 0.2 else {})],
                           className="list-item") for c in checks]
    return html.Div([
        card("Answer", status, dcc.Markdown(answer), wide=True),
        html.Div([card("Grounding check", "Each answer sentence vs the headlines it cites. Orange = weak support.", html.Div(check_rows)),
                  card("Retrieved headlines", "Bar color = category", dcc.Graph(figure=fig, config={"displayModeBar": False}))], className="row"),
        card("Sources", None, sources, wide=True),
    ], className="section")


app.callback(Output("a-out", "children"), Input("a-go", "n_clicks"), Input("a-q", "n_submit"),
             State("a-q", "value"), State("a-snap", "value"), State("theme", "data"), prevent_initial_call=True)(ask_question)


if __name__ == "__main__":
    print(f"\n{config.APP_NAME}: {STATUS['message']}\nSentiment: {SENTIMENT_METHOD}\n"
          f"Open http://127.0.0.1:{config.PORT}\n")
    app.run(debug=config.DEBUG, port=config.PORT)
