"""NewsAPI fetch, snapshot archive and the headline DataFrame."""
import json
from datetime import datetime

import pandas as pd

from . import config


# fetching
def fetch_live_headlines(api_key: str):
    """Class P1, made robust. Returns (all_results dict, list of error strings)."""
    errors = []
    try:
        from newsapi import NewsApiClient
        newsapi = NewsApiClient(api_key=api_key)
    except Exception as exc:  # package missing or bad key format
        return {}, [f"NewsAPI client could not start: {exc}"]

    all_results = {}
    for cat in config.CATEGORIES:
        try:
            headlines = newsapi.get_top_headlines(category=cat, country="us")
            if headlines.get("status") != "ok":
                errors.append(f"{cat}: {headlines.get('message', 'bad status')}")
                continue
            all_results[cat] = headlines.get("articles") or []
        except Exception as exc:  # NewsAPIException, timeouts, ...
            errors.append(f"{cat}: {exc}")
    return all_results, errors


def save_snapshot(all_results: dict) -> str:
    """Archive a live fetch so it becomes one point on the trends timeline."""
    now = datetime.now()
    label = now.strftime("%Y-%m-%d %H:%M")
    path = config.SNAPSHOT_DIR / f"{now.strftime('%Y-%m-%d_%H%M')}.json"
    payload = {"snapshot": label, "source_notebook": None,
               "fetched_by": "live NewsAPI fetch", "articles": all_results}
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    return label


def load_snapshots():
    """Read every archived snapshot. Corrupt files are skipped, not fatal."""
    snapshots = []
    for path in sorted(config.SNAPSHOT_DIR.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload.get("articles"), dict):
                snapshots.append(payload)
        except (json.JSONDecodeError, OSError):
            continue
    return snapshots


# DataFrame (P3)
def clean_title(title) -> str:
    """Class P2 trick: drop the ' - Publisher' suffix from a headline."""
    if not isinstance(title, str):
        return ""
    return title.rsplit(" - ", 1)[0].strip()


def build_dataframe(snapshots) -> pd.DataFrame:
    """Class P3 extended: one row per (snapshot, category, article)."""
    rows = []
    for snap in snapshots:
        for category, articles in snap["articles"].items():
            for article in articles or []:
                title = clean_title(article.get("title"))
                # NewsAPI returns "[Removed]" placeholders for deleted articles
                if not title or title == "[Removed]":
                    continue
                source = article.get("source") or {}
                rows.append({
                    "snapshot": snap["snapshot"],
                    "category": category,
                    "title": title,
                    "publishedAt": article.get("publishedAt"),
                    "url": article.get("url") or "",
                    "author": article.get("author"),
                    "source": source.get("name") or "Unknown source",
                    "description": article.get("description") or "",
                })

    df = pd.DataFrame(rows, columns=["snapshot", "category", "title", "publishedAt",
                                     "url", "author", "source", "description"])
    if df.empty:
        return df
    df["publishedAt"] = pd.to_datetime(df["publishedAt"], utc=True, errors="coerce")
    df["author_missing"] = df["author"].isna() | (df["author"].astype(str).str.strip() == "")
    # Same article can be filed under two categories; uid lets us count it once.
    df["uid"] = df["snapshot"] + "|" + df["url"].where(df["url"] != "", df["title"])
    return df.reset_index(drop=True)


def get_news_data():
    """
    Entry point used by the app.
    Returns (df, status) where status describes the data mode for the UI banner.
    """
    status = {"mode": "demo", "message": "", "live_errors": []}

    if not config.DEMO_MODE:
        all_results, errors = fetch_live_headlines(config.NEWS_API_KEY)
        status["live_errors"] = errors
        n_live = sum(len(v) for v in all_results.values())
        if n_live > 0:
            label = save_snapshot(all_results) if config.SAVE_SNAPSHOTS else None
            status["mode"] = "live"
            status["message"] = f"Live: {n_live} headlines fetched from NewsAPI"
            if not config.SAVE_SNAPSHOTS:
                # keep the fresh fetch in memory only
                df = build_dataframe(load_snapshots() + [{
                    "snapshot": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "articles": all_results}])
                return df, status
            status["message"] += f" and archived as snapshot {label}"
        else:
            status["message"] = ("Live fetch failed, showing saved snapshots instead. "
                                 + (errors[0] if errors else ""))
    else:
        status["message"] = ("Demo mode: saved class snapshots (no API key set). "
                             "Add NEWS_API_KEY to .env for live news.")

    df = build_dataframe(load_snapshots())
    if df.empty:
        status["message"] = "No headlines available: data/snapshots is empty and no live fetch succeeded."
    return df, status
