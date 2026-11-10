"""NewsAPI fetch, snapshot archive and the headline DataFrame."""
import json
from datetime import datetime, timedelta

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


def save_snapshot(all_results: dict, day: str, fetched_by: str = "live NewsAPI fetch") -> str:
    """One file per day (data/snapshots/YYYY-MM-DD.json); a later fetch on the same day replaces it."""
    payload = {"snapshot": day, "source_notebook": None, "fetched_by": fetched_by, "articles": all_results}
    (config.SNAPSHOT_DIR / f"{day}.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")
    return day


def todays_file_is_fresh(today: str) -> bool:
    """True if today's snapshot was saved less than LIVE_REFRESH_MINUTES ago (saves API calls on restart)."""
    path = config.SNAPSHOT_DIR / f"{today}.json"
    if not path.exists():
        return False
    age_minutes = (datetime.now().timestamp() - path.stat().st_mtime) / 60
    return age_minutes < config.LIVE_REFRESH_MINUTES


def backfill_past_days(api_key: str, days: list):
    """
    Top headlines cannot look back in time, so missing past days come from NewsAPI's
    article search: one request per category, limited to that category's US sources,
    then each article is filed under the day it was published. Runs once; files are saved.
    """
    if not days:
        return []
    errors = []
    try:
        from newsapi import NewsApiClient
        newsapi = NewsApiClient(api_key=api_key)
        sources = newsapi.get_sources(language="en", country="us").get("sources", [])
    except Exception as exc:
        return [f"backfill could not start: {exc}"]

    by_day = {d: {cat: [] for cat in config.CATEGORIES} for d in days}
    for cat in config.CATEGORIES:
        ids = [s["id"] for s in sources if s.get("category") == cat][:20]   # API allows 20 sources per request
        if not ids:
            continue
        try:
            found = newsapi.get_everything(sources=",".join(ids), from_param=days[0], to=days[-1],
                                           language="en", sort_by="popularity", page_size=100)
        except Exception as exc:
            errors.append(f"{cat}: {exc}")
            continue
        for article in found.get("articles") or []:
            day = (article.get("publishedAt") or "")[:10]
            if day in by_day:
                by_day[day][cat].append(article)

    for day, articles in by_day.items():
        if any(articles.values()):
            save_snapshot(articles, day, "NewsAPI article search: articles published that day")
    return errors


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
    snapshots = None

    if not config.DEMO_MODE:
        today = datetime.now().date()
        window = [(today - timedelta(days=i)).isoformat() for i in range(config.DAYS_BACK, -1, -1)]

        # fill in any missing past days once (files are saved, so this does not repeat)
        saved_days = {p.stem for p in config.SNAPSHOT_DIR.glob("*.json")}
        status["live_errors"] += backfill_past_days(config.NEWS_API_KEY, [d for d in window[:-1] if d not in saved_days])

        if todays_file_is_fresh(window[-1]):
            status["mode"] = "live"
            status["message"] = "Live: today's headlines (fetched within the last hour)"
        else:
            all_results, errors = fetch_live_headlines(config.NEWS_API_KEY)
            status["live_errors"] += errors
            n_live = sum(len(v) for v in all_results.values())
            if n_live > 0:
                status["mode"] = "live"
                status["message"] = f"Live: {n_live} headlines fetched from NewsAPI"
                if config.SAVE_SNAPSHOTS:
                    save_snapshot(all_results, window[-1])
                else:
                    # keep the fresh fetch in memory only
                    snapshots = [s for s in load_snapshots() if s["snapshot"] != window[-1]]
                    snapshots.append({"snapshot": window[-1], "articles": all_results})
            else:
                status["message"] = ("Live fetch failed, showing saved snapshots instead. "
                                     + (errors[0] if errors else ""))

        if status["mode"] == "live":
            # live mode shows today and the past DAYS_BACK days only
            snapshots = [s for s in (snapshots or load_snapshots()) if str(s["snapshot"])[:10] in window]
    else:
        status["message"] = ("Demo mode: saved class snapshots (no API key set). "
                             "Add NEWS_API_KEY to .env for live news.")
        # class snapshots are the ones saved from a class notebook
        snapshots = [s for s in load_snapshots() if s.get("source_notebook")]

    df = build_dataframe(snapshots if snapshots is not None else load_snapshots())
    if df.empty:
        status["message"] = "No headlines available: data/snapshots is empty and no live fetch succeeded."
    return df, status
