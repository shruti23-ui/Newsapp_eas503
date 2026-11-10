"""Yahoo Finance, Bluesky and Tavily. Each call returns (data, status) and never raises."""
import json

import requests

import config

# 1. Stocks (P8)
SYMBOLS = ["NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "GOOG", "AVGO", "META", "TSLA",
           "BRK-B", "WMT", "LLY", "JPM", "XOM", "JNJ", "V", "COST", "MA", "ORCL",
           "MU", "NFLX", "CVX", "ABBV", "BAC", "PLTR", "AMD", "CAT", "PG", "HD", "KO",
           "CSCO", "GE", "MRK", "AMAT", "LRCX", "MS", "RTX", "GS", "UNH", "WFC", "PM",
           "INTC", "GEV", "LIN", "IBM", "TMUS", "MCD", "PEP", "VZ", "AXP"]


def _saved_prices():
    files = sorted(config.DEMO_DIR.glob("prices_*.json"))
    if not files:
        return {}
    return json.loads(files[-1].read_text(encoding="utf-8")).get("prices", {})


def get_stock_prices(use_live: bool):
    """Returns ({symbol: price}, status)."""
    if use_live and config.ENABLE_LIVE_STOCKS:
        try:
            import yfinance as yf
            # One batched request instead of the class loop of 50 fast_info calls
            closes = yf.download(SYMBOLS, period="5d", progress=False,
                                 auto_adjust=False)["Close"].ffill().iloc[-1]
            prices = {s: float(v) for s, v in closes.items() if v == v}  # v == v drops NaN
            if prices:
                return prices, "Yahoo Finance, live"
        except Exception as exc:
            return _saved_prices(), f"Saved prices (live fetch failed: {type(exc).__name__})"
    return _saved_prices(), "Saved prices from the Nov 3 class notebook"


# 2. Bluesky (class bluesky.py)
BLUESKY_HANDLES = ["washingtonpost.com", "nytimes.com", "espn.com",
                   "theguardian.com", "cnn.com", "wsj.com"]
BLUESKY_URL = "https://public.api.bsky.app/xrpc/app.bsky.feed.getAuthorFeed"


def get_bluesky_posts(limit_per_handle: int = 2):
    """Class Bluesky code with timeouts, skipped reposts and per-handle error handling."""
    rows, failures = [], []
    for handle in BLUESKY_HANDLES:
        try:
            resp = requests.get(BLUESKY_URL, timeout=6, params={
                "actor": handle, "limit": limit_per_handle * 3, "filter": "posts_no_replies"})
            feed = resp.json().get("feed", [])
        except (requests.RequestException, ValueError):
            failures.append(handle)
            continue
        kept = 0
        for item in feed:
            if "reason" in item:           # a repost of someone else's post
                continue
            post = item.get("post", {})
            uri = post.get("uri", "")
            rows.append({
                "source": handle,
                "text": post.get("record", {}).get("text", ""),
                "createdAt": post.get("record", {}).get("createdAt", ""),
                "link": f"https://bsky.app/profile/{handle}/post/{uri.rsplit('/', 1)[-1]}" if uri else "",
            })
            kept += 1
            if kept >= limit_per_handle:
                break
    if not rows:
        return [], "Bluesky unreachable right now"
    status = f"{len(rows)} posts from Bluesky"
    if failures:
        status += f" ({len(failures)} handles failed)"
    return rows, status


# 3. Tavily corroboration (EC 3 add-on)
TAVILY_CACHE = config.CACHE_DIR / "tavily_cache.json"


def corroborate_claim(claim: str, max_results: int = 5):
    """Search recent news for a claim with Tavily. Returns (sources, status). Cached."""
    try:
        cache = json.loads(TAVILY_CACHE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        cache = {}
    if claim in cache:
        return cache[claim], "Tavily (cached)"
    if not config.TAVILY_API_KEY:
        return [], "Add TAVILY_API_KEY to .env to check claims against the web"
    try:
        resp = requests.post("https://api.tavily.com/search", timeout=20,
                             headers={"Authorization": f"Bearer {config.TAVILY_API_KEY}"},
                             json={"query": claim[:380], "topic": "news",
                                   "time_range": "week", "max_results": max_results})
        resp.raise_for_status()
        results = resp.json().get("results", [])
    except (requests.RequestException, ValueError) as exc:
        return [], f"Tavily request failed ({type(exc).__name__})"
    sources = [{"title": r.get("title", ""), "url": r.get("url", ""),
                "domain": (r.get("url", "").split("/") + ["", "", ""])[2].replace("www.", ""),
                "score": round(float(r.get("score") or 0), 2),
                "published_date": r.get("published_date") or "date unknown"}
               for r in results]
    cache[claim] = sources
    TAVILY_CACHE.write_text(json.dumps(cache, indent=1), encoding="utf-8")
    return sources, "Tavily, live"
