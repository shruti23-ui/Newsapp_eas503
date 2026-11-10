"""Save today's NewsAPI top headlines (and a Gemini summary) as one snapshot. Run daily by GitHub Actions."""
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # so "buffalo" can be imported

from buffalo import ai_summary, config, data_pipeline  # noqa: E402


def main():
    if not config.NEWS_API_KEY:
        print("NEWS_API_KEY is not set")
        return 1
    results, errors = data_pipeline.fetch_live_headlines(config.NEWS_API_KEY)
    for err in errors:
        print("warning:", err)
    n = sum(len(articles) for articles in results.values())
    if n == 0:
        print("no headlines fetched, nothing saved")
        return 1

    day = date.today().isoformat()
    data_pipeline.save_snapshot(results, day)
    print(f"saved {n} headlines as data/snapshots/{day}.json")

    # a summary for the same day, so the Newsroom Check has something to audit
    titles = data_pipeline.build_dataframe([{"snapshot": day, "articles": results}])["title"].drop_duplicates().tolist()
    _, status = ai_summary.summarise_snapshot(day, titles)
    print("summary:", status)
    return 0


if __name__ == "__main__":
    sys.exit(main())
