"""Settings read from environment variables / .env."""
import os
from pathlib import Path

try:
    # python-dotenv reads key=value pairs from a .env file into os.environ
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # the app still works if python-dotenv is missing
    pass

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
SNAPSHOT_DIR = DATA_DIR / "snapshots"   # archive of every NewsAPI fetch
DEMO_DIR = DATA_DIR / "demo"            # saved class outputs (prices, Gemini JSON)
CACHE_DIR = DATA_DIR / "cache"          # zero-shot + Tavily caches
SUMMARY_DIR = DATA_DIR / "summaries"    # one Gemini summary per snapshot
for d in (SNAPSHOT_DIR, DEMO_DIR, CACHE_DIR, SUMMARY_DIR):
    d.mkdir(parents=True, exist_ok=True)


def _flag(name: str, default: str = "false") -> bool:
    """Read a true/false environment variable."""
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


NEWS_API_KEY = os.getenv("NEWS_API_KEY", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "").strip()
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "").strip()
ELEVENLABS_MODEL = os.getenv("ELEVENLABS_MODEL", "eleven_multilingual_v2").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite").strip()

# Demo mode = use the saved class snapshots only, no network calls.
# It is switched on automatically when no NewsAPI key is configured.
DEMO_MODE = _flag("DEMO_MODE") or not NEWS_API_KEY

ENABLE_ZSC = _flag("ENABLE_ZSC", "true")          # BART zero-shot (needs transformers)
ENABLE_LIVE_STOCKS = _flag("ENABLE_LIVE_STOCKS", "true")
SAVE_SNAPSHOTS = _flag("SAVE_SNAPSHOTS", "true")  # archive each live fetch
PORT = int(os.getenv("PORT", "8080"))
DEBUG = _flag("DASH_DEBUG")

APP_NAME = "The Buffalo Times"

CATEGORIES = ["business", "entertainment", "health", "science",
              "sports", "technology", "general"]
