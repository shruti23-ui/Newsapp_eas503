"""Gemini rewrite, Ask (retrieval + citations) and the ElevenLabs bulletin. Results are cached."""
import base64
import hashlib
import json
import os
import re

import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from . import config
from .analytics import split_sentences

CACHE = config.CACHE_DIR
AUDIO_DIR = CACHE / "audio"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)


def _key(*parts):
    return hashlib.sha1("||".join(parts).encode("utf-8")).hexdigest()[:16]


def _gemini(prompt: str):
    """One Gemini call. Returns (text or None, status)."""
    if not config.GEMINI_API_KEY:
        return None, "Add GEMINI_API_KEY to .env"
    try:
        import google.generativeai as genai
        genai.configure(api_key=config.GEMINI_API_KEY)
        reply = genai.GenerativeModel(config.GEMINI_MODEL).generate_content(prompt)
        return getattr(reply, "text", ""), f"Gemini ({config.GEMINI_MODEL})"
    except Exception as exc:
        return None, f"Gemini call failed ({type(exc).__name__})"


def _load(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


# Fix: balanced rewrite
REWRITE_PROMPT = """You are the editor of "The Buffalo Times".
An audit showed the previous summary ignored some categories.
Write a short radio news bulletin that covers EVERY category below, giving
each category space roughly in proportion to its number of headlines.
Use ONLY facts stated in the headlines. Do not add outside knowledge.

Return only valid JSON, no Markdown:
{{"intro": "one sentence",
  "sections": [{{"category": "business", "text": "2-3 sentences"}}],
  "outro": "one sentence"}}

Headlines by category:
{blocks}
"""


def rewrite_balanced(snapshot_label, df_snap):
    """Returns (bulletin dict or None, status). Cached per snapshot."""
    path = CACHE / f"balanced_{_key(snapshot_label)}.json"
    cached = _load(path)
    if cached:
        return cached, "Balanced bulletin (cached)"
    blocks = "\n".join(
        f"[{cat}] ({len(g)} headlines)\n" + "\n".join(f"- {t}" for t in g["title"].head(15))
        for cat, g in df_snap.drop_duplicates("title").groupby("category"))
    text, status = _gemini(REWRITE_PROMPT.format(blocks=blocks))
    if text is None:
        return None, status
    data = None
    try:
        data = json.loads(text[text.find("{"): text.rfind("}") + 1])
    except ValueError:
        pass
    if not data or not data.get("sections"):
        return None, "Gemini reply was not valid JSON"
    path.write_text(json.dumps(data, indent=1), encoding="utf-8")
    return data, status


def bulletin_as_summary(bulletin):
    """Same shape as a Gemini summary, so analytics.audit_summary can score it."""
    return {"clusters": [{"title": s.get("category", "section"), "summary": s.get("text", "")}
                         for s in bulletin.get("sections", [])],
            "global_summary": f"{bulletin.get('intro', '')} {bulletin.get('outro', '')}".strip()}


# Ask The Buffalo Times
def retrieve(question, articles, k=8):
    """TF-IDF cosine retrieval. articles: DataFrame with title/snapshot/category/url."""
    arts = articles.drop_duplicates("uid").reset_index(drop=True)
    if arts.empty or not question.strip():
        return arts.head(0).assign(score=[])
    vec = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit(list(arts["title"]) + [question])
    scores = cosine_similarity(vec.transform([question]), vec.transform(arts["title"])).ravel()
    arts = arts.assign(score=scores).sort_values("score", ascending=False)
    return arts[arts["score"] > 0].head(k).reset_index(drop=True)


ASK_PROMPT = """Answer the question using ONLY the numbered headlines below.
Cite every claim with its headline number in square brackets, like [2].
If the headlines do not answer the question, say so plainly.
Keep the answer under 120 words.

Question: {q}

Headlines:
{h}
"""


def ask(question, hits):
    """Returns (answer text or None, status). Cached per question + sources."""
    if hits.empty:
        return None, "No matching headlines"
    numbered = "\n".join(f"[{i + 1}] ({r.snapshot}, {r.category}) {r.title}"
                         for i, r in enumerate(hits.itertuples()))
    path = CACHE / f"ask_{_key(question.lower().strip(), numbered)}.json"
    cached = _load(path)
    if cached:
        return cached["answer"], "Gemini (cached)"
    text, status = _gemini(ASK_PROMPT.format(q=question, h=numbered))
    if text is None:
        return None, status
    path.write_text(json.dumps({"question": question, "answer": text}), encoding="utf-8")
    return text, status


def grounding(answer, hits):
    """For each answer sentence: which headlines it cites and how similar it is to them."""
    titles = list(hits["title"])
    rows = []
    if not answer or not titles:
        return rows
    vec = TfidfVectorizer(stop_words="english").fit(titles + [answer])
    T = vec.transform(titles)
    for sent in split_sentences(answer):
        cites = sorted({int(n) for n in re.findall(r"\[(\d+)\]", sent) if 0 < int(n) <= len(titles)})
        clean = re.sub(r"\[\d+\]", "", sent)
        sims = cosine_similarity(vec.transform([clean]), T).ravel()
        cited_sim = max((sims[c - 1] for c in cites), default=0.0)
        rows.append({"sentence": sent, "cites": cites, "support": float(cited_sim)})
    return rows


# Listen: ElevenLabs radio bulletin
# Premade voices that free ElevenLabs accounts can use through the API
DEFAULT_VOICES = ["JBFqnCBsd6RMkjVDRZzb", "pNInz6obpgDQGcFmaJgB",
                  "EXAVITQu4vr4xnSDxMaL", "ErXwobaYiN019PkySvjV"]


def _voices():
    env = os.getenv("ELEVENLABS_VOICES", "").strip()
    return [v.strip() for v in env.split(",") if v.strip()] or DEFAULT_VOICES


def _tts(text, voice_id):
    resp = requests.post(f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}", timeout=60,
                         headers={"xi-api-key": config.ELEVENLABS_API_KEY,
                                  "Content-Type": "application/json", "Accept": "audio/mpeg"},
                         json={"text": text, "model_id": config.ELEVENLABS_MODEL})
    resp.raise_for_status()
    return resp.content


def _tts_error(exc):
    """Readable status for a failed ElevenLabs call, using ElevenLabs' own reason when it sends one."""
    if isinstance(exc, requests.HTTPError):
        is_json = "json" in exc.response.headers.get("Content-Type", "")
        detail = exc.response.json().get("detail", {}) if is_json else {}
        reason = detail.get("message", "") if isinstance(detail, dict) else str(detail)
        return f"ElevenLabs error {exc.response.status_code}: {reason[:120]}"
    return f"ElevenLabs request failed ({type(exc).__name__})"


def speak(text):
    """Read one text aloud with the anchor voice. Returns (data URI or None, status). Cached per text."""
    voice = _voices()[0]
    path = AUDIO_DIR / f"story_{_key(text, voice)}.mp3"
    if not path.exists():
        if not config.ELEVENLABS_API_KEY:
            return None, "Add ELEVENLABS_API_KEY to .env to listen"
        try:
            path.write_bytes(_tts(text, voice))
        except requests.RequestException as exc:
            return None, _tts_error(exc)
    return "data:audio/mpeg;base64," + base64.b64encode(path.read_bytes()).decode(), "ElevenLabs"


def radio_bulletin(bulletin):
    """Anchor voice for intro/outro, other voices rotate per section. Returns (data URI, script, status)."""
    voices = _voices()
    script = [("Anchor", 0, bulletin.get("intro", ""))]
    for i, sec in enumerate(bulletin.get("sections", [])):
        script.append((sec.get("category", "news").title() + " desk", 1 + i % (len(voices) - 1), sec.get("text", "")))
    script.append(("Anchor", 0, bulletin.get("outro", "")))
    script = [s for s in script if s[2].strip()]

    path = AUDIO_DIR / f"bulletin_{_key(json.dumps(script), ','.join(voices))}.mp3"
    if not path.exists():
        if not config.ELEVENLABS_API_KEY:
            return None, script, "Add ELEVENLABS_API_KEY to .env to hear the bulletin"
        try:
            audio = b"".join(_tts(text, voices[v]) for _, v, text in script)
        except requests.RequestException as exc:
            return None, script, _tts_error(exc)
        path.write_bytes(audio)
        status = "ElevenLabs, live"
    else:
        status = "ElevenLabs (cached)"
    uri = "data:audio/mpeg;base64," + base64.b64encode(path.read_bytes()).decode()
    return uri, script, status
