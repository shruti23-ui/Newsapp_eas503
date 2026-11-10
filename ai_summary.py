"""Gemini story clusters + global summary, parsed from JSON (class P6/P7)."""
import json
import re

import config

PROMPT_TEMPLATE = """
You are a news assistant. I will give you a bunch of news headlines.

Task:
1. Group related headlines into story clusters.
2. For each cluster, give:
- a short title
- a 2-3 sentence summary.
3. Then give me a 3-4 sentence global summary of today's news.

Return only valid JSON in this exact structure, with no Markdown or extra text:
{{
  "clusters": [
    {{
      "title": "Short story title",
      "summary": "Two-sentence summary."
    }}
  ],
  "global_summary": "Three-sentence global summary."
}}


If no usable headlines are provided, return:
{{"clusters": [], "global_summary": "No usable headlines provided."}}

Headlines:
{headlines}
"""


def parse_summary_json(text: str):
    """Pull the JSON object out of a model reply. Returns dict or None."""
    if not text:
        return None
    text = re.sub(r"```(?:json)?", "", text).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    clusters = [c for c in data.get("clusters", [])
                if isinstance(c, dict) and c.get("title")]
    return {"clusters": clusters, "global_summary": str(data.get("global_summary", ""))}


def _summary_path(snapshot_label: str):
    return config.SUMMARY_DIR / (snapshot_label.replace(" ", "_").replace(":", "") + ".json")


def load_summaries():
    """All saved summaries, keyed by the snapshot label they were written for."""
    out = {}
    for path in sorted(config.SUMMARY_DIR.glob("*.json")):
        parsed = parse_summary_json(path.read_text(encoding="utf-8"))
        if parsed:
            label = path.stem
            if "_" in label:                     # 2026-11-06_1405 -> 2026-11-06 14:05
                day, hm = label.split("_", 1)
                label = f"{day} {hm[:2]}:{hm[2:]}"
            out[label] = parsed
    return out


def summarise_snapshot(snapshot_label: str, headlines):
    """Live Gemini call (class P7 prompt). Saves the parsed JSON next to its snapshot.
    Returns (summary or None, status text)."""
    if not config.GEMINI_API_KEY:
        return None, "no GEMINI_API_KEY set"
    if not headlines:
        return None, "no headlines to summarise"
    try:
        import google.generativeai as genai
        genai.configure(api_key=config.GEMINI_API_KEY)
        model = genai.GenerativeModel(config.GEMINI_MODEL)
        response = model.generate_content(PROMPT_TEMPLATE.format(headlines="\n".join(headlines)))
        parsed = parse_summary_json(getattr(response, "text", ""))
    except Exception as exc:
        return None, f"Gemini call failed ({type(exc).__name__})"
    if not parsed:
        return None, "Gemini reply was not valid JSON"
    _summary_path(snapshot_label).write_text(json.dumps(parsed, indent=1), encoding="utf-8")
    return parsed, f"Gemini ({config.GEMINI_MODEL}), live"
