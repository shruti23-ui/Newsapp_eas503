"""Tag every saved headline with a news frame using zero-shot classification.

Frames from Semetko & Valkenburg (2000). Needs: pip install -r requirements-zsc.txt
Writes data/framing/frames.csv and, once, data/framing/hand_labels.csv (50 headlines for you to label).
Run from the project root:  python scripts/tag_frames.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # so "buffalo" can be imported

from buffalo import analytics, config, data_pipeline  # noqa: E402

# what the model reads for each frame: "This headline is about <description>."
DESCRIPTIONS = {
    "conflict": "conflict or disagreement",
    "economic": "money, business or economic consequences",
    "human interest": "a personal or human-interest story",
    "morality": "morality, ethics or religion",
    "responsibility": "who is responsible or to blame for a problem",
}
FRAMES_FILE = config.FRAMING_DIR / "frames.csv"
HAND_FILE = config.FRAMING_DIR / "hand_labels.csv"


def main():
    df = data_pipeline.build_dataframe(data_pipeline.load_snapshots())
    arts = analytics.unique_articles(df)[["uid", "snapshot", "category", "title"]]

    done = pd.read_csv(FRAMES_FILE) if FRAMES_FILE.exists() else pd.DataFrame(columns=["uid"])
    todo = arts[~arts["uid"].isin(done["uid"])]
    print(f"{len(arts)} headlines, {len(todo)} still to tag")

    if len(todo):
        from transformers import pipeline
        classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        name_of = {text: name for name, text in DESCRIPTIONS.items()}
        results = classifier(todo["title"].tolist(), candidate_labels=list(DESCRIPTIONS.values()),
                             hypothesis_template="This headline is about {}.", batch_size=16)
        todo = todo.assign(frame=[name_of[r["labels"][0]] for r in results],
                           score=[round(float(r["scores"][0]), 3) for r in results])
        done = pd.concat([done, todo], ignore_index=True)
        done.to_csv(FRAMES_FILE, index=False)
        print(f"saved {len(done)} tagged headlines to {FRAMES_FILE.relative_to(config.BASE_DIR)}")

    if not HAND_FILE.exists():
        # a fixed random sample, written WITHOUT the model's answer so your labels stay independent
        sample = arts.sample(50, random_state=503)[["uid", "category", "title"]].assign(my_frame="")
        sample.to_csv(HAND_FILE, index=False)
        print(f"wrote {HAND_FILE.relative_to(config.BASE_DIR)}: fill in my_frame with one of {analytics.FRAMES}")


if __name__ == "__main__":
    main()
