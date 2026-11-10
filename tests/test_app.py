"""Offline smoke tests: pytest -q"""
import base64
import io
import os
import sys
import zipfile

import numpy as np

os.environ["DEMO_MODE"] = "true"          # never touch the network in tests
os.environ["ENABLE_ZSC"] = "false"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import analytics  # noqa: E402
import app as A  # noqa: E402
import newsroom  # noqa: E402
import nlp  # noqa: E402

FOCUS = "2026-11-02"


def test_data_loaded():
    assert len(A.SNAPS) >= 4
    assert A.DF["uid"].nunique() > 400


def test_entity_rules_fix_known_mislabels():
    labels = dict(A.E_ALL.drop_duplicates("entity")[["entity", "label"]].values)
    assert labels.get("Donald Trump") == "PERSON"
    assert labels.get("Nvidia") == "ORG"
    assert "AI" not in labels


def test_briefing_and_entities():
    assert len(A.update_briefing(FOCUS, None, "dark")[5]) > 100
    assert A.update_entities("ALL", None, ["PERSON", "ORG", "PLACE"], 15, "light")[2]


def test_echo_rises_and_terms():
    means = A.ECHO.groupby("snapshot")["echo"].mean().dropna()
    assert means.is_monotonic_increasing            # the real finding: 0.12 -> 0.16 -> 0.19
    assert not analytics.term_shift(A.DF, FOCUS).empty


def test_audit_finds_missing_categories():
    kpis = A.update_audit(FOCUS, 0.10, "dark")[0]
    assert kpis[1].children[1].children.startswith("3")   # 3 of 7 categories left out


def test_no_key_paths_do_not_crash():
    import config
    if not config.GEMINI_API_KEY:
        assert "Could not create the rewrite" in A.fix_summary(1, FOCUS, 0.10, "dark")[0].children
    dl = A.download_report(1, FOCUS, 0.10, None, None)
    assert "claims.csv" in zipfile.ZipFile(io.BytesIO(base64.b64decode(dl["content"]))).namelist()


def test_sentence_split_keeps_us():
    assert len(analytics.split_sentences("Talks in the U.S. continue. Markets rose.")) == 2


def test_grounding_flags_unsupported_sentence():
    hits = newsroom.retrieve("What is happening with Iran?", A.DF)
    checks = newsroom.grounding("Aliens landed in Buffalo [1].", hits)
    assert checks[0]["support"] < 0.2


def test_kappa_behaves():
    df = A.DF.copy()
    df["sentiment_method"] = "zsc"
    df["zsc_sentiment"] = df["polarity"].apply(nlp.polarity_to_label)
    assert analytics.sentiment_agreement(df)[1] == 1.0
    df["zsc_sentiment"] = np.random.default_rng(0).choice(nlp.SENTIMENTS, len(df))
    assert abs(analytics.sentiment_agreement(df)[1]) < 0.15


def test_opinion_map_and_confidence_column():
    out = A.update_briefing(FOCUS, None, "dark")
    assert len(out[4].data) >= 5                      # one scatter trace per category
    assert "subjectivity" in out[5][0] and "confidence" in out[5][0]
