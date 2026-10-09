import pytest

from app.services import semantic_scale
from app.services.banner_test import AD_ATTITUDES, aggregate, assign_attitudes


def _scored(banner, person, score, top2):
    return {
        "banner": banner, "person_id": person, "segment": "ИП", "click_intent": 4,
        "trust": 4, "relevance": 4, "clarity": 4, "understood": True, "objections": [],
        "ssr_click_intent": score, "ssr_click_intent_top2": top2,
        "ssr_click_intent_pmf": [0.1, 0.2, 0.3, 0.2, 0.2],
    }


def test_ranking_prefers_text_based_score_and_reports_confidence():
    answers = []
    for person in range(20):
        answers.append(_scored("A", person, 2.0 + (person % 3) * 0.1, 0.2))
        answers.append(_scored("B", person, 3.5 + (person % 3) * 0.1, 0.6))
    stats = aggregate(answers, ["A", "B"])

    assert stats["ranking_metric"] == "ssr_click_intent"
    assert stats["ranking"] == ["B", "A"]
    assert stats["confidence"]["B"]["win_pct"] == 100
    low, high = stats["confidence"]["B"]["ci"]
    assert 3.5 <= low <= high <= 3.7
    overall = stats["banners"]["B"]["overall"]
    assert overall["ssr_click_intent_pct"] == 60
    assert overall["ssr_click_intent_dist"] == [10, 20, 30, 20, 20]


def test_ranking_falls_back_to_direct_scores_for_old_answers():
    answers = [
        {"banner": "A", "person_id": 1, "segment": "ИП", "click_intent": 2, "objections": []},
        {"banner": "B", "person_id": 1, "segment": "ИП", "click_intent": 5, "objections": []},
    ]
    stats = aggregate(answers, ["A", "B"])
    assert stats["ranking_metric"] == "click_intent"
    assert stats["ranking"] == ["B", "A"]


def test_attitudes_keep_shares():
    panel = [{"id": i} for i in range(100)]
    assign_attitudes(panel)
    texts = [p["ad_attitude"] for p in panel]
    for share, text in AD_ATTITUDES:
        assert texts.count(text) == round(share * 100)


def test_rate_texts_orders_clear_answers():
    pytest.importorskip("fastembed")
    try:
        ratings = semantic_scale.rate_texts("click", [
            "Пролистаю, мне это совсем не нужно.",
            "",
            "Нажму обязательно, давно искал именно это.",
        ])
    except Exception as error:  # model not downloadable offline
        pytest.skip(f"embedding model unavailable: {error}")
    assert ratings[1] is None
    assert ratings[0]["score"] < ratings[2]["score"]
    assert abs(sum(ratings[0]["pmf"]) - 1) < 0.01
