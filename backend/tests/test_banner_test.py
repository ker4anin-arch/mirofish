from app.services.banner_test import aggregate


def _answer(banner, segment, click, trust=3, understood=True):
    return {
        "banner": banner, "segment": segment, "click_intent": click, "trust": trust,
        "relevance": 3, "clarity": 4, "understood": understood, "objections": ["дорого"],
    }


def test_aggregate_ranks_banners_and_splits_segments():
    answers = [
        _answer("A", "ИП", 4), _answer("A", "ИП", 5), _answer("A", "Новички", 2, understood=False),
        _answer("B", "ИП", 2), _answer("B", "Новички", 1),
    ]
    stats = aggregate(answers, ["A", "B"])

    assert stats["ranking"] == ["A", "B"]
    overall = stats["banners"]["A"]["overall"]
    assert overall["n"] == 3
    assert overall["click_intent"] == round((4 + 5 + 2) / 3, 2)
    assert overall["would_click_pct"] == 67
    assert overall["understood_pct"] == 67
    assert stats["banners"]["A"]["segments"]["ИП"]["n"] == 2
    assert stats["banners"]["B"]["top_objections"] == ["дорого"]


def test_aggregate_handles_banner_without_answers():
    stats = aggregate([_answer("A", "ИП", 3)], ["A", "B"])
    assert stats["banners"]["B"]["overall"]["n"] == 0
    assert stats["ranking"] == ["A"]
