import io

from PIL import Image

from app.services.landing_test import aggregate, normalize_answer, slice_page


def _png(width, height):
    out = io.BytesIO()
    Image.new("RGB", (width, height), (200, 200, 200)).save(out, format="PNG")
    return out.getvalue()


def test_slice_page_cuts_a_tall_desktop_page_into_screens():
    screens = slice_page(_png(1440, 4500), "desktop")
    # 1440x4500 -> scaled to 1280 wide = 4000 tall -> 5 screens of 800.
    assert len(screens) == 5
    with Image.open(io.BytesIO(screens[0])) as first:
        assert first.size == (1280, 800)


def test_slice_page_caps_very_long_pages():
    screens = slice_page(_png(780, 40000), "mobile")
    assert len(screens) == 10


def test_normalize_answer_ignores_screens_after_leaving():
    person = {"id": 1, "segment": "ИП"}
    answer = normalize_answer({
        "first_screen": {"understood": True, "offer_in_own_words": "касса", "for_me": 4, "want_to_scroll": 3},
        "screens": [
            {"screen": 1, "impression": "ок", "convincing": 4},
            {"screen": 2, "impression": "отзывы", "convincing": 2, "doubt": "фейк"},
            {"screen": 3, "impression": "не видел", "convincing": 5},
        ],
        "stopped_at": 2,
        "would_apply": 2,
        "trust": "3",
        "objections": "дорого",
    }, person, "A", total=4)

    assert answer["stopped_at"] == 2
    assert answer["screen_scores"] == {"1": 4, "2": 2}
    assert answer["trust"] == 3
    assert answer["objections"] == ["дорого"]


def test_aggregate_builds_reach_curve_and_ranking():
    def row(label, stopped, apply):
        return {"banner": label, "segment": "ИП", "stopped_at": stopped, "would_apply": apply,
                "understood": True, "screen_scores": {"1": 4}, "objections": []}

    stats = aggregate(
        [row("A", None, 5), row("A", 2, 2), row("B", 1, 1), row("B", 1, 2)],
        ["A", "B"],
        {"A": 3, "B": 3},
    )
    assert stats["ranking"] == ["A", "B"]
    assert stats["banners"]["A"]["overall"]["reach"] == [100, 100, 50]
    assert stats["banners"]["A"]["overall"]["read_to_end_pct"] == 50
    assert stats["banners"]["B"]["overall"]["reach"] == [100, 0, 0]
    assert stats["banners"]["A"]["overall"]["apply_pct"] == 50
