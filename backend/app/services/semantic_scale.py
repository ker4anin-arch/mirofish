"""
Semantic similarity rating (SSR) for panel answers.

Asking a model for a number ("rate 1-5") gives unrealistic answers: nearly
everyone says 4 and the spread is tiny. Following Maier et al. 2025
(arXiv:2510.08338), panel members answer in their own words instead, and the
text is mapped to a 1-5 distribution by embedding similarity to anchor
statements for each scale point:

    p(r) ~ (sim(text, anchor_r) - min_r sim + eps) ** (1 / T)

averaged over several anchor sets. The expected value is the score; p(4)+p(5)
is the "would do it" share. Embeddings come from the local multilingual model
already baked into the image, so this adds no API cost.

``confidence`` bootstraps panel members to show how stable the ranking is:
the share of resamples in which each variant wins and a 95% interval for its
mean. That replaces repeated paid runs.
"""

from __future__ import annotations

import os
import random
import threading
from typing import Any, Dict, Iterable, List, Optional

from ..utils.logger import get_logger

logger = get_logger("mirofish.semantic_scale")

MODEL_NAME = os.environ.get(
    "SSR_EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
TEMPERATURE = 0.5
EPSILON = 0.01

# Anchor statements per scale, one list of five (1 = lowest ... 5 = highest)
# per set. Several sets smooth out the quirks of any single wording.
ANCHORS: Dict[str, List[List[str]]] = {
    "click": [
        ["Я точно не нажму, просто пролистаю дальше.",
         "Скорее всего не нажму, мне это не очень интересно.",
         "Может быть нажму, а может и нет, не уверен.",
         "Скорее всего нажму, чтобы посмотреть подробнее.",
         "Точно нажму, мне это нужно прямо сейчас."],
        ["Ноль интереса, даже не задержу взгляд.",
         "Вряд ли кликну, разве что случайно.",
         "Не знаю, зависит от настроения.",
         "Пожалуй, кликну и посмотрю, что там.",
         "Обязательно кликну, это именно то, что я искал."],
        ["Пролистаю, мне это совсем неинтересно.",
         "Скорее пролистаю, не цепляет.",
         "Не знаю, может гляну, может нет.",
         "Скорее нажму, интересно.",
         "Нажму сразу, очень интересно и нужно."],
        ["Игнорирую, это мне не нужно.",
         "Сомневаюсь, что мне это нужно.",
         "Нейтрально, без особого интереса.",
         "Интересно, хочу узнать подробнее.",
         "Очень хочу, нажимаю немедленно."],
    ],
    "apply": [
        ["Точно не оставлю заявку, ухожу со страницы.",
         "Скорее всего не буду оставлять заявку.",
         "Пока не решил, может быть оставлю заявку.",
         "Скорее всего оставлю заявку.",
         "Обязательно оставлю заявку прямо сейчас."],
        ["Нет, это мне не подходит, закрываю.",
         "Вряд ли, слишком много сомнений.",
         "Подумаю, вернусь позже, если вспомню.",
         "Пожалуй, попробую, выглядит подходяще.",
         "Да, беру, это именно то, что мне нужно."],
        ["Ни за что не стану этим пользоваться.",
         "Сомневаюсь, что стану этим пользоваться.",
         "Нейтрально, нужно сравнить с другими вариантами.",
         "Хочу попробовать, выглядит убедительно.",
         "Готов начать сразу, полностью убедили."],
    ],
    "trust": [
        ["Похоже на развод, совсем не доверяю.",
         "Скорее не доверяю, есть подвох.",
         "Не знаю, можно ли этому доверять.",
         "В целом доверяю, выглядит надёжно.",
         "Полностью доверяю, это солидно и честно."],
        ["Обман и навязчивая реклама.",
         "Звучит сомнительно, верится с трудом.",
         "Обычная реклама, ни доверия, ни недоверия.",
         "Выглядит честно и серьёзно.",
         "Абсолютно надёжно, известный и проверенный бренд."],
        ["Не верю ни одному слову.",
         "Слишком хорошо, чтобы быть правдой.",
         "Нужно проверить, пока непонятно.",
         "Верю, выглядит правдоподобно.",
         "Верю полностью, никаких сомнений."],
    ],
    "relevance": [
        ["Мне это совершенно не нужно, не моя тема.",
         "Скорее не про меня.",
         "Возможно пригодится когда-нибудь.",
         "Это мне актуально.",
         "Это ровно то, что мне сейчас нужно."],
        ["Меня это не касается.",
         "Малополезно для меня.",
         "Отчасти про меня, отчасти нет.",
         "Полезно для моей ситуации.",
         "Очень нужно, давно ищу именно это."],
        ["Никакого отношения ко мне.",
         "Вряд ли мне пригодится.",
         "Не уверен, нужно ли мне это.",
         "Скорее нужно, есть такая задача.",
         "Крайне актуально, решает мою проблему."],
    ],
}

_model = None
_model_lock = threading.Lock()
_anchor_cache: Dict[str, Any] = {}


def _get_model():
    global _model
    with _model_lock:
        if _model is None:
            from fastembed import TextEmbedding

            logger.info("Loading embedding model for semantic scales: %s", MODEL_NAME)
            _model = TextEmbedding(MODEL_NAME)
        return _model


def _embed(texts: List[str]):
    import numpy as np

    model = _get_model()
    with _model_lock:
        vectors = np.array(list(model.embed(texts)), dtype="float32")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def _anchors(scale: str):
    if scale not in _anchor_cache:
        _anchor_cache[scale] = [_embed(anchor_set) for anchor_set in ANCHORS[scale]]
    return _anchor_cache[scale]


def rate_texts(scale: str, texts: List[str]) -> List[Optional[Dict[str, Any]]]:
    """Return {"score", "top2", "pmf"} per text (None for empty text)."""

    import numpy as np

    results: List[Optional[Dict[str, Any]]] = [None] * len(texts)
    indices = [i for i, text in enumerate(texts) if text and text.strip()]
    if not indices:
        return results
    vectors = _embed([texts[i].strip()[:1000] for i in indices])
    anchor_sets = _anchors(scale)
    points = np.arange(1, 6)
    for row, index in zip(vectors, indices):
        pmfs = []
        for anchors in anchor_sets:
            sims = anchors @ row
            weights = (sims - sims.min() + EPSILON) ** (1 / TEMPERATURE)
            pmfs.append(weights / weights.sum())
        pmf = np.mean(pmfs, axis=0)
        results[index] = {
            "score": round(float((points * pmf).sum()), 3),
            "top2": round(float(pmf[3] + pmf[4]), 4),
            "pmf": [round(float(p), 4) for p in pmf],
        }
    return results


def score_answers(answers: List[Dict[str, Any]], fields: Dict[str, tuple]) -> None:
    """Add ``ssr_<name>``, ``ssr_<name>_top2`` and ``ssr_<name>_pmf`` in place.

    fields: {output name: (anchor scale, answer text field)}.
    """

    for name, (scale, text_field) in fields.items():
        texts = [str(answer.get(text_field) or "") for answer in answers]
        for answer, rating in zip(answers, rate_texts(scale, texts)):
            if rating is None:
                continue
            answer[f"ssr_{name}"] = rating["score"]
            answer[f"ssr_{name}_top2"] = rating["top2"]
            answer[f"ssr_{name}_pmf"] = rating["pmf"]


# --------------------------------------------------------------- statistics


def summarize(rows: Iterable[Dict[str, Any]], name: str) -> Dict[str, Any]:
    """Mean score, "would do it" share and averaged distribution (percent)."""

    rows = [row for row in rows if isinstance(row.get(f"ssr_{name}"), (int, float))]
    if not rows:
        return {}
    n = len(rows)
    pmf = [0.0] * 5
    for row in rows:
        for i, p in enumerate(row.get(f"ssr_{name}_pmf") or [0] * 5):
            pmf[i] += p / n
    return {
        f"ssr_{name}": round(sum(row[f"ssr_{name}"] for row in rows) / n, 2),
        f"ssr_{name}_pct": round(100 * sum(row[f"ssr_{name}_top2"] for row in rows) / n),
        f"ssr_{name}_dist": [round(100 * p) for p in pmf],
    }


def confidence(
    answers: List[Dict[str, Any]],
    labels: List[str],
    key: str,
    iterations: int = 1000,
    seed: int = 0,
) -> Dict[str, Dict[str, Any]]:
    """Bootstrap over panel members: win share and 95% interval per variant.

    The same people see every variant, so members are resampled as a whole.
    """

    by_person: Dict[Any, Dict[str, float]] = {}
    for answer in answers:
        value = answer.get(key)
        if isinstance(value, (int, float)):
            by_person.setdefault(answer.get("person_id"), {})[answer["banner"]] = float(value)
    people = list(by_person.values())
    if not people or not labels:
        return {}

    rng = random.Random(seed)
    wins = {label: 0.0 for label in labels}
    means: Dict[str, List[float]] = {label: [] for label in labels}
    for _ in range(iterations):
        totals = {label: 0.0 for label in labels}
        counts = {label: 0 for label in labels}
        for _ in range(len(people)):
            person = people[rng.randrange(len(people))]
            for label, value in person.items():
                if label in totals:
                    totals[label] += value
                    counts[label] += 1
        current = {label: totals[label] / counts[label] for label in labels if counts[label]}
        for label, value in current.items():
            means[label].append(value)
        if current:
            best = max(current.values())
            leaders = [label for label, value in current.items() if value == best]
            for label in leaders:
                wins[label] += 1 / len(leaders)

    result = {}
    for label in labels:
        values = sorted(means[label])
        if not values:
            continue
        low = values[int(0.025 * (len(values) - 1))]
        high = values[int(0.975 * (len(values) - 1))]
        result[label] = {
            "win_pct": round(100 * wins[label] / iterations),
            "ci": [round(low, 2), round(high, 2)],
        }
    return result
