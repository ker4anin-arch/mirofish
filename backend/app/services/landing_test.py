"""
Landing page test: a synthetic audience panel scrolls through page mockups.

A full page is too tall to read as one image, so each uploaded page is cut
into viewport-sized screens. Every panel member arrives "from an ad", sees the
screens in order in one vision request and reports what the first screen
says, how each block lands, where they would stop scrolling and whether they
would apply. Numbers (reach per screen, apply intent) are aggregated in code;
the LLM writes the qualitative comparison.

Shares storage, listing and the panel builder with banner_test; tests are
told apart by meta["mode"] == "landing".
"""

from __future__ import annotations

import base64
import concurrent.futures
import io
import json
import os
import random
import statistics
import threading
import uuid
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..utils.locale import get_language_instruction, get_locale, set_locale
from ..utils.logger import get_logger
from . import banner_test, semantic_scale
from .banner_test import (
    BannerTestRunner, PLACEMENTS, SSR_REPORT_NOTE, _read_json, _test_dir, _write_json,
    attitude_line, load_meta, rank, save_meta,
)

logger = get_logger("mirofish.landing_test")

MAX_VARIANTS = 3
MAX_SCREENS = 10
# Free-text answers scored by semantic similarity: {name: (anchor scale, text field)}.
SSR_FIELDS = {
    "would_apply": ("apply", "apply_thoughts"),
    "trust": ("trust", "trust_thoughts"),
}

DEVICES = {
    # width the page is scaled to, and screen height for that width
    "desktop": {"width": 1280, "screen_height": 800, "label": "компьютер (браузер на десктопе)"},
    "mobile": {"width": 780, "screen_height": 1690, "label": "смартфон (мобильный браузер)"},
}


# ---------------------------------------------------------------- slicing


def slice_page(raw: bytes, device: str) -> List[bytes]:
    """Scale a full-page screenshot to the device width and cut it into screens."""

    from PIL import Image

    spec = DEVICES.get(device, DEVICES["desktop"])
    with Image.open(io.BytesIO(raw)) as image:
        image.load()
        if image.mode == "RGBA":
            background = Image.new("RGB", image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])
            image = background
        elif image.mode != "RGB":
            image = image.convert("RGB")
        width = spec["width"]
        height = max(1, round(image.height * width / image.width))
        image = image.resize((width, height))

        screen_height = spec["screen_height"]
        count = max(1, -(-height // screen_height))
        if count > MAX_SCREENS:
            # Very long pages: fewer, taller screens rather than dropping content.
            count = MAX_SCREENS
            screen_height = -(-height // count)

        screens = []
        for index in range(count):
            top = index * screen_height
            bottom = min(height, top + screen_height)
            if bottom - top < screen_height * 0.15 and screens:
                break  # a sliver at the end adds nothing
            out = io.BytesIO()
            image.crop((0, top, width, bottom)).save(out, format="PNG", optimize=True)
            screens.append(out.getvalue())
        return screens


def create_test(
    *,
    title: str,
    goal: str,
    placement: str,
    device: str,
    panel_size: int,
    audience_text: str,
    pages: List[Dict[str, Any]],
) -> str:
    if not pages:
        raise ValueError("нужен хотя бы один макет страницы")
    if len(pages) > MAX_VARIANTS:
        raise ValueError(f"не больше {MAX_VARIANTS} вариантов страницы")
    if not audience_text.strip():
        raise ValueError("нужно описание аудитории")
    device = device if device in DEVICES else "desktop"
    placement = placement if placement in PLACEMENTS else "other"
    panel_size = max(5, min(int(panel_size), banner_test.MAX_PANEL_SIZE))

    test_id = f"bt_{uuid.uuid4().hex[:12]}"
    directory = _test_dir(test_id)
    os.makedirs(directory, exist_ok=True)

    variants = []
    for index, page in enumerate(pages):
        label = chr(ord("A") + index)
        screens = slice_page(page["data"], device)
        files = []
        for number, data in enumerate(screens, start=1):
            filename = f"page_{label}_screen_{number}.png"
            with open(os.path.join(directory, filename), "wb") as f:
                f.write(data)
            files.append(filename)
        variants.append({
            "index": index,
            "label": label,
            "name": (page.get("name") or f"Вариант {label}")[:120],
            "screens": files,
            # The first screen doubles as the card thumbnail.
            "file": files[0],
        })

    with open(os.path.join(directory, "audience.txt"), "w", encoding="utf-8") as f:
        f.write(audience_text)

    save_meta(test_id, {
        "test_id": test_id,
        "mode": "landing",
        "title": (title or goal or "Тест лендинга")[:200],
        "goal": goal,
        "placement": placement,
        "placement_text": PLACEMENTS[placement],
        "device": device,
        "device_text": DEVICES[device]["label"],
        "panel_size": panel_size,
        "banner_count": len(variants),
        "banners": variants,
        "status": "queued",
        "progress": 0,
        "message": "",
        "created_at": datetime.now().isoformat(timespec="seconds"),
    })
    return test_id


# ----------------------------------------------------------------- runner


class LandingTestRunner(BannerTestRunner):
    """Reuses the banner runner's panel builder, LLM helper and meta updates."""

    def _screen_urls(self, variant: Dict[str, Any]) -> List[str]:
        urls = []
        for filename in variant["screens"]:
            with open(os.path.join(self.dir, filename), "rb") as f:
                urls.append("data:image/png;base64," + base64.b64encode(f.read()).decode())
        return urls

    def _ask_page(self, person: Dict[str, Any], variant: Dict[str, Any], meta: Dict[str, Any], urls: List[str]) -> Dict[str, Any]:
        total = len(urls)
        system = (
            "Ты — живой человек, а не ассистент. Отвечай строго от лица описанного персонажа, его "
            "словами и с его взглядами. Веди себя как обычный посетитель: большинство людей не "
            "дочитывают страницу до конца и уходят, если быстро не нашли причину остаться. "
            "Верни только JSON.\n" + get_language_instruction()
        )
        prompt = f"""Кто ты:
{person['persona']}
{attitude_line(person)}
Ситуация: ты увидел рекламу ({meta['placement_text']}) и перешёл на страницу. Устройство: {meta['device_text']}.
Что предлагают по задумке автора: {meta.get('goal') or 'не указано'} (ты этого заранее не знаешь — суди по странице).
Ниже {total} экранов страницы по порядку, как ты бы их пролистывал (экран 1 — то, что видно сразу).
Первый экран оцени так, будто смотришь на него 3–5 секунд. Если в какой-то момент ты бы ушёл —
укажи этот экран в "stopped_at", а экраны после него не оценивай. Уйти быстро или остаться
равнодушным — нормально, так поступает большинство посетителей.

Верни JSON (поля *_thoughts — своими словами, честно, 1–2 фразы, без цифр):
{{
  "first_screen": {{
    "understood": true или false — понял ли за 5 секунд, что предлагают,
    "offer_in_own_words": "что предлагают, своими словами",
    "for_me": 1-5 (насколько это для тебя),
    "want_to_scroll": 1-5
  }},
  "screens": [
    {{"screen": 1, "impression": "что подумал на этом экране", "convincing": 1-5, "doubt": "что смутило или пусто"}}
    // по одному объекту на каждый просмотренный экран
  ],
  "stopped_at": номер экрана, на котором ушёл бы, или null — если дочитал до конца,
  "stop_reason": "почему ушёл (или пусто)",
  "apply_thoughts": "оставил бы ты заявку (совершил целевое действие) — и почему",
  "trust_thoughts": "веришь ли ты этой странице и компании — и почему",
  "trust": 1-5,
  "would_apply": 1-5 (оставил бы заявку / совершил целевое действие),
  "decision_reason": "главная причина решения в 1-2 фразах",
  "objections": ["что мешает решиться"],
  "suggestions": ["что изменить на странице"]
}}"""
        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        for number, url in enumerate(urls, start=1):
            content.append({"type": "text", "text": f"Экран {number} из {total}:"})
            content.append({"type": "image_url", "image_url": {"url": url}})
        data = self._chat_json([
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ], temperature=0.8)
        return normalize_answer(data, person, variant["label"], total)

    def _collect_answers(self, meta: Dict[str, Any], panel: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        urls = {variant["label"]: self._screen_urls(variant) for variant in meta["banners"]}
        jobs = [(person, variant) for person in panel for variant in meta["banners"]]
        answers: List[Dict[str, Any]] = []
        locale = get_locale()
        done = 0
        lock = threading.Lock()

        def run(job):
            set_locale(locale)
            person, variant = job
            try:
                return self._ask_page(person, variant, meta, urls[variant["label"]])
            except Exception as error:  # noqa: BLE001 - one answer must not sink the test
                logger.warning("Landing answer failed (%s, %s): %s", person["name"], variant["label"], error)
                return None

        path = os.path.join(self.dir, "answers.jsonl")
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.parallel) as pool, \
                open(path, "w", encoding="utf-8") as out:
            for answer in pool.map(run, jobs):
                with lock:
                    done += 1
                    if answer:
                        answers.append(answer)
                        out.write(json.dumps(answer, ensure_ascii=False) + "\n")
                    if done % 5 == 0 or done == len(jobs):
                        self._update(progress=int(30 + 55 * done / len(jobs)),
                                     message=f"Просмотр страниц: {done}/{len(jobs)}")
        return answers

    def _write_report(self, meta: Dict[str, Any], stats: Dict[str, Any], answers: List[Dict[str, Any]]) -> str:
        by_variant = defaultdict(list)
        for answer in answers:
            by_variant[answer["banner"]].append(answer)
        samples = []
        for variant in meta["banners"]:
            rows = by_variant.get(variant["label"], [])
            picked = random.Random(42).sample(rows, min(len(rows), 20))
            samples.append({
                "variant": f"{variant['label']} ({variant['name']}, экранов: {len(variant['screens'])})",
                "answers": [
                    {k: row.get(k) for k in ("segment", "offer_in_own_words", "stopped_at", "stop_reason",
                                             "ssr_would_apply", "would_apply", "apply_thoughts", "decision_reason",
                                             "screen_notes", "objections", "suggestions")}
                    for row in picked
                ],
            })
        prompt = f"""Ты — аналитик конверсии сайтов. Синтетическая панель из {len(set(a['person_id'] for a in answers))} человек
пришла с рекламы ({meta['placement_text']}) и пролистала посадочные страницы. Устройство: {meta['device_text']}.
Что продвигаем: {meta.get('goal') or 'не указано'}.

Посчитанные метрики (reach — доля дошедших до экрана; convincing — средняя убедительность экрана 1-5;
would_apply — намерение оставить заявку 1-5), общие и по сегментам:
{json.dumps(stats, ensure_ascii=False)}

{SSR_REPORT_NOTE}

Выборка ответов участников (screen_notes — впечатления по экранам):
{json.dumps(samples, ensure_ascii=False)}

Напиши отчёт в Markdown:
## Итог — какой вариант сильнее и почему (2-4 предложения; опирайся на цифры)
## Первый экран — понятно ли за 5 секунд, что и для кого; как люди пересказывают оффер
## Где уходят — на каких экранах теряем людей и почему (по каждому варианту)
## Что работает и что нет — сильные и слабые блоки
## Сегменты — где реакции сегментов расходятся
## Главные возражения
## Что изменить — конкретные правки по экранам (заголовки, порядок блоков, доказательства, CTA)
Не выдумывай чисел сверх данных. Помни: это синтетическая панель — модели дочитывают внимательнее людей,
поэтому абсолютные цифры завышены, важнее разница между вариантами и места, где проседает.
Верни JSON: {{"markdown": "..."}}"""
        data = self._chat_json([
            {"role": "system", "content": "Ты пишешь ясные практичные отчёты. Верни только JSON.\n" + get_language_instruction()},
            {"role": "user", "content": prompt},
        ], temperature=0.4)
        return str(data.get("markdown") or "")

    def run(self) -> None:
        try:
            meta = load_meta(self.test_id)
            self._update(status="running", progress=2, message="Планируем панель аудитории...")
            panel = self._build_panel(meta)
            if not panel:
                raise RuntimeError("не удалось создать участников панели")
            answers = self._collect_answers(meta, panel)
            if not answers:
                raise RuntimeError("модель не вернула ни одного ответа")
            self._update(progress=86, message="Переводим ответы в шкалы...")
            self._apply_semantic_scales(answers, SSR_FIELDS)
            self._update(progress=88, message="Считаем метрики и пишем отчёт...")
            screen_counts = {v["label"]: len(v["screens"]) for v in meta["banners"]}
            stats = aggregate(answers, [v["label"] for v in meta["banners"]], screen_counts)
            _write_json(os.path.join(self.dir, "stats.json"), stats)
            report = self._write_report(meta, stats, answers)
            with open(os.path.join(self.dir, "report.md"), "w", encoding="utf-8") as f:
                f.write(report)
            self._update(status="completed", progress=100, message="Готово",
                         answers_count=len(answers), panel_actual=len(panel))
        except Exception as error:  # noqa: BLE001 - surfaced to the UI
            logger.exception("Landing test %s failed", self.test_id)
            self._update(status="failed", message=str(error)[:500])


def start_test(test_id: str) -> None:
    locale = get_locale()

    def target():
        set_locale(locale)
        LandingTestRunner(test_id).run()

    threading.Thread(target=target, name=f"landing-test-{test_id}", daemon=True).start()


# ------------------------------------------------------------- answers


def _score(value: Any) -> Optional[int]:
    try:
        return max(1, min(5, int(round(float(value)))))
    except (TypeError, ValueError):
        return None


def normalize_answer(data: Dict[str, Any], person: Dict[str, Any], label: str, total: int) -> Dict[str, Any]:
    first = data.get("first_screen") if isinstance(data.get("first_screen"), dict) else {}
    understood = first.get("understood")
    stopped = data.get("stopped_at")
    try:
        stopped = int(stopped) if stopped not in (None, "", "null") else None
    except (TypeError, ValueError):
        stopped = None
    if stopped is not None:
        stopped = max(1, min(total, stopped))

    screen_scores: Dict[int, int] = {}
    notes = []
    for item in data.get("screens") or []:
        if not isinstance(item, dict):
            continue
        try:
            number = int(item.get("screen"))
        except (TypeError, ValueError):
            continue
        if not 1 <= number <= total or (stopped is not None and number > stopped):
            continue
        score = _score(item.get("convincing"))
        if score is not None:
            screen_scores[number] = score
        impression = str(item.get("impression") or "")[:200]
        doubt = str(item.get("doubt") or "")[:200]
        if impression or doubt:
            notes.append(f"{number}: {impression}" + (f" | сомнение: {doubt}" if doubt else ""))

    def as_list(value):
        if isinstance(value, list):
            return [str(v)[:300] for v in value if str(v).strip()][:5]
        return [str(value)[:300]] if value else []

    return {
        "person_id": person["id"],
        "segment": person["segment"],
        "banner": label,
        "screens_total": total,
        "understood": understood if isinstance(understood, bool) else str(understood).lower() in ("true", "да", "1"),
        "offer_in_own_words": str(first.get("offer_in_own_words") or "")[:400],
        "for_me": _score(first.get("for_me")),
        "want_to_scroll": _score(first.get("want_to_scroll")),
        "stopped_at": stopped,
        "stop_reason": str(data.get("stop_reason") or "")[:300],
        "screen_scores": {str(k): v for k, v in sorted(screen_scores.items())},
        "screen_notes": notes[:MAX_SCREENS],
        "trust": _score(data.get("trust")),
        "would_apply": _score(data.get("would_apply")),
        "decision_reason": str(data.get("decision_reason") or "")[:400],
        "apply_thoughts": str(data.get("apply_thoughts") or "")[:500],
        "trust_thoughts": str(data.get("trust_thoughts") or "")[:500],
        "objections": as_list(data.get("objections")),
        "suggestions": as_list(data.get("suggestions")),
    }


# ------------------------------------------------------------ aggregate


def _mean(values: List[Optional[int]]) -> Optional[float]:
    values = [v for v in values if isinstance(v, int)]
    return round(statistics.mean(values), 2) if values else None


def _pct(part: int, whole: int) -> Optional[int]:
    return round(100 * part / whole) if whole else None


def _summarize(rows: List[Dict[str, Any]], screens: int) -> Dict[str, Any]:
    n = len(rows)
    reach = []
    convincing = []
    for number in range(1, screens + 1):
        reached = sum(1 for r in rows if r.get("stopped_at") is None or r["stopped_at"] >= number)
        reach.append(_pct(reached, n))
        convincing.append(_mean([r.get("screen_scores", {}).get(str(number)) for r in rows]))
    applies = [r.get("would_apply") for r in rows if isinstance(r.get("would_apply"), int)]
    summary = {
        "n": n,
        "would_apply": _mean([r.get("would_apply") for r in rows]),
        "apply_pct": _pct(sum(1 for v in applies if v >= 4), len(applies)),
        "understood_pct": _pct(sum(1 for r in rows if r.get("understood")), n),
        "for_me": _mean([r.get("for_me") for r in rows]),
        "want_to_scroll": _mean([r.get("want_to_scroll") for r in rows]),
        "trust": _mean([r.get("trust") for r in rows]),
        "read_to_end_pct": _pct(sum(1 for r in rows if r.get("stopped_at") is None), n),
        "reach": reach,
        "convincing": convincing,
    }
    for name in SSR_FIELDS:
        summary.update(semantic_scale.summarize(rows, name))
    return summary


def aggregate(answers: List[Dict[str, Any]], labels: List[str], screen_counts: Dict[str, int]) -> Dict[str, Any]:
    by_variant: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for answer in answers:
        by_variant[answer["banner"]].append(answer)

    variants = {}
    for label in labels:
        rows = by_variant.get(label, [])
        screens = screen_counts.get(label, 1)
        segments = defaultdict(list)
        for row in rows:
            segments[row.get("segment") or "Аудитория"].append(row)
        objections = Counter(o.strip().lower() for r in rows for o in r.get("objections", []) if o.strip())
        variants[label] = {
            "screens": screens,
            "overall": _summarize(rows, screens),
            "segments": {name: _summarize(seg, screens) for name, seg in sorted(segments.items())},
            "top_objections": [o for o, _ in objections.most_common(8)],
        }

    return {"mode": "landing", "banners": variants, **rank(variants, labels, answers, "would_apply")}
