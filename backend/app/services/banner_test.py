"""
Banner test: a synthetic audience panel looks at banner images.

Unlike the social simulation, banners are seen for a second or two and
either clicked or ignored. So this runs as a panel survey instead:

1. Build personas from the uploaded audience description (CrowdGenerator).
2. Show every banner to every persona through a vision-capable LLM and
   collect a structured first-impression answer.
3. Aggregate numbers per banner and per segment in code, then let the LLM
   write a qualitative comparison.

State lives on disk under uploads/banner_tests/<test_id>/ so results survive
restarts.
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

from openai import OpenAI

from ..config import Config
from ..utils.locale import get_language_instruction, get_locale, set_locale
from ..utils.logger import get_logger
from ..utils.openai_chat_compat import create_chat_completion, extract_chat_completion_text
from . import semantic_scale
from .crowd_generator import CrowdGenerator

logger = get_logger("mirofish.banner_test")

BANNER_TESTS_DIR = os.path.join(Config.UPLOAD_FOLDER, "banner_tests")
ALLOWED_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
MAX_BANNERS = 5
MAX_PANEL_SIZE = 300
MAX_IMAGE_SIDE = 1280

PLACEMENTS = {
    "vk_feed": "лента ВКонтакте (рекламный пост между постами друзей и сообществ)",
    "telegram": "Telegram-канал (рекламный пост в канале)",
    "website": "баннер на сайте или в медиа (рядом со статьёй)",
    "rsya": "Рекламная сеть Яндекса (баннер на стороннем сайте)",
    "outdoor": "наружная реклама (билборд или экран на улице, видно несколько секунд)",
    "other": "рекламный баннер",
}

SCORE_FIELDS = ("click_intent", "trust", "relevance", "clarity")

# Free-text answers scored by semantic similarity: {name: (anchor scale, text field)}.
SSR_FIELDS = {
    "click_intent": ("click", "click_thoughts"),
    "trust": ("trust", "trust_thoughts"),
    "relevance": ("relevance", "relevance_thoughts"),
}

# Models are more attentive and agreeable than people. Each panel member gets
# an everyday attitude to advertising, in realistic proportions, so the panel
# includes the people who barely look and the ones looking for a catch.
AD_ATTITUDES = (
    (0.30, "Обычно рекламу почти не замечаешь и пролистываешь не глядя; задерживаешься, только если зацепило мгновенно."),
    (0.25, "К рекламе относишься скептически: ищешь подвох и мелкий шрифт в условиях."),
    (0.30, "К рекламе нейтрален: смотришь, только если тема актуальна для тебя прямо сейчас."),
    (0.15, "Открыт к новому и иногда кликаешь на рекламу из любопытства."),
)


def assign_attitudes(panel: List[Dict[str, Any]], seed: int = 0) -> None:
    """Give every member an ad attitude, keeping the AD_ATTITUDES shares."""

    slots: List[str] = []
    for share, text in AD_ATTITUDES:
        slots.extend([text] * round(share * len(panel)))
    while len(slots) < len(panel):
        slots.append(AD_ATTITUDES[2][1])
    random.Random(seed).shuffle(slots)
    for person, attitude in zip(panel, slots):
        person["ad_attitude"] = attitude


def attitude_line(person: Dict[str, Any]) -> str:
    attitude = person.get("ad_attitude")
    return f"\nТвоё обычное отношение к рекламе: {attitude}\n" if attitude else ""


# --------------------------------------------------------------------- storage


def _test_dir(test_id: str) -> str:
    if not test_id or not all(c.isalnum() or c == "_" for c in test_id):
        raise ValueError("invalid test id")
    return os.path.join(BANNER_TESTS_DIR, test_id)


def _write_json(path: str, data: Any) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _read_json(path: str, default: Any = None) -> Any:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def load_meta(test_id: str) -> Optional[Dict[str, Any]]:
    return _read_json(os.path.join(_test_dir(test_id), "meta.json"))


def save_meta(test_id: str, meta: Dict[str, Any]) -> None:
    meta["updated_at"] = datetime.now().isoformat(timespec="seconds")
    _write_json(os.path.join(_test_dir(test_id), "meta.json"), meta)


def list_tests(limit: int = 50) -> List[Dict[str, Any]]:
    if not os.path.isdir(BANNER_TESTS_DIR):
        return []
    metas = []
    for name in os.listdir(BANNER_TESTS_DIR):
        meta = _read_json(os.path.join(BANNER_TESTS_DIR, name, "meta.json"))
        if meta:
            metas.append({k: meta.get(k) for k in (
                "test_id", "mode", "title", "status", "progress", "created_at", "banner_count", "panel_size"
            )})
    metas.sort(key=lambda m: m.get("created_at") or "", reverse=True)
    return metas[:limit]


def prepare_image(raw: bytes) -> bytes:
    """Validate an uploaded image and downscale it to keep requests small."""

    from PIL import Image

    with Image.open(io.BytesIO(raw)) as image:
        image.load()
        if image.mode not in ("RGB", "RGBA"):
            image = image.convert("RGBA")
        image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
        if image.mode == "RGBA":
            background = Image.new("RGB", image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])
            image = background
        out = io.BytesIO()
        image.save(out, format="PNG", optimize=True)
        return out.getvalue()


def create_test(
    *,
    title: str,
    goal: str,
    placement: str,
    panel_size: int,
    audience_text: str,
    banners: List[Dict[str, Any]],
) -> str:
    """Persist inputs and return the new test id. banners: [{name, data(bytes)}]."""

    if not banners:
        raise ValueError("at least one banner is required")
    if len(banners) > MAX_BANNERS:
        raise ValueError(f"at most {MAX_BANNERS} banners")
    if not audience_text.strip():
        raise ValueError("audience description is required")
    panel_size = max(5, min(int(panel_size), MAX_PANEL_SIZE))
    placement = placement if placement in PLACEMENTS else "other"

    test_id = f"bt_{uuid.uuid4().hex[:12]}"
    directory = _test_dir(test_id)
    os.makedirs(directory, exist_ok=True)

    stored = []
    for index, banner in enumerate(banners):
        filename = f"banner_{index + 1}.png"
        with open(os.path.join(directory, filename), "wb") as f:
            f.write(prepare_image(banner["data"]))
        label = chr(ord("A") + index)
        stored.append({
            "index": index,
            "label": label,
            "name": (banner.get("name") or f"Вариант {label}")[:120],
            "file": filename,
        })

    with open(os.path.join(directory, "audience.txt"), "w", encoding="utf-8") as f:
        f.write(audience_text)

    save_meta(test_id, {
        "test_id": test_id,
        "title": (title or goal or "Тест баннеров")[:200],
        "goal": goal,
        "placement": placement,
        "placement_text": PLACEMENTS[placement],
        "panel_size": panel_size,
        "banner_count": len(stored),
        "banners": stored,
        "status": "queued",
        "progress": 0,
        "message": "",
        "created_at": datetime.now().isoformat(timespec="seconds"),
    })
    return test_id


# ---------------------------------------------------------------------- runner


class BannerTestRunner:
    def __init__(self, test_id: str, parallel: int = 8):
        self.test_id = test_id
        self.dir = _test_dir(test_id)
        self.parallel = parallel
        self.client = OpenAI(api_key=Config.LLM_API_KEY, base_url=Config.LLM_BASE_URL)
        self.model = Config.LLM_MODEL_NAME
        self._meta_lock = threading.Lock()

    # -- helpers ---------------------------------------------------------

    def _update(self, **fields: Any) -> None:
        with self._meta_lock:
            meta = load_meta(self.test_id) or {}
            meta.update(fields)
            save_meta(self.test_id, meta)

    def _image_url(self, banner: Dict[str, Any]) -> str:
        with open(os.path.join(self.dir, banner["file"]), "rb") as f:
            return "data:image/png;base64," + base64.b64encode(f.read()).decode()

    def _chat_json(self, messages: List[Dict[str, Any]], temperature: float = 0.7) -> Dict[str, Any]:
        last_error: Optional[Exception] = None
        for attempt in range(3):
            try:
                response = create_chat_completion(
                    self.client,
                    model=self.model,
                    messages=messages,
                    response_format={"type": "json_object"},
                    temperature=temperature,
                )
                text = extract_chat_completion_text(response).strip()
                if text.startswith("```"):
                    text = text.strip("`")
                    text = text[text.find("{"):]
                return json.loads(text)
            except Exception as error:  # noqa: BLE001 - retried
                last_error = error
        raise RuntimeError(f"LLM call failed: {last_error}")

    # -- steps -----------------------------------------------------------

    def _build_panel(self, meta: Dict[str, Any]) -> List[Dict[str, Any]]:
        with open(os.path.join(self.dir, "audience.txt"), encoding="utf-8") as f:
            audience_text = f.read()
        requirement = (
            f"Тест восприятия рекламных баннеров. Что продвигаем: {meta.get('goal') or 'не указано'}. "
            f"Где показывается: {meta['placement_text']}."
        )

        def progress(current: int, total: int, _segment: str) -> None:
            self._update(progress=int(5 + 25 * current / max(total, 1)),
                         message=f"Создание участников панели: {current}/{total}")

        profiles, entities = CrowdGenerator().generate(
            simulation_requirement=requirement,
            entities=[],
            total=meta["panel_size"],
            start_user_id=0,
            parallel_count=5,
            progress_callback=progress,
            context_text=audience_text,
        )
        panel = [
            {
                "id": profile.user_id,
                "name": profile.name,
                "age": profile.age,
                "gender": profile.gender,
                "profession": profile.profession,
                "segment": (entity.attributes or {}).get("segment") or "Аудитория",
                "persona": profile.persona,
            }
            for profile, entity in zip(profiles, entities)
        ]
        assign_attitudes(panel)
        _write_json(os.path.join(self.dir, "panel.json"), panel)
        return panel

    def _ask(self, person: Dict[str, Any], banner: Dict[str, Any], meta: Dict[str, Any], image_url: str) -> Dict[str, Any]:
        system = (
            "Ты — живой человек, а не ассистент. Отвечай строго от лица описанного персонажа, "
            "его словами и с его взглядами. Не будь вежливее и внимательнее, чем этот человек в жизни. "
            "Верни только JSON.\n" + get_language_instruction()
        )
        prompt = f"""Кто ты:
{person['persona']}
{attitude_line(person)}
Ситуация: ты листаешь и видишь этот баннер — {meta['placement_text']}.
Обычно на рекламу смотрят 1–2 секунды; оценивай так, как отреагировал бы на самом деле.
Если тебе неинтересно или ты бы его даже не заметил — так и скажи, так реагирует большинство людей.

Ответь в JSON (поля *_thoughts — своими словами, честно, 1–2 фразы, без цифр):
{{
  "noticed_first": "что бросилось в глаза в первую секунду",
  "understood": true или false — понял ли, что именно предлагают,
  "offer_in_own_words": "что, по-твоему, предлагают (своими словами)",
  "click_thoughts": "нажал бы ты на этот баннер или пролистал — и почему",
  "trust_thoughts": "веришь ли ты этому предложению и бренду — и почему",
  "relevance_thoughts": "нужно ли это лично тебе сейчас — и почему",
  "clarity": 1-5,
  "trust": 1-5 (1 = похоже на развод/навязчивую рекламу, 5 = доверяю),
  "relevance": 1-5 (насколько это нужно лично тебе),
  "click_intent": 1-5 (1 = точно пролистаю, 5 = точно нажму),
  "emotion": "одно-два слова: что почувствовал",
  "reaction": "1-2 фразы — что подумал или сказал бы вслух",
  "objections": ["что смутило или оттолкнуло"],
  "suggestions": ["что изменить, чтобы ты кликнул"]
}}"""
        data = self._chat_json([
            {"role": "system", "content": system},
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_url}},
            ]},
        ], temperature=0.8)
        answer = {"person_id": person["id"], "segment": person["segment"], "banner": banner["label"]}
        for field in SCORE_FIELDS:
            try:
                answer[field] = max(1, min(5, int(round(float(data.get(field))))))
            except (TypeError, ValueError):
                answer[field] = None
        understood = data.get("understood")
        answer["understood"] = understood if isinstance(understood, bool) else str(understood).lower() in ("true", "да", "1")
        for field in ("noticed_first", "offer_in_own_words", "emotion", "reaction",
                      "click_thoughts", "trust_thoughts", "relevance_thoughts"):
            answer[field] = str(data.get(field) or "")[:500]
        for field in ("objections", "suggestions"):
            values = data.get(field) or []
            answer[field] = [str(v)[:300] for v in values][:5] if isinstance(values, list) else [str(values)[:300]]
        return answer

    def _collect_answers(self, meta: Dict[str, Any], panel: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        images = {banner["label"]: self._image_url(banner) for banner in meta["banners"]}
        jobs = [(person, banner) for person in panel for banner in meta["banners"]]
        answers: List[Dict[str, Any]] = []
        locale = get_locale()
        done = 0
        lock = threading.Lock()

        def run(job):
            set_locale(locale)
            person, banner = job
            try:
                return self._ask(person, banner, meta, images[banner["label"]])
            except Exception as error:  # noqa: BLE001 - one answer must not sink the test
                logger.warning("Banner answer failed (%s, %s): %s", person["name"], banner["label"], error)
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
                                     message=f"Показ баннеров: {done}/{len(jobs)} ответов")
        return answers

    def _write_report(self, meta: Dict[str, Any], stats: Dict[str, Any], answers: List[Dict[str, Any]]) -> str:
        by_banner = defaultdict(list)
        for answer in answers:
            by_banner[answer["banner"]].append(answer)
        samples = []
        for banner in meta["banners"]:
            rows = by_banner.get(banner["label"], [])
            picked = random.Random(42).sample(rows, min(len(rows), 25))
            samples.append({
                "banner": f"{banner['label']} ({banner['name']})",
                "answers": [
                    {k: row.get(k) for k in ("segment", "ssr_click_intent", "click_intent", "trust", "understood",
                                             "offer_in_own_words", "click_thoughts", "reaction",
                                             "objections", "suggestions")}
                    for row in picked
                ],
            })
        prompt = f"""Ты — аналитик рекламы. Синтетическая панель из {len(set(a['person_id'] for a in answers))} человек
посмотрела баннеры. Что продвигаем: {meta.get('goal') or 'не указано'}. Где показ: {meta['placement_text']}.

Посчитанные метрики (средние 1-5, доля понявших оффер), общие и по сегментам:
{json.dumps(stats, ensure_ascii=False)}

{SSR_REPORT_NOTE}

Выборка ответов участников:
{json.dumps(samples, ensure_ascii=False)}

Напиши отчёт в Markdown:
## Итог — какой вариант сильнее и почему (2-4 предложения; опирайся на цифры)
## Сравнение вариантов — по каждому: что работает, что мешает, кто его понял и кто нет
## Сегменты — где мнения сегментов расходятся
## Частые непонимания и возражения
## Что изменить — конкретные правки для каждого баннера (заголовок, оффер, визуал, CTA)
Не выдумывай чисел сверх данных. Помни: это синтетическая панель, абсолютные значения
завышены, важнее разница между вариантами. Верни JSON: {{"markdown": "..."}}"""
        data = self._chat_json([
            {"role": "system", "content": "Ты пишешь ясные практичные отчёты. Верни только JSON.\n" + get_language_instruction()},
            {"role": "user", "content": prompt},
        ], temperature=0.4)
        return str(data.get("markdown") or "")

    def _apply_semantic_scales(self, answers: List[Dict[str, Any]], fields: Dict[str, tuple]) -> None:
        """Score free-text answers on 1-5 scales; the test still completes without it."""

        try:
            semantic_scale.score_answers(answers, fields)
        except Exception as error:  # noqa: BLE001 - optional enrichment
            logger.warning("Semantic scale scoring failed, using direct scores only: %s", error)
            return
        with open(os.path.join(self.dir, "answers.jsonl"), "w", encoding="utf-8") as out:
            for answer in answers:
                out.write(json.dumps(answer, ensure_ascii=False) + "\n")

    # -- entry point -----------------------------------------------------

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
            stats = aggregate(answers, [b["label"] for b in meta["banners"]])
            _write_json(os.path.join(self.dir, "stats.json"), stats)
            report = self._write_report(meta, stats, answers)
            with open(os.path.join(self.dir, "report.md"), "w", encoding="utf-8") as f:
                f.write(report)
            self._update(status="completed", progress=100, message="Готово",
                         answers_count=len(answers), panel_actual=len(panel))
        except Exception as error:  # noqa: BLE001 - surfaced to the UI
            logger.exception("Banner test %s failed", self.test_id)
            self._update(status="failed", message=str(error)[:500])


def start_test(test_id: str) -> None:
    locale = get_locale()

    def target():
        set_locale(locale)
        BannerTestRunner(test_id).run()

    threading.Thread(target=target, name=f"banner-test-{test_id}", daemon=True).start()


# ------------------------------------------------------------------ aggregate

SSR_REPORT_NOTE = (
    "Метрики ssr_* посчитаны не из цифр, которые поставила модель, а из свободных ответов участников "
    "(сходство текста с эталонными фразами шкалы) — они ближе к реальным людям, опирайся в первую "
    "очередь на них; прямые оценки 1-5 обычно завышены. ssr_*_pct — доля склонных сделать действие. "
    "confidence: win_pct — в скольких процентах перевыборок панели вариант лидирует, ci — 95% интервал. "
    "Если у лидера win_pct ниже 70, прямо скажи, что разница между вариантами ненадёжна. "
    "В тексте отчёта не пиши технические имена полей (ssr_click_intent, win_pct, ci и т.п.) — называй "
    "их по-человечески: «желание кликнуть», «склонны кликнуть», «шанс, что вариант лучший», «интервал»."
)


def _summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"n": len(rows)}
    for field in SCORE_FIELDS:
        values = [row[field] for row in rows if isinstance(row.get(field), int)]
        summary[field] = round(statistics.mean(values), 2) if values else None
    summary["understood_pct"] = round(100 * sum(1 for r in rows if r.get("understood")) / len(rows)) if rows else None
    clicks = [row["click_intent"] for row in rows if isinstance(row.get("click_intent"), int)]
    summary["would_click_pct"] = round(100 * sum(1 for v in clicks if v >= 4) / len(clicks)) if clicks else None
    for name in SSR_FIELDS:
        summary.update(semantic_scale.summarize(rows, name))
    return summary


def aggregate(answers: List[Dict[str, Any]], labels: List[str]) -> Dict[str, Any]:
    by_banner: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for answer in answers:
        by_banner[answer["banner"]].append(answer)

    banners = {}
    for label in labels:
        rows = by_banner.get(label, [])
        segments = defaultdict(list)
        for row in rows:
            segments[row.get("segment") or "Аудитория"].append(row)
        objections = Counter(
            o.strip().lower() for row in rows for o in row.get("objections", []) if o.strip()
        )
        banners[label] = {
            "overall": _summarize(rows),
            "segments": {name: _summarize(seg_rows) for name, seg_rows in sorted(segments.items())},
            "top_objections": [o for o, _ in objections.most_common(8)],
        }

    return {"banners": banners, **rank(banners, labels, answers, "click_intent")}


def rank(variants: Dict[str, Any], labels: List[str], answers: List[Dict[str, Any]], key: str) -> Dict[str, Any]:
    """Rank by the text-based score when every variant has it, else the direct one."""

    if all(variants[label]["overall"].get(f"ssr_{key}") is not None for label in labels):
        key = f"ssr_{key}"
    ranked = sorted(
        (label for label in labels if variants[label]["overall"].get(key) is not None),
        key=lambda label: variants[label]["overall"][key],
        reverse=True,
    )
    return {
        "ranking": ranked,
        "ranking_metric": key,
        "confidence": semantic_scale.confidence(answers, labels, key) if len(labels) > 1 else {},
    }
