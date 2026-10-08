"""
Banner test API (separate from the graph/simulation/report flow).

POST /api/banner-test                 multipart: audience (files), audience_text,
                                      banners (images), goal, placement, panel_size, title
GET  /api/banner-test/list
GET  /api/banner-test/<test_id>       status, inputs, stats and report when ready
GET  /api/banner-test/<test_id>/image/<label>
GET  /api/banner-test/<test_id>/answers
"""

import os
import tempfile

from flask import jsonify, request, send_file

from . import banner_bp
from ..services import banner_test, landing_test
from ..utils.file_parser import FileParser
from ..utils.logger import get_logger

logger = get_logger("mirofish.api.banner")

AUDIENCE_EXTENSIONS = {"pdf", "md", "txt", "markdown"}


def _ext(filename: str) -> str:
    return filename.rsplit(".", 1)[-1].lower() if "." in filename else ""


def _audience_text_from_files(files) -> str:
    parts = []
    for storage in files:
        if not storage or not storage.filename:
            continue
        ext = _ext(storage.filename)
        if ext not in AUDIENCE_EXTENSIONS:
            raise ValueError(f"unsupported audience file: {storage.filename}")
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, f"audience.{ext}")
            storage.save(path)
            parts.append(FileParser.extract_text(path))
    return "\n\n".join(part for part in parts if part)


@banner_bp.route("", methods=["POST"])
def create_banner_test():
    try:
        audience_text = _audience_text_from_files(request.files.getlist("audience"))
        extra = (request.form.get("audience_text") or "").strip()
        if extra:
            audience_text = f"{audience_text}\n\n{extra}".strip()

        banners = []
        for storage in request.files.getlist("banners"):
            if not storage or not storage.filename:
                continue
            if _ext(storage.filename) not in banner_test.ALLOWED_IMAGE_EXTENSIONS:
                return jsonify({"success": False, "error": f"Неподдерживаемый формат: {storage.filename} (нужны PNG, JPG, WEBP)"}), 400
            name = os.path.splitext(os.path.basename(storage.filename))[0]
            banners.append({"name": name, "data": storage.read()})

        try:
            panel_size = int(request.form.get("panel_size") or 100)
        except ValueError:
            return jsonify({"success": False, "error": "panel_size должен быть числом"}), 400

        if (request.form.get("mode") or "banner") == "landing":
            test_id = landing_test.create_test(
                title=request.form.get("title") or "",
                goal=request.form.get("goal") or "",
                placement=request.form.get("placement") or "other",
                device=request.form.get("device") or "desktop",
                panel_size=panel_size,
                audience_text=audience_text,
                pages=banners,
            )
            landing_test.start_test(test_id)
        else:
            test_id = banner_test.create_test(
                title=request.form.get("title") or "",
                goal=request.form.get("goal") or "",
                placement=request.form.get("placement") or "other",
                panel_size=panel_size,
                audience_text=audience_text,
                banners=banners,
            )
            banner_test.start_test(test_id)
        return jsonify({"success": True, "data": {"test_id": test_id}})
    except ValueError as error:
        return jsonify({"success": False, "error": str(error)}), 400
    except Exception as error:  # noqa: BLE001
        logger.exception("Failed to create banner test")
        return jsonify({"success": False, "error": str(error)}), 500


@banner_bp.route("/list", methods=["GET"])
def list_banner_tests():
    return jsonify({"success": True, "data": banner_test.list_tests()})


@banner_bp.route("/<test_id>", methods=["GET"])
def get_banner_test(test_id):
    try:
        meta = banner_test.load_meta(test_id)
    except ValueError:
        meta = None
    if not meta:
        return jsonify({"success": False, "error": "Тест не найден"}), 404
    directory = banner_test._test_dir(test_id)
    data = dict(meta)
    if meta.get("status") == "completed":
        data["stats"] = banner_test._read_json(os.path.join(directory, "stats.json"), {})
        try:
            with open(os.path.join(directory, "report.md"), encoding="utf-8") as f:
                data["report"] = f.read()
        except OSError:
            data["report"] = ""
    return jsonify({"success": True, "data": data})


@banner_bp.route("/<test_id>/image/<label>", methods=["GET"])
def get_banner_image(test_id, label):
    try:
        meta = banner_test.load_meta(test_id)
    except ValueError:
        meta = None
    banner = next((b for b in (meta or {}).get("banners", []) if b["label"] == label), None)
    if not banner:
        return jsonify({"success": False, "error": "Баннер не найден"}), 404
    return send_file(os.path.join(banner_test._test_dir(test_id), banner["file"]), mimetype="image/png")


@banner_bp.route("/<test_id>/screen/<label>/<int:number>", methods=["GET"])
def get_landing_screen(test_id, label, number):
    try:
        meta = banner_test.load_meta(test_id)
    except ValueError:
        meta = None
    variant = next((b for b in (meta or {}).get("banners", []) if b["label"] == label), None)
    screens = (variant or {}).get("screens") or []
    if not 1 <= number <= len(screens):
        return jsonify({"success": False, "error": "Экран не найден"}), 404
    return send_file(os.path.join(banner_test._test_dir(test_id), screens[number - 1]), mimetype="image/png")


@banner_bp.route("/<test_id>/answers", methods=["GET"])
def get_banner_answers(test_id):
    import json

    try:
        path = os.path.join(banner_test._test_dir(test_id), "answers.jsonl")
    except ValueError:
        return jsonify({"success": False, "error": "Тест не найден"}), 404
    answers = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            answers = [json.loads(line) for line in f if line.strip()]
    return jsonify({"success": True, "data": answers})
