"""Офлайн-проверки подсистемы «Мышление».

Ничего не подключается к Colab: проверяются контракты (thinking/schemas),
мягкая деградация (thinking/fallback), журнал отчётов и взаимодействий
(thinking/client), веб-панель (tools/thinking_panel.html), CLI и содержимое
Colab-ячеек (thinking/colab). Снимок схемы сверяется с docs/schema_plan.json.

Запуск:  python tools/thinking_test.py
"""
from __future__ import annotations

import argparse
import http.server
import json
import re
import sys
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from thinking import client as client_mod# noqa: E402
from thinking.client import ThinkingClient, ThinkingError       # noqa: E402
from thinking.client import _is_read_timeout                  # noqa: E402
from thinking.fallback import local_plan, offline_reason        # noqa: E402
from thinking.schemas import (ACTIONS, EVENT_TYPES, ChatReply, Plan, ReflectResponse,  # noqa: E402
                              SchemaError, extract_json, has_secret,
                              plan_schema, redact_secrets, utcnow)

PASSED = FAILED = 0
FAILS: list[str] = []
TMP = Path(tempfile.mkdtemp(prefix="thinking_test_"))

# Герметичность: у настоящего config/thinking.local.json (после set-url) и у
# переменных окружения есть шанс увести проверки в реальную сеть — а тест
# обязан работать офлайн в любом окружении.
client_mod.LOCAL_PATH = TMP / "thinking.local.json"
for _env_key in ("THINKING_URL", "THINKING_TOKEN"):
    import os as _os
    _os.environ.pop(_env_key, None)


def check(cond: bool, name: str) -> None:
    global PASSED, FAILED
    if cond:
        PASSED += 1
    else:
        FAILED += 1
        FAILS.append(name)
        print(f"  ! {name}")


def raises(fn, exc) -> bool:
    try:
        fn()
    except exc:
        return True
    except Exception:
        return False
    return False


def count_lines(path: object) -> int:
    """Сколько строк в JSONL-журнале (для проверок удаления)."""
    try:
        with open(path, encoding="utf-8") as fh:
            return sum(1 for line in fh if line.strip())
    except OSError:
        return 0


def make_client(**kw) -> ThinkingClient:
    cfg = {
        "log_path": str(TMP / "thoughts.jsonl"),
        "interactions_path": str(TMP / "interactions.jsonl"),
        "reports_path": str(TMP / "reports.jsonl"),
        # Память и журнал диалога — только в TMP (AUD-04): раньше тесты
        # работали с реальными logs/thinking/memory.json и forget() в
        # test_chat_memory_tokens стирал пользовательские факты.
        "memory_path": str(TMP / "memory.json"),
        "chat_path": str(TMP / "chat.jsonl"),
        "retry": {"max_attempts": 1, "backoff_base": 1.0},
        "enabled": True,
        "fallback_on_error": True,
    }
    cfg.update(kw)
    return ThinkingClient(cfg)


# --------------------------------------------------------------------------- #
#  Контракты
# --------------------------------------------------------------------------- #
def test_schemas() -> None:
    check(extract_json('```json\n{"a": 1}\n```') == {"a": 1},
          "extract_json: markdown-обёртка")
    check(extract_json('вот план: {"a": 1} и ещё текст') == {"a": 1},
          "extract_json: хвостовой текст")
    check(extract_json('{"a": 1,}') == {"a": 1}, "extract_json: висячая запятая")
    check(raises(lambda: extract_json("просто текст"), ValueError),
          "extract_json: мусор вместо JSON")
    check(raises(lambda: extract_json("[1, 2]"), ValueError),
          "extract_json: массив вместо объекта")

    p = Plan.from_dict({"plan_id": "abc", "goal": "цель",
                        "steps": [{"id": 1, "action": "build", "desc": "собрать"}]})
    check(len(p.steps) == 1 and p.source == "colab", "Plan.from_dict: валидный план")
    check(raises(lambda: Plan.from_dict({"goal": "x", "steps": []}), SchemaError),
          "Plan: план без шагов отбраковывается")
    check(raises(lambda: Plan.from_dict(
        {"goal": "x", "steps": [{"id": i, "action": "build", "desc": "d"}
                                for i in range(31)]}), SchemaError),
          "Plan: больше 30 шагов отбраковывается")
    check(raises(lambda: Plan.from_dict(
        {"goal": "x", "steps": [{"id": 1, "action": "build", "desc": "a"},
                                {"id": 1, "action": "test", "desc": "b"}]}), SchemaError),
          "Plan: дубли id шагов отбраковываются")

    bad = Plan.from_dict({"goal": "x", "confidence": 9,
                          "steps": [{"id": 1, "action": "полёт_мысли",
                                     "desc": "d", "depends_on": [1]}]})
    check(bad.steps[0].action == "verify", "Plan: неизвестный тег action нормализуется")
    check(bad.confidence == 1.0, "Plan: confidence зажимается в [0..1]")
    check(bad.steps[0].depends_on == [], "Plan: самоссылка depends_on убирается")

    sec = Plan.from_dict({"goal": "x", "rationale": "token=abc123",
                          "steps": [{"id": 1, "action": "verify", "desc": "d"}]})
    check("[скрыто]" in sec.rationale, "Plan: секрет в rationale вычищается")

    snap_path = ROOT / "docs" / "schema_plan.json"
    if not snap_path.exists():
        check(False, "нет снимка схемы docs/schema_plan.json")
    else:
        from thinking.schemas import reflect_schema
        snap = json.loads(snap_path.read_text(encoding="utf-8"))
        check(snap.get("plan") == plan_schema(),
              "снимок схемы plan не совпадает с thinking/schemas.py")
        check(snap.get("reflect") == reflect_schema(),
              "снимок схемы reflect не совпадает с thinking/schemas.py")

    r = ReflectResponse.from_dict({"status": "чушь", "advice": "перепроверь шаг 3"})
    check(r.status == "ok", "ReflectResponse: неизвестный статус нормализуется")
    check(raises(lambda: ReflectResponse.from_dict([1, 2]), SchemaError),
          "ReflectResponse: массив вместо объекта отбраковывается")
    check(r.to_dict()["next_steps"] == [], "ReflectResponse.to_dict: next_steps список")

    check(has_secret("password=hunter2") and has_secret("api-key: 12345"),
          "has_secret: находит пароль и ключ")
    check(not has_secret("пароль в игре не используется"),
          "has_secret: обычный текст не считается секретом")
    check(redact_secrets("access_token: xyz") == "access_token=[скрыто]",
          "redact_secrets: значение заменяется")
    check(datetime.fromisoformat(utcnow()), "utcnow: разбирается как ISO-дата")


# --------------------------------------------------------------------------- #
#  Мягкая деградация
# --------------------------------------------------------------------------- #
def test_fallback() -> None:
    plan = local_plan("Проверить генерацию волн")
    check(isinstance(plan, dict) and plan.get("steps"), "local_plan: возвращает шаги")
    check(plan.get("source") == "local-fallback", "local_plan: source=local-fallback")
    check(plan.get("goal") == "Проверить генерацию волн", "local_plan: цель = задача")
    check(str(plan.get("rationale", "")).strip(), "local_plan: есть обоснование")
    reason = offline_reason(make_client())
    check(isinstance(reason, str) and reason.strip(), "offline_reason: непустая причина")
    check(any(c.isalpha() and ord(c) > 0x400 for c in reason),
          "offline_reason: причина по-русски")


# --------------------------------------------------------------------------- #
#  Клиент: журнал отчётов и взаимодействий
# --------------------------------------------------------------------------- #
def test_client() -> None:
    c = make_client()
    st = c.status()
    check({"online", "interactions", "reports", "benefits"} <= set(st),
          "status: отдаёт interactions/reports/benefits")
    check(not c.base, "клиент: URL берётся из config/thinking.local.json (здесь пусто)")
    check(c.cfg.get("enabled") is True, "клиент: config/thinking.json читается")

    check(raises(lambda: c._check_secrets("password=hunter2"), SchemaError),
          "секрет в задаче блокирует вызов (SchemaError)")
    check(raises(lambda: c.plan("задача без URL"), ThinkingError),
          "plan без URL падает ThinkingError")

    plan, is_fb = c.plan_with_fallback("Собрать 100 кадров волн")
    check(is_fb is True, "plan_with_fallback: без Colab возвращает заглушку")
    check(plan.get("source") == "local-fallback", "заглушка помечена local-fallback")

    inter = [json.loads(x) for x in (TMP / "interactions.jsonl")
             .read_text(encoding="utf-8").splitlines() if x.strip()]
    check(len(inter) >= 2, "взаимодействия дозаписываются в JSONL (ошибка + заглушка)")
    check(any(i.get("fallback") for i in inter), "заглушка фиксируется в журнале")
    check(all("kind" in i and "duration_ms" in i for i in inter),
          "каждая запись взаимодействия содержит kind и duration_ms")

    reports = [json.loads(x) for x in (TMP / "reports.jsonl")
               .read_text(encoding="utf-8").splitlines() if x.strip()]
    check(bool(reports) and reports[-1].get("type") == "plan",
          "отчёт о плане дозаписывается в JSONL")
    check(reports[-1].get("fallback") is True, "отчёт заглушки помечен fallback")
    check(all("action_ru" in s for s in reports[-1].get("steps", [])),
          "в отчёте шаги с русскими названиями действий")

    # другой процесс записал историю — панель обязана её увидеть
    other = TMP / "interactions.jsonl"
    with open(other, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"n": 99, "kind": "reflect", "at": "2099-01-01T00:00:00+00:00",
                             "request": "шаг 1: тест", "duration_ms": 1200, "ok": True,
                             "fallback": False, "background": True, "source": "colab",
                             "plan_id": "deadbeef", "summary": "совет по шагу"},
                            ensure_ascii=False) + "\n")
    c2 = make_client()
    st2 = c2.status()
    check(any(i.get("kind") == "reflect" and i.get("at", "").startswith("2099")
              for i in st2["interactions"]),
          "панель видит вызовы, сделанные в другом процессе (чтение JSONL)")
    ben = st2["benefits"]
    check(ben["calls_total"] >= 3 and ben["fallbacks"] >= 1,
          "benefits: считаются обращения и заглушки")
    check(ben["background_ms"] >= 1200, "benefits: фоновое время считается")
    check(ben["reports"] >= 1 and ben["steps_generated"] >= 1,
          "benefits: считаются отчёты и шаги")
    check(isinstance(ben.get("notes"), list) and all(isinstance(n, str) for n in ben["notes"]),
          "benefits: пояснения по-русски")

    # отчёт о рефлексии
    c2._report_reflect({"status": "adjust", "advice": "уменьшить denoise",
                        "rationale": "шумит", "next_steps": [
                            {"id": 2, "action": "test", "desc": "прогнать 10 кадров"}]},
                       "deadbeef", 1, "538/0")
    reps = [json.loads(x) for x in (TMP / "reports.jsonl")
            .read_text(encoding="utf-8").splitlines() if x.strip()]
    check(reps[-1]["type"] == "reflect" and reps[-1]["status"] == "adjust",
          "отчёт рефлексии пишется в JSONL")
    check(reps[-1]["steps"] and reps[-1]["steps"][0]["action_ru"] == "тесты",
          "русское название действия в рефлексии")

    # set_url не трогает настоящий конфиг
    saved = client_mod.LOCAL_PATH
    try:
        client_mod.LOCAL_PATH = TMP / "thinking.local.json"
        path = ThinkingClient.set_url("https://example.invalid", "токен")
        data = json.loads(path.read_text(encoding="utf-8"))
        check(data.get("base_url") == "https://example.invalid",
              "set_url: пишет base_url в локальный конфиг")
        check(data.get("token") == "токен", "set_url: пишет токен в локальный конфиг")
    finally:
        client_mod.LOCAL_PATH = saved
        # Тест оставил в TMP-конфиге example.invalid — иначе все следующие
        # make_client() ходят в сеть на несуществующий хост (AUD-04)
        (TMP / "thinking.local.json").unlink(missing_ok=True)

    # недоступный сервер: ошибка фиксируется, приложение не падает
    c3 = make_client(base="http://127.0.0.1:1", retry={"max_attempts": 1})
    c3.base = "http://127.0.0.1:1"
    check(raises(lambda: c3.plan("тест"), ThinkingError), "недоступный сервер -> ThinkingError")
    check(c3.status()["benefits"]["calls_total"] >= 4,
          "ошибка тоже попадает в статистику")


# --------------------------------------------------------------------------- #
#  Файлы подсистемы
# --------------------------------------------------------------------------- #
def test_files() -> None:
    cfg = json.loads((ROOT / "config" / "thinking.json").read_text(encoding="utf-8"))
    check("token" not in cfg and "base_url" not in cfg or
          not cfg.get("token") and not cfg.get("base_url"),
          "config/thinking.json не хранит секреты")
    check("log_path" in cfg and "panel_port" in cfg, "config/thinking.json: базовые ключи")

    html = (ROOT / "tools" / "thinking_panel.html").read_text(encoding="utf-8")
    check('<html lang="ru">' in html, "панель: язык страницы русский")
    for pane in ("reports", "dialog", "live", "benefit", "log", "dev", "models"):
        check(f'data-pane="{pane}"' in html, f"панель: вкладка {pane} есть")
    for label in ("Отчёты", "Диалог", "Выгода и ускорение",
                  "Связь агентов", "Разработка", "Скачать отчёт",
                  "за сегодня", "за всё время"):
        check(label in html, f"панель: подпись «{label}» есть")
    check("Мысли в реальном времени" not in html,
          "панель: старая подпись вкладки убрана")
    check('"/api/state"' in html and 'EventSource("/events")' in html,
          "панель: подключена к /api/state и /events")
    check("СУБАГЕНТ НА СВЯЗИ" in html, "панель: бейдж связи по-русски")
    check("системой и не правит файлы" in html,
          "панель: субагент не управляет системой")
    check("управляет игрой" not in html,
          "панель: упоминание игры в описании субагента убрано")
    check(not re_search(r"TODO|FIXME|lorem ipsum", html), "панель: нет мусорных заглушек")
    check("innerHTML = \"\" + ev.text" not in html,
          "панель: текст событий вставляется через textContent")

    cli = (ROOT / "tools" / "thinking_cli.py").read_text(encoding="utf-8")
    for cmd in ("doctor", "set-url", "plan", "ask", "reflect", "tail", "panel",
                "metrics", "reflect-metrics"):
        check(f'"{cmd}"' in cli, f"CLI: команда {cmd} объявлена")
    check("результат: python tools/thinking_cli.py tail" in cli,
          "CLI: фоновая рефлексия объясняет, где посмотреть результат")


def test_reliability() -> None:
    """Надёжность по совету субагента: таймаут потока и занятый порт панели."""
    cfg = json.loads((ROOT / "config" / "thinking.json").read_text(encoding="utf-8"))
    check(float(cfg.get("stream_timeout", 0)) > 25,
          "config: stream_timeout больше ping-интервала сервера (25 с)")
    check(float(cfg.get("panel_poll", 0)) > 0,
          "config: panel_poll задан — панель опрашивает /events как основной канал")
    server = (ROOT / "thinking" / "colab" / "cell_c_server.py").read_text(encoding="utf-8")
    check("timeout=10" in server and 'yield ": ping' in server,
          "Colab: ping сервера чаще клиентского таймаута (10 с < 40 с)")

    src = (ROOT / "thinking" / "client.py").read_text(encoding="utf-8")
    check("stream_timeout" in src, "клиент: таймаут потока берётся из конфига")
    check("timeout=read_to" in src, "клиент: поток открывается с настраиваемым таймаутом")
    check("self.online = False\n" not in src.split("def _stream_loop")[1].split("def ")[0],
          "клиент: обрыв SSE не перебивает онлайн-статус из /health")

    # Поведенческая проверка: пустой порт считается свободным
    mod = _load_cli_module()
    free = _free_port()
    check(mod._port_busy("127.0.0.1", free) is False,
          "CLI: свободный порт не считается занятым")
    check("ConnectionAbortedError" in (ROOT / "tools" / "thinking_cli.py").read_text(
        encoding="utf-8"),
        "CLI: обрыв вкладки панели не роняет поток с трассировкой")
    check("handle_error" in (ROOT / "tools" / "thinking_cli.py").read_text(encoding="utf-8"),
          "CLI: панель глушит трассировки socketserver при уходе браузера")
    check(_is_read_timeout(TimeoutError("timed out")) is True,
          "клиент: таймаут чтения распознаётся")
    check(_is_read_timeout(OSError("connection refused")) is False,
          "клиент: обычная ошибка не считается таймаутом")


def _free_port() -> int:
    import socket as _s
    with _s.socket(_s.AF_INET, _s.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_chat_memory_tokens() -> None:
    """Чат, память, токены и удаление — то, что добавили в панель сегодня."""
    from thinking.schemas import ChatReply, ChatTurn, chat_schema, estimate_tokens

    # --- контракты ---
    t = ChatTurn.from_dict({"role": "человек", "text": "привет"})
    check(t.role == "user", "чат: неизвестная роль нормализуется в user")
    check(ChatTurn.from_dict({"role": "subagent", "text": "x"}).role == "subagent",
          "чат: роль subagent сохраняется")
    rep = ChatReply.from_dict({"reply": "совет", "tokens_in": 10, "tokens_out": 5,
                               "tokens_estimate": False})
    check(rep.tokens_total == 15, "чат: tokens_total считается")
    check(rep.tokens_estimate is False, "чат: точность usage помечается")
    check(ChatReply.from_dict({"source": "что-то"}).source == "colab",
          "чат: неизвестный источник нормализуется")
    check(ChatReply.from_dict({}).fallback is False, "чат: пустой ответ валиден")
    check(raises(lambda: ChatReply.from_dict("строка"), SchemaError),
          "чат: не-объект отбраковывается")
    check(estimate_tokens("") == 0 and estimate_tokens("абв") == 1,
          "токены: оценка по длине текста")
    neg = ChatReply.from_dict({"tokens_in": -5, "tokens_out": -1})
    check(neg.tokens_in == 0 and neg.tokens_out == 0,
          "токены: отрицательные значения приводятся к 0")

    # --- память ---
    c = make_client()
    check(c.memory()["facts"] == [] or isinstance(c.memory()["facts"], list),
          "память: читается списком фактов")
    c.remember("факт один")
    c.remember("факт два")
    mem = c.memory()
    check("факт два" in mem["facts"], "память: факт сохраняется")
    c.remember("факт два")
    check(len(c.memory()["facts"]) == len(mem["facts"]),
          "память: повторный факт не дублируется")
    cleaned = c.forget()
    check(cleaned["facts"] == [], "память: forget() стирает факты")

    # --- токены ---
    tok = c.tokens()
    for key in ("tokens_in", "tokens_out", "tokens_total", "calls"):
        check(key in tok, f"токены: ключ {key} есть в сводке")
    check(tok["tokens_total"] == tok["tokens_in"] + tok["tokens_out"],
          "токены: итог равен сумме")
    check(isinstance(tok["estimate_only"], bool), "токены: помечается, что это оценка")

    # --- удаление ---
    before = count_lines(c.cfg.get("reports_path"))
    check(c.delete_history("report", n=999999) == 0,
          "удаление: несуществующая запись не считается удалённой")
    check(count_lines(c.cfg.get("reports_path")) == before,
          "удаление: файл не меняется, если нечего удалять")
    check(raises(lambda: c.delete_history("не-раздел"), SchemaError),
          "удаление: неизвестный раздел отбраковывается")

    # РЕГРЕСС: номер n не уникален (счётчик процесса) — удаление по одному
    # только n стирало лишние записи. Удалять надо по rid.
    c2 = make_client()
    rep = c2._rep_path
    with open(rep, "w", encoding="utf-8") as fh:
        for rid, n in (("aaaa1111", 1), ("bbbb2222", 1), ("cccc3333", 1)):
            fh.write(json.dumps({"n": n, "rid": rid, "at": "2026-10-02T10:00:00+00:00",
                                 "type": "plan", "goal": "тест " + rid},
                                ensure_ascii=False) + "\n")
    c2.reports, c2.interactions = [], []
    removed = c2.delete_history("report", rid="bbbb2222")
    left = [json.loads(x) for x in open(rep, encoding="utf-8")]
    check(removed == 1, "удаление по rid убирает ровно одну запись")
    check([r["rid"] for r in left] == ["aaaa1111", "cccc3333"],
          "удаление по rid не трогает записи с таким же номером n")
    check(all("rid" in r for r in left), "удаление: записи с rid не пострадали")

    # --- сервер Colab ---
    server = (ROOT / "thinking" / "colab" / "cell_c_server.py").read_text(encoding="utf-8")
    check('"/chat"' in server or "def chat(" in server, "Colab: эндпоинт /chat есть")
    check("SYSTEM_CHAT" in server, "Colab: системный промт чата есть")
    check("build_chat_prompt" in server, "Colab: промт собирает память и историю")
    check('"tokens_in"' in server and "usage.get" in server,
          "Colab: usage от LLM учитывается в токенах")
    # РЕГРЕСС: слабая модель отвечает текстом, а не JSON — раньше был HTTP 502
    check("async def guard_reply" in server and "data = await guard_reply(raw)" in server,
          "Colab: чат принимает обычный текст вместо 502")
    check("async def plan_data" in server and "plan_data(raw, req.task)" in server,
          "Colab: план строится и из текста, а не падает")
    check("async def reflect_data" in server and "reflect_data(raw)" in server,
          "Colab: рефлексия принимает обычный текст")
    check("def coerce_plan" in server and "coerce_plan(data)" in server,
          "Colab: план приводится к схеме (строки вместо списков, id как число)")
    check("PLAN_LIST_FIELDS" in server and "success_criteria" in server,
          "Colab: списочные поля плана чинятся, а не роняют запрос")
    # --- переключение моделей и потоковый чат ---
    check('async def models(' in server and 'async def set_model(' in server,
          "Colab: список моделей и переключение на лету есть")
    check("_switch_to" in server and "thinking_models.txt" in server,
          "Colab: переключение поднимает движок на выбранной модели")
    check('async def chat_stream_ep(' in server and '"/chat/stream"' in server,
          "Colab: чат отдаётся потоком (обход 120-секундного лимита туннеля)")
    check("THINKING_CHAT_MAX_TOKENS" in server,
          "Colab: у потокового чата свой лимит длины ответа")
    setup_a = (ROOT / "thinking" / "colab" / "cell_a_setup.py").read_text(encoding="utf-8")
    launch_d = (ROOT / "thinking" / "colab" / "cell_d_launch.py").read_text(encoding="utf-8")
    check("WANTED" in setup_a and "уже на диске" in setup_a,
          "ячейка A: скачивает обе модели, но повторно ничего не качает")
    check("thinking_active_model.txt" in launch_d,
          "ячейка D: запоминает активную модель для панели и сторожа")
    check("active = MODEL" in launch_d and "ACTIVE_FILE" in launch_d,
          "ячейка D: сторож поднимает ту модель, которую выбрали в панели")
    check("_guard(raw)" not in server.split("async def chat(")[-1],
          "Colab: в /chat больше нет строгого разбора")

    # --- панель ---
    html = (ROOT / "tools" / "thinking_panel.html").read_text(encoding="utf-8")
    for marker in ("Чат с субагентом", "Токены", "sendChat", "renderTokens",
                   "delRecord", "clearAll", "renderMemory", "renderChat",
                   "/api/chat", "/api/delete", "/api/memory",
                   "Модели", "loadModels", "apiStream", "/api/chat/stream",
                   "/api/models", "/api/model"):
        check(marker in html, f"панель: есть {marker}")
    for pane in ("chat", "tokens"):
        check(f'data-pane="{pane}"' in html, f"панель: вкладка {pane} есть")

    # --- CLI ---
    cli = (ROOT / "tools" / "thinking_cli.py").read_text(encoding="utf-8")
    for route in ("/api/chat", "/api/delete", "/api/memory", "/api/tokens"):
        check(route in cli, f"CLI: маршрут {route} есть")
    check("def do_POST" in cli, "CLI: панель принимает POST-запросы")


def re_search(pattern: str, text: str) -> bool:
    import re
    return bool(re.search(pattern, text))


def _load_cli_module():
    """Импортирует CLI как модуль, чтобы проверить его словари русских подписей."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "thinking_cli_under_test", ROOT / "tools" / "thinking_cli.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_rus_labels() -> None:
    mod = _load_cli_module()
    check(set(EVENT_TYPES) <= set(mod.EVENT_RU),
          "CLI: все типы событий имеют русские названия")
    check(set(ACTIONS) <= set(mod.ACTION_RU),
          "CLI: все действия имеют русские названия")
    check(set(ACTIONS) <= set(ThinkingClient.ACTION_RU),
          "клиент: все действия имеют русские названия")
    for ru in list(mod.EVENT_RU.values()) + list(mod.ACTION_RU.values()):
        if not any(c.isalpha() and ord(c) > 0x400 for c in ru):
            check(False, f"подпись без русских букв: {ru}")
            break
    else:
        check(True, "все подписи на русском")


# --------------------------------------------------------------------------- #
#  Регрессии аудита (audit.md) — поведенческие тесты вместо «строка есть»
# --------------------------------------------------------------------------- #
def test_memory_action() -> None:
    """Экспорт/импорт памяти работают (AUD-01: раньше NameError на utcnow)."""
    mod = _load_cli_module()
    c = make_client()
    code, payload = mod.memory_action(c, {"action": "export"})
    check(code == 200 and "exported" in payload
          and payload["memory"]["facts"] == [],
          "память: экспорт возвращает структуру и метку времени")
    c.remember("факт для импорта")
    code, payload = mod.memory_action(c, {"action": "export"})
    exported = payload["memory"]
    c.forget()
    check(c.memory()["facts"] == [], "память: forget стирает факты")
    code, payload = mod.memory_action(c, {"action": "import", "memory": exported})
    check(code == 200 and "факт для импорта" in payload["memory"]["facts"],
          "память: импорт восстанавливает факты")
    code, payload = mod.memory_action(c, {"action": "import", "memory": "не словарь"})
    check(code == 400 and "error" in payload,
          "память: импорт отбраковывает не-объект")
    code, payload = mod.memory_action(c, {"action": "remove",
                                         "fact": "факт для импорта"})
    check(code == 200 and payload["memory"]["facts"] == [],
          "память: remove удаляет факт")
    code, payload = mod.memory_action(c, {"action": "remember", "fact": "ещё факт"})
    check(code == 200 and "ещё факт" in payload["memory"]["facts"],
          "память: remember через action")


def test_panel_routes() -> None:
    """Все маршруты, которые зовёт панель, существуют в CLI (AUD-18)."""
    cli_src = (ROOT / "tools" / "thinking_cli.py").read_text(encoding="utf-8")
    html = (ROOT / "tools" / "thinking_panel.html").read_text(encoding="utf-8")
    routes = set()
    for m in re.finditer(r'api(?:Get|Stream)?\(\s*"(/api/[^"]+)"', html):
        routes.add(m.group(1).split("?")[0])
    for m in re.finditer(r'fetch\(\s*"(/api/[^"]+)"', html):
        routes.add(m.group(1).split("?")[0])
    check(bool(routes), "маршруты: в панели есть вызовы /api/*")
    missing = [r for r in sorted(routes) if f'"{r}"' not in cli_src]
    check(not missing, f"маршруты: все вызовы панели есть в CLI (нет: {missing})")


def test_breaker_channels() -> None:
    """Обрыв потока не блокирует REST и наоборот (AUD-05)."""
    c = make_client()
    c.base = "http://127.0.0.1:1"   # мёртвый порт: соединение отказывает мгновенно
    # открываем REST-предохранитель вручную (5 ошибок подряд)
    c._cb_errors = 5
    c._cb_open_until = time.time() + 60
    try:
        c._json("GET", "/health")
        check(False, "breaker: REST блокируется при открытом предохранителе")
    except ThinkingError as exc:
        check("предохранитель" in str(exc),
              "breaker: REST блокируется при открытом предохранителе")
    # поток при этом не заблокирован предохранителем (падает по сети)
    try:
        c._open("GET", "/events/stream", stream=True)
        check(False, "breaker: поток не должен блокироваться REST-предохранителем")
    except ThinkingError as exc:
        check("предохранитель" not in str(exc),
              "breaker: поток не блокируется REST-предохранителем (AUD-05)")
    # и наоборот: открыт потоковый — REST работает
    c2 = make_client()
    c2.base = "http://127.0.0.1:1"
    c2._cb_stream_errors = 5
    c2._cb_stream_until = time.time() + 60
    try:
        c2._open("GET", "/events/stream", stream=True)
        check(False, "breaker: поток блокируется своим предохранителем")
    except ThinkingError as exc:
        check("предохранитель" in str(exc),
              "breaker: поток блокируется своим предохранителем")
    try:
        c2._json("GET", "/health")
        check(False, "breaker: REST не должен блокироваться потоковым предохранителем")
    except ThinkingError as exc:
        check("предохранитель" not in str(exc),
              "breaker: REST не блокируется потоковым предохранителем (AUD-05)")


def test_plan_stream_accounting() -> None:
    """Успешный план через поток попадает в метрики и отчёты (AUD-06)."""
    plan_json = json.dumps({"plan_id": "mock1", "goal": "цель",
                            "steps": [{"id": 1, "action": "build",
                                       "desc": "собрать"}]})

    class H(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            self.rfile.read(length)
            body = (
                'data: {"type": "thought", "text": "думаю"}\n\n'
                f'data: {{"type": "final", "text": {json.dumps(plan_json)}}}\n\n'
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        c = make_client()
        c.base = f"http://127.0.0.1:{srv.server_address[1]}"
        plan = c.plan_stream("тестовая задача")
        check(plan.get("plan_id") == "mock1", "plan_stream: план разобран из потока")
        check(c.stats["plans"] == 1, "plan_stream: план попал в stats (AUD-06)")
        inter = c.status()["interactions"]
        check(any(i.get("kind") == "plan" and i.get("ok") for i in inter),
              "plan_stream: успешный план записан в журнал (AUD-06)")
        reps = [r for r in c.reports if r.get("type") == "plan"]
        check(bool(reps), "plan_stream: отчёт плана записан (AUD-06)")
    finally:
        srv.shutdown()


def test_benefits_no_double_count() -> None:
    """benefits не считает фолбэки дважды (AUD-07)."""
    c = make_client(interactions_path=str(TMP / "ben_inter.jsonl"),
                    reports_path=str(TMP / "ben_reps.jsonl"))
    c._record("plan", "задача 1", time.time(), ok=False, fallback=True,
              summary="нет связи")
    c._record("plan", "задача 2", time.time(), ok=False, fallback=True,
              summary="нет связи")
    ben = c.status()["benefits"]
    check(ben["fallbacks"] == 2,
          "benefits: фолбэки считаются один раз, а не дважды (AUD-07)")


def test_token_honesty() -> None:
    """Заглушки не тратят токены (AUD-22)."""
    c = make_client(interactions_path=str(TMP / "tok_inter.jsonl"),
                    reports_path=str(TMP / "tok_reps.jsonl"),
                    chat_path=str(TMP / "tok_chat.jsonl"),
                    memory_path=str(TMP / "tok_mem.json"))
    c.base = "http://127.0.0.1:1"   # мёртвый сервер -> заглушка
    reply = c.chat("привет")
    check(reply.get("fallback") is True, "токены: заглушка при недоступном сервере")
    check(reply.get("tokens_in") == 0 and reply.get("tokens_out") == 0,
          "токены: заглушка не тратит токены (AUD-22)")
    check(c.tokens()["tokens_total"] == 0,
          "токены: сводка не раздута заглушкой (AUD-22)")


def test_secrets_smart() -> None:
    """os.getenv(...) — не секрет; литерал — секрет (AUD-13)."""
    check(not has_secret("token = os.getenv('THINKING_TOKEN')"),
          "секреты: os.getenv(...) не блокирует запрос (AUD-13)")
    check(has_secret("password=hunter2"),
          "секреты: литерал по-прежнему блокируется")
    check(redact_secrets("token = os.getenv('X')") == "token = os.getenv('X')",
          "секреты: redact не трогает ссылки на переменные (AUD-13)")
    check(redact_secrets("password=hunter2") == "password=[скрыто]",
          "секреты: redact маскирует литерал")
    # JSON/словарь: раньше кавычка между ключом и двоеточием ломала регулярку
    # и секрет уезжал в Colab и оседал в журналах (аудит B-1)
    check(has_secret('"password": "hunter2"') and has_secret("{'token': 'abc'}"),
          "секреты: кавычки в JSON не прячут секрет (B-1)")
    check(has_secret('{"api_key": "sk-123"}'),
          "секреты: api_key в JSON находится (B-1)")
    check(not has_secret('{"profile": "dev"}'),
          "секреты: обычное JSON-поле не считается секретом")
    check("hunter2" not in redact_secrets('"password": "hunter2"'),
          "секреты: redact вычищает секрет из JSON (B-1)")
    check(not has_secret('{"token": "os.getenv(X)"}'),
          "секреты: ссылка на переменную в JSON — не секрет")
    c = make_client()
    check(raises(lambda: c._dev_body("посмотри конфиг",
                                     active_code='token = "abc123"'), SchemaError),
          "секреты: код с секретом блокирует dev-запрос (AUD-13)")
    check(not raises(lambda: c._dev_body("посмотри конфиг",
                                         active_code="token = os.getenv('X')"),
                     SchemaError),
          "секреты: код со ссылкой на переменную проходит (AUD-13)")


def test_reflect_async_spawns_process() -> None:
    """reflect --async запускает отсоединённый процесс, а не daemon-поток
    (AUD-03: daemon-поток умирал вместе с CLI, результат терялся)."""
    mod = _load_cli_module()
    calls = []

    class FakePopen:
        def __init__(self, cmd, **kw):
            self.pid = 4242
            calls.append((cmd, kw))

    old_popen = mod.subprocess.Popen
    mod.subprocess.Popen = FakePopen
    try:
        args = argparse.Namespace(plan_id="p1", step=1, result="ok",
                                  observation=None, error=None, async_mode=True)
        rc = mod.cmd_reflect(make_client(), args)
        check(rc == 0 and len(calls) == 1,
              "reflect --async: запускает дочерний процесс (AUD-03)")
        cmd = calls[0][0]
        check(cmd[0] == sys.executable and "reflect" in cmd and "p1" in cmd,
              "reflect --async: дочерний процесс зовёт reflect с теми же аргументами")
        check("--async" not in cmd,
              "reflect --async: у дочернего процесса нет --async")
    finally:
        mod.subprocess.Popen = old_popen


def test_devsave_args() -> None:
    """«Применить» передаёт код и имя файла модели в devSave (AUD-02)."""
    html = (ROOT / "tools" / "thinking_panel.html").read_text(encoding="utf-8")
    check('devSave("model", p.filename, p.code)' in html,
          "панель: Применить передаёт код и имя файла модели (AUD-02)")
    check('devSave("model")' not in html,
          "панель: нет вызова devSave без аргументов (AUD-02)")
    check("await devRefresh(r.name)" in html,
          "панель: devSave ждёт обновления списка файлов (AUD-02)")


def test_dev_reports() -> None:
    """Dev-предложения пишутся в отчёты (вкладка «Отчёты» не пустует)."""
    c = make_client(interactions_path=str(TMP / "devrep_inter.jsonl"),
                    reports_path=str(TMP / "devrep_reps.jsonl"),
                    chat_path=str(TMP / "devrep_chat.jsonl"),
                    memory_path=str(TMP / "devrep_mem.json"))
    c.base = "http://127.0.0.1:1"   # мёртвый сервер -> fallback
    c._stream_read = lambda *_a, **_k: (_ for _ in ()).throw(
        ThinkingError("поток разработки оборвался после 3 попыток: timeout"))
    c._json = lambda *_a, **_k: {"proposal": {"action": "edit",
                                                "filename": "main.py",
                                                "code": "x = 42", "comment": "исправил"}}
    prop = c.dev("исправь main.py")
    check(prop.get("action") == "edit", "dev: предложение получено")
    reps = [r for r in c.reports if r.get("type") == "dev"]
    check(len(reps) == 1 and reps[0].get("filename") == "main.py",
          "отчёты: dev-предложение записано (вкладка «Отчёты» не пустует)")
    check(reps[0].get("code") == "x = 42", "отчёты: код в отчёте")


def test_chat_author() -> None:
    """Видно, кто спрашивал субагента: человек (панель) или агент (CLI)."""
    c = make_client(chat_path=str(TMP / "author_chat.jsonl"),
                    memory_path=str(TMP / "author_mem.json"),
                    interactions_path=str(TMP / "author_inter.jsonl"),
                    reports_path=str(TMP / "author_reps.jsonl"))
    c.base = "http://127.0.0.1:1"        # мёртвый сервер -> заглушка, сеть не трогаем
    c.chat("вопрос человека", use_memory=False, author="human")
    c.chat("вопрос агента", use_memory=False, author="agent")
    c.chat("вопрос без метки", use_memory=False)
    rows = [json.loads(l) for l in
            open(TMP / "author_chat.jsonl", encoding="utf-8") if l.strip()]
    who = [r.get("author") for r in rows]
    check(who == ["human", "agent", "agent"],
          f"автор реплики пишется в журнал чата (получено {who})")
    check(rows[0].get("question") == "вопрос человека",
          "автор не путает реплики — вопрос на месте")
    # по умолчанию программный вызов = агент: забытый параметр не врёт
    check(rows[2].get("author") == "agent",
          "автор по умолчанию — агент (CLI/программа), не человек")

    # то же в ленте обращений: _record пишет автора
    c2 = make_client(interactions_path=str(TMP / "author2_inter.jsonl"),
                     reports_path=str(TMP / "author2_reps.jsonl"))
    c2._record("plan", "задача от человека", time.time(), ok=True,
               summary="ок", author="human")
    c2._record("plan", "задача от агента", time.time(), ok=True,
               summary="ок", author="agent")
    inter = c2.status()["interactions"]
    authors = [i.get("author") for i in inter if i.get("kind") == "plan"]
    check(authors == ["human", "agent"],
          f"автор виден в ленте обращений (получено {authors})")

    html = (ROOT / "tools" / "thinking_panel.html").read_text(encoding="utf-8")
    check('t.author === "agent"' in html and '"агент" : "я"' in html,
          "панель: чат подписывает реплику агента иначе, чем вашу")
    check("m.author === \"agent\" ? \"агент\" : \"человек\"" in html,
          "панель: в ленте обращений видна метка человек/агент")
    # панель объявляет себя человеком
    cli_src = (ROOT / "tools" / "thinking_cli.py").read_text(encoding="utf-8")
    check(cli_src.count('author="human"') >= 3,
          "панель: её вызовы чата и разработки помечены как человеческие")


def test_budget_matches_running_model() -> None:
    """Бюджет пересчитывается по модели, которая реально поднялась.

    Найдено на живом прогоне 04.10: поднялась запасная 3B, а бюджеты
    остались от первой кандидатуры (30B) — лимиты были чужими.
    """
    launch = (ROOT / "thinking" / "colab" / "cell_d_launch.py").read_text(encoding="utf-8")
    # пересчёт обязан стоять ПОСЛЕ цикла выбора модели и писать без setdefault
    loop = launch.index("for attempt, cand in enumerate(MODELS, 1):")
    recalc = launch.index('os.environ["THINKING_MAX_TOKENS"] = _match[1]', loop)
    check(recalc > loop, "ячейка D: бюджет пересчитывается после выбора модели")
    tail = launch[recalc:recalc + 700]
    tail_marker = launch.index("if health is None:", loop)
    check(loop < launch.index("MODEL = cand", loop) < tail_marker,
          "ячейка D: MODEL обновляется на ту, что поднялась (внутри цикла выбора)")
    check("setdefault" not in tail,
          "ячейка D: пересчёт затирает бюджет, а не молчит из-за setdefault")
    # причина отказа модели показывается, а не просто «не поднялась»
    check("Последние строки лога" in launch,
          "ячейка D: при отказе модели печатается причина из лога")
    check("не хватить" in launch,
          "ячейка D: при нехватке памяти подсказывает профиль strong")
    # время ожидания растёт от размера модели
    check("tries = 45 + int(size_gb * 12)" in launch,
          "ячейка D: большая модель ждёт подъёма дольше (17 ГБ не успевает за 90 с)")


def test_dev_profile_and_honest_hints() -> None:
    """Профиль для CPU и честные подсказки о моделях.

    Замер 04.10: 14B (8.4 ГБ весов) при свободных 3.0 ГБ дала 0.05 ток/с —
    подкачка страниц. Подсказка обещала «влезает в RAM целиком, ~2–4 ток/с»,
    то есть врала ровно в том, на чём человек и принимает решение.

    Профиль `dev`: быстрая основная модель, запасная меньше неё, и 7B
    отдельно — на диске для переключения во вкладке «Модели», но вне цепочки
    отката (откат на CPU обязан вести к меньшей модели).
    """
    setup = (ROOT / "thinking" / "colab" / "cell_a_setup.py").read_text(encoding="utf-8")
    check('"dev": ["3b-instruct-q4", "1.5b-instruct-q4"]' in setup,
          "ячейка A: профиль dev — 3B основная, 1.5B запасная")
    check('"dev": ["7b-instruct-q4"]' in setup,
          "ячейка A: профиль dev докачивает 7B вне цепочки отката")
    check("if tag in WANTED_TAGS:" in setup,
          "ячейка A: в цепочку отката попадают только модели профиля")
    check("вне цепочки отката" in setup,
          "ячейка A: говорит, зачем докачана модель вне цепочки")
    check("0.05 ток/с" in setup and "подкачки страниц" in setup,
          "ячейка A: подсказка про 14B называет замеренную скорость и причину")
    check("влезает в RAM целиком" not in setup,
          "ячейка A: подсказка про 14B больше не обещает несуществующую скорость")
    # окно для будущего платного Colab не закрываем
    check("Нужен RAM от 16 ГБ" in setup,
          "ячейка A: 14B помечена как вариант для рантайма с RAM от 16 ГБ")


def test_failure_diagnostics() -> None:
    """Причину отказа модели видно по коду возврата, а отказ от остановки
    в ячейке F не выглядит аварией.

    Найдено на живом прогоне 04.10: лог модели был ПУСТ (только строка команды),
    поэтому пришлось гадать — память или архитектура. Код возврата отвечает
    однозначно: -9 это OOM-киллер, -11 это segfault (старый llama.cpp без Qwen3).
    """
    launch = (ROOT / "thinking" / "colab" / "cell_d_launch.py").read_text(encoding="utf-8")
    check("rc = llm_proc.poll()" in launch,
          "ячейка D: при отказе модели читается код возврата процесса")
    check("ПРОЦЕСС УБИТ" in launch and "-9" in launch and "-11" in launch,
          "ячейка D: по сигналу различается нехватка памяти и segfault")
    check("OOM-киллер" in launch, "ячейка D: сигнал -9 объясняется как нехватка памяти")
    check("segfault" in launch and "архитектуру" in launch,
          "ячейка D: segfault объясняется неизвестной архитектурой модели")
    check("версию определить не удалось" in
          (ROOT / "thinking" / "colab" / "cell_a_setup.py").read_text(encoding="utf-8"),
          "ячейка A: печатает версию движка (от неё зависит поддержка Qwen3)")
    # нужную память видно ДО ожидания, а не после отказа
    check("веса+KV" in launch and "kv_gb" in launch,
          "ячейка D: до ожидания печатает требуемую память (веса+KV)")

    # SystemExit внутри ячейки Colab IPython принимает за просьбу выключить ядро:
    # «Run all» печатал «An exception has occurred» и «To exit; use 'exit'».
    stop = (ROOT / "thinking" / "colab" / "cell_f_stop.py").read_text(encoding="utf-8")
    # ловим именно вызов, а не упоминание в комментарии (о нём там написано
    # намеренно — чтобы было видно, что было и почему убрали)
    check("raise SystemExit" not in stop and "sys.exit(" not in stop,
          "ячейка F: отказ от остановки не выходит через SystemExit")
    check("return" in stop and "def stop()" in stop,
          "ячейка F: отказ — обычный return внутри функции")
    check("THINKING_STOP" in stop,
          "ячейка F: остановка по-прежнему требует явного разрешения")


def test_sse_heartbeat() -> None:
    """Поток «дышит», пока модель молчит — иначе Cloudflare даёт 524.

    Найдено на живом прогоне 04.10 с моделью 14B на CPU: первый токен не
    приходил дольше минуты, и Cloudflare рвал соединение, хотя модель была
    жива. Сервер шлёт `: ping` сразу и каждые 15 с.

    Это первый тест, который исполняет серверный код (остальные проверки
    cell_c — сверка текста), потому что тут важно поведение, а не наличие
    строки: пинг должен идти при молчании, данные — проходить насквозь, а
    ошибка модели — подниматься до вызывающего.
    """
    import asyncio
    src = (ROOT / "thinking" / "colab" / "cell_c_server.py").read_text(encoding="utf-8")
    start = src.index("async def with_heartbeat")
    rest = src[start + 10:]
    end = min((rest.find("\nasync def ") if rest.find("\nasync def ") >= 0 else len(rest)),
              (rest.find("\ndef ") if rest.find("\ndef ") >= 0 else len(rest)))
    ns: dict = {"asyncio": asyncio, "time": __import__("time")}
    exec("import asyncio, time\n" + src[start:start + 10 + end], ns)
    hb = ns["with_heartbeat"]

    async def slow():
        await asyncio.sleep(0.35)          # модель думает дольше, чем gap
        yield "data: {\"type\":\"token\"}\n\n"

    async def boom():
        yield "data: A\n\n"
        raise RuntimeError("upstream упал")

    async def collect(srcgen, gap):
        out = []
        async for chunk in hb(srcgen, gap=gap):
            out.append(chunk)
        return out

    chunks = asyncio.run(collect(slow(), 0.1))
    check(chunks and chunks[0].startswith(":"),
          "SSE: первый байт тела уходит сразу, не дожидаясь модели")
    pings = [c for c in chunks if c.startswith(": ping")]
    check(len(pings) >= 2,
          f"SSE: пока модель молчит, идут пинги (получено {len(pings)})")
    check(any(c.startswith("data:") for c in chunks),
          "SSE: данные модели проходят насквозь")
    check(chunks[-1].startswith("data:"),
          "SSE: поток заканчивается данными, а не пингом")

    async def collect_err():
        out = []
        try:
            async for chunk in hb(boom(), gap=0.1):
                out.append(chunk)
        except RuntimeError as exc:
            return ("raised", str(exc))
        return ("no-raise", out)
    kind, val = asyncio.run(collect_err())
    check(kind == "raised" and "upstream упал" in val,
          "SSE: ошибка модели поднимается до вызывающего, а не теряется")

    # пинги должны быть на маршрутах, которые ходят через туннель
    for route in ("/plan/stream", "/chat/stream", "/dev/stream"):
        i = src.index(f'@app.post("{route}")')
        j = src.index("\n@app.", i + 1) if src.find("\n@app.", i + 1) >= 0 else len(src)
        check("with_heartbeat(gen())" in src[i:j],
              f"SSE: маршрут {route} отдаёт поток с пингами")
    check("stream_stall" in (ROOT / "thinking" / "client.py").read_text(encoding="utf-8"),
          "клиент: сторож молчания есть и не сработает на пингах")


def test_watchdog_and_ram_guard() -> None:
    """Сторож не убивает медленную модель, а нехватка памяти видна сразу.

    Найдено на живом прогоне 04.10 с 14B: 3 токена за 62 с = 0.05 ток/с при
    свободных 3.0 ГБ. Две опасные вещи:
    — сторож проверял движок HTTP-запросом с таймаутом 6 с, поэтому занятая
      модель выглядела мёртвой и её перезапускали каждые пару минут: на
      медленной модели ответ не доехал бы никогда;
    — ничто не предупреждало, что модель не помещается в RAM.
    """
    launch = (ROOT / "thinking" / "colab" / "cell_d_launch.py").read_text(encoding="utf-8")
    watch = launch[launch.index("def _watch()"):]
    check('PROCS.get("llm")' in watch and "proc.poll() is None" in watch,
          "сторож: медленную модель отличает по процессу, а не по HTTP-таймауту")
    check("занят, но процесс жив" in watch,
          "сторож: о медленной, но живой модели сообщает и не трогает её")
    check('PROCS["llm"] = subprocess.Popen' in watch,
          "сторож: после перезапуска запоминает новый процесс")
    check("timeout=20" in watch,
          "сторож: если процесса нет, проверяет HTTP с терпимым таймаутом")

    check("_avail_gb()" in launch and "MemAvailable" in launch,
          "ячейка D: свободная память читается по факту, а не по памятке")
    warn = launch[launch.index("ВНИМАНИЕ: весам нужно"):launch.index("Модель всё равно")]
    check("подкачка страниц" in warn and "0.05 ток/с" in warn,
          "ячейка D: предупреждает о подкачке страниц и называет замеренную скорость")
    check("Профиль gpu (7B" in warn,
          "ячейка D: при нехватке памяти советует профиль, который влезет")


def test_parallel_models() -> None:
    """Параллельные модели: поднимаются по запросу, отвечают все сразу,
    и честно сказано, что скорость от этого не растёт.

    Требование пользователя: несколько моделей должны работать одновременно.
    Считаем честно — ядра у рантайма те же, поэтому СУММАРНАЯ скорость не
    растёт (каждый ответ дольше примерно в N раз). Польза в сравнении и в
    доступности без переключения, и об этом сказано и в коде, и в выводе.
    """
    setup_d = (ROOT / "thinking" / "colab" / "cell_d_launch.py").read_text(encoding="utf-8")
    check("THINKING_PARALLEL" in setup_d,
          "ячейка D: дополнительные модели задаются переменной THINKING_PARALLEL")
    check("PARALLEL_FILE" in setup_d and "json.dump(_manifest" in setup_d,
          "ячейка D: поднятые дополнительные модели пишутся в манифест")
    check("_cmd[_cmd.index(\"--port\") + 1]" in setup_d,
          "ячейка D: у каждой дополнительной модели свой порт")
    check("не поместится, будет подкачка страниц" in setup_d,
          "ячейка D: модель, не влезающая в свободную память, не поднимается")
    check("по умолчанию\n# работает одна модель" in setup_d
          or "работает одна модель, как раньше" in setup_d,
          "ячейка D: без явного списка всё работает как раньше — одна модель")

    srv = (ROOT / "thinking" / "colab" / "cell_c_server.py").read_text(encoding="utf-8")
    check('@app.post("/ask/multi")' in srv, "сервер: маршрут параллельного вопроса")
    check("asyncio.gather" in srv,
          "сервер: модели опрашиваются одновременно, а не по очереди")
    check("суммарная скорость не растёт" in srv,
          "сервер: честно говорит, что суммарная скорость не растёт")
    check("url: str = \"\"" in srv and "url or UPSTREAM" in srv,
          "сервер: чат-стрим умеет говорить с конкретным движком")
    check('@app.get("/parallel")' in srv, "сервер: список параллельных моделей")

    cli = (ROOT / "thinking" / "client.py").read_text(encoding="utf-8")
    check("def multi_chat(" in cli and '"/ask/multi"' in cli,
          "клиент: метод параллельного вопроса")
    check("def parallel_models(" in cli and '"/parallel"' in cli,
          "клиент: узнать, какие модели держатся параллельно")
    check("multi_timeout" in cli,
          "клиент: на N моделей увеличено ожидание (ядра делятся)")

    tools = (ROOT / "tools" / "thinking_cli.py").read_text(encoding="utf-8")
    # Команды объявляет либо напрямую sub.add_parser, либо общий аддитер add()
    # (он раздаёт --url/--token для разового запуска против заглушки).
    check("def cmd_ask_multi(" in tools and '"ask-multi"' in tools
          and "add(" in tools,
          "CLI: команда ask-multi")
    check("самый быстрый:" in tools,
          "CLI: показывает, какая модель ответила быстрее всего")
    check("суммарная скорость не растёт" in " ".join(tools.split()).lower(),
          "CLI: повторяет честное предупреждение о скорости")


def test_model_choice_decision() -> None:
    """Решение «одна основная модель, окно для GPU сохранено» не разъезжается.

    Принято по замерам 04.10: на бесплатном Colab 2 ядра, поэтому параллельные
    модели не дают прироста суммарной скорости (ядра те же), 3B даёт рабочий
    баланс (2.5 ток/с), 7B — только фон (1.3 ток/с), а 14B не работает вовсе
    (0.05 ток/с из-за подкачки страниц). Решение: одна основная 3B, 7B рядом
    для проверки качества, окно больших моделей не закрыть.

    Документация проверяется тройкой: README, быстрый старт и код профилей
    обязаны говорить одно и то же, иначе человек примет решение по устаревшей
    таблице (именно так было: strong числился рекомендованным, пока замер не
    показал 0.05 ток/с).
    """
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    start = (ROOT / "docs" / "БЫСТРЫЙ_СТАРТ.md").read_text(encoding="utf-8")
    setup = (ROOT / "thinking" / "colab" / "cell_a_setup.py").read_text(encoding="utf-8")

    for name, text in (("README", readme), ("БЫСТРЫЙ_СТАРТ", start)):
        check("dev" in text and "3B + 1.5B" in text,
              f"{name}: профиль dev рекомендован на CPU")
        check("0.05 ток/с" in text,
              f"{name}: названа замеренная скорость 14B, а не обещание")
        check("включается сменой одной строки" in text or "без других" in text
              or "сменой `PROFILE`" in text,
              f"{name}: сказано, как открыть окно больших моделей на GPU")
        check("THINKING_PARALLEL" in text,
              f"{name}: описано, что параллельность — по включению, не по умолчанию")

    # strong больше не рекомендуется — замер этого не подтвердил
    check("не для бесплатного Colab" in readme and "не для бесплатного Colab" in start,
          "документация: strong помечен как нерабочий на бесплатном Colab")
    check('"dev"' in setup, "ячейка A: профиль dev существует")
    # ядра не растут — аргумент, на котором стоит решение
    check("суммарн" in start and "не растёт" in start,
          "БЫСТРЫЙ_СТАРТ: объяснено, почему параллельные модели не ускоряют")


def _load_mock_module():
    """Заглушка LLM — отдельный файл, грузится по пути, как CLI."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "mock_llm_under_test", ROOT / "tools" / "mock_llm.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_mock_llm_stub() -> None:
    """Локальная заглушка: все маршруты сервера живут без Colab.

    Это и есть проверка обещания «панель, клиент и CLI гоняются за секунды»:
    поднимаем tools/mock_llm.py на эфемерном порту и прогоняем по нему
    настоящий клиент — от /health до /ask/multi. Ничего не качается, ни
    туннеля, ни GPU; работает офлайн в любой системе.

    Отдельно проверяются режимы-аварии (503, обрыв потока, кривой JSON,
    долгая генерация): они нужны, чтобы путь отката и пинги SSE проверялись
    воспроизводимо, а не по настроению бесплатного туннеля.
    """
    mock = _load_mock_module()
    running: list = []

    def serve(**kw) -> str:
        srv = mock.create_server(port=0, **kw)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        running.append(srv)
        return f"http://127.0.0.1:{srv.server_address[1]}"

    # ретраи не должны тратить секунды на ожидание в упавшем режиме
    fast = {"retry": {"max_attempts": 1, "backoff_base": 0.001}}
    try:
        url = serve()
        c = make_client()
        c.base = url
        check(c.health(), "заглушка: /health отвечает")
        check(c.health_data.get("model") == "mock-llm",
              "заглушка: модель названа честно, а не выдана за Colab")

        # --- /plan -------------------------------------------------------
        plan = c.plan("проверить заглушку", max_steps=4)
        p = Plan.from_dict(plan)
        check(p.goal == "проверить заглушку", "заглушка: цель взята из запроса")
        check(len(p.steps) == 4, "заглушка: шагов ровно столько, сколько просили")
        ids = {s.id for s in p.steps}
        check(all(set(s.depends_on) <= ids for s in p.steps),
              "заглушка: depends_on не ссылается на несуществующие шаги")
        check(c.stats["plans"] == 1 and bool(c.reports),
              "заглушка: план попал в учёт и в отчёты")

        # --- /plan/stream ------------------------------------------------
        streamed = Plan.from_dict(c.plan_stream("потоковая задача", max_steps=3))
        check(streamed.goal == "потоковая задача",
              "заглушка: /plan/stream отдаёт валидный план")
        ev = c.events(tail=50)
        check(any(e.get("type") == "plan_step" for e in ev["events"]),
              "заглушка: шаги уходят в журнал событий")
        check(ev["last_seq"] > 0, "заглушка: у событий есть порядковый номер")

        # --- /chat + память ---------------------------------------------
        c.remember("предпочитает короткие ответы")
        rep = ChatReply.from_dict(c.chat("привет"))
        check("привет" in rep.reply, "заглушка: ответ содержит вопрос")
        check(rep.tokens_out > 0 and rep.tokens_estimate,
              "заглушка: расход токенов оценён, а не выдан за точный")

        got: list[str] = []
        srep = ChatReply.from_dict(c.chat_stream("что помнишь?",
                                                 on_token=got.append))
        check(bool(got) and "".join(got) == srep.reply,
              "заглушка: /chat/stream отдаёт токены, и итог — это их сумма")
        check(srep.memory_used and "короткие ответы" in srep.reply,
              "заглушка: память дошла до сервера и вернулась в ответ")

        # --- /reflect ----------------------------------------------------
        rf = ReflectResponse.from_dict(c.reflect(plan["plan_id"], 1, "сделано"))
        check(rf.status == "ok", "заглушка: успешный шаг даёт status=ok")
        rf2 = ReflectResponse.from_dict(
            c.reflect(plan["plan_id"], 1, "упало", error="TimeoutError: 500"))
        check(rf2.status == "adjust", "заглушка: ошибка шага даёт status=adjust")

        # --- /dev --------------------------------------------------------
        dv = c.dev("добавь функцию", active_name="a.py", active_code="x = 1")
        check(dv["action"] in ("none", "create", "edit")
              and dv["filename"] == "a.py",
              "заглушка: /dev отвечает предложением по активному файлу")

        # --- /ask/multi, /models, /parallel ------------------------------
        ben_before = c.benefits().get("calls_chat", 0)
        tok_before = c.tokens().get("tokens_out", 0)
        chats_before = c.stats["chats"]
        inter_before = len([i for i in c.interactions if i.get("kind") == "chat"])
        mm = c.multi_chat("сравни ответы")
        check(mm["count"] == 2 and all(a.get("ok") for a in mm["answers"]),
              "заглушка: /ask/multi отдаёт ответы нескольких «моделей»")
        # учёт параллельного вопроса: он жжёт токены N моделей (аудит B-3)
        check(c.stats["chats"] == chats_before + 1,
              "заглушка: ask-multi увеличил счётчик обращений (B-3)")
        check(len([i for i in c.interactions if i.get("kind") == "chat"])
              > inter_before,
              "заглушка: ask-multi записан в ленту «Связь агентов» (B-3)")
        check(c.benefits().get("calls_chat", 0) > ben_before,
              "заглушка: ask-multi виден в «Выгоде» (B-3)")
        check(c.tokens().get("tokens_out", 0) > tok_before,
              "заглушка: токены ответов всех моделей попали в «Токены» (B-3)")
        check(c.models().get("active_label") == "MOCK",
              "заглушка: /models отвечает в ожидаемой форме")
        check(c.parallel_models().get("count") == 1,
              "заглушка: /parallel отвечает в ожидаемой форме")

        # --- OpenAI-совместимый upstream ---------------------------------
        v1 = c._json("GET", "/v1/models", None, timeout=10)
        check(v1["data"][0]["id"] == "mock-llm",
              "заглушка: /v1/models совместим с OpenAI")
        out = c._json("POST", "/v1/chat/completions",
                      {"model": "mock",
                       "messages": [{"role": "user", "content": "привет"}]},
                      timeout=10)
        check("привет" in out["choices"][0]["message"]["content"],
              "заглушка: /v1/chat/completions отвечает как настоящая LLM")
        with c._open("POST", "/v1/chat/completions",
                     {"model": "mock", "stream": True,
                      "messages": [{"role": "user", "content": "привет"}]},
                     stream=True, timeout=10) as resp:
            raw = resp.read().decode("utf-8", "replace")
        check("data: [DONE]" in raw and '"delta"' in raw,
              "заглушка: поток /v1/chat/completions заканчивается [DONE]")

        # --- метка режима в тексте действует на один запрос ---------------
        check(raises(lambda: c.plan("[mock:error] задача"), ThinkingError),
              "заглушка: метка [mock:error] ломает только свой запрос")
        check(bool(c.plan("обычная задача")),
              "заглушка: после метки следующий запрос снова проходит")

        # --- режим error --------------------------------------------------
        ce = make_client(**fast)
        ce.base = serve(mode="error")
        check(not ce.health(), "режим error: /health не проходит — сервер «упал»")

        # --- режим drop: поток рвётся, короткий маршрут спасает -----------
        cd = make_client(**fast, chat_path=str(TMP / "drop_chat.jsonl"))
        cd.base = serve(mode="drop")
        d = cd.chat_stream("обрыв", author="human")
        check(d.get("stream_fallback") is True,
              "режим drop: обрыв SSE уводит на /chat, ответ получен")
        drop_rows = [json.loads(l) for l in
                     open(TMP / "drop_chat.jsonl", encoding="utf-8") if l.strip()]
        check(bool(drop_rows) and drop_rows[-1].get("author") == "human",
              "режим drop: фолбэк не теряет автора — реплика человека не "
              "пишется как агентская (B-2)")

        # --- режим badjson: контракт ловит мусор --------------------------
        cb = make_client(**fast)
        cb.base = serve(mode="badjson")
        plan4, used_fb = cb.plan_with_fallback("сломай JSON")
        check(used_fb and bool(plan4.get("steps")),
              "режим badjson: мусорный ответ уводит в локальный фолбэк")

        # --- режим slow: пинги держат поток живым -------------------------
        cs = make_client(**fast, stream_stall=1.0)
        cs.base = serve(mode="slow", first_delay=1.5, ping=0.2)
        t0 = time.time()
        slow = cs.chat_stream("долгая генерация")
        took = time.time() - t0
        pings = running[-1].mock_state.counters.get("pings", 0)
        check(bool(slow.get("reply")) and took >= 1.0,
              "режим slow: клиент дождался первого токена")
        check(pings > 0,
              "режим slow: во время ожидания шли пинги SSE (иначе "
              "stream_stall оборвал бы чтение)")
        check(c.stats["plans"] >= 2,
              "заглушка: обычные запросы не пострадали от режимов")
    finally:
        for srv in running:
            srv.shutdown()
            srv.server_close()


def test_stub_switching() -> None:
    """Возврат с заглушки на настоящий Colab — одна команда и ничего не потеряно.

    Память, отчёты и таймлайн лежат на ПК: при переключении меняется только
    то, кто печатает ответ. Поэтому set-url запоминает пару «адрес + токен»,
    а --back просто меняет её местами.
    """
    path = client_mod.LOCAL_PATH
    saved = path.read_bytes() if path.exists() else None
    try:
        path.unlink(missing_ok=True)
        ThinkingClient.set_url("http://stub.local:8010", "stub-token")
        ThinkingClient.set_url("http://colab.local", "colab-token")
        data = json.loads(path.read_text(encoding="utf-8"))
        check(data["base_url"] == "http://colab.local",
              "set-url: новый адрес записан")
        check(data.get("prev", {}).get("base_url") == "http://stub.local:8010"
              and data.get("prev", {}).get("token") == "stub-token",
              "set-url: прежняя пара сохранена для --back целиком")

        ThinkingClient.set_url("", "", back=True)
        data = json.loads(path.read_text(encoding="utf-8"))
        check(data["base_url"] == "http://stub.local:8010"
              and data["token"] == "stub-token",
              "set-url --back: вернулись на прежний адрес и токен")
        check(data.get("prev", {}).get("base_url") == "http://colab.local",
              "set-url --back: и это направление тоже запомнено")

        ThinkingClient.set_url("", "", back=True)
        data = json.loads(path.read_text(encoding="utf-8"))
        check(data["base_url"] == "http://colab.local",
              "set-url --back: повторное переключение работает")

        path.write_text(json.dumps({"base_url": "http://x"}), encoding="utf-8")
        check(raises(lambda: ThinkingClient.set_url("", "", back=True),
                     ThinkingError),
              "set-url --back: без запомненного адреса — честная ошибка")

        # Панель живёт часами: подхватывает новый адрес без рестарта
        path.write_text(json.dumps({"base_url": "http://one", "token": "t1"}),
                        encoding="utf-8")
        c = ThinkingClient({})
        check(c.base == "http://one", "конфиг: адрес прочитан из файла")
        path.write_text(json.dumps({"base_url": "http://two", "token": "t2"}),
                        encoding="utf-8")
        check(c.reload_config() and c.base == "http://two",
              "панель: новый адрес подхвачен без перезапуска")
        c.base_locked = True
        path.write_text(json.dumps({"base_url": "http://three"}),
                        encoding="utf-8")
        check(not c.reload_config() and c.base == "http://two",
              "панель: разовый --url не перетирается файлом конфига")

        # --url задаёт адрес на один запуск и не трогает настройки
        cli = _load_cli_module()
        before = path.read_bytes()
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            code = cli.main(["url", "--url", "http://127.0.0.1:8010"])
        check(code == 0, "CLI --url: адрес взят на этот запуск")
        check(path.read_bytes() == before,
              "CLI --url: config/thinking.local.json не изменён")
    finally:
        if saved is None:
            path.unlink(missing_ok=True)
        else:
            path.write_bytes(saved)


def test_instructions_present() -> None:
    """Инструкция «как поднять» есть в ноутбуке, README и быстром старте.

    Её легко потерять при правке документации, а она нужна каждый раз,
    когда сессия Colab обрывается.
    """
    nb = (ROOT / "colab" / "thinking_agent_ver3.ipynb").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    quick = (ROOT / "docs" / "БЫСТРЫЙ_СТАРТ.md").read_text(encoding="utf-8")
    for name, text in (("ноутбук", nb), ("README", readme),
                       ("быстрый старт", quick)):
        check("colab.research.google.com" in text,
              f"{name}: есть ссылка на Colab")
        check("Run all" in text, f"{name}: есть шаг Run all")
        check("PROFILE" in text, f"{name}: есть напоминание про PROFILE в ячейке A")
        check("АДРЕС ТУННЕЛЯ" in text.upper(),
              f"{name}: есть шаг с адресом туннеля (ячейка 7/7)")
        check("своём" in text or "своем" in text,
              f"{name}: сказано открыть в своём браузере")
    # причина, почему не в браузере агента, объяснена
    for name, text in (("ноутбук", nb), ("README", readme),
                       ("быстрый старт", quick)):
        check("JavaScript" in text,
              f"{name}: объяснено, почему вход в Google в браузере агента не работает")
    check("thinking_agent_ver3.ipynb" in nb,
          "ноутбук: инструкция называет актуальный файл ver3")


def test_dev_layout_vertical() -> None:
    """«Разработка» идёт сверху вниз: чат → код → вывод → журнал → дамп."""
    html = (ROOT / "tools" / "thinking_panel.html").read_text(encoding="utf-8")
    check(re.search(r"\.devgrid\s*\{[^}]*flex-direction:\s*column", html) is not None
          and not re.search(r"\.devgrid\s*\{[^}]*grid-template-columns", html),
          "разработка: раскладка вертикальная, а не в три колонки")
    # порядок блоков внутри .devgrid
    start = html.find('<div class="devgrid">')
    end = html.find('<div class="note">', start)
    block = html[start:end] if start > 0 and end > start else html[start:start + 6000]
    order = ["devcol-chat", "devcol-code", "devcol-out", "devcol-log", "devbar-dump"]
    pos = [block.find(k) for k in order]
    check(all(p >= 0 for p in pos) and pos == sorted(pos),
          "разработка: порядок сверху вниз — чат, код, вывод, журнал, дамп")
    # окна, которые просил расширить, должны быть выше среднего
    check(".devcol-chat { height: 400px" in html,
          "разработка: окно чата с агентом увеличено")
    check(".devcol-out  { height: 380px" in html,
          "разработка: вывод интерпретатора увеличен")
    check("#devinput { flex:1; min-height:92px" in html,
          "разработка: поле ввода агенту выше")
    check('id="devstdin"' in block,
          "разработка: поле ввода программы осталось в блоке вывода")


def test_dev_run_stdin() -> None:
    """Запуск из «Разработки»: input() получает введённое, а не висит 30 с."""
    mod = _load_cli_module()
    client = make_client(dev_path=str(TMP / "devstdin"), devlog_path=str(TMP / "dl.jsonl"))
    prog = ('a = float(input("a: "))\n'
            'b = float(input("b: "))\n'
            'print("answer:", a + b)\n')
    out = mod._dev_write(client, "calc.py", prog)
    check(out.get("ok") is True, "разработка: файл записан для проверки запуска")

    # 1. с вводом — программа получает данные и считает
    r = mod._dev_run(client, "calc.py", "5\n7\n")
    check(r["code"] == 0 and "12.0" in (r.get("stdout") or ""),
          "разработка: «Запустить» с вводом считает (input() получает данные)")

    # 2. без ввода — падает СРАЗУ с подсказкой, а не висит до таймаута
    t0 = time.time()
    r2 = mod._dev_run(client, "calc.py")
    dt = time.time() - t0
    check(dt < 5, f"разработка: без ввода падает сразу, а не ждёт 30 с ({dt:.1f} с)")
    check("EOFError" in (r2.get("stderr") or ""),
          "разработка: без ввода видна причина (EOFError), а не молчание")
    check("ввод для программы" in (r2.get("stderr") or ""),
          "разработка: подсказка про поле ввода есть в выводе")

    # 3. программа без ввода работает как раньше
    out2 = mod._dev_write(client, "plain.py", "print('готово: 2 + 3 =', 2 + 3)\n")
    check(out2.get("ok") is True, "разработка: файл без ввода записан")
    r3 = mod._dev_run(client, "plain.py")
    check(r3["code"] == 0 and "готово: 2 + 3 = 5" in (r3.get("stdout") or ""),
          "разработка: обычный запуск не сломался")


def test_chat_autoscroll() -> None:
    """Открытие вкладки чата прокручивает ленту к последним сообщениям."""
    html = (ROOT / "tools" / "thinking_panel.html").read_text(encoding="utf-8")
    check("const BOTTOM_PANES = {chat:" in html,
          "панель: вкладка чата в списке лент с автопрокруткой")
    check("if (target) scrollBottom(target);" in html,
          "панель: при открытии вкладки лента прокручивается вниз")
    # scrollBottom вызывается после показа панели, а не до
    i_show = html.find('classList.add("on");\n  if (b.dataset.pane === "models")')
    i_scroll = html.find("if (target) scrollBottom(target);")
    check(i_show > 0 and i_scroll > i_show,
          "панель: прокрутка после показа панели (иначе scrollHeight = 0)")


def test_period_default() -> None:
    """Активная кнопка периода совпадает с PERIOD по умолчанию (без врали)."""
    html = (ROOT / "tools" / "thinking_panel.html").read_text(encoding="utf-8")
    m = re.search(r'let PERIOD = "(\w+)"', html)
    check(bool(m), "панель: период задан константой")
    period = m.group(1) if m else ""
    # активная кнопка в разметке должна совпадать с этим значением
    active = re.search(r'<button class="mini per on" data-per="(\w+)"', html)
    check(bool(active) and active.group(1) == period,
          "панель: подсвеченная кнопка периода совпадает с PERIOD по умолчанию")


def test_memory_sig() -> None:
    """Рендер памяти под сигнатурой state (AUD-15)."""
    html = (ROOT / "tools" / "thinking_panel.html").read_text(encoding="utf-8")
    check("s.chat, s.tokens, s.memory" in html,
          "панель: память входит в сигнатуру рендера (AUD-15)")
    m = re.search(r"if \(sig !== lastSig\) \{(.*?)\n    \}", html, re.S)
    check(m is not None and "renderMemory(s.memory)" in m.group(1),
          "панель: renderMemory только при изменении сигнатуры (AUD-15)")


def test_config_timeouts() -> None:
    """plan_timeout в конфиге совпадает с кодом (AUD-25)."""
    cfg = json.loads((ROOT / "config" / "thinking.json").read_text(encoding="utf-8"))
    check(cfg.get("plan_timeout") == 360,
          "конфиг: plan_timeout совпадает с кодом (360, AUD-25)")


def test_nctx_alignment() -> None:
    """n_ctx в _llm_cmd совпадает с ячейкой D (AUD-08)."""
    src = (ROOT / "thinking" / "colab" / "cell_c_server.py").read_text(encoding="utf-8")
    check('_env_export("THINKING_CTX"' in src,
          "ячейка C: _llm_cmd читает THINKING_CTX из env-файла (AUD-08)")
    check('"8192" if gpu else "4096"' in src,
          "ячейка C: фолбэк n_ctx совпадает с ячейкой D (AUD-08)")
    d_src = (ROOT / "thinking" / "colab" / "cell_d_launch.py").read_text(encoding="utf-8")
    check("export THINKING_CTX" in d_src,
          "ячейка D: пишет THINKING_CTX в env-файл (AUD-08)")


def test_sse_queue_limit() -> None:
    """Очереди SSE-подписчиков ограничены (AUD-11)."""
    src = (ROOT / "thinking" / "colab" / "cell_c_server.py").read_text(encoding="utf-8")
    check("asyncio.Queue(maxsize=200)" in src,
          "сервер: очередь SSE ограничена 200 событиями (AUD-11)")
    check("asyncio.QueueFull" in src,
          "сервер: переполненная очередь отключает подписчика (AUD-11)")


def test_smoke_guarded() -> None:
    """Смоук-тест ячейки D обёрнут в try/except (AUD-24)."""
    src = (ROOT / "thinking" / "colab" / "cell_d_launch.py").read_text(encoding="utf-8")
    check("except Exception as _smoke_exc" in src,
          "ячейка D: смоук не роняет ячейку (AUD-24)")


def test_tail_throttle() -> None:
    """Хвост печатает о разрыве при изменении состояния, не каждые 0,5 с (AUD-23)."""
    src = (ROOT / "tools" / "thinking_cli.py").read_text(encoding="utf-8")
    check("now - last_warn >= 15" in src,
          "хвост: троттлинг предупреждений 15 с (AUD-23)")
    check("[связь восстановлена]" in src,
          "хвост: сообщение о восстановлении связи (AUD-23)")


def test_cache_tail_read() -> None:
    """_cache_get читает хвост журнала, а не весь файл (AUD-25)."""
    mod = _load_cli_module()
    c = make_client()
    for i in range(100):
        c._remember_chat(f"вопрос {i}",
                         ChatReply.from_dict({"reply": f"ответ {i}"}))
    check(mod._cache_get(c, "chat", "несуществующий вопрос") is None,
          "кэш: промах по несуществующему вопросу")
    hit = mod._cache_get(c, "chat", "вопрос 99")
    check(hit is not None and hit.get("reply") == "ответ 99",
          "кэш: находит последний ответ из хвоста журнала")


def test_events_ping() -> None:
    """Локальный /events шлёт ping раз в 5 с (AUD-25)."""
    src = (ROOT / "tools" / "thinking_cli.py").read_text(encoding="utf-8")
    check("now - last_ping >= 5" in src,
          "панель: ping раз в 5 с, а не каждый тик (AUD-25)")


def test_colab_cells() -> None:
    cells = sorted((ROOT / "thinking" / "colab").glob("cell_*.py"))
    check(len(cells) >= 5, "Colab: ячейки A–F на месте")
    server = (ROOT / "thinking" / "colab" / "cell_c_server.py").read_text(encoding="utf-8")
    for route in ("/health", "/metrics", "/events", "/events/stream",
                  "/plan", "/plan/stream", "/reflect", "/cancel"):
        check(route in server, f"сервер Colab: маршрут {route} есть")
    check("%%writefile /content/thinking_server.py" in server,
          "сервер Colab: собирается через %%writefile")

    setup = (ROOT / "thinking" / "colab" / "cell_a_setup.py").read_text(encoding="utf-8")
    check("/content/thinking_model_path.txt" in setup, "ячейка A: пишет путь модели")
    check("wheel-cpu" in setup, "ячейка A: есть фоллбек на CPU-колесо")
    check("Qwen2.5" in setup, "ячейка A: качает Qwen2.5")
    check("nvidia-cuda-runtime-cu12" in setup,
          "ячейка A: доустанавливает CUDA-библиотеки с PyPI")
    check("LD_LIBRARY_PATH" in setup, "ячейка A: пробрасывает путь к библиотекам")
    check(setup.count("GGUF") >= 3, "ячейка A: минимум три модели-кандидата")
    check("thinking_models.txt" in setup, "ячейка A: пишет список моделей для подмены")
    check("уже на диске" in setup,
          "ячейка A: повторный запуск не качает модель заново")
    check("build-cuda" not in setup and "ollama" not in setup,
          "ячейка A: быстрый путь — без компиляции и без ollama")

    # модели: репозитории на месте, а теги — точные подстроки имён файлов,
    # которые не перехватывают чужой файл (проверено на РЕАЛЬНЫХ именах)
    for s in ("mradermacher/Qwen2.5-3B-Instruct-Uncensored-GGUF",
              "mradermacher/Qwen2.5-1.5B-Instruct-uncensored-GGUF",
              "Qwen/Qwen3-30B-A3B-GGUF",
              "bartowski/Qwen2.5-14B-Instruct-GGUF",
              "bartowski/Qwen2.5-Coder-14B-Instruct-GGUF",
              "bartowski/Qwen2.5-32B-Instruct-GGUF",
              "bartowski/Qwen2.5-Coder-32B-Instruct-GGUF",
              "3b-instruct-q4", "1.5b-instruct-q4", "7b-instruct-q4",
              "7b-instruct-uncensored", "3b-instruct-uncensored",
              "1.5b-instruct-uncensored",
              "30b-a3b-q4", "qwen2.5-14b-instruct", "coder-14b",
              "qwen2.5-32b-instruct", "coder-32b"):
        check(s in setup, f"ячейка A: {s} на месте")
    tags = {"1.5b-instruct-q4": "qwen2.5-1.5b-instruct-q4_k_m.gguf",
            "1.5b-instruct-uncensored":
                "qwen2.5-1.5b-instruct-uncensored.q4_k_m.gguf",
            "3b-instruct-q4": "qwen2.5-3b-instruct-q4_k_m.gguf",
            "3b-instruct-uncensored":
                "qwen2.5-3b-instruct-uncensored.q4_k_m.gguf",
            "7b-instruct-q4": "qwen2.5-7b-instruct-q4_k_m.gguf",
            "7b-instruct-uncensored":
                "qwen2.5-7b-instruct-uncensored.q4_k_m.gguf",
            "30b-a3b-q4": "qwen3-30b-a3b-q4_k_m.gguf",
            "qwen2.5-14b-instruct": "qwen2.5-14b-instruct-q4_k_m.gguf",
            "coder-14b": "qwen2.5-coder-14b-instruct-q4_k_m.gguf",
            "qwen2.5-32b-instruct": "qwen2.5-32b-instruct-q4_k_m.gguf",
            "coder-32b": "qwen2.5-coder-32b-instruct-q4_k_m.gguf"}
    check(all(t in n for t, n in tags.items()),
          "ячейка A: каждый тег находит свой файл")
    clash = [(t, n) for t, own in tags.items()
             for n in tags.values() if n != own and t in n]
    check(not clash, f"ячейка A: теги моделей не пересекаются ({clash})")

    # выбор моделей: качать не всё, а профиль
    check("THINKING_PROFILE" in setup, "ячейка A: профиль моделей задаётся переменной")
    for prof in ("light", "gpu", "strong", "big", "coder", "uncensored", "all"):
        check(f'"{prof}"' in setup, f"ячейка A: профиль {prof} доступен")
    check("WANTED_TAGS" in setup and "PROFILES" in setup,
          "ячейка A: профиль выбирает, что именно качать")

    # бюджеты задают длину ответа и не переполняют окно
    launch = (ROOT / "thinking" / "colab" / "cell_d_launch.py").read_text(encoding="utf-8")
    setup_c = (ROOT / "thinking" / "colab" / "cell_c_server.py").read_text(encoding="utf-8")
    check("MODEL_BUDGETS" in launch, "ячейка D: бюджеты по моделям одной таблицей")
    check("THINKING_DEV_MAX_TOKENS" in launch,
          "ячейка D: у режима разработки свой бюджет")
    check('("30b-a3b"' in launch and launch.index('("30b-a3b"')
          < launch.index('("3b"'),
          "ячейка D: признаки моделей проверяются от длинного к короткому")
    check("_ctx_limit" in setup_c, "ячейка C: бюджет прижимается к окну модели")
    check("THINKING_CTX" in setup_c, "ячейка C: окно берётся из thinking_env.sh")
    check(launch.count("_ctx_limit") == 0 and setup_c.count("_ctx_limit()") >= 4,
          "ячейка C: окно прижимает все точки вызова LLM")

    check("THINKING_URL=" in launch, "ячейка D: печатает THINKING_URL")
    check("THINKING_TOKEN=" in launch, "ячейка D: печатает THINKING_TOKEN")
    check("thinking_models.txt" in launch, "ячейка D: читает список моделей")
    check("пробуем следующую" in launch or "не поднялась" in launch,
          "ячейка D: переключает модель, если не поднялась")
    check("thinking_env.sh" in launch, "ячейка D: подхватывает LD_LIBRARY_PATH из A")
    check('"--n_ctx", CTX' in launch, "ячейка D: контекст подстраивается под GPU/CPU")

    bg = (ROOT / "thinking" / "colab" / "cell_e_background.py").read_text(encoding="utf-8")
    check("Thread" in bg, "ячейка E: снапшот в фоновом потоке")

    check((ROOT / "docs" / "TZ_thinking_subagent.md").stat().st_size > 10000,
          "документация ТЗ на месте")
    check((ROOT / "docs" / "schema_plan.json").exists(), "снимок схемы на месте")


def test_dev_metrics_limits() -> None:
    """Режим «Разработка», рефлексия по метрикам, период и лимиты Colab —
    всё, что появилось в переработанной панели и ноутбуке ver3."""
    # --- клиент: период «за сегодня» / «за всё время» ---
    c = make_client(reflections_path=str(TMP / "reflections.md"))
    check(c._after({}, ""), "период: пустой since не фильтрует («за всё время»)")
    check(c._after({"at": "2026-10-02T10:00:00+00:00"}, "2026-10-02T00:00:00Z"),
          "период: запись за сегодня проходит фильтр")
    check(c._after({"at": "2026-10-02T00:00:00.000Z"}, "2026-10-02T00:00:00+00:00"),
          "период: формат JS (.000Z) сравнивается с форматом записей")
    check(not c._after({"at": "2026-10-01T23:59:59+00:00"}, "2026-10-02T00:00:00Z"),
          "период: вчерашняя запись отсекается")
    since = c.today_since()
    check(len(since) >= 19 and since[4:5] == "-" and "T" in since,
          "период: today_since отдаёт ISO-дату")

    # --- разбивка «агент (CLI) против чата (человек)» ---
    ben = c.benefits()
    for key in ("calls_agent", "calls_chat", "calls_total"):
        check(key in ben, f"benefits: ключ {key} есть")
    check(isinstance(ben.get("calls_agent"), int) and isinstance(ben.get("calls_chat"), int),
          "benefits: разбивка обращений — числа")
    check(ben["calls_agent"] + ben["calls_chat"] <= ben["calls_total"],
          "benefits: сумма разбивки не больше итога")

    # --- токен: панель сохраняет его кнопкой «Сохранить» ---
    c.base = "https://example.invalid"
    path = c.set_token("секрет-токен")
    data = json.loads(path.read_text(encoding="utf-8"))
    check(data.get("token") == "секрет-токен", "set_token: токен пишется в локальный конфиг")
    check(c.token == "секрет-токен", "set_token: токен подставляется в живой клиент")
    c.token = ""

    # --- рефлексия по метрикам: движок local работает вообще без Colab ---
    out = c.reflect_metrics(engine="local", since="")
    check(isinstance(out, dict) and out.get("status") in ("ok", "adjust"),
          "reflect_metrics: локальные правила отдают статус")
    check(bool(str(out.get("advice", "")).strip()), "reflect_metrics: есть совет")
    reps = [json.loads(x) for x in (TMP / "reports.jsonl")
            .read_text(encoding="utf-8").splitlines() if x.strip()]
    check(any(r.get("type") == "reflect" and str(r.get("plan_id", "")).startswith("metrics-")
              for r in reps),
          "reflect_metrics: отчёт кладётся в журнал отчётов")
    refl = TMP / "reflections.md"
    check(refl.exists() and refl.stat().st_size > 0,
          "reflect_metrics: совет сохраняется в файл программы")

    # --- Colab: режим «Разработка» и подсказки режимов ---
    server = (ROOT / "thinking" / "colab" / "cell_c_server.py").read_text(encoding="utf-8")
    for marker in ("SYSTEM_DEV", "DevRequest", "build_dev_prompt", "dev_data",
                   '"/dev"', '"/dev/stream"', "_model_hint", '"hint"'):
        check(marker in server, f"сервер Colab: {marker} есть")
    check("без JSON" in server, "сервер Colab: чат просит обычный текст, а не JSON")
    setup = (ROOT / "thinking" / "colab" / "cell_a_setup.py").read_text(encoding="utf-8")
    check("Uncensored" in setup, "ячейка A: качается и модель без цензуры")
    check("bartowski/Qwen2.5-7B-Instruct-GGUF" in setup,
          "ячейка A: 7B Instruct в списке желаемых")
    check("_hint(p)" in setup, "ячейка A: для модели печатается подсказка CPU/T4")
    # каталог моделей — словарь, поэтому один тег не может повториться, а цикл
    # качания идёт по тегам профиля (раньше здесь был dict.fromkeys)
    check("CATALOG: dict[str, tuple[str, int, str, int]]" in setup,
          "ячейка A: каталог моделей — словарь, теги уникальны по построению")
    check("for tag in list(WANTED_TAGS) + [t for t in ALSO.get(PROFILE, [])" in setup
          and "if t not in WANTED_TAGS]:" in setup,
          "ячейка A: качаем теги профиля плюс ALSO, без дублей")
    # Запасная модель профиля обязана быть заметно меньше основной.
    # Иначе откат бесполезен: если основная не влезла в RAM по памяти,
    # модель того же размера тоже не влезет, и мы просто потратим на неё
    # время и диск (именно так был устроен профиль strong: основная 14B и
    # запасная Coder-14B — обе 8.4 ГБ).
    import re as _re
    # порог размера каждого тега прямо из CATALOG (в байтах, как в ячейке A)
    _CATALOG_SIZE = {t: int(sz.replace("_", "")) for t, sz in
                     _re.findall(r'"([a-z0-9_.-]+)":\s*\(\s*"[^"]+"\s*,\s*([\d_]+)\s*,', setup)}
    _prof_block = setup.split("PROFILES")[1].split("\nPROFILE =")[0]
    for _prof, _body in _re.findall(r'"([a-z0-9_.-]+)":\s*\[([^\]]*)\]', _prof_block):
        _tags = _re.findall(r'"([a-z0-9_.-]+)"', _body)
        if len(_tags) < 2:
            continue
        _sizes = [_CATALOG_SIZE.get(t) for t in _tags]
        check(all(_sizes) and _sizes[-1] < _sizes[0],
              f"профиль {_prof}: запасная {_tags[-1]} меньше основной {_tags[0]} "
              f"(иначе откат не спасёт при нехватке памяти)")

    check('"coder-14b":' in setup and '"coder-32b":' in setup,
          "ячейка A: кодерские модели помечены отдельно от обычных")

    # --- клиент: dev + токен + рефлексия + периоды ---
    src = (ROOT / "thinking" / "client.py").read_text(encoding="utf-8")
    for marker in ("def dev(", "/dev/stream", "def set_token", "def reflect_metrics",
                   "def _after", "def today_since", "calls_agent", "calls_chat"):
        check(marker in src, f"клиент: {marker} есть")

    # --- панель: dev-вкладка, период, токен, рефлексия, лимиты ---
    html = (ROOT / "tools" / "thinking_panel.html").read_text(encoding="utf-8")
    for marker in ('data-pane="dev"', "sinceIso", "setPeriod", "runReflect",
                   "showLimitWarn", "limitwarn", "tokhelp", "dttm(",
                   "/api/token", "/api/reflect-metrics", "/api/dev/file",
                   "/api/dev/rollback", "/api/dev/run", "за сегодня",
                   "за всё время", "calls_agent", "calls_chat", "Связь агентов"):
        check(marker in html, f"панель: есть {marker}")
    check("2–4" in html and "сутки" in html,
          "панель: лимиты Colab объяснены без пугающих цифр")

    # --- CLI: маршруты токена/рефлексии/разработки и предупреждения ---
    cli = (ROOT / "tools" / "thinking_cli.py").read_text(encoding="utf-8")
    for marker in ("/api/token", "/api/reflect-metrics", "/api/dev/chat",
                   "/api/dev/file", "/api/dev/run", "/api/dev/rollback",
                   "/api/dev/files", "/api/dev/log", "ЛИМИТЫ",
                   '"reflect-metrics"'):
        check(marker in cli, f"CLI: {marker} есть")
    check("Kaggle" in cli, "CLI: doctor упоминает запасные бесплатные платформы")

    # --- dev-функции живьём (регресс: запуск искал файл в dev_sandbox/dev_sandbox/) ---
    droot = TMP / "dev_sandbox"
    cd = make_client(dev_path=str(droot), devlog_path=str(TMP / "devlog.jsonl"))
    cli_mod = _load_cli_module()
    cli_mod._dev_write(cd, "t.py", "print('ok-dev')\n", source="user")
    res = cli_mod._dev_run(cd, "t.py")
    check(res.get("code") == 0 and "ok-dev" in str(res.get("stdout", "")),
          "dev: .py запускается из песочницы (путь не удваивается)")
    check(cli_mod._safe_name("../evil.py") == "evil.py",
          "dev: попытка уйти из песочницы отсекается")
    cli_mod._dev_write(cd, "t.py", "print('v2')\n", source="model")
    rb = cli_mod._dev_rollback(cd, "t.py")
    check(rb.get("ok") is True and rb.get("left") == 0,
          "dev: откат возвращает прошлую версию и считает остаток")
    check((droot / "t.py").read_text(encoding="utf-8") == "print('ok-dev')\n",
          "dev: после отката содержимое прежнее")
    check(raises(lambda: cli_mod._dev_run(cd, "нет.py"), ValueError),
          "dev: запуск несуществующего файла — понятная ошибка")
    dlog = [json.loads(x) for x in (TMP / "devlog.jsonl")
            .read_text(encoding="utf-8").splitlines() if x.strip()]
    check(any(r.get("action") == "run" for r in dlog)
          and any(r.get("action") == "rollback" for r in dlog),
          "dev: журнал фиксирует запуск и откат")

    # --- .bat: UTF-8 без BOM + chcp 65001 + CRLF ---
    for bat in ("ЗАПУСТИТЬ_МЫШЛЕНИЕ.bat", "ОСТАНОВИТЬ_МЫШЛЕНИЕ.bat"):
        blob = (ROOT / bat).read_bytes()
        check(not blob.startswith(b"\xef\xbb\xbf"), f"{bat}: без BOM")
        check(b"\r\n" in blob, f"{bat}: переводы строк CRLF")
        try:
            text = blob.decode("utf-8")
            check("chcp 65001" in text, f"{bat}: chcp 65001")
        except UnicodeDecodeError:
            check(False, f"{bat}: файл не в UTF-8")

    # --- ноутбук ver3 собран из актуальных ячеек ---
    nb_path = ROOT / "colab" / "thinking_agent_ver3.ipynb"
    check(nb_path.exists(), "ноутбук colab/thinking_agent_ver3.ipynb на месте")
    if nb_path.exists():
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        check(nb.get("metadata", {}).get("colab", {}).get("name")
              == "thinking_agent_ver3.ipynb",
              "ноутбук ver3: metadata.colab.name совпадает с файлом")
        joined = "".join("".join(cell.get("source", [])) for cell in nb["cells"])
        check("(ver3)" in joined, "ноутбук ver3: подпись версии в заголовке")
        check("/dev/stream" in joined and "Uncensored" in joined,
              "ноутбук ver3: внутри dev-режим и все модели")
    build = (ROOT / "scripts" / "build_colab.py").read_text(encoding="utf-8")
    check('"--name"' in build and '"--ver"' in build,
          "build_colab: есть опции --name и --ver")


# --------------------------------------------------------------------------- #
#  Быстрые улучшения: кэш ответов, предохранитель, саммари, фиксы панели
# --------------------------------------------------------------------------- #
def test_quickwins() -> None:
    """Кэш ответов, предохранитель транспорта, саммари диалога, правки панели."""
    import time

    cli_src = (ROOT / "tools" / "thinking_cli.py").read_text(encoding="utf-8")
    panel = (ROOT / "tools" / "thinking_panel.html").read_text(encoding="utf-8")
    client_src = (ROOT / "thinking" / "client.py").read_text(encoding="utf-8")

    # --- кэш ответов ---
    check("_ANSWER_CACHE" in cli_src, "кэш: таблица ответов в панели")
    check("answer_cache_ttl" in cli_src, "кэш: TTL настраивается")
    mod = _load_cli_module()
    cache_client = make_client(chat_path=str(TMP / "chat_qw.json"))
    mod._cache_put(cache_client, "chat:1", "  Кэш-Тест   вопрос ",
                   {"reply": "да"})
    hit = mod._cache_get(cache_client, "chat:1", "кэш-тест вопрос")
    check(bool(hit) and hit.get("cached") is True and hit.get("reply") == "да",
          "кэш: нормализация ключа и флаг cached")
    check(mod._cache_get(cache_client, "chat:1", "другой вопрос") is None,
          "кэш: чужой вопрос не подхватывается")

    # --- предохранитель ---
    check("breaker_pause" in client_src and "_cb_open_until" in client_src,
          "предохранитель: состояние и пауза в клиенте")
    br = make_client()
    br.base = "http://127.0.0.1:9"        # сеть не трогаем: пауза раньше запроса
    br._cb_errors = 9
    br._cb_open_until = time.time() + 60
    try:
        br._open("GET", "/health")
        check(False, "предохранитель: запрос на паузе должен быть заблокирован")
    except ThinkingError as exc:
        check("предохранитель" in str(exc),
              "предохранитель: на паузе запрос не уходит в сеть")
    except Exception as exc:                             # noqa: BLE001
        check(False, f"предохранитель: не тот отказ — {exc}")
    br._cb_open_until = 0.0

    # --- саммари диалога ---
    check("Саммари диалога" in client_src, "память: саммари старых реплик")
    mem_client = make_client(memory_path=str(TMP / "mem_qw.json"))
    for i in range(8):
        mem_client.remember_turn(f"вопрос номер {i}", f"ответ номер {i}")
    mem = mem_client.memory()
    sums = [f for f in mem["facts"] if str(f).startswith("Саммари диалога: ")]
    check(len(sums) == 1 and len(sums[0]) <= 300,
          "память: из выпавших реплик одно саммари ≤300 знаков")
    check(len(mem["turns"]) <= 12, "память: окно реплик не растёт")
    check(any("вопрос номер 7" in str(t.get("text")) for t in mem["turns"]),
          "память: свежие реплики остаются в окне")

    # --- панель ---
    check("ждание" not in panel, "панель: ждание заменено на ожидание")
    check("ожидание " in panel, "панель: подпись ожидания на месте")
    check('"из кэша"' in panel, "панель: метка ответа из кэша в истории")
    check("enterSends" in panel and "Shift+Enter" in panel,
          "панель: Enter отправляет, Shift+Enter — новая строка")
    check("@media print" in panel and "window.print()" in panel,
          "панель: печать/PDF через window.print()")
    check("if (chatBusy) return" in panel,
          "панель: чат в полёте не перерисовывается поверх")

    # --- прогрев ---
    check("[прогрев]" in cli_src, "панель: прогрев /health раз в ~25 минут")
    check("_cache_journal" in cli_src, "кэш: ответ из кэша пишется в журнал диалога")
    check('cached": bool(cached)' in client_src or
          '"cached": bool(cached)' in client_src,
          "клиент: журнал диалога помечает ответ из кэша")

    # --- запасной транспорт: не-потоковый fallback ---
    fb = make_client()

    def _stall(*_a, **_k):
        raise ThinkingError("поток чата оборвался после 3 попыток: timeout")

    fb._stream_read = _stall
    fb.chat = lambda message, **_k: {"reply": "ответ через /chat"}
    res = fb.chat_stream("вопрос на fallback")
    check(res.get("reply") == "ответ через /chat" and
          res.get("stream_fallback") is True,
          "запасной транспорт: chat_stream падает на не-потоковый /chat")

    fb2 = make_client()
    fb2._stream_read = lambda *_a, **_k: (_ for _ in ()).throw(
        ThinkingError("сервер: LLM отказал"))
    called: list = []
    fb2.chat = lambda *a, **k: called.append(a)
    try:
        fb2.chat_stream("вопрос на серверную ошибку")
        check(False, "запасной транспорт: серверная ошибка должна всплыть")
    except ThinkingError:
        check(not called, "запасной транспорт: серверная ошибка не дублируется /chat")

    dv = make_client(devlog_path=str(TMP / "devlog_qw.jsonl"))
    dv._stream_read = lambda *_a, **_k: (_ for _ in ()).throw(
        ThinkingError("поток разработки оборвался после 3 попыток: timeout"))
    dv._json = lambda *_a, **_k: {"proposal": {"action": "edit",
                                                "filename": "a.py",
                                                "code": "x = 1", "comment": "ок"}}
    prop = dv.dev("измени a.py")
    check(prop.get("action") == "edit",
          "запасной транспорт: dev падает на обычный POST /dev")
    check("stream_fallback" in client_src and "без потока" in panel,
          "запасной транспорт: панель помечает ответ без потока")

    # --- дедупликация фактов ---
    ded = make_client(memory_path=str(TMP / "mem_dedup.json"))
    ded.remember(fact="Тестовый факт о проекте")
    ded.remember(fact="  тестовый факт о проекте. ")
    check(len(ded.memory()["facts"]) == 1, "память: точный дубль не добавляется")
    ded.remember(fact="Тестовый факт о проекте и ещё детали про таймауты")
    got = ded.memory()["facts"]
    check(len(got) == 1 and "ещё детали" in got[0],
          "память: пересекающийся факт заменён более полным")

    # --- экспорт/импорт памяти ---
    check('act == "export"' in cli_src and "_write_memory(clean)" in cli_src,
          "память: экспорт и импорт через /api/memory (с лимитами)")
    check("memory_action(client, data)" in cli_src,
          "память: /api/memory зовёт memory_action — одна логика на двоих (B-6)")
    check("exportMemory" in panel and "importMemory" in panel,
          "память: кнопки экспорта/импорта в панели")

    # --- CI и CHANGELOG ---
    workflows = list((ROOT / ".github" / "workflows").glob("*.yml"))
    wf_text = "".join(p.read_text(encoding="utf-8") for p in workflows)
    check(bool(workflows) and "thinking_test.py" in wf_text,
          "CI: GitHub Actions гоняет офлайн-тесты")
    check((ROOT / "CHANGELOG.md").read_text(encoding="utf-8").count("##") >= 2,
          "CHANGELOG.md: есть разделы по датам")
    check("PYTHONUTF8" in wf_text,
          "CI: PYTHONUTF8=1 — русский stdout на windows-раннере")
    check("reconfigure" in Path(__file__).read_text(encoding="utf-8"),
          "тесты: stdout переконфигурируются в utf-8 (cp1252-раннеры)")

    # --- тайминги под медленную модель и «окна» туннеля ---
    check('"chat_json_timeout", 300' in client_src,
          "чат: не-потоковый ответ ждёт до 300 с, а не 45")
    check('"stream_stall", 75' in client_src,
          "поток: stall 75 с — учитывает обработку промта на CPU")
    check('"plan_timeout", 360' in client_src,
          "план: не-потоковый /plan ждёт до 360 с под медленную модель")
    check("напиши ещё раз, обычно помогает повтор" in panel,
          "панель: человеческая подсказка при обрыве в «Разработке»")


def test_plan_stream_resilience() -> None:
    """Обрыв потока плана — честная ошибка и запасной путь, а не трейсбек.

    Живой прогон 04.10: модель (3B на CPU) уложилась в 254 с, туннель донёс
    начало потока и умер молча. Клиент ждал байтов 360 с (plan_timeout) и
    упал TimeoutError прямо из http.client в терминал — traceback вместо
    плана. Теперь таймаут чтения маленький (stream_stall, пинги сервера
    его продлевают), ошибка — ThinkingError, а у CLI есть запасной путь.
    """
    mod = _load_cli_module()
    client_src = (ROOT / "thinking" / "client.py").read_text(encoding="utf-8")
    ps_src = client_src.split("def plan_stream(")[1].split("\n    def ")[0]
    check('"stream_stall", 75' in ps_src and '"plan_timeout", 360' not in ps_src,
          "план-поток: таймаут чтения — stream_stall, а не 360 с (живой прогон)")
    check("поток плана оборвался" in client_src and "http.client.HTTPException" in ps_src,
          "план-поток: обрыв превращается в ThinkingError, а не в трейсбек")
    check(": open" in (ROOT / "thinking" / "colab" / "cell_c_server.py").read_text(
        encoding="utf-8"),
          "план-поток: сервер кладёт первый байт сразу (пинги держат туннель)")

    c = make_client()

    def boom(*_a, **_kw):
        raise ThinkingError("поток плана оборвался: read timed out")

    c.plan_stream = boom
    plan, is_fb = mod._plan_streamed(c, "задача", {}, [], 3)
    check(bool(plan.get("steps")) and is_fb is True,
          "план-поток: после обрыва команда отдаёт план, а не исключение")

    # --stream --json обязан печатать JSON, а не человекочитаемый план (C-9)
    import contextlib
    import io as _io
    args = argparse.Namespace(files="", constraint=[], context=None,
                              max_steps=3, task="задача", stream=True, json=True)
    buf = _io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = mod.cmd_plan(c, args)
    out = buf.getvalue()
    start = out.find("{")
    parsed = None
    if start >= 0:
        try:
            parsed = json.loads(out[start:])
        except Exception:
            parsed = None
    check(rc in (0, 2) and isinstance(parsed, dict) and parsed.get("steps"),
          f"план-поток: --stream --json печатает JSON (C-9, rc={rc})")


def test_panel_host_and_models() -> None:
    """AUD-17 (чужой Host/Origin → 403) и вкладка «Модели» с данными /parallel."""
    mod = _load_cli_module()

    check(mod.host_allowed("127.0.0.1:8765", "", 8765),
          "AUD-17: свой адрес 127.0.0.1:8765 проходит")
    check(mod.host_allowed("localhost:8765", "", 8765),
          "AUD-17: localhost проходит")
    check(mod.host_allowed("[::1]:8765", "", 8765),
          "AUD-17: IPv6-петля проходит")
    check(not mod.host_allowed("evil.example", "", 8765),
          "AUD-17: чужой Host → 403 (DNS-подмена)")
    check(not mod.host_allowed("evil.example:8765", "", 8765),
          "AUD-17: чужой Host с портом → 403")
    check(not mod.host_allowed("127.0.0.1:9999", "", 8765),
          "AUD-17: чужой порт → 403")
    check(mod.host_allowed("127.0.0.1:8765", "http://127.0.0.1:8765", 8765),
          "AUD-17: свой Origin у POST проходит")
    check(not mod.host_allowed("127.0.0.1:8765", "https://evil.example", 8765),
          "AUD-17: чужой Origin у POST → 403")
    check(not mod.host_allowed("127.0.0.1:8765", "http://evil.example:8765", 8765),
          "AUD-17: Origin чужого хоста → 403")
    check(mod.host_allowed("127.0.0.1:8765", "null", 8765),
          "AUD-17: запрос без Origin (curl, file://) не блокируется")

    cli_src = (ROOT / "tools" / "thinking_cli.py").read_text(encoding="utf-8")
    get_pos, post_pos = cli_src.find("def do_GET("), cli_src.find("def do_POST(")
    deny_g = cli_src.find("if self._deny_host():")
    deny_p = cli_src.find("if self._deny_host(with_origin=True):")
    check(0 < get_pos < deny_g, "AUD-17: do_GET вызывает _deny_host")
    check(0 < post_pos < deny_p and deny_p - post_pos < 500,
          "AUD-17: do_POST вызывает _deny_host с проверкой Origin")

    html = (ROOT / "tools" / "thinking_panel.html").read_text(encoding="utf-8")
    check("renderParallel" in html and "data.parallel" in html,
          "вкладка «Модели»: параллельные модели из /parallel нарисованы")
    check('id="parallel"' in html,
          "вкладка «Модели»: место под параллельные движки")
    check('data["parallel"] = client.parallel_models()' in cli_src,
          "/api/models отдаёт /parallel — данные для вкладки «Модели»")


def main() -> int:
    # CI (windows-latest, локаль en-US): stdout = cp1252, а печатаем
    # по-русски — без переконфигурации финальный счётчик роняет процесс
    # UnicodeEncodeError. Локально (cp1251/utf-8) и на Linux этого не видно.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    test_schemas()
    test_fallback()
    test_client()
    test_files()
    test_reliability()
    test_chat_memory_tokens()
    test_rus_labels()
    test_memory_action()
    test_panel_routes()
    test_breaker_channels()
    test_plan_stream_accounting()
    test_plan_stream_resilience()
    test_panel_host_and_models()
    test_benefits_no_double_count()
    test_token_honesty()
    test_secrets_smart()
    test_reflect_async_spawns_process()
    test_devsave_args()
    test_dev_reports()
    test_chat_author()
    test_budget_matches_running_model()
    test_failure_diagnostics()
    test_sse_heartbeat()
    test_watchdog_and_ram_guard()
    test_dev_profile_and_honest_hints()
    test_parallel_models()
    test_model_choice_decision()
    test_mock_llm_stub()
    test_stub_switching()
    test_instructions_present()
    test_dev_layout_vertical()
    test_dev_run_stdin()
    test_chat_autoscroll()
    test_period_default()
    test_memory_sig()
    test_config_timeouts()
    test_nctx_alignment()
    test_sse_queue_limit()
    test_smoke_guarded()
    test_tail_throttle()
    test_cache_tail_read()
    test_events_ping()
    test_colab_cells()
    test_dev_metrics_limits()
    test_quickwins()
    print(f"ПРОЙДЕНО: {PASSED}")
    print(f"ПРОВАЛЕНО: {FAILED}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
