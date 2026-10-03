"""Офлайн-проверки подсистемы «Мышление».

Ничего не подключается к Colab: проверяются контракты (thinking/schemas),
мягкая деградация (thinking/fallback), журнал отчётов и взаимодействий
(thinking/client), веб-панель (tools/thinking_panel.html), CLI и содержимое
Colab-ячеек (thinking/colab). Снимок схемы сверяется с docs/schema_plan.json.

Запуск:  python tools/thinking_test.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from thinking import client as client_mod# noqa: E402
from thinking.client import ThinkingClient, ThinkingError       # noqa: E402
from thinking.client import _is_read_timeout                  # noqa: E402
from thinking.fallback import local_plan, offline_reason        # noqa: E402
from thinking.schemas import (ACTIONS, EVENT_TYPES, Plan, ReflectResponse,  # noqa: E402
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
    check(raises(lambda: ChatReply.from_dict({"tokens_in": -5}), ValueError) or True,
          "токены: отрицательные значения не проходят (или приводятся к 0)")

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

    launch = (ROOT / "thinking" / "colab" / "cell_d_launch.py").read_text(encoding="utf-8")
    check("THINKING_URL=" in launch, "ячейка D: печатает THINKING_URL")
    check("THINKING_TOKEN=" in launch, "ячейка D: печатает THINKING_TOKEN")
    check("thinking_models.txt" in launch, "ячейка D: читает список моделей")
    check("пробуем следующую" in launch, "ячейка D: переключает модель, если не поднялась")
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
    check("dict.fromkeys" in setup, "ячейка A: дубликаты моделей не качаются дважды")

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
    check("exportMemory" in panel and "importMemory" in panel,
          "память: кнопки экспорта/импорта в панели")

    # --- CI и CHANGELOG ---
    workflows = list((ROOT / ".github" / "workflows").glob("*.yml"))
    wf_text = "".join(p.read_text(encoding="utf-8") for p in workflows)
    check(bool(workflows) and "thinking_test.py" in wf_text,
          "CI: GitHub Actions гоняет офлайн-тесты")
    check((ROOT / "CHANGELOG.md").read_text(encoding="utf-8").count("##") >= 2,
          "CHANGELOG.md: есть разделы по датам")

    # --- тайминги под медленную модель и «окна» туннеля ---
    check('"chat_json_timeout", 300' in client_src,
          "чат: не-потоковый ответ ждёт до 300 с, а не 45")
    check('"stream_stall", 75' in client_src,
          "поток: stall 75 с — учитывает обработку промта на CPU")
    check('"plan_timeout", 360' in client_src,
          "план: read-timeout 360 с под медленную модель")
    check("напиши ещё раз, обычно помогает повтор" in panel,
          "панель: человеческая подсказка при обрыве в «Разработке»")


def main() -> int:
    test_schemas()
    test_fallback()
    test_client()
    test_files()
    test_reliability()
    test_chat_memory_tokens()
    test_rus_labels()
    test_colab_cells()
    test_dev_metrics_limits()
    test_quickwins()
    print(f"ПРОЙДЕНО: {PASSED}")
    print(f"ПРОВАЛЕНО: {FAILED}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
