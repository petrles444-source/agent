#!/usr/bin/env python3
"""Локальная заглушка «Мышления»: панель, клиент и CLI без Colab.

Заглушка подменяет не модель, а **сервер субагента** — тот самый, на что
смотрит `set-url`. Она отвечает на те же маршруты и в том же формате, что
ячейка C на Colab (`/health`, `/plan`, `/plan/stream`, `/reflect`, `/chat`,
`/chat/stream`, `/dev`, `/ask/multi`, `/models`, `/events`, `/metrics`),
но текст собирается из шаблона за миллисекунды. Поэтому разработка идёт
локально: без туннеля, без GPU, без десятков секунд на каждый запрос.

Дополнительно отвечает на `/v1/models` и `/v1/chat/completions`
(OpenAI-совместимо) — чтобы заглушку можно было подставить и как upstream
для настоящего сервера:  THINKING_UPSTREAM=http://127.0.0.1:8010

Режимы (--mode):
    normal   — обычные ответы с небольшой задержкой (по умолчанию)
    slow     — первый токен через --first-delay секунд: имитация 14B на CPU,
               чтобы пинги SSE и stream_stall проверялись без ожидания вручную
    error    — все запросы отвечают 503: путь отката и фолбэк
    drop     — потоки обрываются на середине, обычные маршруты работают:
               ровно та авария, при которой клиент падает с SSE на /chat
    badjson  — маршруты генерации отдают не-JSON: контракт ловит мусор

Метки в тексте запроса переключают режим на один вызов (и вычищаются):
    [mock:slow]  [mock:error]  [mock:drop]  [mock:badjson]

Как вернуться на настоящий Colab:

    python tools/thinking_cli.py set-url --back
    python tools/thinking_cli.py set-url https://<адрес>.trycloudflare.com <токен>

Заглушка сама ничего не пишет в config/thinking.local.json: переключение
всегда явное, поэтому «уйти обратно» — это одна команда. Ни память
(logs/thinking/memory.json), ни отчёты, ни таймлайн при переключении не
трогаются: они лежат на ПК, а меняется только то, кто печатает ответ.

Запуск:  python tools/mock_llm.py [--port 8010] [--mode normal]
Только стандартная библиотека, зависимостей нет.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Optional
from urllib.parse import parse_qs, urlparse

MODES = ("normal", "slow", "error", "drop", "badjson")
HINTS = {
    "[mock:slow]": "slow",
    "[mock:error]": "error",
    "[mock:drop]": "drop",
    "[mock:badjson]": "badjson",
}
# Русский текст ~= 2 символа на токен, JSON/английский ~= 4; как в
# thinking/schemas.py берём консервативные 3, чтобы не занижать расход.
CHARS_PER_TOKEN = 3
ACTIONS = ("build", "test", "verify", "docs", "prompt", "debug")
CHUNK = 10          # символов в одном «токене» заглушки
MODEL_ID = "mock-llm"


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------- #
#  Тексты ответов (шаблон, а не генерация)
# --------------------------------------------------------------------------- #
def strip_hints(text: str) -> tuple[str, str]:
    """Достаёт метку режима из текста и возвращает (режим, текст без меток)."""
    mode, out = "", str(text or "")
    for mark, name in HINTS.items():
        if mark in out:
            mode, out = name, out.replace(mark, "")
    return mode, out


def probe_text(body: dict) -> str:
    """Где искать метки режима: задача, сообщение, результат, код."""
    return " ".join(str(body.get(k) or "")
                    for k in ("task", "message", "result", "active_code"))


def clip(text: str, max_tokens: int) -> str:
    """Обрезает ответ по бюджету токенов — как настоящая модель."""
    limit = max(60, int(max_tokens or 0) * CHARS_PER_TOKEN)
    text = str(text)
    return text if len(text) <= limit else text[:limit].rstrip() + " …"


def pieces(text: str, chunk: int = CHUNK) -> list[str]:
    chunk = max(1, int(chunk))
    return [text[i:i + chunk] for i in range(0, len(text), chunk)]


def build_plan(task: str, max_steps: int, constraints: list[str],
               max_tokens: int) -> dict:
    """План по шаблону: структура настоящая (проходит Plan.from_dict),
    содержание — от заглушки. Именно структура, а не красота прозы,
    проверяется контрактом."""
    task = str(task or "").strip() or "задача без названия"
    short = task[:80]
    n = max(2, min(5, int(max_steps or 3)))
    templates = [
        ("build", f"Сформулировать цель «{short}» и собрать исходные данные",
         "цель и исходные данные"),
        ("verify", "Проверить ограничения и допущения — ошибка здесь дороже всего",
         "список ограничений"),
        ("test", "Собрать решение и прогнать проверки на живых данных",
         "пройденные проверки"),
        ("docs", "Описать результат и зафиксировать его в отчёте",
         "отчёт о результате"),
        ("build", "Доработать по замечаниям проверки", "исправления"),
    ]
    steps = []
    for i in range(n):
        action, desc, expected = templates[i]
        steps.append({
            "id": i + 1,
            "action": action,
            "desc": desc,
            "inputs": {"task": short} if i == 0 else {},
            "expected_output": expected,
            "retry_policy": "once",
            "depends_on": [i] if i else [],
        })
    return {
        "plan_id": f"mock-{uuid.uuid4().hex[:8]}",
        "goal": task[:500],
        "source": "colab",
        "rationale": ("Заглушка: план собран из шаблона по задаче «" + short +
                      "». Структура настоящая, содержание добавит модель в "
                      "Colab — вернись командой set-url --back.")[:800],
        "confidence": 0.5,
        "sub_goals": [s["desc"][:80] for s in steps[:3]],
        "constraints": [str(c)[:200] for c in (constraints or [])][:10],
        "contradictions": [],
        "success_criteria": ["Проверки пройдены", "Отчёт записан"],
        "unknown_files": [],
        "fallback": "выполнить напрямую, без субагента",
        "steps": steps,
        "created_at": utcnow(),
    }


def build_chat_reply(body: dict, max_tokens: int) -> dict:
    _mode, msg = strip_hints(str(body.get("message") or ""))
    msg = msg.strip()[:300] or "(пустое сообщение)"
    # Клиент кладёт память в context.memory, поле memory у сервера резервное —
    # читаем оба, иначе заглушка не видит факты и эхо памяти не работает.
    ctx = body.get("context") if isinstance(body.get("context"), dict) else {}
    mem = body.get("memory") if isinstance(body.get("memory"), dict) else {}
    if not mem and isinstance(ctx.get("memory"), dict):
        mem = ctx["memory"]
    facts = [str(f) for f in (mem.get("facts") or []) if str(f).strip()]
    used = bool(facts or body.get("profile") or body.get("history") or ctx)
    text = (f"Заглушка отвечает на «{msg}». "
            f"Это локальная модель-заглушка: текст собирается из шаблона, "
            f"поэтому ответ приходит за доли секунды и без Colab. "
            f"Настоящий ответ даст модель на GPU/CPU-рантайме — "
            f"вернись командой set-url --back.")
    if facts:
        text += " Я помню: " + "; ".join(facts[-3:]) + "."
    text = clip(text, max_tokens)
    return {
        "reply": text,
        "rationale": "Заглушка: шаблонный ответ, токены посчитаны по длине.",
        "steps": [{"desc": "Принято"}, {"desc": "Выполнено шаблоном"}],
        "source": "colab",
        "plan_id": "",
        "fallback": False,
        "memory_used": used,
        "tokens_in": max(1, len(str(body.get("message") or "")) // CHARS_PER_TOKEN),
        "tokens_out": max(1, len(text) // CHARS_PER_TOKEN),
        "tokens_estimate": True,
        "duration_ms": 0,
        "at": utcnow(),
    }


def build_reflect(body: dict, max_tokens: int) -> dict:
    err = str(body.get("error") or "").strip()
    obs = str(body.get("observation") or "").strip()
    if err:
        status, advice = "adjust", f"Заглушка: шаг не прошёл ({err[:120]}) — повтори с другой гипотезой."
    else:
        status, advice = "ok", "Заглушка: шаг засчитан, продолжай по плану."
    if obs:
        advice += f" Учло наблюдение: {obs[:120]}."
    return {
        "status": status,
        "rationale": ("Заглушка: рефлексия по шаблону, шаг "
                      f"{body.get('step_id', 0)}.")[:400],
        "advice": clip(advice, max_tokens)[:800],
        "next_steps": [],
        "updated_goal_stack": [],
    }


def build_dev(body: dict, max_tokens: int) -> dict:
    _mode, msg = strip_hints(str(body.get("message") or ""))
    return {
        "action": "none",
        "filename": str(body.get("active_name") or ""),
        "code": "",
        "comment": clip("Заглушка: предложение изменений не сгенерировано — "
                        f"запрос «{(msg or '')[:120]}». Файлы не тронуты, "
                        "контроль остаётся за человеком.", max_tokens)[:1200],
        "tokens_in": max(1, len(msg) // CHARS_PER_TOKEN),
        "tokens_out": 0,
        "tokens_estimate": True,
        "duration_ms": 0,
        "at": utcnow(),
    }


# --------------------------------------------------------------------------- #
#  Состояние заглушки
# --------------------------------------------------------------------------- #
class MockState:
    """Общая память сервера: события, счётчики, режим."""

    def __init__(self, mode: str = "normal", delay: float = 0.01,
                 first_delay: float = 0.0, ping: float = 15.0,
                 max_tokens: int = 700, dev_max_tokens: int = 1600,
                 token: str = "", verbose: bool = False) -> None:
        self.mode = mode if mode in MODES else "normal"
        self.delay = max(0.0, float(delay))
        self.first_delay = max(0.0, float(first_delay))
        # ping = 0 — пинги НЕ шлём вовсе: так воспроизводится живой случай,
        # когда они не доходят через туннель (клиент обязан терпеть до
        # первого байта по stream_stall_first, а не по stream_stall)
        self.ping = max(0.0, float(ping))
        self.max_tokens = int(max_tokens)
        self.dev_max_tokens = int(dev_max_tokens)
        self.token = str(token)
        self.verbose = bool(verbose)
        self.t0 = time.time()
        self.lock = threading.Lock()
        self.seq = 0
        self.events: list[dict] = []
        self.counters = {
            "plans": 0, "reflects": 0, "chats": 0, "errors": 0,
            "bad_json": 0, "dropped": 0, "requests": 0,
            "tokens_in": 0, "tokens_out": 0,
        }
        self.llm_ms: list[int] = []
        # «Манифест» дополнительных движков — как его пишет ячейка D
        # (cell_c:1279-1293). Держим в состоянии, чтобы /parallel и
        # /ask/multi считали одно и то же: у настоящего сервера
        # targets = [(основная, UPSTREAM)] + _parallel_models()
        # (cell_c:1322-1324), и оба маршрута обязаны сходиться.
        self.parallel: list[dict] = [
            {"label": "MOCK-7B", "path": "mock://7b", "url": "local-mock"}]

    # -------------------------------------------------------------- события --
    def emit(self, etype: str, text: str, **extra: Any) -> dict:
        with self.lock:
            self.seq += 1
            ev = {"seq": self.seq, "type": etype, "text": str(text)[:800],
                  # ключ ts, как у настоящего сервера (cell_c:245): панель
                  # читает ev.ts, и на ключе «at» лента мыслей показывала
                  # бы пустое время (аудит A-4, 04-tests-mock)
                  "ts": utcnow(), **extra}
            self.events.append(ev)
            if len(self.events) > 500:
                del self.events[:-500]
            return ev

    def count(self, key: str, n: int = 1) -> None:
        with self.lock:
            self.counters[key] = self.counters.get(key, 0) + n

    def timed(self, ms: int) -> None:
        with self.lock:
            self.llm_ms.append(int(ms))
            if len(self.llm_ms) > 200:
                del self.llm_ms[:-200]

    def mode_for(self, hinted: str) -> str:
        """Режим конкретного запроса: метка в тексте важнее глобального."""
        return hinted if hinted in MODES else self.mode

    def snapshot(self) -> dict:
        with self.lock:
            return {"seq": self.seq, "counters": dict(self.counters),
                    "events": list(self.events[-200:]),
                    "llm_ms": list(self.llm_ms)}


def _metrics(state: MockState) -> dict:
    snap = state.snapshot()
    c, ms = snap["counters"], snap["llm_ms"]
    calls = c["plans"] + c["reflects"] + c["chats"]
    return {
        "events": snap["seq"],
        "plans_total": c["plans"],
        "reflects": c["reflects"],
        "chats_total": c["chats"],
        "errors": c["errors"],
        "bad_json": c["bad_json"],
        "subscribers": 0,
        "llm_ms_avg": int(sum(ms) / len(ms)) if ms else None,
        "llm_ms_p95": sorted(ms)[int(len(ms) * 0.95)] if len(ms) > 5 else None,
        "queue": 1,
        "tokens_in": c["tokens_in"],
        "tokens_out": c["tokens_out"],
        "tokens_total": c["tokens_in"] + c["tokens_out"],
        "usage_calls": calls,
        "usage_exact": True,
        # лишние ключи контракт не ломает — он их просто не читает
        "mock": True,
        "mode": state.mode,
        "uptime_s": int(time.time() - state.t0),
    }


def _health(state: MockState) -> dict:
    return {
        "status": "ok",
        "model": MODEL_ID,
        "upstream": "local-mock",
        "uptime_s": int(time.time() - state.t0),
        "plans": state.counters["plans"],
        "events": state.seq,
        "gpu": "mock",
        "vram_used_gb": None,
        "mock": True,
        "mode": state.mode,
    }


INDEX_HTML = (
    '<!doctype html><meta charset="utf-8">'
    "<title>Заглушка «Мышление»</title>"
    "<body style=\"font-family:system-ui;max-width:44rem;margin:3rem auto\">"
    "<h1>Заглушка работает</h1>"
    "<p>Это локальная заглушка LLM: панель, клиент и CLI гоняются без Colab. "
    "Настоящий адрес возвращается командой "
    "<code>python tools/thinking_cli.py set-url --back</code>.</p>"
    "<p>Маршруты: <code>/health</code> <code>/metrics</code> "
    "<code>/plan</code> <code>/plan/stream</code> <code>/chat</code> "
    "<code>/chat/stream</code> <code>/reflect</code> <code>/dev</code> "
    "<code>/ask/multi</code> <code>/models</code> <code>/events</code> "
    "<code>/v1/models</code> <code>/v1/chat/completions</code></p>"
    "</body>"
)


# --------------------------------------------------------------------------- #
#  HTTP-обработчик
# --------------------------------------------------------------------------- #
def make_handler(state: MockState) -> type:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "MockLLM/1.0"
        # Метка режима из тела запроса — заполняется в do_POST
        _hinted = ""
        # Пинги SSE возможны только после заголовков потока: до них клиент
        # ждёт статус-код, и сырые байты ": ping" сломали бы разбор.
        _streaming = False

        # ------------------------------------------------------------ утилиты
        def log_message(self, fmt: str, *args: Any) -> None:
            if state.verbose:
                sys.stderr.write("mock: " + (fmt % args) + "\n")

        @property
        def _route(self) -> str:
            return urlparse(self.path).path

        @property
        def _query(self) -> dict:
            return parse_qs(urlparse(self.path).query)

        def _send(self, code: int, obj: Any,
                  ctype: str = "application/json; charset=utf-8") -> None:
            if isinstance(obj, bytes):
                body = obj
            elif isinstance(obj, (dict, list)):
                body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            else:
                body = str(obj).encode("utf-8")
            try:
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.close_connection = True
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                self.close_connection = True

        def _sse_open(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            # буферизация прокси убивает поток — отключаем, как на Colab
            self.send_header("X-Accel-Buffering", "no")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            self._streaming = True

        def _sse(self, payload: Any) -> None:
            data = payload if isinstance(payload, str) \
                else json.dumps(payload, ensure_ascii=False)
            self.wfile.write(f"data: {data}\n\n".encode("utf-8"))
            self.wfile.flush()

        def _ping(self) -> None:
            """Комментарий SSE: держит соединение живым между токенами."""
            self.wfile.write(b": ping\n\n")
            self.wfile.flush()
            state.count("pings")

        def _read_body(self) -> dict:
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length > 0 else b""
            if state.verbose:
                sys.stderr.write(f"mock: POST {self.path} cl={length} "
                                 f"raw={raw[:120]!r}\n")
            if not raw:
                return {}
            try:
                data = json.loads(raw.decode("utf-8", "replace"))
            except json.JSONDecodeError:
                return {}
            return data if isinstance(data, dict) else {}

        def _authorized(self) -> bool:
            if not state.token:
                return True
            got = self.headers.get("X-Agent-Token", "") or \
                (self._query.get("token") or [""])[0]
            if got == state.token:
                return True
            self._send(401, {"error": "unauthorized"})
            return False

        # ------------------------------------------------- генерация / режим
        def _wait_first(self) -> bool:
            """Пауза до первого токена (режим slow).

            В потоке во время паузы идут пинги `: ping` — именно они держат
            соединение живым, пока модель «думает». Без пингов клиент со
            stream_stall закрыл бы чтение и решил, что сервер умер.
            """
            if state.mode_for(self._hinted) != "slow" or state.first_delay <= 0:
                return True
            end = time.time() + state.first_delay
            if not self._streaming:
                time.sleep(state.first_delay)   # до заголовков писать нельзя
                return True
            while time.time() < end:
                if state.ping <= 0:            # режим «без пингов»
                    time.sleep(max(0.02, end - time.time()))
                    continue
                time.sleep(min(state.ping, max(0.02, end - time.time())))
                try:
                    self._ping()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    return False
            return True

        def _sleep_between(self) -> None:
            if state.delay > 0:
                time.sleep(state.delay)

        def _content_route(self) -> Optional[str]:
            """Маршруты, которые генерируют текст (на них действуют режимы)."""
            p = self._route
            return p if p in ("/plan", "/plan/stream", "/reflect",
                              "/reflect/stream", "/chat",
                              "/chat/stream", "/dev", "/dev/stream",
                              "/ask/multi") else None

        # ------------------------------------------------------------ ответы
        def _bad_json(self) -> None:
            """Прокси отдал HTML вместо JSON.

            Сам сервер так не умеет: `plan_data`/`guard_reply`/`reflect_data`
            мягко деградируют в валидный ответ (cell_c:586-680). Этот режим
            воспроизводит ошибку уже на стороне доставки — Cloudflare/captive
            portal отдаёт 200 с HTML, и клиент обязан это пережить.
            """
            state.count("bad_json")
            state.emit("error", "заглушка: прокси отдал HTML вместо JSON")
            self._send(200, "<html>это не JSON</html>".encode("utf-8"),
                       "text/html; charset=utf-8")

        def _fail(self) -> None:
            state.count("errors")
            state.emit("error", "заглушка: режим error — 503")
            self._send(503, {"error": "заглушка: LLM недоступен (режим error)"})

        def _tokens(self, text: str) -> None:
            for piece in pieces(text):
                self._sleep_between()
                self._sse({"type": "token", "text": piece})

        # ------------------------------------------------------------- GET
        def do_GET(self) -> None:  # noqa: N802
            state.count("requests")
            route = self._route
            if route == "/":
                self._send(200, INDEX_HTML, "text/html; charset=utf-8")
                return
            if route == "/v1/models":
                self._send(200, {
                    "object": "list",
                    "data": [{"id": MODEL_ID, "object": "model",
                              "created": int(state.t0), "owned_by": "local"}],
                })
                return
            # настоящий сервер палит токен на всех своих маршрутах — заглушка
            # обязана вести себя так же, иначе неверный токен в конфиге
            # маскируется «у заглушки и без токена работает» (аудит C-1).
            # /v1/* открыт — это upstream для llama.cpp, у него своих нет.
            if not self._authorized():
                return
            # Режим error НЕ гасит /health: у настоящего сервера /health от
            # LLM не зависит и всегда 200 (cell_c:716-728), поэтому сценарий
            # «health жив, контентные маршруты 503» должен быть воспроизводим
            # (аудит A-4, 04-tests-mock: раньше заглушка гасила всё, и тест
            # закреплял «сервер упал» вместо «LLM упала»).
            if route == "/health":
                self._send(200, _health(state))
            elif route == "/metrics":
                self._send(200, _metrics(state))
            elif route == "/events":
                snap = state.snapshot()
                since = int((self._query.get("since") or ["0"])[0] or 0)
                tail = int((self._query.get("tail") or ["0"])[0] or 0)
                items = [e for e in snap["events"] if e["seq"] > since]
                if tail:
                    items = items[-tail:]
                self._send(200, {"last_seq": snap["seq"], "events": items})
            elif route == "/events/stream":
                if not self._authorized():
                    return
                self._sse_open()
                for ev in state.snapshot()["events"]:
                    try:
                        self._sse(ev)
                    except (BrokenPipeError, ConnectionResetError):
                        return
            elif route == "/models":
                self._send(200, {
                    "active": f"mock://{MODEL_ID}",
                    "active_label": "MOCK",
                    "switching": False,
                    "gpu": False,
                    "models": [{"path": f"mock://{MODEL_ID}", "label": "MOCK",
                                "size_gb": 0.0,
                                "hint": "заглушка: настоящей модели нет",
                                "active": True}],
                })
            elif route == "/parallel":
                self._send(200, {
                    "active": "MOCK",
                    "url": "local-mock",
                    "extra": list(state.parallel),
                    # count = len(extra), как cell_c:1300-1304: это число
                    # ДОПОЛНИТЕЛЬНЫХ движков, а не всех отвечающих
                    "count": len(state.parallel),
                })
            elif route.startswith("/dump/"):
                name = route.split("/dump/", 1)[1]
                self._send(200, f"заглушка: файлов Colab нет (запрошен {name})\n",
                           "text/plain; charset=utf-8")
            else:
                self._send(404, {"error": f"нет маршрута {route}"})

        # ------------------------------------------------------------ POST
        def do_POST(self) -> None:  # noqa: N802
            state.count("requests")
            route = self._route
            body = self._read_body()
            if state.mode == "error":
                self._fail()
                return
            if not self._authorized():
                return
            self._hinted, _ = strip_hints(probe_text(body))
            mode = state.mode_for(self._hinted)
            route = self._content_route() or route

            # --- OpenAI-совместимый upstream (для настоящего сервера) --------
            if route == "/v1/chat/completions":
                self._openai_chat(body, mode)
                return

            # --- режимы, ломающие контракт ---------------------------------
            # Метка в тексте запроса действует так же, как глобальный --mode:
            # иначе [mock:error] в задаче молча игнорировался бы.
            if mode in ("badjson", "error") and self._content_route():
                if mode == "error":
                    self._fail()
                else:
                    self._bad_json()
                return
            if mode == "drop" and route.endswith("/stream"):
                self._drop_stream()
                return

            started = time.time()
            try:
                if route == "/plan":
                    self._plan(body)
                elif route == "/plan/stream":
                    self._plan_stream(body)
                elif route == "/reflect":
                    self._reflect(body)
                elif route == "/reflect/stream":
                    self._reflect_stream(body)
                elif route == "/chat":
                    self._chat(body)
                elif route == "/chat/stream":
                    self._chat_stream(body)
                elif route == "/dev":
                    self._dev(body)
                elif route == "/dev/stream":
                    self._dev_stream(body)
                elif route == "/ask/multi":
                    self._ask_multi(body)
                elif route == "/model":
                    self._send(200, {"ok": True, "active": f"mock://{MODEL_ID}",
                                     "label": "MOCK",
                                     "note": "заглушка: переключать некуда"})
                elif route.startswith("/cancel/"):
                    pid = route.rsplit("/", 1)[-1]
                    self._send(200, {"status": "cancelled", "plan_id": pid})
                else:
                    self._send(404, {"error": f"нет маршрута {route}"})
            except (BrokenPipeError, ConnectionResetError):
                state.count("dropped")
                state.emit("error", f"заглушка: клиент оборвал {route}")
            except Exception as exc:  # noqa: BLE001
                state.count("errors")
                state.emit("error", f"заглушка: {type(exc).__name__}: {exc}"[:300])
                try:
                    self._send(500, {"error": str(exc)[:300]})
                except OSError:
                    pass
            finally:
                state.timed(int((time.time() - started) * 1000))

        # ------------------------------------------------------ маршруты
        def _plan(self, body: dict) -> None:
            _mode, task = strip_hints(str(body.get("task") or ""))
            state.emit("thought", f"Задача: {task[:150]}")
            state.emit("thought", "Декомпозирую на шаги…")
            if not self._wait_first():
                return
            plan = build_plan(task, int(body.get("max_steps") or 4),
                              list(body.get("constraints") or []),
                              state.max_tokens)
            self._finish_plan(plan, stream=False)

        def _plan_stream(self, body: dict) -> None:
            _mode, task = strip_hints(str(body.get("task") or ""))
            state.emit("thought", f"Стрим плана: {task[:150]}")
            plan = build_plan(task, int(body.get("max_steps") or 4),
                              list(body.get("constraints") or []),
                              state.max_tokens)
            self._sse_open()
            if not self._wait_first():
                return
            raw = json.dumps(plan, ensure_ascii=False)
            self._tokens(raw)
            self._finish_plan(plan, stream=True)

        def _finish_plan(self, plan: dict, stream: bool) -> None:
            blob = json.dumps(plan, ensure_ascii=False)
            state.count("plans")
            state.count("tokens_in", max(1, len(blob) // 8 // CHARS_PER_TOKEN))
            state.count("tokens_out", max(1, len(blob) // CHARS_PER_TOKEN))
            if stream:
                self._sse({"type": "rationale", "text": plan["rationale"]})
                for s in plan["steps"]:
                    self._sse({"type": "plan_step",
                               "text": f"[{s['id']}] {s['desc']}",
                               "step_id": s["id"]})
                self._sse({"type": "final",
                           "text": json.dumps(plan, ensure_ascii=False),
                           "plan_id": plan["plan_id"]})
            else:
                state.emit("rationale", plan["rationale"], plan_id=plan["plan_id"])
                for s in plan["steps"]:
                    state.emit("plan_step", f"[{s['id']}] {s['desc']}",
                               plan_id=plan["plan_id"], step_id=s["id"])
                state.emit("final", f"План {plan['plan_id'][:8]} готов: "
                                    f"{len(plan['steps'])} шагов",
                           plan_id=plan["plan_id"])
                self._send(200, plan)

        def _reflect(self, body: dict) -> None:
            state.emit("thought", f"Рефлексия шага {body.get('step_id', 0)}",
                       plan_id=str(body.get("plan_id") or ""))
            if not self._wait_first():
                return
            out = build_reflect(body, state.max_tokens)
            state.count("reflects")
            etype = "final" if out["status"] == "ok" else "contradiction"
            state.emit(etype, f"[{out['status']}] {out['advice']}",
                       plan_id=str(body.get("plan_id") or ""))
            self._send(200, out)

        def _reflect_stream(self, body: dict) -> None:
            """Рефлексия по потоку: как /reflect, но итог уходит в done.

            Нужен, чтобы офлайн-тесты гоняли клиентский фолбэк B-5 так же,
            как настоящий сервер на Colab: токены — по мере генерации,
            итог — событие done с полем response.
            """
            state.emit("thought", f"Рефлексия (поток) шага {body.get('step_id', 0)}",
                       plan_id=str(body.get("plan_id") or ""))
            out = build_reflect(body, state.max_tokens)
            self._sse_open()
            if not self._wait_first():
                return
            self._tokens(f"[{out['status']}] {out['advice']}")
            self._sse({"type": "done", "response": out})
            state.count("reflects")
            etype = "final" if out["status"] == "ok" else "contradiction"
            state.emit(etype, f"[{out['status']}] {out['advice']}",
                       plan_id=str(body.get("plan_id") or ""))

        def _chat(self, body: dict) -> None:
            state.emit("thought", f"Чат: {str(body.get('message') or '')[:120]}")
            if not self._wait_first():
                return
            out = build_chat_reply(body, state.max_tokens)
            state.count("chats")
            state.count("tokens_in", out["tokens_in"])
            state.count("tokens_out", out["tokens_out"])
            state.emit("final", f"[чат] {out['reply'][:200]}")
            self._send(200, out)

        def _chat_stream(self, body: dict) -> None:
            state.emit("thought", f"Чат (поток): {str(body.get('message') or '')[:120]}")
            out = build_chat_reply(body, state.max_tokens)
            self._sse_open()
            if not self._wait_first():
                return
            self._tokens(out["reply"])
            out["duration_ms"] = 0
            self._sse({"type": "done", **out})
            state.count("chats")
            state.count("tokens_in", out["tokens_in"])
            state.count("tokens_out", out["tokens_out"])
            state.emit("final", f"[чат-поток] {out['reply'][:200]}")

        def _dev(self, body: dict) -> None:
            state.emit("thought",
                       f"Разработка: {str(body.get('message') or '')[:120]}")
            if not self._wait_first():
                return
            out = build_dev(body, state.dev_max_tokens)
            state.count("chats")
            state.emit("final", f"[{out['action']}] "
                                f"{out['filename'] or 'без файла'}: "
                                f"{out['comment'][:200]}")
            self._send(200, out)

        def _dev_stream(self, body: dict) -> None:
            state.emit("thought",
                       f"Разработка (поток): {str(body.get('message') or '')[:120]}")
            out = build_dev(body, state.dev_max_tokens)
            self._sse_open()
            if not self._wait_first():
                return
            self._tokens(out["comment"])
            self._sse({"type": "done", **out})
            state.count("chats")
            state.emit("final", f"[{out['action']}] "
                                f"{out['filename'] or 'без файла'}: "
                                f"{out['comment'][:200]}")

        def _ask_multi(self, body: dict) -> None:
            t0 = time.time()
            _mode, msg = strip_hints(str(body.get("message") or ""))
            # цель та же, что у cell_c:1322-1324: основная модель + манифест
            targets = ["MOCK"] + [str(d.get("label") or d.get("path"))
                                  for d in state.parallel]
            state.emit("thought", f"Параллельный вопрос {len(targets)} моделям: "
                                  f"{msg[:100]}")
            if not self._wait_first():
                return
            base = build_chat_reply(body, state.max_tokens)["reply"]
            answers = [
                {"model": label, "ok": True,
                 "answer": base if label == "MOCK"
                 else clip("Заглушка (другая модель): " + base, 400),
                 "seconds": round(time.time() - t0, 1)}   # 1 знак, как cell_c:1339
                for label in targets
            ]
            state.count("chats")
            self._send(200, {
                "answers": answers, "count": len(answers),
                "wall_seconds": round(time.time() - t0, 1),
                # текст note и округление — как у настоящего сервера
                # (cell_c:1348-1350), иначе тест защищал форму заглушки
                "note": "суммарная скорость не растёт: ядра те же, "
                        "ответы идут параллельно",
            })

        # --------------------------------------------------- спец-режимы
        def _drop_stream(self) -> None:
            """Обрыв потока посередине: пара токенов ушла, дальше EOF."""
            state.count("dropped")
            state.emit("error", f"заглушка: обрыв потока {self._route}")
            self._sse_open()
            for piece in ("Заглушка досылает", " токены и"):
                try:
                    self._sse({"type": "token", "text": piece})
                except (BrokenPipeError, ConnectionResetError):
                    return
            # соединение закрывается без "done" — клиент обязан пережить это

        def _openai_chat(self, body: dict, mode: str) -> None:
            """POST /v1/chat/completions — для настоящего сервера как upstream."""
            messages = body.get("messages") if isinstance(body.get("messages"), list) else []
            prompt = "\n".join(str(m.get("content") or "")
                               for m in messages if isinstance(m, dict))
            _hint, prompt = strip_hints(prompt)
            max_tokens = int(body.get("max_tokens") or state.max_tokens)
            reply = build_chat_reply({"message": prompt or "(пусто)"},
                                     max_tokens)["reply"]
            created = int(time.time())
            common = {"id": f"cmpl-{uuid.uuid4().hex[:10]}",
                      "object": "chat.completion" if not body.get("stream")
                      else "chat.completion.chunk",
                      "created": created,
                      "model": str(body.get("model") or MODEL_ID)}
            if not body.get("stream"):
                if mode == "slow" and state.first_delay > 0:
                    time.sleep(state.first_delay)
                self._send(200, {**common, "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": reply},
                    "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": max(1, len(prompt) // CHARS_PER_TOKEN),
                              "completion_tokens": max(1, len(reply) // CHARS_PER_TOKEN),
                              "total_tokens": max(1, len(prompt) // CHARS_PER_TOKEN)
                              + max(1, len(reply) // CHARS_PER_TOKEN)}})
                return
            self._sse_open()
            if not self._wait_first():
                return
            for piece in pieces(reply):
                self._sleep_between()
                self._sse({**common, "choices": [{"index": 0,
                                                  "delta": {"content": piece},
                                                  "finish_reason": None}]})
            self._sse({**common, "choices": [{"index": 0, "delta": {},
                                              "finish_reason": "stop"}]})
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

    return Handler


def create_server(host: str = "127.0.0.1", port: int = 0, **state_kw: Any
                  ) -> ThreadingHTTPServer:
    """Сервер на эфемерном порту — так его поднимает тест."""
    state = MockState(**state_kw)
    srv = ThreadingHTTPServer((host, int(port)), make_handler(state))
    srv.daemon_threads = True
    srv.mock_state = state  # type: ignore[attr-defined]
    return srv


def serve(srv: ThreadingHTTPServer) -> None:
    try:
        srv.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="mock_llm",
        description="Локальная заглушка «Мышление»: разработка без Colab")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8010)
    p.add_argument("--mode", choices=MODES, default="normal",
                   help="режим ответов (по умолчанию normal)")
    p.add_argument("--delay", type=float, default=0.01,
                   help="пауза между токенами, секунд")
    p.add_argument("--first-delay", type=float, default=0.0,
                   help="пауза до первого токена в режиме slow (по умолчанию 60)")
    p.add_argument("--ping", type=float, default=15.0,
                   help="как часто слать : ping в режиме slow")
    p.add_argument("--max-tokens", type=int, default=700)
    p.add_argument("--token", default="",
                   help="если задан — принимается только такой X-Agent-Token")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(argv)

    first_delay = args.first_delay
    if args.mode == "slow" and first_delay <= 0:
        first_delay = 60.0
    srv = create_server(args.host, args.port, mode=args.mode,
                        delay=args.delay, first_delay=first_delay,
                        ping=args.ping, max_tokens=args.max_tokens,
                        token=args.token, verbose=args.verbose)
    url = f"http://{args.host}:{srv.server_address[1]}"
    print(f"ЗАГЛУШКА: {url}   режим: {args.mode}")
    print("  python tools/thinking_cli.py set-url", url, args.token or "<токен>")
    print("  вернуться на Colab: python tools/thinking_cli.py set-url --back")
    print("  остановить: Ctrl+C")
    serve(srv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
