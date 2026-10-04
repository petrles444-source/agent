"""Синхронный клиент субагента «Мышление».

Только стандартная библиотека: urllib + threading. Никакого asyncio,
никаких внешних зависимостей — подсистема должна работать в любом проекте
и не тянуть за собой чужой стек (см. docs/TZ_thinking_subagent.md).

Все вызовы деградируют мягко: единственное место, где бросается исключение, —
`plan_with_fallback` при выключенном fallback. Остальное логируется, и приложение
продолжает работать без Colab.
"""
from __future__ import annotations

import http.client
import json
import logging
import socket
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

from thinking.fallback import local_plan
from thinking.schemas import (MAX_MEMORY_FACTS, MAX_MEMORY_TURNS, ChatReply,
                              ChatTurn, SchemaError, estimate_tokens,
                              has_secret, redact_secrets, utcnow)


def _norm_fact(text: str) -> str:
    """Нормализация факта для дедупликации: регистр, пробелы, края пунктуации."""
    return " ".join(str(text).split()).lower().strip(" .,;:!?…")


def _turn_line(turn: dict) -> str:
    """Одна реплика памяти в компактной строке для промта."""
    who = "субагент" if turn.get("role") == "subagent" else "я"
    return f"{who}: {str(turn.get('text') or '')[:300]}"


def _looks_like_missing_route(exc: Exception) -> bool:
    """Сервер старый и не знает маршрут — это не поломка, а повод сработать."""
    text = str(exc).lower()
    return any(tag in text for tag in
               ("404", "405", "not found", "method not allowed", "no route"))


class TokenLedger:
    """Счётчик токенов процесса: точная цифра, если сервер её отдал,
    иначе оценка по длине текста — и честная пометка, что это оценка."""

    def __init__(self) -> None:
        self.tokens_in = 0
        self.tokens_out = 0
        self.exact_calls = 0
        self.estimate_calls = 0

    def record(self, reply: Any) -> None:
        if int(getattr(reply, "tokens_in", 0) or 0):
            self.tokens_in += int(reply.tokens_in)
        if int(getattr(reply, "tokens_out", 0) or 0):
            self.tokens_out += int(reply.tokens_out)
        if getattr(reply, "tokens_estimate", True):
            self.estimate_calls += 1
        else:
            self.exact_calls += 1

    def as_dict(self) -> dict:
        return {"tokens_in": self.tokens_in, "tokens_out": self.tokens_out,
                "tokens_total": self.tokens_in + self.tokens_out,
                "exact_calls": self.exact_calls,
                "estimate_calls": self.estimate_calls}

log = logging.getLogger("thinking.client")

CONFIG_PATH = Path("config/thinking.json")
LOCAL_PATH = Path("config/thinking.local.json")


class ThinkingError(RuntimeError):
    """Субагент недоступен или ответ невалиден."""


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except Exception as exc:
        log.warning("не прочитать %s: %s", path, exc)
        return {}


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _append_jsonl(path: Path, obj: dict) -> None:
    """Дозапись одной записи в JSONL (журнал отчётов/взаимодействий)."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.stat().st_size > 8 * 1024 * 1024:
            path.replace(path.with_name(path.name + ".1"))
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")
    except Exception as exc:
        log.debug("не дописался %s: %s", path, exc)


def _is_read_timeout(exc: BaseException) -> bool:
    """Чистый обрыв по таймауту чтения, а не падение сервера."""
    if isinstance(exc, (socket.timeout, TimeoutError)):
        return True
    if isinstance(exc, urllib.error.URLError) and isinstance(
            exc.reason, (socket.timeout, TimeoutError)):
        return True
    text = str(exc).lower()
    return "timed out" in text or "timeout" in text


def new_rid() -> str:
    """Уникальный id записи журнала.

    Поле `n` — просто счётчик внутри процесса, у разных процессов он
    совпадает. Для удаления по одной записи нужен настоящий идентификатор.
    """
    return uuid.uuid4().hex[:8]


def _read_jsonl(path: Path, limit: int = 200) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    out.append(obj)
    except Exception as exc:
        log.debug("не прочитан %s: %s", path, exc)
    return out[-limit:]


class ThinkingClient:
    def __init__(self, cfg: Optional[dict] = None):
        import os
        base_cfg = _read_json(CONFIG_PATH)
        local_cfg = _read_json(LOCAL_PATH)
        self.cfg = {**base_cfg, **(cfg or {})}
        self.base = (
            local_cfg.get("base_url")
            or self.cfg.get("base_url")
            or os.environ.get("THINKING_URL", "")
        ).rstrip("/")
        self.token = (
            local_cfg.get("token")
            or self.cfg.get("token")
            or os.environ.get("THINKING_TOKEN", "")
        )
        # --url задал адрес вручную: reload_config() не должен его перетирать
        self.base_locked = False
        self.timeout = float(self.cfg.get("timeout", 45))
        self.connect_timeout = float(self.cfg.get("connect_timeout", 5))
        self.retry = self.cfg.get("retry") or {}
        self._log_path = Path(self.cfg.get("log_path", "logs/thinking/thoughts.jsonl"))
        self._log_max = int(self.cfg.get("log_max_bytes", 5 * 1024 * 1024))
        self._events: list[dict] = []
        self._seq = 0
        self._seq_local = 0            # свои seq, чтобы не ронять события
        self._lock = threading.Lock()  # из одной миллисекунды (аудит C-2)
        self._stop = threading.Event()
        self._reader: Optional[threading.Thread] = None
        self.last_ok: Optional[str] = None
        self.last_error: str = ""
        self.online = False
        # Поток событий — отдельный канал: его обрыв не делает субагента офлайновым
        # (онлайн-статус ведёт REST /health), но обрыв должен быть виден.
        self.stream_ok = False
        self.stream_error: str = ""
        # --- для веб-интерфейса: что я просил и что получил -------------
        # Всё дублируется на диск: вызовы делаются из других процессов
        # (CLI), а панель живёт в своём — иначе история была бы пустой.
        self._inter_path = Path(self.cfg.get(
            "interactions_path", "logs/thinking/interactions.jsonl"))
        self._rep_path = Path(self.cfg.get(
            "reports_path", "logs/thinking/reports.jsonl"))
        self._chat_path = Path(self.cfg.get(
            "chat_path", "logs/thinking/chat.jsonl"))
        self._mem_path = Path(self.cfg.get(
            "memory_path", "logs/thinking/memory.json"))
        self.interactions: list[dict] = []   # хронология моих вызовов
        self.reports: list[dict] = []        # подробные отчёты субагента
        self.chat_log: list[dict] = []       # живой диалог с субагентом
        self.token_stats = TokenLedger()     # токены процесса (чат)
        self.stats = {"plans": 0, "reflects": 0, "fallbacks": 0, "chats": 0,
                      "blocked_ms": 0, "background_ms": 0, "errors": 0}
        self._disk_cache: tuple = (None, [], [])
        # (аудит AUD-20: неиспользуемый _mem_cache удалён — memory() всегда
        # читает файл по mtime, а кэширование давало бы риск устаревших данных)
        # Предохранитель транспорта: N ошибок подряд → пауза вместо спама
        # ретраями, когда туннель мёртв (HTTP 5xx, обрывы, таймауты).
        # Каналы раздельные (AUD-05): REST и длинный поток считают отдельно.
        self._cb_errors = 0
        self._cb_open_until = 0.0
        self._cb_stream_errors = 0
        self._cb_stream_until = 0.0
        self.health_data: dict = {}

    # ------------------------------------------------------------------ #
    #  конфигурация
    # ------------------------------------------------------------------ #
    @staticmethod
    def set_url(base_url: str, token: str = "", back: bool = False) -> Path:
        """Пишет URL/токен в config/thinking.local.json (секреты не в git).

        Текущая пара запоминается в поле «prev», а `set-url --back` меняет
        её местами с текущей. Это и есть возврат с локальной заглушки на
        Colab: одна команда, без копирования адреса и токена из чата, и
        повторный --back снова уводит на заглушку.
        """
        data = _read_json(LOCAL_PATH)
        if back:
            prev = data.get("prev")
            prev = prev if isinstance(prev, dict) else {}
            if not prev.get("base_url"):
                raise ThinkingError(
                    "предыдущий адрес не сохранён — выполни вручную: "
                    "python tools/thinking_cli.py set-url <URL> <TOKEN>")
            cur_base = str(data.get("base_url") or "").rstrip("/")
            cur_token = str(data.get("token") or "")
            data["base_url"] = str(prev.get("base_url") or "").rstrip("/")
            data["token"] = str(prev.get("token") or "")
            if cur_base:
                data["prev"] = {"base_url": cur_base, "token": cur_token}
            else:
                data.pop("prev", None)
            _write_json(LOCAL_PATH, data)
            return LOCAL_PATH
        base = str(base_url or "").rstrip("/")
        old = str(data.get("base_url") or "").rstrip("/")
        if old and old != base:
            # ушли на другой адрес — запоминаем, откуда пришли
            data["prev"] = {"base_url": old,
                            "token": str(data.get("token") or "")}
        data["base_url"] = base
        if token:
            data["token"] = str(token)
        _write_json(LOCAL_PATH, data)
        return LOCAL_PATH

    def reload_config(self) -> bool:
        """Перечитывает адрес/токен из config/thinking.local.json.

        Панель живёт часами и не хочет рестарта после set-url: она сверяет
        файл и подхватывает новый адрес сама. Переопределение через --url
        (base_locked) файлом не перетирается — иначе разовый запуск против
        заглушки молча уехал бы в Colab.
        """
        if getattr(self, "base_locked", False):
            return False
        import os
        local = _read_json(LOCAL_PATH)
        base = str(local.get("base_url")
                   or self.cfg.get("base_url")
                   or os.environ.get("THINKING_URL", "") or "").rstrip("/")
        token = str(local.get("token") or self.cfg.get("token")
                    or os.environ.get("THINKING_TOKEN", "") or "")
        if (base, token) == (self.base, self.token):
            return False
        self.base, self.token = base, token
        return True

    def set_token(self, token: str) -> Path:
        """Сохраняет токен в локальный конфиг и подставляет в живой клиент.

        Нужно только после перезапуска Colab, где ячейка D сгенерировала
        новый токен: панель сама берёт его из config, но старое значение
        приходится заменять вручную — этой кнопкой.
        """
        data = _read_json(LOCAL_PATH)
        # base_url здесь не трогаем (аудит C-4): адресом отвечает set_url.
        # Раньше сохранение токена из панели молча перезаписывало адрес
        # текущим — панель, запущенная с --url на заглушку, прописывала бы
        # её в конфиг навсегда.
        data["token"] = token
        _write_json(LOCAL_PATH, data)
        self.token = token
        return LOCAL_PATH

    # ------------------------------------------------------------------ #
    #  транспорт
    # ------------------------------------------------------------------ #
    def _open(self, method: str, path: str, body: Optional[dict] = None,
              timeout: Optional[float] = None, stream: bool = False):
        if not self.base:
            raise ThinkingError("base_url пуст — выполните set-url или задайте THINKING_URL")
        # Предохранители разведены (аудит AUD-05): пять обрывов длинного
        # потока событий не должны блокировать короткие REST-вызовы (чат,
        # разработка, health) на 10 минут — у каналов свои счётчики и своя
        # пауза. Живой туннель (успех любого канала) сбрасывает оба счётчика.
        gate = self._cb_stream_until if stream else self._cb_open_until
        if time.time() < gate:
            # Туннель уже доказал, что он мёртв: не ходим по нему вхолостую.
            errs = self._cb_stream_errors if stream else self._cb_errors
            raise ThinkingError(
                f"предохранитель{' потока' if stream else ''}: {errs} ошибок подряд, "
                f"пауза ещё {int(gate - time.time())} с")
        url = f"{self.base}{path}"
        payload = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
        headers = {"Accept": "text/event-stream" if stream else "application/json"}
        if payload is not None:
            headers["Content-Type"] = "application/json"
        if self.token:
            headers["X-Agent-Token"] = self.token
        attempts = int(self.retry.get("max_attempts", 3))
        backoff = float(self.retry.get("backoff_base", 1.5))
        cap = float(self.retry.get("max_backoff", 15))
        last: Optional[BaseException] = None
        for i in range(attempts):
            try:
                req = urllib.request.Request(url, data=payload, headers=headers, method=method)
                resp = urllib.request.urlopen(req, timeout=timeout or self.timeout)
                self.online = True
                self.last_ok = utcnow()
                self.last_error = ""
                self._cb_errors = 0      # успех сбрасывает счётчик ошибок
                self._cb_stream_errors = 0  # живой туннель виден обоим каналам
                return resp
            except urllib.error.HTTPError as exc:
                raw = exc.read().decode("utf-8", "replace")[:400]
                if exc.code in (401, 403, 404, 422):
                    # ретрай не поможет
                    self.online = False
                    self.last_error = f"HTTP {exc.code}"
                    raise ThinkingError(f"HTTP {exc.code}: {raw}") from exc
                if exc.code == 524:
                    # Cloudflare не дождался ответа origin за 120 с.
                    # Повтор того же не-потокового маршрута упрётся в тот же
                    # лимит (и начнёт вторую генерацию на сервере, который
                    # ещё думает над первой), поэтому падаем сразу: вызывающий
                    # уйдёт на поток, где пинги каждые 15 с держат соединение
                    # живым и 524 не бывает. Живой прогон 04.10: три ретрая
                    # по 120 с = 360 с мёртвого ожидания вместо секунды.
                    self.last_error = "HTTP 524 (origin timeout, нужен поток)"
                    raise ThinkingError(f"HTTP 524: {raw}") from exc
                last = ThinkingError(f"HTTP {exc.code}: {raw}")
            except Exception as exc:  # таймаут, DNS, обрыв туннеля
                last = exc
            wait = min(cap, backoff ** (i + 1))
            log.warning("thinking %s %s не прошёл (%s), повтор через %.1f с",
                        method, path, last, wait)
            time.sleep(wait)
        limit = int(self.cfg.get("breaker_errors", 5))
        if stream:
            # Длинный поток — отдельный канал: его неудачи копятся только у
            # потока и не трогают онлайн-статус (его ведёт REST /health).
            self._cb_stream_errors += 1
            if self._cb_stream_errors >= limit:
                pause = float(self.cfg.get("breaker_pause", 600))
                self._cb_stream_until = time.time() + pause
                log.warning("предохранитель потока открыт: %d ошибок подряд — пауза %.0f с",
                            self._cb_stream_errors, pause)
        else:
            self.online = False
            self.last_error = str(last)[:200]
            self._cb_errors += 1
            if self._cb_errors >= limit:
                pause = float(self.cfg.get("breaker_pause", 600))
                self._cb_open_until = time.time() + pause
                log.warning("предохранитель открыт: %d ошибок подряд — пауза %.0f с",
                            self._cb_errors, pause)
        raise ThinkingError(f"субагент недоступен: {last}")

    def _json(self, method: str, path: str, body: Optional[dict] = None,
              timeout: Optional[float] = None) -> Any:
        with self._open(method, path, body, timeout) as resp:
            raw = resp.read().decode("utf-8", "replace")
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ThinkingError(f"не-JSON в ответе: {raw[:200]}") from exc

    # ------------------------------------------------------------------ #
    #  API
    # ------------------------------------------------------------------ #
    def health(self, timeout: float = 3.0) -> bool:
        try:
            data = self._json("GET", "/health", timeout=min(timeout, self.connect_timeout or timeout))
            self.online = True
            self.health_data = data
            return True
        except Exception as exc:
            self.online = False
            self.last_error = str(exc)[:200]
            self.health_data = {}
            return False

    def _check_secrets(self, *parts: Any) -> None:
        blob = " ".join(str(p) for p in parts)
        if has_secret(blob):
            raise SchemaError("в запросе обнаружен секрет — в контекст/задачу "
                              "нельзя передавать токены и пароли: уберите "
                              "значение (оставьте ссылку вида os.getenv(...))")

    def plan(self, task: str, context: Optional[dict] = None,
             constraints: Optional[list[str]] = None, max_steps: int = 12,
             author: str = "agent") -> dict:
        self._check_secrets(task, json.dumps(context or {}, ensure_ascii=False),
                            json.dumps(constraints or [], ensure_ascii=False))
        t0 = time.time()
        try:
            out = self._json("POST", "/plan", {
                "task": task,
                "context": context or {},
                "constraints": constraints or [],
                "max_steps": max(1, min(30, int(max_steps))),
            }, timeout=max(self.timeout, float(self.cfg.get("plan_timeout", 360))))
        except Exception as exc:
            self.stats["errors"] += 1
            self._record("plan", task, t0, ok=False, summary=f"не ответил: {exc}"[:400],
                         author=author)
            raise
        if isinstance(out, dict):
            self._account_plan(out, task, t0, author=author)
        return out

    def _account_plan(self, plan: dict, task: str, t0: float,
                      author: str = "agent") -> None:
        """Единый учёт успешного плана — для обычного и потокового вызова.

        Раньше plan_stream не вызывал этот блок (аудит AUD-06): успешный
        план через поток числился ошибкой предыдущего вызова, а в отчёты,
        метрики и учёт токенов не попадал вовсе.
        """
        self._remember_json("plan", plan)
        self._report_plan(plan, task)
        self.stats["plans"] += 1
        self.stats["blocked_ms"] += int((time.time() - t0) * 1000)
        self._record("plan", task, t0, ok=True,
                     summary=self._plan_summary(plan), plan_id=plan.get("plan_id"),
                     author=author)

    def ask(self, question: str, max_steps: int = 4) -> dict:
        return self.plan(question, max_steps=max_steps)

    def plan_stream(self, task: str, context: Optional[dict] = None,
                    constraints: Optional[list[str]] = None, max_steps: int = 12,
                    on_event: Optional[Callable[[dict], None]] = None,
                    author: str = "agent") -> dict:
        """POST /plan/stream: отдаёт токены по мере генерации, возвращает план."""
        self._check_secrets(task, json.dumps(context or {}, ensure_ascii=False))
        t0 = time.time()
        body = {"task": task, "context": context or {},
                "constraints": constraints or [],
                "max_steps": max(1, min(30, int(max_steps)))}
        plan: dict = {}
        # Первый байт приходит сразу: сервер кладёт `: open` и шлёт пинги
        # каждые 15 с, пока модель думает. Поэтому читаем с таймаутом
        # stream_stall, а не plan_timeout: если ни байта, ни пинга не было
        # stream_stall секунд — туннель умер молча. Раньше здесь стояло
        # 360 с, и живой прогон 04.10 провисел на обрыве шесть минут,
        # а сверху упал сырой TimeoutError с трейсбеком (аудит B-5/находка
        # живого прогона).
        stall = max(float(self.cfg.get("stream_stall", 75)), 30.0)
        try:
            with self._open("POST", "/plan/stream", body, stream=True,
                            timeout=stall) as resp:
                buf: list[str] = []
                for raw in resp:
                    line = raw.decode("utf-8", "replace").rstrip("\r\n")
                    if line.startswith("data:"):
                        buf.append(line[5:].strip())
                    elif line.startswith(":"):
                        continue
                    elif line == "" and buf:
                        chunk = "".join(buf)
                        buf = []
                        try:
                            ev = json.loads(chunk)
                        except json.JSONDecodeError:
                            continue
                        if not isinstance(ev, dict):
                            continue
                        etype = ev.get("type")
                        if etype == "error":
                            raise ThinkingError(f"сервер: {ev.get('text')}")
                        if etype == "final" and str(ev.get("text", "")).strip().startswith("{"):
                            try:
                                plan = json.loads(ev["text"])
                            except json.JSONDecodeError:
                                plan = {}
                            continue
                        self._remember_ev(ev)
                        if on_event:
                            try:
                                on_event(ev)
                            except Exception as exc:
                                log.debug("обработчик потока упал: %s", exc)
        except ThinkingError:
            raise
        except (TimeoutError, socket.timeout, OSError,
                http.client.HTTPException) as exc:
            # обрыв на середине — честная ошибка, а не трейсбек из глубин
            # http.client: caller (plan_with_fallback) решит, что делать дальше
            log.warning("поток плана оборвался: %s", exc)
            raise ThinkingError(f"поток плана оборвался: {exc}") from exc
        if plan:
            self._account_plan(plan, task, t0, author=author)
            return plan
        raise ThinkingError("поток завершился без итогового плана")

    def reflect(self, plan_id: str, step_id: int, result: str,
                observation: Optional[str] = None, error: Optional[str] = None,
                timeout: Optional[float] = None, author: str = "agent") -> dict:
        self._check_secrets(result, observation or "", error or "")
        if timeout is None:
            timeout = max(self.timeout, float(self.cfg.get("reflect_timeout", 180)))
        t0 = time.time()
        try:
            out = self._json("POST", "/reflect", {
                "plan_id": plan_id,
                "step_id": int(step_id),
                "result": str(result),
                "observation": observation,
                "error": error,
            }, timeout=timeout)
        except Exception as exc:
            self.stats["errors"] += 1
            self._record("reflect", f"шаг {step_id}: {result[:300]}", t0,
                         ok=False, summary=f"не ответил: {exc}"[:400],
                         plan_id=plan_id, author=author)
            raise
        if isinstance(out, dict):
            self._remember_json("reflect", out)
            self._report_reflect(out, plan_id, step_id, result)
            self.stats["reflects"] += 1
            self._record("reflect", f"шаг {step_id}: {result[:300]}", t0, ok=True,
                         summary=f"[{out.get('status')}] {out.get('advice', '')}"[:400],
                         plan_id=plan_id, author=author)
        return out

    # ------------------------------------------------------------------ #
    #  Рефлексия по метрикам выгоды (зацикленная, с сохранением в папку)
    # ------------------------------------------------------------------ #
    def _save_reflection(self, text: str) -> Path:
        """Дописывает ответ рефлексии в logs/thinking/reflections.md."""
        path = Path(self.cfg.get("reflections_path", "logs/thinking/reflections.md"))
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(text.rstrip() + "\n\n")
        except OSError as exc:
            log.warning("не дописалась рефлексия: %s", exc)
        return path

    def reflect_metrics(self, engine: str = "subagent",
                        since: str = "") -> dict:
        """Рефлексия по метрикам выгода: смотрит на цифры за период и
        предлагает, как их улучшить.

        engine="subagent" — вопрос уходит активной модели Colab;
        engine="local" — офлайн-правила на ПК (работает без Colab).
        Отчёт кладётся в журнал отчётов (вкладка «Отчёты»), совет
        дополнительно сохраняется в папку программы (reflections.md).
        """
        since = since or self.today_since()
        b = self.benefits(since=since)
        plan_id = f"metrics-{since[:10] or 'all'}"
        metrics = (f"обращений всего {b['calls_total']} "
                   f"(агент {b['calls_agent']}, чат {b['calls_chat']}), "
                   f"ошибок {b['errors']}, заглушек {b['fallbacks']}, "
                   f"успешных {b['calls_ok']}, "
                   f"среднее время ответа {b['avg_answer_ms'] / 1000:.1f} с, "
                   f"максимум {b['max_answer_ms'] / 1000:.1f} с, "
                   f"ожидание блокировало работу {b['blocked_ms'] / 1000:.1f} с, "
                   f"в фоне {b['background_ms'] / 1000:.1f} с, "
                   f"отчётов {b['reports']}")
        t0 = time.time()
        if engine == "local" or not self.base:
            # Правила на ПК: честно считаем, что хуже всего, и советуем.
            if b["errors"]:
                status, advice = "adjust", (
                    f"Ошибок за период: {b['errors']} из {b['calls_total']}. "
                    "Чаще всего ломается связь или контракт — проверь doctor "
                    "и уменьши длину ответа (MAX_TOKENS_BY_MODEL).")
            elif b["blocked_ms"] > max(60_000, 3 * max(b["background_ms"], 1)):
                status, advice = "adjust", (
                    f"Ожидание блокировало работу {b['blocked_ms'] / 1000:.0f} с "
                    f"при {b['background_ms'] / 1000:.0f} с в фоне: гоняй "
                    "планы/рефлексии через --async, синхронные вызовы — пауза.")
            elif b["calls_total"] and b["calls_ok"] == b["calls_total"]:
                status, advice = "ok", (
                    "Все обращения успешны, заглушек нет: связь стабильна. "
                    "Следующий шаг — ускорение (T4 для 7B) и больше фоновых "
                    "вызовов, чтобы ожидание не блокировало работу.")
            else:
                status, advice = "adjust", (
                    f"Доля успешных ответов — "
                    f"{100 * b['calls_ok'] // max(b['calls_total'], 1)}%: "
                    "посмотри журнал, где обрывы, и сократи длину промта.")
            out = {"status": status, "advice": advice[:800],
                   "rationale": "локальные правила по метрикам (без Colab)",
                   "next_steps": [], "updated_goal_stack": []}
            self._report_reflect(out, plan_id, 0, metrics)
            self._record("reflect", f"метрики за период: {metrics}", t0, ok=True,
                         fallback=True,
                         summary=f"[{status}] {advice}"[:400], plan_id=plan_id)
            self.stats["reflects"] += 1
        else:
            out = self.reflect(
                plan_id, 0,
                f"МЕТРИКИ ЗА ПЕРИОД: {metrics}",
                observation=("Сравни с прошлым разом и подскажи, что изменить "
                             "в работе субагента, чтобы цифры выросли: "
                             "меньше ошибок и ожидания, больше успешных "
                             "обращений и фоновых вызовов."))
        text = (f"## {utcnow()} — рефлексия по метрикам ({engine})\n"
                f"- Период: {b['period']}\n- Метрики: {metrics}\n"
                f"- Статус: {out.get('status')}\n- Совет: {out.get('advice')}\n"
                f"- Почему: {out.get('rationale')}")
        path = self._save_reflection(text)
        return {"engine": engine, "since": since, "period": b.get("period"),
                "status": out.get("status"), "advice": out.get("advice", ""),
                "rationale": out.get("rationale", ""), "metrics": metrics,
                "saved_to": str(path), "plan_id": plan_id,
                "duration_ms": int((time.time() - t0) * 1000)}

    def cancel(self, plan_id: str) -> dict:
        return self._json("POST", f"/cancel/{plan_id}", {})

    def events(self, since: int = 0, tail: int = 0) -> dict:
        return self._json("GET", f"/events?since={int(since)}&tail={int(tail)}")

    def dump(self, name: str, timeout: float = 30.0) -> bytes:
        """Скачивает серверный файл из белого списка (логи Colab) — для архива."""
        with self._open("GET", f"/dump/{name}", timeout=timeout) as resp:
            return resp.read()

    def metrics(self) -> dict:
        return self._json("GET", "/metrics")

    def plan_with_fallback(self, task: str,
                           stream_error: Optional[BaseException] = None,
                           **kw: Any) -> tuple[dict, bool]:
        """(план, is_fallback). Никогда не бросает исключение, кроме SchemaError.

        stream_error — вызывающий уже пробовал /plan/stream и получил эту
        ошибку: её причина решает, есть ли смысл в не-потоковом маршруте
        (при обрыве транспорта его ждёт тот же лимит Cloudflare — 120 с).
        """
        t0 = time.time()
        stream_kw = {k: v for k, v in kw.items()
                     if k in ("context", "constraints", "max_steps", "author")}
        if self.cfg.get("enabled", True):
            # Поток — основной путь. Первый байт тела уходит сразу, пинги
            # каждые 15 с не дают Cloudflare дать 524, а генерация на CPU
            # спокойно живёт дольше его лимита в 120 с. Не-потоковый /plan
            # при этом гарантированно ловил 524 и сжигал 120 с впустую
            # (живой прогон 04.10: сервер додумал план за 5 с, а клиент успел
            # получить 524 и оборванный поток).
            if stream_error is None:
                try:
                    return self.plan_stream(task, **stream_kw), False
                except SchemaError:
                    raise  # секреты в задаче — не повод молча уходить в fallback
                except Exception as exc:
                    log.warning("поток плана не прошёл: %s", exc)
                    self.last_error = str(exc)[:200]
                    stream_error = exc
            if _is_read_timeout(stream_error):
                # Транспорт мёртв или обрезан прокси: обычный маршрут упрётся
                # в тот же лимит (524 = 120 с ожидания впустую) — сразу
                # локальный план. А ошибку самого сервера (404 у старого
                # ноутбука, мусорный JSON) имеет смысл отдать ему.
                log.warning("транспорт не отвечает (%s) — беру локальный план, "
                            "не-потоковый /plan не повторяю", stream_error)
            else:
                # путь для старых ноутбуков без /plan/stream и для случаев,
                # когда поток упал по причине, а не по транспорту
                try:
                    return self.plan(task, **kw), False
                except SchemaError:
                    raise  # секреты в задаче — не повод молча уходить в fallback
                except Exception as exc:
                    log.warning("plan через субагента не прошёл: %s", exc)
                    self.last_error = str(exc)[:200]
        if self.cfg.get("fallback_on_error", True):
            plan = local_plan(task)
            self._report_plan(plan, task)
            self._record("plan", task, t0, ok=True, fallback=True,
                         summary=(f"Colab недоступен — отдан шаблон-план "
                                  f"«{plan.get('goal', task[:60])}» "
                                  f"из {len(plan.get('steps') or [])} шагов"))
            return plan, True
        raise ThinkingError("субагент недоступен и fallback отключён в конфиге")

    # ------------------------------------------------------------------ #
    #  живой поток (SSE)
    # ------------------------------------------------------------------ #
    def start_stream(self, on_event: Callable[[dict], None]) -> None:
        if self._reader and self._reader.is_alive():
            return
        self._stop.clear()
        self._reader = threading.Thread(target=self._stream_loop,
                                        args=(on_event,), daemon=True)
        self._reader.start()

    def stop_stream(self) -> None:
        self._stop.set()

    def _stream_loop(self, on_event: Callable[[dict], None]) -> None:
        delay = float(self.cfg.get("reconnect_delay", 5))
        # Таймаут чтения обязан быть больше интервала ping сервера (25 с):
        # иначе «тихий» поток всегда рвётся и не доживает до следующего события.
        read_to = max(30.0, float(self.cfg.get("stream_timeout", 40)))
        while not self._stop.is_set():
            try:
                with self._open("GET", "/events/stream",
                                timeout=read_to, stream=True) as resp:
                    self.stream_ok = True
                    self.stream_error = ""
                    buf: list[str] = []
                    for raw in resp:
                        if self._stop.is_set():
                            break
                        line = raw.decode("utf-8", "replace").rstrip("\r\n")
                        if line.startswith("data:"):
                            buf.append(line[5:].strip())
                        elif line.startswith(":"):
                            continue
                        elif line == "" and buf:
                            self._dispatch("".join(buf), on_event)
                            buf = []
                    if buf:
                        self._dispatch("".join(buf), on_event)
            except Exception as exc:                          # noqa: BLE001
                self.stream_ok = False
                self.stream_error = str(exc)[:200]
                # Онлайн-статус ведёт REST /health: обрыв SSE не должен
                # объявлять субагента офлайновым — иначе панель мигает «НЕТ СВЯЗИ».
                if _is_read_timeout(exc):
                    log.debug("поток мыслей: тишина дольше %s с, переподключаюсь", read_to)
                else:
                    log.warning("поток мыслей оборвался: %s", exc)
                time.sleep(delay)

    def _dispatch(self, chunk: str, on_event: Callable[[dict], None]) -> None:
        try:
            ev = json.loads(chunk)
        except json.JSONDecodeError:
            return
        if not isinstance(ev, dict):
            return
        self._remember_ev(ev)
        try:
            on_event(ev)
        except Exception as exc:
            log.debug("обработчик события упал: %s", exc)

    def sync_history(self, tail: int = 100) -> list[dict]:
        """Дотягивает историю по REST — работает даже когда SSE недоступен."""
        try:
            data = self.events(tail=tail)
        except Exception as exc:
            log.warning("sync_history: %s", exc)
            return []
        out = data.get("events") or []
        for ev in out:
            if isinstance(ev, dict):
                self._remember_ev(ev)
        return out

    # ------------------------------------------------------------------ #
    #  буфер панели + журнал
    # ------------------------------------------------------------------ #
    def _remember_json(self, kind: str, data: Any) -> None:
        text = json.dumps(data, ensure_ascii=False, default=str)
        # seq — метка времени в миллисекундах: два события в одну
        # миллисекунду (план и рефлексия подряд) совпадали, и второе молча
        # выбрасывалось дедупликацией по seq (аудит C-2). Счётчик поверх
        # метки коллизию ловит и двигает номер.
        with self._lock:
            self._seq_local = max(int(time.time() * 1000), self._seq_local + 1)
            seq = self._seq_local
        self._remember_ev({"seq": seq, "type": kind,
                           "text": redact_secrets(text)[:4000], "ts": utcnow()})

    def _remember_ev(self, ev: dict) -> None:
        seq = 0
        try:
            seq = int(ev.get("seq") or 0)
        except Exception:
            seq = 0
        with self._lock:
            # Окно дедупа ≥ хвоста опроса (50) и начального батча SSE (50):
            # иначе каждая перезагрузка истории дописывает одни и те же
            # события в буфер и в thoughts.jsonl (журнал раздувался в разы).
            if seq and any(e.get("seq") == seq for e in self._events[-200:]):
                return
            self._seq = max(self._seq, seq)
            ev["text"] = redact_secrets(str(ev.get("text") or ""))
            self._events.append(ev)
            if len(self._events) > 500:
                del self._events[:-500]
        self._append_log(ev)

    @property
    def last_seq(self) -> int:
        with self._lock:
            return self._seq

    @property
    def buffered(self) -> list[dict]:
        with self._lock:
            return list(self._events)

    def _append_log(self, ev: dict) -> None:
        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            if self._log_path.exists() and self._log_path.stat().st_size > self._log_max:
                self._log_path.replace(self._log_path.with_name(self._log_path.name + ".1"))
            with open(self._log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(ev, ensure_ascii=False, default=str) + "\n")
        except Exception as exc:
            log.debug("журнал не дописался: %s", exc)

    def iter_thoughts(self) -> Iterator[dict]:
        """Блокирующий итератор событий (для --follow)."""
        seen = 0
        while not self._stop.is_set():
            with self._lock:
                batch = self._events[seen:]
                seen = len(self._events)
            for ev in batch:
                yield ev
            time.sleep(0.25)

    # ------------------------------------------------------------------ #
    #  состояние для панели
    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    #  отчёты, история взаимодействия и выгода (для веб-интерфейса)
    # ------------------------------------------------------------------ #
    ACTION_RU = {
        "build": "сборка", "test": "тесты", "refactor": "рефакторинг",
        "verify": "проверка", "docs": "документация", "prompt": "промт",
        "debug": "отладка", "release": "релиз",
    }

    @staticmethod
    def _plan_summary(plan: dict) -> str:
        steps = plan.get("steps") or []
        first = steps[0].get("desc") if steps else "—"
        return (f"План из {len(steps)} шагов «{plan.get('goal', '')}». "
                f"Первый шаг: {first}")

    @staticmethod
    def _ru_action(tag: Any) -> str:
        return ThinkingClient.ACTION_RU.get(str(tag), "шаг")

    def _record(self, kind: str, request: str, t0: float, ok: bool,
                summary: str, plan_id: Optional[str] = None,
                fallback: bool = False, background: bool = False,
                author: str = "agent") -> None:
        who = str(author or "agent")[:20]
        rec = {
            "n": len(self.interactions) + 1, "rid": new_rid(),
            "kind": kind,
            "at": utcnow(),
            "author": who,
            "request": request[:700],
            "duration_ms": int((time.time() - t0) * 1000),
            "ok": bool(ok),
            "fallback": bool(fallback),
            "background": bool(background),
            "source": "local-fallback" if fallback else "colab",
            "plan_id": plan_id or "",
            "summary": summary[:900],
        }
        self.interactions.append(rec)
        if len(self.interactions) > 200:
            del self.interactions[:-200]
        if fallback:
            self.stats["fallbacks"] += 1
        _append_jsonl(self._inter_path, rec)
        # в живой ленте сразу видно, чей это вызов — человек или агент
        who_ru = "человек" if who == "human" else "агент"
        self._append_log({"seq": int(time.time() * 1000), "type": "interaction",
                          "text": f"[{kind}/{who_ru}] {request[:200]} -> {summary[:300]}",
                          "ts": rec["at"]})

    def _report_plan(self, plan: dict, task: str) -> None:
        steps = []
        for s in plan.get("steps") or []:
            steps.append({
                "id": s.get("id"),
                "action_ru": self._ru_action(s.get("action")),
                "desc": s.get("desc"),
                "expected": s.get("expected_output") or "",
                "depends": s.get("depends_on") or [],
            })
        rep = {
            "n": len(self.reports) + 1, "rid": new_rid(),
            "type": "plan",
            "at": utcnow(),
            "plan_id": str(plan.get("plan_id", ""))[:8],
            "source": plan.get("source", "colab"),
            "fallback": plan.get("source") == "local-fallback",
            "task": task[:700],
            "goal": plan.get("goal", ""),
            "rationale": plan.get("rationale", ""),
            "confidence": plan.get("confidence"),
            "sub_goals": plan.get("sub_goals") or [],
            "contradictions": plan.get("contradictions") or [],
            "success_criteria": plan.get("success_criteria") or [],
            "unknown_files": plan.get("unknown_files") or [],
            "steps": steps,
        }
        self.reports.append(rep)
        if len(self.reports) > 60:
            del self.reports[:-60]
        _append_jsonl(self._rep_path, rep)

    def _report_dev(self, proposal: dict, message: str) -> None:
        """Отчёт о предложении субагента в режиме разработчика.

        Основная работа субагента — правки кода. Без этого отчёта вкладка
        «Отчёты» пустует, даже когда dev-режим активно используется.
        """
        rep = {
            "n": len(self.reports) + 1, "rid": new_rid(),
            "type": "dev",
            "at": utcnow(),
            "source": proposal.get("source", "colab"),
            "fallback": proposal.get("source") == "local-fallback",
            "task": message[:700],
            "goal": f"[{proposal.get('action', 'none')}] {proposal.get('filename', '')}",
            "rationale": proposal.get("comment", ""),
            "filename": proposal.get("filename", ""),
            "action": proposal.get("action", "none"),
            "code": proposal.get("code", "")[:4000],
        }
        self.reports.append(rep)
        if len(self.reports) > 60:
            del self.reports[:-60]
        _append_jsonl(self._rep_path, rep)

    def _report_reflect(self, out: dict, plan_id: str, step_id: int, result: str) -> None:
        rep = {
            "n": len(self.reports) + 1, "rid": new_rid(),
            "type": "reflect",
            "at": utcnow(),
            "plan_id": str(plan_id)[:8],
            "step_id": step_id,
            "task": f"Рефлексия шага {step_id}: {result[:500]}",
            "status": out.get("status", "ok"),
            "rationale": out.get("rationale", ""),
            "advice": out.get("advice", ""),
            "steps": [{"id": s.get("id"),
                       "action_ru": self._ru_action(s.get("action")),
                       "desc": s.get("desc"),
                       "expected": s.get("expected_output") or "",
                       "depends": s.get("depends_on") or []}
                      for s in out.get("next_steps") or []],
            "goal_stack": out.get("updated_goal_stack") or [],
        }
        self.reports.append(rep)
        if len(self.reports) > 60:
            del self.reports[:-60]
        _append_jsonl(self._rep_path, rep)

    # ------------------------------------------------------------------ #
    #  ЧАТ: человек ↔ субагент, с памятью и учётом токенов
    # ------------------------------------------------------------------ #
    def multi_chat(self, message: str, use_memory: bool = False,
                   timeout: Optional[float] = None,
                   author: str = "agent") -> dict:
        """Один вопрос — всем моделям сразу, ответы рядом.

        Дополнительные движки поднимает ячейка D (THINKING_PARALLEL) и
        описывает в манифесте; сервер опрашивает их параллельно через
        asyncio.gather. Если дополнительных моделей нет, отвечает только
        основная — маршрут не ломается, а просто возвращает один ответ.

        Скорость не растёт: ядра те же, поэтому каждый ответ дольше примерно
        в N раз. Ценность в сравнении, а не в темпе.
        """
        text = str(message or "").strip()
        if not text:
            raise SchemaError("пустое сообщение")
        self._check_secrets(text)
        body: dict[str, Any] = {"message": text[:2000]}
        if use_memory:
            mem = self.memory()
            body["profile"] = str(mem.get("profile") or "")[:800]
            body["context"] = {"memory": {"facts": mem.get("facts", [])[-12:]}}
        read_to = timeout or max(self.timeout,
                                 float(self.cfg.get("chat_json_timeout", 300)),
                                 # N моделей делят ядра: ждать дольше
                                 float(self.cfg.get("multi_timeout", 900)))
        t0 = time.time()
        out = self._json("POST", "/ask/multi", body, timeout=read_to)
        answers = out.get("answers") or []
        if not isinstance(answers, list):
            raise SchemaError(f"/ask/multi: answers — не список: {type(answers).__name__}")
        result = {"answers": answers,
                  "count": int(out.get("count") or len(answers)),
                  "wall_seconds": float(out.get("wall_seconds") or 0.0),
                  "note": str(out.get("note") or "")}
        # маршрут жжёт токены N моделей разом, поэтому обязан учитываться так
        # же, как обычный чат (аудит B-3)
        self._account_multi(text, result, t0, author=author)
        return result

    def _account_multi(self, question: str, out: dict, t0: float,
                       author: str = "agent") -> None:
        """Учёт параллельного вопроса: история, «Связь агентов», «Токены».

        Раньше /ask/multi не попадал ни в историю, ни в «Выгоду», ни в
        «Токены», хотя промт уезжает в каждую модель — заметная цифра.
        В историю кладётся ответ основной модели (первый успешный),
        токены считаются суммарно по всем ответам: промт × N входов.
        """
        answers = [a for a in (out.get("answers") or [])
                   if isinstance(a, dict)]
        ok_ones = [a for a in answers if a.get("ok") and str(a.get("answer") or "").strip()]
        if not ok_ones:
            self._record_chat(question, "", t0, fallback=False,
                              error=str(out.get("note") or "ни одна модель не ответила"),
                              author=author)
            return
        main = str(ok_ones[0].get("answer") or "")
        tin = estimate_tokens(question) * max(1, len(answers))
        tout = sum(estimate_tokens(str(a.get("answer") or ""))
                   for a in ok_ones)
        reply = ChatReply.from_dict(
            {"reply": main, "at": utcnow(), "fallback": False,
             "tokens_in": tin, "tokens_out": tout, "tokens_estimate": True,
             "duration_ms": int((time.time() - t0) * 1000)})
        self._remember_chat(question, reply, author=author)
        self.stats["chats"] += 1
        self._record_chat(question, main, t0, fallback=False,
                          tokens_in=tin, tokens_out=tout, estimate=True,
                          author=author)
        self.token_stats.record(reply)

    def parallel_models(self) -> dict:
        """Какие модели держатся параллельно прямо сейчас."""
        try:
            out = self._json("GET", "/parallel", None, timeout=30)
        except ThinkingError:
            return {"active": "", "extra": [], "count": 0}
        extra = out.get("extra")
        return {"active": str(out.get("active") or ""),
                "extra": extra if isinstance(extra, list) else [],
                "count": int(out.get("count") or 0)}

    def chat(self, message: str, use_memory: bool = True,
             max_steps: int = 4, timeout: Optional[float] = None,
             author: str = "agent") -> dict:
        """Живой диалог с субагентом.

        Сначала пробует серверный /chat (в нём есть память и честный usage).
        Если сервер его ещё не знает (старый Colab) — тот же вопрос уходит
        через /plan, а ответ собирается из полей плана. Фолбэк тот же, что у
        plan_with_fallback: при офлайне отдаётся честная заглушка.
        """
        text = str(message or "").strip()
        if not text:
            raise SchemaError("пустое сообщение")
        self._check_secrets(text)
        t0 = time.time()
        mem = self.memory()
        turns = mem.get("turns", []) if use_memory else []
        context: dict[str, Any] = {}
        if use_memory:
            # память идёт в контексте: субагент видит, что помнит о проекте
            # последние факты важнее старых — в промт уходят они
            context["memory"] = {"facts": mem.get("facts", [])[-12:],
                                 "recent": [_turn_line(t) for t in
                                            turns[-6:]]}
            context["dialog"] = True
        body = {"message": text[:2000], "context": context,
                "max_steps": max(1, min(8, int(max_steps)))}
        if use_memory and mem.get("profile"):
            body["profile"] = str(mem["profile"])[:800]

        out: dict = {}
        # Медленная модель (7B на CPU считает ~1 ток/с) и «окна» на пути до
        # туннеля: не-потоковый чат ждём дольше обычного 45-секундного лимита,
        # иначе честный ответ превращается в заглушку «субагент недоступен».
        read_to = timeout or max(self.timeout,
                                 float(self.cfg.get("chat_json_timeout", 300)))
        try:
            out = self._json("POST", "/chat", body, timeout=read_to)
        except ThinkingError as exc:
            if _looks_like_missing_route(exc):
                # сервер без /chat (старый Colab) — не ошибка, а повод сработать
                out = {}
            else:
                # Colab упал: чат не должен ломать работу — отдаём честную
                # заглушку, как это делает plan_with_fallback.
                self.stats["errors"] += 1
                reply = ChatReply(
                    reply=("Субагент сейчас недоступен — я не могу ответить по-настоящему.\n\n"
                           "Что делать: проверь Colab (python tools/thinking_cli.py doctor), "
                           "при необходимости перезапусти ноутбук и выполни set-url заново.\n"
                           "Пока что решение принимаю я сам — структура задачи не изменилась."),
                    source="local-fallback", fallback=True,
                    memory_used=use_memory and bool(mem.get("facts")),
                    # Модель не вызывалась — токенов не потрачено (AUD-22):
                    # раньше заглушка «тратила» оценку от длины сообщения.
                    tokens_in=0, tokens_out=0,
                    tokens_estimate=False)
                reply.duration_ms = int((time.time() - t0) * 1000)
                self._remember_chat(text, reply, author=author)
                self.stats["chats"] += 1
                self.stats["fallbacks"] += 1
                self._record_chat(text, reply.reply, t0, fallback=True, author=author)
                self.token_stats.record(reply)
                return reply.to_dict()
        if not out:
            try:
                out = self._chat_via_plan(text, context, max_steps)
            except (ThinkingError, SchemaError):
                out = {"reply": "", "source": "local-fallback", "fallback": True,
                       "tokens_in": 0, "tokens_out": 0, "tokens_estimate": False}
        reply = ChatReply.from_dict(out)
        reply.duration_ms = int((time.time() - t0) * 1000)
        if not reply.at:
            reply.at = utcnow()
        if use_memory:
            reply.memory_used = bool(mem.get("facts") or turns)
        if not reply.reply:
            reply.reply = "Субагент промолчал — задача не разобрана."
        self._remember_chat(text, reply, author=author)
        if use_memory and not reply.fallback:
            self._safe_remember_turn(text, reply)
        self.stats["chats"] += 1
        self._record_chat(text, reply.reply, t0, fallback=reply.fallback,
                          plan_id=reply.plan_id, tokens_in=reply.tokens_in,
                          tokens_out=reply.tokens_out,
                          estimate=reply.tokens_estimate, author=author)
        self.token_stats.record(reply)
        return reply.to_dict()

    def models(self) -> dict:
        """Какие модели лежат в Colab и какая активна: {active, models: [...]}.

        Пустой словарь, если сервер старый и маршрута /models ещё нет.
        """
        try:
            data = self._json("GET", "/models", None,
                              timeout=min(self.timeout, 20))
        except ThinkingError:
            return {}
        return data if isinstance(data, dict) else {}

    def set_model(self, model: str) -> dict:
        """Переключает модель в рантайме (панель зовёт это напрямую).

        model — путь к .gguf либо короткое имя: "3b", "1.5B", "7b".
        """
        want = str(model or "").strip()
        if not want:
            raise SchemaError("не указана модель")
        out = self._json("POST", "/model", {"model": want},
                         timeout=max(self.timeout, 300))
        if not isinstance(out, dict):
            raise SchemaError(f"сервер вернул не словарь: {type(out).__name__}")
        return out

    # ------------------------------------------------------------------ #
    #  чтение SSE-потока с повтором при обрыве
    # ------------------------------------------------------------------ #
    def _stream_read(self, path: str, body: dict,
                     on_token: Optional[Callable[[str], None]] = None,
                     on_retry: Optional[Callable[[int, int], None]] = None,
                     label: str = "поток") -> tuple[list[str], dict]:
        """Читает поток events и повторяет запрос при обрыве.

        Беспроводной путь до туннеля Cloudflare местами замирает посреди
        ответа: сервер доделывает генерацию (это видно в журнале событий),
        а клиенту байты не приходят и EOF не закрывается. Повтор почти
        всегда проходит, поэтому: read-timeout stream_stall секунд на
        чтение, затем до stream_retries повторных попыток (всего —
        1 + stream_retries), пока не придёт done.

        on_retry(attempt, total) вызывается перед каждой повторной
        попыткой — интерфейс может сказать человеку «повторяю…».
        Ошибки сервера/схемы (ThinkingError) не повторяются.
        """
        stall = float(self.cfg.get("stream_stall", 75))
        budget = max(stall * 2, float(self.cfg.get("chat_timeout", 600)))
        total = 1 + max(0, int(self.cfg.get("stream_retries", 2)))
        t0 = time.time()
        last: Optional[BaseException] = None
        for attempt in range(1, total + 1):
            pieces: list[str] = []
            done: dict = {}
            try:
                with self._open("POST", path, body, stream=True,
                                timeout=stall) as resp:
                    for raw in resp:
                        line = raw.decode("utf-8", "replace").strip()
                        if not line.startswith("data:"):
                            continue
                        try:
                            ev = json.loads(line[5:].strip())
                        except json.JSONDecodeError:
                            continue
                        if not isinstance(ev, dict):
                            continue
                        etype = ev.get("type")
                        if etype == "token":
                            piece = str(ev.get("text") or "")
                            if piece:
                                pieces.append(piece)
                                if on_token:
                                    try:
                                        on_token(piece)
                                    except Exception as exc:  # обработчик не роняет поток
                                        log.debug("on_token упал: %s", exc)
                        elif etype == "error":
                            raise ThinkingError(f"сервер: {ev.get('text')}")
                        elif etype == "done":
                            done = ev
                            break   # ответ получен: не ждём EOF (keep-alive)
            except ThinkingError:
                raise               # ошибки сервера/схемы не повторяем
            except (OSError, http.client.IncompleteRead) as exc:
                last = exc          # таймаут чтения, обрыв соединения
            if done:
                return pieces, done
            if last is None:
                last = ThinkingError(f"{label} завершился без ответа")
            if attempt >= total or time.time() - t0 > budget:
                break
            log.warning("%s %s оборвался (%s), повтор %d/%d",
                        label, path, last, attempt, total - 1)
            if on_retry:
                try:
                    on_retry(attempt, total)
                except Exception as exc:
                    log.debug("on_retry упал: %s", exc)
            last = None
        raise ThinkingError(f"{label} оборвался после {total} попыток: {last}")

    def chat_stream(self, message: str, on_token: Optional[Callable[[str], None]] = None,
                     author: str = "agent",
                    use_memory: bool = True, max_steps: int = 4,
                    timeout: Optional[float] = None,
                    on_retry: Optional[Callable[[int, int], None]] = None) -> dict:
        """POST /chat/stream: ответ приходит токенами, поэтому длинный текст
        не обрывается 120-секундным лимитом туннеля, а человек видит его сразу.

        on_token получает очередной кусок текста. Возвращает тот же словарь,
        что и chat(), поэтому вызывающий код может идти по старой схеме.
        """
        text = str(message or "").strip()
        if not text:
            raise SchemaError("пустое сообщение")
        self._check_secrets(text)
        t0 = time.time()
        mem = self.memory()
        context: dict[str, Any] = {}
        if use_memory:
            # последние факты важнее старых — в промт уходят они
            context["memory"] = {"facts": mem.get("facts", [])[-12:],
                                 "recent": [_turn_line(t) for t in mem["turns"][-6:]]}
            context["dialog"] = True
        body = {"message": text[:2000], "context": context,
                "max_steps": max(1, min(8, int(max_steps)))}
        if use_memory and mem.get("profile"):
            body["profile"] = str(mem["profile"])[:800]

        try:
            pieces, done = self._stream_read(
                "/chat/stream", body, on_token=on_token, on_retry=on_retry,
                label="поток чата")
            if not done:
                raise ThinkingError("поток чата завершился без ответа")
        except ThinkingError as exc:
            # Запасной транспорт (решение от 03.10): при обрывах «окнами»
            # короткий не-потоковый /chat проходит там, где SSE умирает.
            # Ошибки самого сервера не повторяем — он ответит тем же.
            if str(exc).startswith("сервер:"):
                raise
            log.warning("поток чата не прошёл (%s) — пробую /chat без потока", exc)
            # author пробрасываем: диалог человека, спасённый фолбэком,
            # иначе уходит в журнал как агентский (аудит B-2)
            out = self.chat(text, use_memory=use_memory, max_steps=max_steps,
                            timeout=timeout, author=author)
            out["stream_fallback"] = True     # панель покажет «без потока»
            return out
        out = dict(done)
        if not str(out.get("reply") or "").strip():
            out["reply"] = "".join(pieces).strip()
        reply = ChatReply.from_dict(out)
        if not reply.tokens_in and not reply.tokens_out:
            # usage у потока сервер не отдаёт (стрим llama.cpp его не пишет) —
            # оцениваем по длине текста, честно помечая оценкой (AUD-22):
            # иначе стриминговые чаты не входили в учёт токенов вовсе.
            reply.tokens_in = estimate_tokens(text)
            reply.tokens_out = estimate_tokens(reply.reply)
            reply.tokens_estimate = True
        reply.duration_ms = int((time.time() - t0) * 1000)
        if not reply.at:
            reply.at = utcnow()
        if use_memory:
            reply.memory_used = bool(mem.get("facts") or mem.get("turns"))
        self._remember_chat(text, reply, author=author)
        if use_memory and not reply.fallback:
            self._safe_remember_turn(text, reply)
        self.stats["chats"] += 1
        self._record_chat(text, reply.reply, t0, fallback=reply.fallback,
                          plan_id=reply.plan_id, tokens_in=reply.tokens_in,
                          tokens_out=reply.tokens_out,
                          estimate=reply.tokens_estimate, author=author)
        self.token_stats.record(reply)
        return reply.to_dict()

    # ------------------------------------------------------------------ #
    #  РЕЖИМ РАЗРАБОТЧИКА: модель работает с файлами кода на ПК
    # ------------------------------------------------------------------ #
    def _dev_body(self, message: str, files: Optional[list] = None,
                  active_name: str = "", active_code: str = "") -> dict:
        text = str(message or "").strip()
        if not text:
            raise SchemaError("пустое сообщение")
        # Активный файл уезжает в промт целиком — его тоже проверяем:
        # секрет в открытом редакторе не должен уезжать в Colab (AUD-13).
        self._check_secrets(text, active_code or "")
        return {
            "message": text[:3000],
            "files": [str(f)[:200] for f in (files or [])][:80],
            "active_name": str(active_name)[:200],
            "active_code": str(active_code or "")[:8000],
        }

    @staticmethod
    def _dev_result(out: dict) -> dict:
        """Нормализует ответ сервера в предложение {action, filename, code, comment}."""
        prop = out.get("proposal") if isinstance(out.get("proposal"), dict) else out
        action = str(prop.get("action") or "none").lower()
        if action not in ("create", "edit", "none"):
            action = "none"
        return {
            "action": action,
            "filename": str(prop.get("filename") or "")[:200].strip(),
            "code": str(prop.get("code") or ""),
            "comment": str(prop.get("comment") or prop.get("reply") or "")[:1200],
            "tokens_in": int(out.get("tokens_in") or 0),
            "tokens_out": int(out.get("tokens_out") or 0),
            "tokens_estimate": bool(out.get("tokens_estimate", True)),
            "at": utcnow(),
        }

    def dev(self, message: str, files: Optional[list] = None,
            active_name: str = "", active_code: str = "",
            on_token: Optional[Callable[[str], None]] = None,
            timeout: Optional[float] = None,
            on_retry: Optional[Callable[[int, int], None]] = None,
            author: str = "agent") -> dict:
        """Запрос в режиме разработчика: модель предлагает изменение файла.

        Ничего не пишет на диск — только возвращает предложение
        {action: create|edit|none, filename, code, comment}. Применение —
        отдельное действие с подтверждением человека (контроль сохраняется).
        Сначала пробует поток /dev/stream (не рвётся 120-секундным лимитом
        туннеля), при старом сервере — обычный POST /dev.
        """
        body = self._dev_body(message, files, active_name, active_code)
        t0 = time.time()
        read_to = max(self.timeout, float(self.cfg.get("dev_timeout", 600)))
        pieces: list[str] = []
        done: dict = {}
        try:
            pieces, done = self._stream_read(
                "/dev/stream", body, on_token=on_token, on_retry=on_retry,
                label="поток разработки")
        except ThinkingError as exc:
            if str(exc).startswith("сервер:"):
                # Сервер сам ответил отказом — повторять бессмысленно.
                self.stats["errors"] += 1
                self._record("dev", body["message"], t0, ok=False,
                             summary=f"не ответил: {exc}"[:400], author=author)
                raise
            if not _looks_like_missing_route(exc):
                # Поток оборвался после попыток: короткий не-потоковый
                # POST /dev в «плохие окна» проходит там, где SSE умирает.
                log.warning("поток разработки не прошёл (%s) — пробую /dev без потока",
                            exc)
            # Нет потокового маршрута либо он не прошёл — обычный POST /dev
            try:
                out = self._json("POST", "/dev", body, timeout=timeout or read_to)
                done = out if isinstance(out, dict) else {}
            except ThinkingError as exc2:
                if _looks_like_missing_route(exc2):
                    raise ThinkingError(
                        "сервер Colab не знает режим разработчика /dev — "
                        "пересобери ноутбук ver3 и выполни ячейки C+D") from exc2
                self.stats["errors"] += 1
                self._record("dev", body["message"], t0, ok=False,
                             summary=f"не ответил: {exc2}"[:400], author=author)
                raise
        if not done:
            raise ThinkingError("режим разработчика не вернул ответа")
        proposal = self._dev_result(done)
        if proposal["action"] == "none" and not proposal["comment"] and pieces:
            proposal["comment"] = "".join(pieces)[-1200:]
        self._remember_json("dev", {"request": body["message"], **proposal})
        self._record("dev", body["message"], t0, ok=True,
                     summary=f"[{proposal['action']}] {proposal['filename']}: "
                             f"{proposal['comment']}"[:400], author=author)
        self._report_dev(proposal, body["message"])
        return proposal

    def _chat_via_plan(self, text: str, context: dict, max_steps: int) -> dict:
        """Старый путь: /chat недоступен — спрашиваем /plan и собираем ответ."""
        task = text
        if context.get("memory", {}).get("facts"):
            facts = "; ".join(str(f) for f in context["memory"]["facts"][-6:])
            task = f"{text}\nЧТО Я ПОМНЮ О ПРОЕКТЕ: {facts}"
        plan = self.plan(task, context=context, max_steps=max_steps)
        if not isinstance(plan, dict) or plan.get("source") == "local-fallback":
            # Заглушка: модель не отвечала — токенов не потрачено (AUD-22)
            return {"reply": "", "source": "local-fallback", "fallback": True,
                    "tokens_in": 0, "tokens_out": 0, "tokens_estimate": False}
        steps = plan.get("steps") or []
        lines = [str(s.get("desc") or "") for s in steps][:4]
        reply = plan.get("rationale") or plan.get("goal") or ""
        if lines:
            reply = (reply + "\n\n" if reply else "") + "\n".join(
                f"{i + 1}. {s}" for i, s in enumerate(lines))
        if plan.get("contradictions"):
            reply += "\n\n⚠ " + "; ".join(str(c) for c in plan["contradictions"][:2])
        return {"reply": reply.strip(), "rationale": plan.get("rationale", ""),
                "steps": [{"desc": s} for s in lines],
                "source": plan.get("source", "colab"),
                "plan_id": str(plan.get("plan_id", ""))[:64],
                "tokens_in": estimate_tokens(task),
                "tokens_out": estimate_tokens(reply),
                "tokens_estimate": True}

    # ------------------------------------------------------------------ #
    #  Память: факты о проекте + последние реплики чата
    # ------------------------------------------------------------------ #
    def memory(self) -> dict:
        """Читает память с диска. Формат — простой JSON, руками правится."""
        data = _read_json(self._mem_path)
        turns = [t for t in (data.get("turns") or []) if isinstance(t, dict)]
        return {"profile": str(data.get("profile") or "")[:800],
                "facts": [str(f)[:300] for f in (data.get("facts") or [])
                          if str(f).strip()][-MAX_MEMORY_FACTS:],
                "turns": turns[-MAX_MEMORY_TURNS:],
                "updated": str(data.get("updated") or "")}

    def remember(self, fact: str = "", profile: str = "") -> dict:
        """Добавляет факт (и/или описание проекта) в память.

        Дедупликация: точный дубль (после нормализации) не добавляется,
        а пересекающийся факт заменяется более полным — вместо двух почти
        одинаковых строк остаётся одна.
        """
        fact = str(fact or "").strip()
        self._check_secrets(fact, profile)
        mem = self.memory()
        if profile.strip():
            mem["profile"] = profile.strip()[:800]
        if fact:
            new = _norm_fact(fact)
            facts = list(mem["facts"])
            if not any(_norm_fact(f) == new for f in facts):
                merged = False
                for i, old in enumerate(facts):
                    o = _norm_fact(old)
                    # пересечение: одна строка целиком содержится в другой
                    if o and o != new and min(len(o), len(new)) >= 12 and \
                            (o in new or new in o):
                        if len(new) > len(o):     # оставляем более полный:
                            # сравнение идёт по нормализованным строкам —        # иначе сырой факт длиннее, а по смыслу короче
                            facts[i] = fact[:300]
                        merged = True
                        log.info("память: факт объединён с похожим — %s", fact[:80])
                        break
                if not merged:
                    facts.append(fact[:300])
                mem["facts"] = facts[-MAX_MEMORY_FACTS:]
            else:
                log.info("память: дубль не добавлен — %s", fact[:80])
        mem["updated"] = utcnow()
        self._write_memory(mem)
        return mem

    def _safe_remember_turn(self, text: str, reply) -> None:
        """Запоминает реплику в диалог, не роняя чат из-за ошибки записи."""
        try:
            self.remember_turn(text, reply.reply, plan_id=reply.plan_id)
        except OSError as exc:
            log.warning("реплика не записана в память: %s", exc)

    def remember_turn(self, question: str, answer: str,
                      fallback: bool = False, plan_id: str = "") -> None:
        mem = self.memory()
        turns = mem.get("turns", [])
        turns.append(ChatTurn(role="user", text=str(question)[:2000],
                              at=utcnow()).to_dict())
        turns.append(ChatTurn(role="subagent", text=str(answer)[:4000],
                              at=utcnow(), plan_id=str(plan_id)[:64],
                              fallback=bool(fallback)).to_dict())
        overflow = turns[:-MAX_MEMORY_TURNS]
        mem["turns"] = turns[-MAX_MEMORY_TURNS:]
        if overflow:
            # Старые реплики не выбрасываем молча: сжимаем в одно саммари в
            # конце facts — оно уходит в промт (последние 12 фактов).
            mem["facts"] = self._summarize_turns(mem.get("facts") or [], overflow)
        mem["updated"] = utcnow()
        self._write_memory(mem)

    @staticmethod
    def _summarize_turns(facts: list[str], overflow: list[dict]) -> list[str]:
        """Сжимает выпавшие из окна реплики в одно саммари-факт (≤300 знаков).

        Саммари одно: старое дописывается в начало нового, а хвост (свежее)
        остаётся, пока не упрётся в лимит факта.
        """
        prefix = "Саммари диалога: "
        parts = []
        for t in overflow:
            txt = " ".join(str(t.get("text") or "").split())
            if not txt:
                continue
            who = "я" if t.get("role") == "user" else "субагент"
            parts.append(f"{who}: {txt[:70]}")
        if not parts:
            return facts
        rest = [f for f in facts if not f.startswith(prefix)]
        old = [f for f in facts if f.startswith(prefix)]
        merged = ((old[-1][len(prefix):] + " | ") if old else "") + "; ".join(parts)
        room = 300 - len(prefix) - 1
        if len(merged) > room:
            merged = "…" + merged[-room:]
        return (rest + [prefix + merged])[-MAX_MEMORY_FACTS:]

    def forget(self, facts: bool = True, turns: bool = True) -> dict:
        """Стирает память по частям — чтобы не терять контекст целиком."""
        mem = self.memory()
        if facts:
            mem["facts"] = []
        if turns:
            mem["turns"] = []
        mem["updated"] = utcnow()
        self._write_memory(mem)
        return mem

    def _write_memory(self, mem: dict) -> None:
        _write_json(self._mem_path, mem)

    def _remember_chat(self, question: str, reply: ChatReply,
                       cached: bool = False, author: str = "agent") -> None:
        # author: кто спрашивал — "human" (панель) или "agent" (агент через CLI).
        # Без этого в истории не отличить мой вопрос от вашего: оба лежали
        # рядом с подписью «я», и вкладка «Связь агентов» вводила в
        # заблуждение, чей это был вызов.
        self.chat_log.append({"at": reply.at, "question": question[:2000],
                              "reply": reply.reply[:4000],
                              "author": str(author or "human")[:20],
                              "fallback": reply.fallback,
                              "tokens_in": reply.tokens_in,
                              "tokens_out": reply.tokens_out,
                              "tokens_estimate": reply.tokens_estimate,
                              "cached": bool(cached),
                              "duration_ms": reply.duration_ms})
        if len(self.chat_log) > 60:
            del self.chat_log[:-60]
        _append_jsonl(self._chat_path, dict(self.chat_log[-1]))

    def _record_chat(self, question: str, reply: str, t0: float,
                     fallback: bool, error: str = "", plan_id: str = "",
                     tokens_in: int = 0, tokens_out: int = 0,
                     estimate: bool = True, author: str = "agent") -> None:
        who = str(author or "human")[:20]
        rec = {"n": len(self.interactions) + 1, "rid": new_rid(), "kind": "chat", "at": utcnow(),
               "request": question[:700],
               "author": who,
               "duration_ms": int((time.time() - t0) * 1000),
               "ok": bool(reply) or bool(error), "fallback": bool(fallback),
               "background": False, "source": "local-fallback" if fallback else "colab",
               "plan_id": plan_id or "",
               "summary": (f"чат: {reply[:400]}" if reply else f"ошибка чата: {error}")[:900],
               "tokens_in": int(tokens_in), "tokens_out": int(tokens_out),
               "tokens_estimate": bool(estimate)}
        self.interactions.append(rec)
        if len(self.interactions) > 200:
            del self.interactions[:-200]
        if fallback:
            self.stats["fallbacks"] += 1
        _append_jsonl(self._inter_path, rec)

    # ------------------------------------------------------------------ #
    #  Период: «за сегодня» (по умолчанию) или «за всё время»
    # ------------------------------------------------------------------ #
    @staticmethod
    def _after(rec: dict, since: str) -> bool:
        """Попадает ли запись в выбранный период.

        `since` присылает панель — локальная полночь, переведённая в UTC
        (ISO). Сравниваем по первым 19 символам («ГГГГ-ММ-ДДЧЧ:ММ:СС»),
        поэтому форматы `+00:00` у записей и `.000Z` у JS не мешают.
        """
        if not since:
            return True
        return str(rec.get("at") or "")[:19] >= str(since)[:19]

    @staticmethod
    def today_since() -> str:
        """Локальная полночь текущих суток в UTC-ISO (для фильтра «сегодня»)."""
        from datetime import datetime, timezone
        local_mid = datetime.now().astimezone().replace(
            hour=0, minute=0, second=0, microsecond=0)
        return local_mid.astimezone(timezone.utc).isoformat(timespec="seconds")

    # ------------------------------------------------------------------ #
    #  Учёт токенов
    # ------------------------------------------------------------------ #
    def tokens(self, since: str = "") -> dict:
        """Сводка по токенам: из чатов (точные или оценка) + оценка по планам.

        since — начало периода (см. `_after`): панель по умолчанию спрашивает
        «за сегодня», кнопка «за всё время» передаёт пустую строку.
        """
        inter, reps = self._load_history()
        if since:
            inter = [i for i in inter if self._after(i, since)]
            reps = [r for r in reps if self._after(r, since)]
        calls = []
        tin = tout = 0
        exact = 0
        for i in inter:
            if not i.get("ok") and not i.get("tokens_in"):
                continue
            a = int(i.get("tokens_in") or 0)
            b = int(i.get("tokens_out") or 0)
            if a or b:
                calls.append({"at": i.get("at", ""), "kind": i.get("kind", ""),
                              "in": a, "out": b,
                              "estimate": bool(i.get("tokens_estimate", True))})
                tin += a
                tout += b
                if not i.get("tokens_estimate", True):
                    exact += 1
        # планы и рефлексии: сервер их usage не отдаёт — считаем по тексту
        for r in reps:
            if r.get("fallback"):
                continue
            text = json.dumps(r, ensure_ascii=False)
            a, b = estimate_tokens(text), estimate_tokens(text[:1500])
            calls.append({"at": r.get("at", ""), "kind": r.get("type", ""),
                          "in": a, "out": b, "estimate": True})
            tin += a
            tout += b
        return {"tokens_in": tin, "tokens_out": tout, "tokens_total": tin + tout,
                "calls": len(calls), "exact_calls": exact,
                "estimate_only": exact == 0,
                "note": ("оценка по длине текста: сервер не отдаёт usage"
                         if exact == 0 else
                         f"точные цифры только у {exact} вызовов, остальное — оценка"),
                "history": calls[-40:]}

    # ------------------------------------------------------------------ #
    #  Удаление записей (кнопки в панели)
    # ------------------------------------------------------------------ #
    def delete_history(self, kind: str = "report", n: Optional[int] = None,
                       before: str = "", rid: str = "", at: str = "") -> int:
        """Удаляет из журнала отчётов/диалогов: по rid, по номеру+времени,
        по времени или всё.

        rid — уникальный идентификатор записи. Поле `n` уникальным НЕ является
        (это счётчик внутри процесса, у каждого процесса он начинается с 1),
        поэтому удаление только по `n` сносило лишнее. Поэтому: если задан rid —
        удаляем ровно одну запись; иначе по `n` — только совпадение n И времени.
        """
        if kind not in ("report", "dialog", "chat", "all"):
            raise SchemaError(f"неизвестный раздел: {kind}")
        targets: list[Path] = []
        if kind in ("report", "all"):
            targets.append(self._rep_path)
        if kind in ("dialog", "all"):
            targets.append(self._inter_path)
        if kind in ("chat", "all"):
            targets.append(self._chat_path)
        removed = 0
        for path in targets:
            if not path.exists():
                continue
            kept: list[str] = []
            for line in _read_jsonl(path, 100000):
                hit = False
                if rid:
                    hit = str(line.get("rid") or "") == rid
                elif n is not None:
                    # номер + метка времени: номер сам по себе не уникален
                    hit = (int(line.get("n", -1)) == int(n)
                           and str(line.get("at", "")) == str(at))
                elif before:
                    hit = str(line.get("at", "")) < str(before)
                else:
                    hit = True                      # чистка раздела целиком
                if hit:
                    removed += 1
                else:
                    kept.append(json.dumps(line, ensure_ascii=False, default=str))
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
            tmp.replace(path)
        if rid:
            self.reports = [r for r in self.reports if r.get("rid") != rid]
            self.interactions = [r for r in self.interactions if r.get("rid") != rid]
        self._disk_cache = (None, [], [])
        return removed

    # ------------------------------------------------------------------ #
    #  история из памяти + диска (вызовы идут из других процессов)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _key(rec: dict) -> tuple:
        return (rec.get("at", ""), rec.get("kind") or rec.get("type", ""),
                str(rec.get("request") or rec.get("task") or "")[:120])

    def _load_history(self) -> tuple[list[dict], list[dict]]:
        """Возвращает (взаимодействия, отчёты), объединяя память и JSONL."""
        try:
            mt = (self._inter_path.stat().st_mtime if self._inter_path.exists() else 0.0,
                  self._rep_path.stat().st_mtime if self._rep_path.exists() else 0.0)
        except OSError:
            mt = (0.0, 0.0)
        if self._disk_cache[0] == mt:
            inter, reps = self._disk_cache[1], self._disk_cache[2]
        else:
            inter = [r for r in _read_jsonl(self._inter_path, 400) if "kind" in r]
            reps = [r for r in _read_jsonl(self._rep_path, 200) if "type" in r]
            seen = {self._key(r) for r in inter}
            for r in self.interactions:
                if self._key(r) not in seen:
                    inter.append(r)
                    seen.add(self._key(r))
            seen_r = {self._key(r) for r in reps}
            for r in self.reports:
                if self._key(r) not in seen_r:
                    reps.append(r)
                    seen_r.add(self._key(r))
            inter.sort(key=lambda r: r.get("at", ""))
            reps.sort(key=lambda r: r.get("at", ""))
            self._disk_cache = (mt, inter, reps)
        # обогащаем статистикой процесса (fallback/фон видны только в памяти)
        return list(inter), list(reps)

    def benefits(self, since: str = "") -> dict:
        """Честная оценка того, что дали субагенты (без прикрас).

        since — начало периода: панель по умолчанию считает «за сегодня».
        """
        inter, reps = self._load_history()
        if since:
            inter = [i for i in inter if self._after(i, since)]
            reps = [r for r in reps if self._after(r, since)]
        total = len(inter)
        ok = sum(1 for i in inter if i.get("ok"))
        # Обращения агента (план/рефлексия/разработка) и обращения человека
        # через окно чата считаются раздельно: у них разная цена ожидания.
        calls_agent = sum(1 for i in inter
                          if i.get("kind") in ("plan", "reflect", "dev"))
        calls_chat = sum(1 for i in inter if i.get("kind") == "chat")
        fb = sum(1 for i in inter if i.get("fallback"))
        plans = sum(1 for i in inter if i.get("kind") == "plan" and i.get("ok"))
        reflects = sum(1 for i in inter if i.get("kind") == "reflect" and i.get("ok"))
        blocked = sum(i.get("duration_ms", 0) for i in inter if not i.get("background"))
        bg = sum(i.get("duration_ms", 0) for i in inter if i.get("background"))
        if not inter and not since:
            # Журнала ещё нет (свежий процесс/удалённый файл) — берём
            # счётчики процесса. Раньше они прибавлялись к сумме по журналу
            # всегда, и «за всё время» считало каждый фолбэк дважды (AUD-07).
            fb = self.stats["fallbacks"]
            bg = self.stats["background_ms"]
        avg = (sum(i.get("duration_ms", 0) for i in inter) // total) if total else 0
        steps_total = sum(len(r.get("steps") or []) for r in reps
                          if r.get("type") == "plan")
        slowest = max((i.get("duration_ms", 0) for i in inter), default=0)
        fastest = min((i.get("duration_ms", 0) for i in inter), default=0)
        return {
            "calls_total": total,
            "calls_agent": calls_agent,
            "calls_chat": calls_chat,
            "calls_ok": ok,
            "plans": plans,
            "reflects": reflects,
            "fallbacks": fb,
            "errors": total - ok,
            "steps_generated": steps_total,
            "avg_answer_ms": avg,
            "min_answer_ms": fastest,
            "max_answer_ms": slowest,
            "blocked_ms": blocked,
            "background_ms": bg,
            "reports": len(reps),
            "period": "с сегодняшнего утра" if since else "за всё время",
            "notes": [
                (f"Обращений за период: всего {total} — агент просил субагента "
                 f"{calls_agent} раз(а) (план/рефлексия/код), а человек написал "
         f"в чат {calls_chat} раз(а). Это разные очереди: у агента "
                 f"ожидание блокирует работу, у чата — просто диалог."),
                (f"Средний ответ субагента — {avg / 1000:.1f} с "
                 f"(от {fastest / 1000:.1f} до {slowest / 1000:.1f} с). За это время "
                 f"основной агент успевает сделать несвязанную работу, поэтому "
                 f"выигрыш — только если гонять запросы параллельно."),
                (f"Ожидание заблокировало работу на {blocked / 1000:.1f} с, "
                 f"выполнено в фоне {bg / 1000:.1f} с — только фон даёт чистое "
                 f"ускорение, синхронные вызовы его не дают."),
                (f"Собрано {steps_total} шагов плана и {reflects} рефлексий: это "
                 f"готовая структура (цель, порядок, критерии, ожидаемые артефакты), "
                 f"которую агенту не пришлось выдумывать за счёт своего контекста."),
                (f"Заглушка при недоступном Colab сработала {fb} раз(а) — "
                 f"приложение продолжало работать без остановок."),
                (f"Отчётов в журнале: {len(reps)}; ошибок контракта: "
                 f"{self.stats['errors']}."),
            ],
        }

    def status(self, since: str = "") -> dict:
        """Состояние для панели. since — начало периода «за сегодня» (см. _after)."""
        inter, reps = self._load_history()
        if since:
            inter = [i for i in inter if self._after(i, since)]
            reps = [r for r in reps if self._after(r, since)]
        chat = _read_jsonl(self._chat_path, 60)
        if since:
            chat = [c for c in chat if self._after(c, since)]
        return {
            "online": bool(self.online),
            "base": self.base,
            "has_token": bool(self.token),
            # сам токен наружу не отдаём (аудит B-8): панели он не нужен —
            # её запросы идут через сервер панели, который токен держит сам.
            # Для вида хватает факта наличия и последних двух символов.
            "token_tail": self.token[-2:] if self.token else "",
            "since": since,
            "last_ok": self.last_ok,
            "last_error": self.last_error,
            "stream_ok": bool(self.stream_ok),
            "stream_error": self.stream_error,
            "last_seq": self.last_seq,
            "buffered": len(self._events),
            "log_path": str(self._log_path),
            "health": getattr(self, "health_data", {}) or {},
            "interactions": inter[-80:],
            "reports": reps[-40:],
            "benefits": self.benefits(since=since),
            "memory": self.memory(),
            "chat": chat,
            "tokens": self.tokens(since=since),
        }
