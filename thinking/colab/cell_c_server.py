%%writefile /content/thinking_server.py
# === Субагент «Мышление» — API (ячейка C) ===
# FastAPI + llama.cpp/vLLM. Субагент ТОЛЬКО советует: никаких инструментов,
# вызовов API изображений и записи файлов у него нет.
# Контракты полей должны совпадать с thinking/schemas.py (ловит tools/thinking_test.py).
from __future__ import annotations

import asyncio
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from typing import Any, Optional

import httpx
from contextlib import asynccontextmanager
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------- конфиг ---
_ENV_CACHE: dict = {"mt": None, "text": ""}


def _env_text() -> str:
    """Содержимое /content/thinking_env.sh, кэшированное по mtime.

    Раньше файл открывался на КАЖДЫЙ вызов `_env_export`, а тот зовётся из
    `_ctx_limit`, который есть в каждом контентном маршруте: постоянный
    фоновый I/O посреди генерации (аудит B, 03-colab).
    """
    try:
        mt = os.path.getmtime("/content/thinking_env.sh")
    except OSError:
        _ENV_CACHE["mt"], _ENV_CACHE["text"] = None, ""
        return ""
    if _ENV_CACHE["mt"] != mt:
        try:
            with open("/content/thinking_env.sh", encoding="utf-8") as fh:
                _ENV_CACHE["text"] = fh.read()
        except OSError:
            _ENV_CACHE["text"] = ""
        _ENV_CACHE["mt"] = mt
    return _ENV_CACHE["text"]


def _env_export(key: str, default: str = "") -> str:
    """Последнее значение `export KEY=...` из /content/thinking_env.sh.

    Туда ячейка D пишет THINKING_CTX — сервер и сторож должны брать одно и то же
    значение окна (AUD-08), и бюджет ответа тоже прижимается к нему.
    """
    tail = f"export {key}="
    for line in _env_text().splitlines():
        if line.startswith(tail):
            default = line.split("=", 1)[1].strip().strip('"').strip("'")
    return default


UPSTREAM = os.environ.get("THINKING_UPSTREAM",
                          "http://127.0.0.1:8001/v1/chat/completions")
MODEL_NAME = os.environ.get("THINKING_MODEL", "thinking")
TOKEN = os.environ.get("THINKING_TOKEN", "")
LLM_TIMEOUT = float(os.environ.get("THINKING_TIMEOUT", "300"))
MAX_TOKENS = int(os.environ.get("THINKING_MAX_TOKENS", "700"))


def _ctx_limit() -> int:
    """Сколько токенов ответа влезает в окно движка.

    Бюджеты в ячейке D задают длину ответа, а не «боязнь долгого ответа».
    Но окно модели (n_ctx) — жёсткий предел: промт плюс ответ должны в него
    уместиться, иначе llama.cpp обрезает промт или отказывает. Резервируем
    ~1200 токенов под системный промт, память и код активного файла.
    """
    try:
        gpu = os.path.exists("/dev/nvidia0") or bool(shutil.which("nvidia-smi"))
    except Exception:
        gpu = False
    ctx = int(_env_export("THINKING_CTX", "8192" if gpu else "4096") or 4096)
    return max(256, ctx - 1200)


MAX_TOKENS = min(MAX_TOKENS, _ctx_limit())
RPS = int(os.environ.get("THINKING_RPS", "5"))
CONCURRENCY = int(os.environ.get("THINKING_CONCURRENCY", "1"))
MAX_EVENTS, MAX_PLANS, PLAN_TTL_S = 1000, 64, 3600

ACTIONS = {"build", "test", "refactor", "verify", "docs", "prompt", "debug", "release"}
RETRY = {"none", "once", "exponential"}
SECRET_RE = re.compile(
    r"\b(password|passwd|secret|api[_-]?key|access[_-]?token|private[_-]?key|token)\b"
    r"""["']?\s*[=:]\s*["']?[^\s"',}]+""",
    re.I,
)
# Аудит A-2 (правка B-1 была применена только к thinking/schemas.py): без
# допуска кавычек JSON-литерал «"password": "hunter2"» проходил мимо редьюсера
# и уезжал в события/отчёты/дамп. Паттерн и список «ссылок, а не секретов»
# держим один в один с schemas.py — тест test_secret_redactor_parity сверяет.
_SECRET_REF = ("os.getenv(", "os.environ[", "environ[", "getenv(",
               "self.", "cfg[", "config.", "settings.", "args.",
               '""', "''", "None", "True", "False")


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def redact(text: str) -> str:
    def _sub(m: "re.Match[str]") -> str:
        full, key = m.group(0), m.group(1) or ""
        tail = full.split(key, 1)[-1] if key in full else full
        if tail.lstrip("=: \t\"'").startswith(_SECRET_REF):
            return m.group(0)        # «token = os.getenv(…)» — это код (AUD-13)
        return f"{key}=[скрыто]"
    return SECRET_RE.sub(_sub, text or "")


# --------------------------------------------------------------- схемы ----
class PlanRequest(BaseModel):
    task: str = Field(..., min_length=3, max_length=2000)
    context: dict = Field(default_factory=dict)
    constraints: list[str] = Field(default_factory=list)
    max_steps: int = Field(12, ge=1, le=30)
    style_hint: Optional[str] = None


class PlanStep(BaseModel):
    id: int
    action: str = "verify"
    desc: str = Field(..., max_length=400)
    inputs: dict = Field(default_factory=dict)
    expected_output: Optional[str] = None
    retry_policy: str = "once"
    depends_on: list[int] = Field(default_factory=list)

    @field_validator("action", mode="before")
    @classmethod
    def _action(cls, v: Any) -> str:
        v = str(v or "").strip().lower()
        return v if v in ACTIONS else "verify"

    @field_validator("retry_policy", mode="before")
    @classmethod
    def _retry(cls, v: Any) -> str:
        v = str(v or "").strip().lower()
        return v if v in RETRY else "once"


class Plan(BaseModel):
    plan_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source: str = "colab"
    rationale: str = ""
    confidence: float = 0.7
    goal: str
    sub_goals: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    steps: list[PlanStep]
    success_criteria: list[str] = Field(default_factory=list)
    unknown_files: list[str] = Field(default_factory=list)
    fallback: str = "выполнить напрямую, без субагента"
    created_at: str = Field(default_factory=utcnow)

    @field_validator("steps")
    @classmethod
    def _steps(cls, v: list[PlanStep]) -> list[PlanStep]:
        if not v:
            raise ValueError("план без шагов")
        if len(v) > 30:
            raise ValueError("слишком много шагов")
        ids = [s.id for s in v]
        if len(ids) != len(set(ids)):
            raise ValueError(f"дублирующиеся id шагов: {ids}")
        return v

    @field_validator("rationale", "goal", mode="before")
    @classmethod
    def _clean(cls, v: Any) -> Any:
        return redact(str(v or "")) if isinstance(v, str) else v


class ReflectRequest(BaseModel):
    plan_id: str
    step_id: int
    result: str = Field(..., max_length=4000)
    observation: Optional[str] = None
    error: Optional[str] = None


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    context: dict = Field(default_factory=dict)
    history: list[dict] = Field(default_factory=list)
    memory: dict = Field(default_factory=dict)
    profile: Optional[str] = None
    max_steps: int = Field(4, ge=1, le=8)


class DevRequest(BaseModel):
    """Запрос режима «Разработка»: модель работает с кодом на ПК человека."""
    message: str = Field(..., min_length=1, max_length=3000)
    files: list[str] = Field(default_factory=list)
    active_name: str = ""
    active_code: str = ""


class ChatResponse(BaseModel):
    reply: str = ""
    rationale: str = ""
    steps: list[dict] = Field(default_factory=list)
    source: str = "colab"
    plan_id: str = ""
    fallback: bool = False
    memory_used: bool = False
    tokens_in: int = 0
    tokens_out: int = 0
    tokens_estimate: bool = True
    duration_ms: int = 0
    at: str = Field(default_factory=utcnow)

    @field_validator("reply", "rationale", mode="before")
    @classmethod
    def _clean(cls, v: Any) -> Any:
        return redact(str(v or "")) if isinstance(v, str) else v


class ReflectResponse(BaseModel):
    status: str = "ok"
    advice: str = ""
    rationale: str = ""
    next_steps: list[PlanStep] = Field(default_factory=list)
    updated_goal_stack: list[str] = Field(default_factory=list)


# ------------------------------------------------------------- состояние ---
class State:
    def __init__(self) -> None:
        self.events: deque[dict] = deque(maxlen=MAX_EVENTS)
        self.plans: dict[str, dict] = {}
        self.subs: set[asyncio.Queue] = set()
        self.seq = 0
        self.t0 = time.time()
        self.sem = asyncio.Semaphore(CONCURRENCY)
        self.win_t, self.win_n = time.time(), 0
        # переключение моделей на лету: процесс движка и флаг «идёт переключение»
        self.llm_proc: Optional[subprocess.Popen] = None
        self.switching = False
        self.stats = {"plans": 0, "reflects": 0, "chats": 0, "errors": 0,
                      "bad_json": 0, "llm_ms": deque(maxlen=200),
                      # честный учёт токенов: usage приходит не от каждого
                      # вызова, поэтому держим и «чистые» вызовы с usage
                      "tokens_in": 0, "tokens_out": 0, "usage_calls": 0,
                      # Телеметрия потоков (аудит 03): без неё обрыв на
                      # доставке невозможно отличить от падения генератора —
                      # ровно та задача, которая встала после живого прогона
                      # 04.10 (сервер додумал план, байты не доехали).
                      "stream_aborted": 0, "stream_first_ms": deque(maxlen=200),
                      "sem_timeouts": 0, "sem_wait_ms": deque(maxlen=200)}


S = State()   # единственный синглтон: снаружи глобальных переменных нет
app = FastAPI(title="Субагент «Мышление»", version="1.1")


@app.middleware("http")
async def cors(request: Request, call_next):
    resp = await call_next(request)
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type, X-Agent-Token"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return resp


@app.options("/{path:path}")
async def options(path: str):
    return {"ok": True}


# Сколько ждать свободного слота генерации, прежде чем честно отказать.
# 120 с = лимит Cloudflare на не-потоковый ответ: дольше всё равно бессмысленно.
SEM_WAIT_S = float(os.environ.get("THINKING_SEM_WAIT", "120"))


@asynccontextmanager
async def sem_slot():
    """Слот генерации с честным отказом вместо бесконечной очереди.

    CONCURRENCY=1 сериализует всю генерацию. Без таймаута запрос, чей
    предшественник умер молча (без EOF — живой симптом 04.10), стоял бы
    в очереди под пинги: сервер докручивает чужую генерацию, клиент не может
    отличить «стою в очереди» от «думаю», а его повтор упирается в то же
    ожидание — это самая правдоподобная причина, почему все три попытки
    плана умерли одинаково. Теперь по истечении SEM_WAIT_S — 503, который
    клиент честно показывает и повторяет.
    """
    t = time.monotonic()
    try:
        await asyncio.wait_for(S.sem.acquire(), timeout=SEM_WAIT_S)
    except asyncio.TimeoutError:
        S.stats["sem_timeouts"] += 1
        raise HTTPException(503, "LLM занят другим запросом — повторите позже")
    S.stats["sem_wait_ms"].append(int((time.monotonic() - t) * 1000))
    try:
        yield
    finally:
        S.sem.release()


def _avg(values) -> Optional[int]:
    """Среднее по выборке; пустая выборка → None, а не 0 (0 читался бы как «мгновенно»)."""
    vals = list(values)
    return int(sum(vals) / len(vals)) if vals else None


# -------------------------------------------------------------- события ---
async def emit(type_: str, text: str, **kw) -> dict:
    S.seq += 1
    ev = {"seq": S.seq, "type": type_, "text": redact(str(text)), "ts": utcnow(), **kw}
    S.events.append(ev)
    dead = []
    for q in list(S.subs):
        try:
            q.put_nowait(ev)
        except asyncio.QueueFull:
            # Подписчик отстал на 200 событий — рвём соединение: клиент
            # переподключится через /events?since= и доберёт хвост (AUD-11).
            # Раньше очередь была без лимита и медленный клиент копил
            # события в памяти сервера бесконечно.
            # Сентинел None обязателен: без него генератор подписчика вечно
            # ждал q.get() и никогда не получал EOF — комментарий про
            # «клиент переподключится» был неверен (аудит B).
            try:
                q.get_nowait()          # освобождаем место под сентинел
            except asyncio.QueueEmpty:
                pass
            try:
                q.put_nowait(None)     # None = «закрыть поток»
            except asyncio.QueueFull:
                pass
            dead.append(q)
        except Exception:
            dead.append(q)
    for q in dead:
        S.subs.discard(q)
    return ev


# --------------------------------------------------- разбор JSON от LLM ---
def extract_json(text: str) -> dict:
    t = (text or "").strip()
    if not t:
        raise ValueError("пустой ответ LLM")
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", t).strip()
    try:
        out = json.loads(t)
        if isinstance(out, dict):
            return out
    except json.JSONDecodeError:
        pass
    start = t.find("{")
    if start < 0:
        raise ValueError("в ответе LLM нет '{'")
    depth, in_str, esc = 0, False, False
    for i in range(start, len(t)):
        ch = t[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                chunk = t[start:i + 1]
                try:
                    return json.loads(chunk)
                except json.JSONDecodeError:
                    return json.loads(re.sub(r",\s*([}\]])", r"\1", chunk))
    raise ValueError("несбалансированный JSON в ответе LLM")


# --------------------------------------------------------------- промты ---
SYSTEM_PLAN = """Ты — советник главного агента: опытный техлид, который помогает
распланировать задачу до начала работы. Тип проекта и технологию указывает
сам агент в поле КОНТЕКСТ — не привязывайся к одной технологии.
Ты только советуешь: инструментов, файлов и действий у тебя нет.
Отвечай ТОЛЬКО валидным JSON, без markdown-обёрток и текста вокруг.
Схема:
{"goal": str, "rationale": str, "confidence": 0.0..1.0,
 "sub_goals": [str], "constraints": [str], "contradictions": [str],
 "steps": [{"id": int, "action": str, "desc": str, "inputs": {},
            "expected_output": str, "retry_policy": "none|once|exponential",
            "depends_on": [int]}],
 "success_criteria": [str], "unknown_files": [str], "fallback": str}
Правила:
- goal, sub_goals, desc, contradictions, success_criteria, rationale — только по-русски.
- action — латинский тег ровно из: build test refactor verify docs prompt debug release.
- Упоминай только файлы из блока РАЗРЕШЁННЫЕ ФАЙЛЫ; если нужен другой — внеси его
  в unknown_files, но не выдумывай путь.
- Шаги проверяемые, по порядку, не больше указанного числа. Полей вне схемы не добавляй.
- rationale — 2-3 фразы по-русски: почему выбран именно такой план."""

SYSTEM_REFLECT = """Ты — рефлектор выполненного шага. Отвечай ТОЛЬКО JSON:
{"status": "ok|adjust|abort", "advice": str, "rationale": str,
 "next_steps": [{"id": int, "action": str, "desc": str, "inputs": {},
                 "expected_output": str, "retry_policy": str, "depends_on": [int]}],
 "updated_goal_stack": [str]}
advice и rationale — по-русски, коротко (до 240 символов). Ничего кроме JSON."""


SYSTEM_CHAT = """Ты — субагент «Мышление», собеседник человека-разработчика.
Отвечай обычным живым текстом по-русски — БЕЗ JSON, без markdown-обёрток
и кода-примеров вокруг ответа. Просто пиши, как обычный собеседник.
Правила:
- Сначала прямой ответ, потом (если уместно) 1-3 конкретных действия.
  Без воды и вступлений; до 700 символов, если вопрос не требует большего.
- Если в блоке ПАМЯТЬ есть факты о проекте — опирайся на них и не переспрашивай
  то, что уже известно.
- Если в блоке ДИАЛОГ есть прошлые реплики — не повторяй то, что уже обсудили.
- Ты советуешь: не выдумывай пути файлов, не обещай выполнить работу сам.
- Если вопрос про код и изменения файлов — скажи, что для этого есть вкладка
  «Разработка» (там модель меняет файлы подтверждением человека).
- Если не хватает данных — задай один-два уточняющих вопроса."""

CHAT_TURNS_MAX = 12
CHAT_HISTORY_CHARS = 2400


def build_chat_prompt(req: "ChatRequest") -> str:
    """Промт чата: память + последние реплики + сам вопрос."""
    facts = [str(f) for f in (req.memory.get("facts") or [])][-12:]
    mem_block = ""
    if req.profile:
        mem_block += f"О ПРОЕКТЕ: {str(req.profile)[:400]}\n"
    if facts:
        mem_block += "ПАМЯТЬ (факты, которые я уже сказал):\n"
        mem_block += "\n".join(f"- {f}" for f in facts)[:1200] + "\n"
    dialog = ""
    turns = [t for t in (req.history or []) if isinstance(t, dict)][-CHAT_TURNS_MAX:]
    if turns:
        lines = []
        for t in turns:
            who = "субагент" if t.get("role") == "subagent" else "я"
            lines.append(f"{who}: {str(t.get('text') or '')[:400]}")
        dialog = "ДИАЛОГ (недавно обсуждали):\n" + "\n".join(lines)[-CHAT_HISTORY_CHARS:] + "\n"
    ctx = json.dumps(req.context or {}, ensure_ascii=False)[:1500]
    return (mem_block + dialog +
            f"КОНТЕКСТ: {ctx}\n"
            f"ВОПРОС: {req.message[:2000]}\n"
            "Отвечай обычным текстом, без JSON.")


def build_plan_prompt(req: PlanRequest) -> str:
    ctx = json.dumps(req.context, ensure_ascii=False)
    files = req.context.get("files") or []
    tail = " | ".join(f"{e['type']}: {e['text'][:70]}" for e in list(S.events)[-10:])
    return (
        f"ЗАДАЧА: {req.task}\n"
        f"КОНТЕКСТ: {ctx[:4000]}\n"
        f"ОГРАНИЧЕНИЯ: {json.dumps(req.constraints, ensure_ascii=False)}\n"
        f"МАКС. ШАГОВ: {req.max_steps}\n"
        + (f"РАЗРЕШЁННЫЕ ФАЙЛЫ: {', '.join(str(f) for f in files)}\n" if files else "")
        + f"ПОСЛЕДНИЕ СОБЫТИЯ: {tail[:500]}\n"
        "Ответ — только JSON по схеме из system-промта."
    )


SYSTEM_DEV = """Ты — режим «Разработка» субагента «Мышление»: ты работаешь с кодом
человека, который физически лежит на его компьютере (ты сам файлы не трогаешь —
только присылаешь изменение, а применит его человек кнопкой).
Отвечай ТОЛЬКО валидным JSON, без markdown-обёрток и текста вокруг.
Схема:
{"action": "create|edit|none", "filename": str, "code": str, "comment": str}
Правила:
- action="create" — новый файл в рабочей папке; action="edit" — замена
  СУЩЕСТВУЮЩЕГО файла из списка ФАЙЛЫ (имя возьми ровно оттуда);
  action="none" — вопрос или обсуждение без изменения кода (code пустой).
- filename — только имя файла (без каталогов), обычно с расширением .py.
- code — ПОЛНОЕ новое содержимое файла целиком (не диф, не фрагмент):
  иначе применение частями непонятно. Только код, без markdown и пояснений
  внутри кода.
- comment — по-русски, 1-3 предложения: что изменено и что это даёт.
- Пиши рабочий Python: без TODO, без заглушек, с обработкой ошибок там,
  где они нужны; стандартная библиотека (у человека только она).
- Если данных мало или просят невозможное — action="none" и задай вопрос
  в comment. Ничего кроме JSON."""


def build_dev_prompt(req: "DevRequest") -> str:
    """Промт режима разработчика: файлы + открытый файл + запрос человека."""
    files = ", ".join(str(f) for f in (req.files or [])[:80]) or "—"
    out = (f"ФАЙЛЫ В РАБОЧЕЙ ПАПКЕ: {files}\n"
           f"ЗАПРОС: {req.message[:3000]}\n")
    if req.active_name:
        out += (f"ОТКРЫТ СЕЙЧАС ФАЙЛ: {req.active_name}\n"
                f"ЕГО КОД:\n{req.active_code[:8000]}\n")
    return out + "Ответ — только JSON по схеме из system-промта."


async def dev_data(raw: str) -> dict:
    """Разбор ответа режима разработчика: JSON либо честный текст."""
    try:
        data = extract_json(raw)
        action = str(data.get("action") or "none").lower()
        if action not in ("create", "edit", "none"):
            action = "none"
        return {"action": action,
                "filename": str(data.get("filename") or "")[:200],
                "code": str(data.get("code") or "")[:20000],
                "comment": redact(str(data.get("comment") or ""))[:1200]}
    except Exception:
        S.stats["bad_json"] += 1
        text = _plain_reply(raw)
        # Модель ответила текстом — пусть идёт комментарием, файлы не трогаем
        return {"action": "none", "filename": "", "code": "",
                "comment": redact(text)[:1200]}


# ------------------------------------------------------- вызовы LLM -------
async def chat_json(system: str, user: str, max_tokens: int = MAX_TOKENS) -> str:
    text, _ = await chat_json_usage(system, user, max_tokens)
    return text


async def chat_json_usage(system: str, user: str,
                          max_tokens: int = MAX_TOKENS) -> tuple[str, dict]:
    """Как chat_json, но возвращает ещё и usage — по нему считаются токены.

    llama.cpp-сервер отдаёт usage не всегда, поэтому при отсутствии честно
    отдаём пустой dict: ПК посчитает оценку по длине текста сам.
    """
    async with sem_slot():
        t = time.time()
        async with httpx.AsyncClient(timeout=LLM_TIMEOUT) as c:
            r = await c.post(UPSTREAM, json={
                "model": MODEL_NAME, "stream": False, "temperature": 0.2,
                "max_tokens": max_tokens,
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": user}],
            })
            r.raise_for_status()
            data = r.json()
        S.stats["llm_ms"].append(int((time.time() - t) * 1000))
    usage = data.get("usage") or {}
    if isinstance(usage, dict) and usage:
        S.stats["tokens_in"] += int(usage.get("prompt_tokens") or 0)
        S.stats["tokens_out"] += int(usage.get("completion_tokens") or 0)
        S.stats["usage_calls"] += 1
    return data["choices"][0]["message"]["content"], (usage if isinstance(usage, dict) else {})


async def with_heartbeat(src, gap: float = 15.0):
    """Отдаёт SSE-комментарии, пока модель молчит.

    Cloudflare рвёт соединение по 524, если не получил первые байты ТЕЛА
    ответа дольше ~120 с. Заголовки приходят мгновенно, а тело — только
    когда модель выдаст первый токен, и на 14B на CPU это больше минуты
    (04.10: `ask` и обычный `/chat` оба падали в 524, хотя модель жива).
    Поэтому шлём `: ping` сразу и далее каждые `gap` секунд: это валидный
    SSE-комментарий, который прокси пропускает, а клиент игнорирует.

    Наполнение идёт отдельной задачей: иначе таймаут отменял бы чтение
    модели на полуслове и поток вёлся бы в никуда.

    Плюс телеметрия (аудит 03): первый некий байт (TTFT) и факт обрыва.
    Раньше сервер вообще не знал, закончился ли поток `done`-событием или
    его вырезали посередине — отличить «туннель убил» от «клиент закрыл»
    было нечем, и живой симптом 04.10 объяснялся только гипотезами.
    """
    q: asyncio.Queue = asyncio.Queue()
    done = object()
    t0 = time.monotonic()
    started = False
    completed = False

    async def _pump() -> None:
        try:
            async for item in src:
                await q.put(item)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await q.put(exc)
        await q.put(done)

    task = asyncio.create_task(_pump())
    try:
        yield ": open\n\n"                     # первый байт тела — сразу
        while True:
            try:
                item = await asyncio.wait_for(q.get(), timeout=gap)
            except asyncio.TimeoutError:
                yield f": ping {int(time.time())}\n\n"
                continue
            if item is done:
                completed = True               # генератор доработал до конца
                return
            if isinstance(item, BaseException):
                raise item
            if not started:
                started = True                 # первый реальный байт = TTFT
                S.stats["stream_first_ms"].append(
                    int((time.monotonic() - t0) * 1000))
            yield item
    finally:
        task.cancel()
        if not completed:
            # клиент ушёл, соединение умерло молча или генератор упал —
            # раньше это выглядело как успешный поток
            S.stats["stream_aborted"] = int(S.stats.get("stream_aborted", 0)) + 1


async def chat_stream(system: str, user: str, max_tokens: int = MAX_TOKENS,
                      url: str = ""):
    async with httpx.AsyncClient(timeout=LLM_TIMEOUT) as c:
        async with c.stream("POST", url or UPSTREAM, json={
            "model": MODEL_NAME, "stream": True, "temperature": 0.2,
            "max_tokens": max_tokens,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
        }) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if not line.startswith("data:"):
                    continue
                chunk = line[5:].strip()
                if chunk == "[DONE]":
                    break
                try:
                    d = json.loads(chunk)
                except Exception:
                    continue
                piece = d.get("choices", [{}])[0].get("delta", {}).get("content")
                if piece:
                    yield piece


# ------------------------------------------------- токен + rate limit -----
def _auth(x_agent_token: str, query_token: str) -> None:
    got = x_agent_token or query_token
    if TOKEN and got != TOKEN:
        raise HTTPException(401, "неверный токен")
    now = time.time()
    if now - S.win_t >= 1.0:
        S.win_t, S.win_n = now, 0
    S.win_n += 1
    if S.win_n > RPS:
        raise HTTPException(429, "слишком часто (rate limit)")


async def _guard(raw: str) -> dict:
    try:
        return extract_json(raw)
    except Exception as exc:
        S.stats["bad_json"] += 1
        try:
            # сырой вывод LLM уходит в дамп (/dump/last_bad_json.txt) — через
            # редьюсер, как и весь остальной вывод сервера (аудит A-2)
            with open("/content/last_bad_json.txt", "w", encoding="utf-8") as fh:
                fh.write(redact(raw or ""))
        except Exception:
            pass
        await emit("error", f"LLM вернула не-JSON: {exc}")
        raise HTTPException(502, f"LLM вернула не-JSON: {exc}")


def _plain_reply(raw: str) -> str:
    """Ответ без JSON — просто текст.

    Маленькие модели (1.5B) часто отвечают обычной фразой вместо JSON.
    Раньше такой ответ считался поломкой (HTTP 502 «пустой ответ»), хотя
    человек-то получил осмысленный текст. Теперь текст идёт как есть,
    а счётчик plain_text показывает, что модель ушла от формата.
    """
    t = (raw or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", t).strip()
    return t


async def guard_reply(raw: str) -> dict:
    """Как _guard, но не падает на обычном тексте: отдаёт {'reply': текст}."""
    try:
        data = extract_json(raw)
        if str(data.get("reply") or "").strip():
            return data
    except Exception as exc:
        S.stats["bad_json"] += 1
        await emit("error", f"LLM ответила текстом, а не JSON: {exc}")
    text = _plain_reply(raw)
    if not text:
        raise HTTPException(502, "пустой ответ чата")
    # Частая форма: модель обернула JSON-объект в текст, но не довела его до
    # конца (обрезали по max_tokens). Тогда разбираем поле reply вручную.
    m = re.search(r'"reply"\s*:\s*"(.*?)"\s*(?:,|})', text, re.S)
    if m:
        text = m.group(1).replace('\\"', '"').replace("\\n", "\n").strip()
    S.stats["plain_text"] = int(S.stats.get("plain_text", 0)) + 1
    return {"reply": text, "rationale": "", "steps": []}


PLAN_LIST_FIELDS = ("sub_goals", "constraints", "contradictions",
                    "success_criteria", "unknown_files")


def coerce_plan(data: dict) -> dict:
    """Мягко приводит план к серверной схеме — ДО `Plan(**data)`.

    Раньше приводило только типы списков и id, а дубли id, пустой список
    шагов, `desc` длиннее 400 и не-списки в `depends_on`/`inputs` всё равно
    роняли валидацию в HTTP 500 — хотя слабая 3B отдаёт именно такие
    варианты. Теперь любую задачу обязан пережить тут, а не падением
    (аудит B, 03-colab: «coerce_plan не гарантирует валидность»).
    """
    out = dict(data or {})

    # --- текстовые поля: длина и тип -------------------------------------
    goal = str(out.get("goal") or "").strip() or "(задача не названа)"
    out["goal"] = goal[:500]
    out["rationale"] = str(out.get("rationale") or "")[:800]
    for key, limit in (("fallback", 500), ("source", 40), ("plan_id", 80),
                       ("created_at", 40)):
        if key in out and out[key] is not None:
            out[key] = str(out[key])[:limit]
    try:
        conf = float(out.get("confidence", 0.7))
    except (TypeError, ValueError):
        conf = 0.7
    out["confidence"] = max(0.0, min(1.0, conf))

    # --- списки строк -----------------------------------------------------
    for key in PLAN_LIST_FIELDS:
        v = out.get(key)
        if v is None:
            out[key] = []
        elif isinstance(v, str):
            out[key] = [v[:500]] if v.strip() else []
        elif isinstance(v, list):
            out[key] = [str(x)[:500] for x in v if str(x).strip()][:50]
        else:
            out[key] = [str(v)]

    # --- шаги -------------------------------------------------------------
    steps = out.get("steps")
    if not isinstance(steps, list):
        steps = [steps] if steps else []
    norm: list[dict] = []
    used: set[int] = set()
    for s in steps:
        item = dict(s) if isinstance(s, dict) else {"desc": str(s)}
        try:
            sid = int(item.get("id"))
        except (TypeError, ValueError):
            sid = len(norm) + 1
        while sid in used:                     # дубликат id → ValidationError
            sid += 1
        used.add(sid)
        item["id"] = sid
        desc = str(item.get("desc") or "").strip() or f"шаг {len(norm) + 1}"
        item["desc"] = desc[:400]
        if not isinstance(item.get("inputs"), dict):
            item["inputs"] = {}
        dep = item.get("depends_on")
        if isinstance(dep, list):
            clean: list[int] = []
            for d in dep:
                try:
                    clean.append(int(d))
                except (TypeError, ValueError):
                    continue
            item["depends_on"] = [d for d in clean if d != sid]
        else:
            item["depends_on"] = []
        norm.append(item)
    if not norm:                               # «план без шагов» → 500
        norm = [{"id": 1, "action": "verify", "desc": "выполнить задачу напрямую"}]
    norm = norm[:30]                           # больше 30 → ValidationError
    ids = {it["id"] for it in norm}
    for it in norm:                            # ссылки только на живые шаги
        it["depends_on"] = [d for d in it.get("depends_on", [])
                            if d in ids and d != it["id"]]
    out["steps"] = norm
    return out


async def plan_data(raw: str, task: str) -> dict:
    """Как _guard, но обычный текст модели превращается в план из одного шага.

    Иначе слабая модель (1.5B) давала бы ошибку вместо полезного ответа,
    хотя рассуждать она может — просто не в том формате.
    """
    try:
        data = extract_json(raw)
        if isinstance(data.get("steps"), list) and data["steps"]:
            return coerce_plan(data)
    except Exception as exc:
        S.stats["bad_json"] += 1
        await emit("error", f"LLM ответила текстом, а не JSON: {exc}")
    text = _plain_reply(raw)
    if not text:
        raise HTTPException(502, "пустой ответ плана")
    S.stats["plain_text"] = int(S.stats.get("plain_text", 0)) + 1
    return coerce_plan({
        "goal": str(task or "")[:300] or "(задача не названа)",
        "rationale": text[:600],
        # id у шага — число: серверная схема жёстко требует int,
        # строковый "s1" приводил к 500 на валидации.
        "steps": [{"id": 1, "desc": text[:300]}],
        "confidence": 0.3,
    })


async def reflect_data(raw: str) -> dict:
    """Как _guard, но текстовая рефлексия тоже считается ответом."""
    try:
        return extract_json(raw)
    except Exception as exc:
        S.stats["bad_json"] += 1
        await emit("error", f"LLM ответила текстом, а не JSON: {exc}")
    text = _plain_reply(raw)
    if not text:
        raise HTTPException(502, "пустой ответ разбора")
    S.stats["plain_text"] = int(S.stats.get("plain_text", 0)) + 1
    return {"status": "ok", "advice": text[:800], "rationale": "", "next_steps": []}


def _store(plan: Plan) -> None:
    S.plans[plan.plan_id] = plan.model_dump()
    if len(S.plans) > MAX_PLANS:
        for key in sorted(S.plans, key=lambda k: S.plans[k].get("created_at", ""))[:8]:
            S.plans.pop(key, None)


# ---------------------------------------------------------- эндпоинты -----
@app.get("/", response_class=HTMLResponse)
async def index(token: str = ""):
    # токен уезжает в JS через json.dumps: кавычки/переводы строки в нём
    # экранируются, и ссылка вида /?token=";evil// не исполняется
    # (аудит B-7)
    return PANEL_HTML.replace("__TOKEN__", json.dumps(str(token)))


# GPU/VRAM с кэшем на 30 с: `import torch` внутри обработчика — сотни мс
# блокировки цикла событий, а /health панель дёргает каждые 2,5 с, и
# тормозились вместе с ним пинги живых потоков (аудит B, 03-colab).
_GPU_INFO: dict = {"gpu": "cpu", "vram": None, "at": 0.0}


def _gpu_info() -> tuple:
    """GPU/VRAM с кэшем на 30 с.

    `import torch` внутри обработчика — сотни мс блокировки цикла событий,
    а /health панель дёргает каждые 2,5 с: тормозились и пинги живых потоков
    (аудит B, 03-colab).
    """
    if time.time() - _GPU_INFO["at"] > 30:
        gpu, vram = "cpu", None
        try:
            import torch
            if torch.cuda.is_available():
                gpu = torch.cuda.get_device_name(0)
                vram = round(torch.cuda.memory_allocated() / 2 ** 30, 2)
        except Exception:
            pass
        _GPU_INFO.update(gpu=gpu, vram=vram, at=time.time())
    return _GPU_INFO["gpu"], _GPU_INFO["vram"]


@app.get("/health")
async def health(x_agent_token: str = Header(default=""), token: str = ""):
    _auth(x_agent_token, token)
    gpu, vram = _gpu_info()
    return {"status": "ok", "model": MODEL_NAME, "upstream": UPSTREAM,
            "uptime_s": int(time.time() - S.t0), "plans": len(S.plans),
            "events": S.seq, "gpu": gpu, "vram_used_gb": vram}


@app.get("/metrics")
async def metrics(x_agent_token: str = Header(default=""), token: str = ""):
    _auth(x_agent_token, token)
    ms = list(S.stats["llm_ms"])
    calls = S.stats["plans"] + S.stats["reflects"] + S.stats["chats"]
    return {"events": S.seq, "plans_total": S.stats["plans"],
            "reflects": S.stats["reflects"], "chats_total": S.stats["chats"],
            "errors": S.stats["errors"],
            "bad_json": S.stats["bad_json"], "subscribers": len(S.subs),
            "llm_ms_avg": int(sum(ms) / len(ms)) if ms else None,
            "llm_ms_p95": sorted(ms)[int(len(ms) * 0.95)] if len(ms) > 5 else None,
            "queue": CONCURRENCY,
            # usage отдаёт не каждый вызов — панель помечает такие цифры оценкой
            "tokens_in": S.stats["tokens_in"], "tokens_out": S.stats["tokens_out"],
            "tokens_total": S.stats["tokens_in"] + S.stats["tokens_out"],
            "usage_calls": S.stats["usage_calls"],
            "usage_exact": bool(calls) and S.stats["usage_calls"] >= calls,
            # телеметрия потоков: сколько оборвалось, как долго ждали очередь
            "stream_aborted": S.stats["stream_aborted"],
            "stream_first_ms_avg": _avg(S.stats["stream_first_ms"]),
            "sem_wait_ms_avg": _avg(S.stats["sem_wait_ms"]),
            "sem_timeouts": S.stats["sem_timeouts"]}


@app.get("/events")
async def events(since: int = 0, tail: int = 0,
                 x_agent_token: str = Header(default=""), token: str = ""):
    _auth(x_agent_token, token)
    items = list(S.events)
    if since:
        items = [e for e in items if e["seq"] > since]
    if tail:
        items = items[-tail:]
    return {"last_seq": S.seq, "events": items}


@app.get("/events/stream")
async def events_stream(x_agent_token: str = Header(default=""), token: str = ""):
    _auth(x_agent_token, token)

    async def gen():
        for e in list(S.events)[-50:]:
            yield f"data: {json.dumps(e, ensure_ascii=False)}\n\n"
        # Очередь с лимитом: медленный подписчик не должен копить события
        # в памяти сервера (AUD-11) — при переполнении его отключит emit()
        q: asyncio.Queue = asyncio.Queue(maxsize=200)
        S.subs.add(q)
        try:
            while True:
                try:
                    # ping каждые 10 с: туннель и прокси рвут «тихое» соединение,
                    # а панель на ПК держит read-timeout потока 40 с
                    ev = await asyncio.wait_for(q.get(), timeout=10)
                    if ev is None:     # сентинел от emit(): очередь переполнена
                        return
                    yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
                except asyncio.TimeoutError:
                    yield ": ping\n\n"
        finally:
            S.subs.discard(q)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


@app.post("/plan")
async def plan(req: PlanRequest, x_agent_token: str = Header(default=""), token: str = ""):
    _auth(x_agent_token, token)
    await emit("thought", f"Задача: {req.task[:150]}")
    await emit("thought", "Декомпозирую на шаги…")
    try:
        raw = await chat_json(SYSTEM_PLAN, build_plan_prompt(req))
        data = await plan_data(raw, req.task)
        p = Plan(**data)
        _store(p)
        S.stats["plans"] += 1
        await emit("rationale", p.rationale or "(без объяснения)", plan_id=p.plan_id)
        for c in p.contradictions:
            await emit("contradiction", c, plan_id=p.plan_id)
        for s in p.steps:
            await emit("plan_step", f"[{s.id}] {s.desc}", plan_id=p.plan_id, step_id=s.id)
        await emit("final", f"План {p.plan_id[:8]} готов: {len(p.steps)} шагов",
                   plan_id=p.plan_id)
        return p
    except HTTPException:
        raise
    except httpx.HTTPError as exc:
        S.stats["errors"] += 1
        await emit("error", f"LLM backend недоступен: {type(exc).__name__}")
        raise HTTPException(503, f"upstream недоступен: {type(exc).__name__}")
    except Exception as exc:
        S.stats["errors"] += 1
        await emit("error", str(exc)[:300])
        raise HTTPException(500, str(exc)[:300])


@app.post("/plan/stream")
async def plan_stream(req: PlanRequest, x_agent_token: str = Header(default=""), token: str = ""):
    _auth(x_agent_token, token)
    await emit("thought", f"Стрим плана: {req.task[:150]}")

    async def gen():
        buf: list[str] = []
        try:
            async with sem_slot():
                async for piece in chat_stream(SYSTEM_PLAN, build_plan_prompt(req)):
                    buf.append(piece)
                    yield f"data: {json.dumps({'type': 'token', 'text': piece}, ensure_ascii=False)}\n\n"
            data = await plan_data("".join(buf), req.task)
            p = Plan(**data)
            _store(p)
            S.stats["plans"] += 1
            yield f"data: {json.dumps({'type': 'rationale', 'text': p.rationale}, ensure_ascii=False)}\n\n"
            for s in p.steps:
                yield ("data: " + json.dumps({"type": "plan_step",
                                              "text": f"[{s.id}] {s.desc}",
                                              "step_id": s.id},
                                             ensure_ascii=False) + "\n\n")
            final = json.dumps(p.model_dump(), ensure_ascii=False, default=str)
            yield ("data: " + json.dumps({"type": "final", "text": final,
                                          "plan_id": p.plan_id},
                                         ensure_ascii=False) + "\n\n")
        except Exception as exc:
            S.stats["errors"] += 1
            yield ("data: " + json.dumps({"type": "error", "text": str(exc)[:300]},
                                         ensure_ascii=False) + "\n\n")

    return StreamingResponse(with_heartbeat(gen()), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


def _reflect_plan_json(req: ReflectRequest) -> str:
    """Текст плана для промта рефлексии (или честная причина его отсутствия)."""
    stored = S.plans.get(req.plan_id)
    return json.dumps(stored, ensure_ascii=False, default=str) if stored \
        else "план не найден (возможно, после рестарта сервера)"


def _reflect_user(req: ReflectRequest, plan_json: str) -> str:
    """Пользовательская часть промта рефлексии — общая для обоих маршрутов."""
    return (f"ПЛАН: {plan_json[:3000]}\nШАГ: {req.step_id}\n"
            f"РЕЗУЛЬТАТ: {req.result[:2000]}\n"
            f"НАБЛЮДЕНИЕ: {req.observation or '-'}\n"
            f"ОШИБКА: {req.error or '-'}")


async def reflect_payload(req: ReflectRequest, raw: str) -> ReflectResponse:
    """Собирает ответ рефлексии из сырого текста LLM.

    Контракт один и тот же для /reflect и /reflect/stream (аудит B-5):
    где бы генерация ни шла, дальше один и тот же разбор и те же лимиты.
    """
    data = await reflect_data(raw)
    status = data.get("status")
    if status not in ("ok", "adjust", "abort"):
        status = "ok"
    steps = []
    for item in (data.get("next_steps") or [])[:8]:
        try:
            steps.append(PlanStep(**item))
        except Exception:
            continue
    return ReflectResponse(
        status=status,
        advice=redact(str(data.get("advice") or ""))[:800],
        rationale=redact(str(data.get("rationale") or ""))[:400],
        next_steps=steps,
        updated_goal_stack=[str(x) for x in (data.get("updated_goal_stack") or [])][:12],
    )


@app.post("/reflect")
async def reflect(req: ReflectRequest, x_agent_token: str = Header(default=""), token: str = ""):
    _auth(x_agent_token, token)
    plan_json = _reflect_plan_json(req)
    await emit("thought", f"Рефлексия шага {req.step_id}",
               plan_id=req.plan_id, step_id=req.step_id)
    try:
        raw = await chat_json(SYSTEM_REFLECT,
                              _reflect_user(req, plan_json),
                              max_tokens=min(500, _ctx_limit()))
        out = await reflect_payload(req, raw)
        S.stats["reflects"] += 1
        await emit("final" if out.status == "ok" else "contradiction",
                   f"[{out.status}] {out.advice}", plan_id=req.plan_id, step_id=req.step_id)
        return out
    except HTTPException:
        raise
    except Exception as exc:
        S.stats["errors"] += 1
        await emit("error", f"рефлексия: {str(exc)[:200]}")
        raise HTTPException(500, str(exc)[:300])


@app.post("/reflect/stream")
async def reflect_stream(req: ReflectRequest, x_agent_token: str = Header(default=""), token: str = ""):
    """Рефлексия по потоку — переживает генерацию дольше 120 с (аудит B-5).

    Не-потоковый /reflect отдаёт тело только после всей генерации, а на CPU
    рефлексия занимает 100-200 с: прокси обрезает такой ответ на 120-й
    секунде (524) и результат до ПК не доезжает вовсе. Поток кладёт первый
    байт сразу, пинги каждые 15 с держат соединение живым, а итог приходит
    событием done — тело читается столько, сколько нужно.
    """
    _auth(x_agent_token, token)
    plan_json = _reflect_plan_json(req)
    await emit("thought", f"Рефлексия (поток) шага {req.step_id}",
               plan_id=req.plan_id, step_id=req.step_id)

    async def gen():
        buf: list[str] = []
        try:
            async with sem_slot():
                async for piece in chat_stream(SYSTEM_REFLECT,
                                               _reflect_user(req, plan_json),
                                               min(500, _ctx_limit())):
                    buf.append(piece)
                    yield ("data: " + json.dumps({"type": "token", "text": piece},
                                                 ensure_ascii=False) + "\n\n")
            out = await reflect_payload(req, "".join(buf))
            S.stats["reflects"] += 1
            await emit("final" if out.status == "ok" else "contradiction",
                       f"[{out.status}] {out.advice}",
                       plan_id=req.plan_id, step_id=req.step_id)
            yield ("data: " + json.dumps({"type": "done",
                                          "response": out.model_dump()},
                                         ensure_ascii=False, default=str) + "\n\n")
        except Exception as exc:
            S.stats["errors"] += 1
            await emit("error", f"рефлексия: {str(exc)[:200]}")
            yield ("data: " + json.dumps({"type": "error", "text": str(exc)[:300]},
                                         ensure_ascii=False) + "\n\n")

    return StreamingResponse(with_heartbeat(gen()), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


@app.post("/cancel/{plan_id}")
async def cancel(plan_id: str, x_agent_token: str = Header(default=""), token: str = ""):
    _auth(x_agent_token, token)
    S.plans.pop(plan_id, None)
    await emit("thought", f"План {plan_id[:8]} отменён")
    return {"status": "cancelled", "plan_id": plan_id}


@app.post("/chat")
async def chat(req: ChatRequest, x_agent_token: str = Header(default=""), token: str = ""):
    """Живой диалог человека с субагентом: память + последние реплики + ответ."""
    _auth(x_agent_token, token)
    t0 = time.time()
    memory = req.memory if isinstance(req.memory, dict) else {}
    facts = [str(f) for f in (memory.get("facts") or []) if str(f).strip()]
    used = bool(facts or req.profile or req.history)
    await emit("thought", f"Чат: {req.message[:120]}")
    try:
        raw, usage = await chat_json_usage(SYSTEM_CHAT, build_chat_prompt(req), 500)
        data = await guard_reply(raw)
        reply = str(data.get("reply") or "").strip()
        if not reply:                      # модель вернула пустоту — честно об этом
            raise HTTPException(502, "пустой ответ чата")
        steps = []
        for item in (data.get("steps") or [])[:5]:
            desc = item.get("desc") if isinstance(item, dict) else item
            if str(desc or "").strip():
                steps.append({"desc": str(desc)[:300]})
        S.stats["chats"] += 1
        tin = int(usage.get("prompt_tokens") or 0)
        tout = int(usage.get("completion_tokens") or 0)
        out = ChatResponse(reply=reply, rationale=str(data.get("rationale") or "")[:600],
                           steps=steps, source="colab", memory_used=used,
                           tokens_in=tin, tokens_out=tout,
                           tokens_estimate=not bool(usage),
                           duration_ms=int((time.time() - t0) * 1000))
        await emit("final", f"[чат] {reply[:200]}")
        if tin or tout:
            await emit("thought", f"токены: в {tin}, out {tout}")
        return out
    except HTTPException:
        raise
    except httpx.HTTPError as exc:
        S.stats["errors"] += 1
        await emit("error", f"чат: upstream недоступен: {type(exc).__name__}")
        raise HTTPException(503, "LLM backend недоступен")
    except Exception as exc:
        S.stats["errors"] += 1
        await emit("error", f"чат: {str(exc)[:200]}")
        raise HTTPException(500, str(exc)[:300])


# ------------------------------------------- выгрузка логов (для архива) ----
DUMP_FILES = {
    "llm.log": "/content/llm.log",
    "api.log": "/content/thinking_api.log",
    "tunnel.log": "/content/tunnel.log",
    "server.py": "/content/thinking_server.py",
    "snapshot.json": "/content/thinking_snapshot.json",
    "last_bad_json.txt": "/content/last_bad_json.txt",
    "model_path.txt": "/content/thinking_model_path.txt",
}


# ------------------------------------------------- переключение моделей ---
MODELS_FILE = "/content/thinking_models.txt"
ACTIVE_MODEL_FILE = "/content/thinking_active_model.txt"
LLM_LOG = "/content/llm.log"
LLM_HOST, LLM_PORT = "127.0.0.1", 8001
S.llm_proc: Optional[subprocess.Popen] = None   # noqa: E305


def _model_list() -> list[str]:
    """Модели, доступные рантайму: то, что перечислила ячейка A/D."""
    try:
        with open(MODELS_FILE, encoding="utf-8") as fh:
            items = [ln.strip() for ln in fh if ln.strip()]
    except OSError:
        items = []
    return [p for p in items if os.path.exists(p)]


def _active_model() -> str:
    try:
        with open(ACTIVE_MODEL_FILE, encoding="utf-8") as fh:
            cur = fh.read().strip()
            if cur and os.path.exists(cur):
                return cur
    except OSError:
        pass
    # Не угадываем по списку: иначе панель решит, что активна первая модель,
    # а в рантайме может крутиться другая — и переключение не сработает.
    return ""


def _model_label(path: str) -> str:
    """Короткое имя для интерфейса: qwen2.5-3b-instruct-q4_k_m → 3B.

    Разные сборки различаются суффиксом (UNC — uncensored), иначе в панели
    две 7B выглядят одинаково и переключение выбирает не ту.
    """
    name = os.path.basename(path).lower()
    size = ""
    for tag in ("0.5b", "1.5b", "3b", "7b", "14b"):
        if tag in name:
            size = tag.upper()
            break
    if size:
        if "uncensored" in name or "unfiltered" in name:
            return size + "-UNC"
        if "dolphin" in name:
            return size + "-DOLPHIN"
        return size
    return os.path.splitext(os.path.basename(path))[0][:24]


def _model_hint(path: str, gpu: Optional[bool] = None) -> str:
    """Подсказка нужного режима (CPU / T4 GPU) — видна в панели и CLI.

    Пользователь меняет hardware accelerator в Colab, поэтому подсказка
    учитывает и модель, и то, что запущено прямо сейчас.
    """
    if gpu is None:
        gpu = os.path.exists("/dev/nvidia0") or bool(shutil.which("nvidia-smi"))
    name = os.path.basename(path).lower()
    here = "сейчас T4/GPU" if gpu else "сейчас CPU"
    unc = " Сборка без цензуры." if "uncensored" in name else ""
    if any(tag in name for tag in ("7b", "8b", "14b")):
        if gpu:
            return (f"7B · {here} — считается быстро (~40–60 ток/с).{unc}")
        return (f"7B · {here} — на CPU это ~1 ток/с, очень медленно. "
                f"Для лучшей работы смени Runtime → Change runtime type → "
                f"T4 GPU.{unc}")
    if "3b" in name:
        return (f"3B · {here} — на CPU ~2–3 ток/с, на T4 ~40 ток/с; "
                f"компромисс скорости и качества.{unc}")
    if "1.5b" in name or "0.5b" in name:
        return (f"1.5B · {here} — самая быстрая на CPU (~6 ток/с), "
                f"лучший выбор без видеокарты.{unc}")
    return f"{here}; на T4 любая модель считается заметно быстрее.{unc}"


def _llm_cmd(model_path: str) -> list[str]:
    """Та же строка запуска, что и в ячейке D, но с нужной моделью."""
    gpu = os.path.exists("/dev/nvidia0") or bool(shutil.which("nvidia-smi"))
    # n_ctx — из thinking_env.sh (пишет ячейка D), чтобы запуск и сторожевый
    # перезапуск не разошлись (AUD-08); фолбэк — те же значения, что в D
    ctx = _env_export("THINKING_CTX", "8192" if gpu else "4096")
    return [sys.executable, "-m", "llama_cpp.server",
            "--model", model_path,
            "--n_ctx", ctx,
            "--n_gpu_layers", "-1",
            "--host", LLM_HOST, "--port", str(LLM_PORT),
            "--model_alias", MODEL_NAME,
            "--n_threads", "4", "--n_batch", "512", "--verbose", "False"]


async def _llm_ready(tries: int = 90, gap: float = 2.0) -> bool:
    url = f"http://{LLM_HOST}:{LLM_PORT}/v1/models"
    for _ in range(tries):
        try:
            async with httpx.AsyncClient(timeout=5) as c:
                r = await c.get(url)
            if r.status_code == 200:
                return True
        except httpx.HTTPError:
            pass
        await asyncio.sleep(gap)
    return False


async def _switch_to(path: str) -> dict:
    """Убивает текущий движок и поднимает на выбранной модели."""
    S.switching = True
    try:
        await emit("thought", f"Переключаю модель на {_model_label(path)}…")
        # Убиваем только ОСНОВНОЙ движок (:8001): широкий паттерн сносил
        # параллельные llm2/llm3, а /parallel продолжал читать устаревший
        # манифест — панель показывала модели, которых уже нет (аудит B)
        subprocess.run(["pkill", "-f", f"llama_cpp.server.*--port {LLM_PORT}"],
                       capture_output=True)
        await asyncio.sleep(2)
        # with закрывает родительскую копию хендлера — у процесса своя,
        # а хендлеров, утекающих на каждое переключение, больше нет (C-6)
        with open(LLM_LOG, "a", encoding="utf-8") as log:
            S.llm_proc = subprocess.Popen(_llm_cmd(path), stdout=log,
                                          stderr=subprocess.STDOUT)
        if not await _llm_ready():
            raise RuntimeError(f"{_model_label(path)} не поднялась")
        try:
            with open(ACTIVE_MODEL_FILE, "w", encoding="utf-8") as fh:
                fh.write(path)
        except OSError:
            pass
        S.stats["model_switches"] = int(S.stats.get("model_switches", 0)) + 1
        await emit("final", f"Модель переключена: {_model_label(path)}")
        return {"ok": True, "active": path, "label": _model_label(path)}
    finally:
        S.switching = False


@app.get("/models")
async def models(x_agent_token: str = Header(default=""), token: str = ""):
    """Какие модели лежат в рантайме и какая активна сейчас (+ подсказка
    нужного режима CPU/T4 для каждой)."""
    _auth(x_agent_token, token)
    items = _model_list()
    gpu = os.path.exists("/dev/nvidia0") or bool(shutil.which("nvidia-smi"))
    active = _active_model()
    return {
        "active": active,
        "active_label": _model_label(active) if items and active else "",
        "switching": bool(S.switching),
        "gpu": bool(gpu),
        "models": [{"path": p, "label": _model_label(p),
                    "size_gb": round(os.path.getsize(p) / 2 ** 30, 2),
                    "hint": _model_hint(p, gpu),
                    "active": p == active} for p in items],
    }


class ModelRequest(BaseModel):
    model: str = ""


@app.post("/model")
async def set_model(req: ModelRequest, x_agent_token: str = Header(default=""),
                    token: str = ""):
    """Переключение модели на лету: панель зовёт это напрямую."""
    _auth(x_agent_token, token)
    want = req.model.strip()
    items = _model_list()
    if not items:
        raise HTTPException(404, "в рантайме нет моделей (выполни ячейку A)")
    target = next((p for p in items if p == want), None)
    if target is None:                       # прислали label: "3b", "1.5B"
        low = want.lower()
        target = next((p for p in items if _model_label(p).lower() == low), None)
    if target is None:
        target = next((p for p in items if low and low in os.path.basename(p).lower()), None)
    if target is None:
        raise HTTPException(404, f"модель «{want}» не найдена; есть: "
                                  + ", ".join(_model_label(p) for p in items))
    if target == _active_model() and await _llm_ready(tries=2, gap=1.0):
        return {"ok": True, "active": target, "label": _model_label(target),
                "note": "уже активна"}
    try:
        return await _switch_to(target)
    except Exception as exc:                 # noqa: BLE001
        S.stats["errors"] += 1
        await emit("error", f"переключение не удалось: {exc}")
        raise HTTPException(500, str(exc)[:300])


PARALLEL_FILE = "/content/parallel_models.json"


def _parallel_models() -> list[dict]:
    """Дополнительные движки, поднятые ячейкой D (по одному на модель).

    Читаются из манифеста, а не из памяти: манифест пишет ячейка D, а сервер
    может перезапускаться сторожем и должен подхватить список заново.
    """
    try:
        with open(PARALLEL_FILE, encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        return []
    return [d for d in data if isinstance(d, dict) and d.get("url")]


@app.get("/parallel")
async def parallel(x_agent_token: str = Header(default=""), token: str = ""):
    """Какие модели держатся параллельно и отвечают прямо сейчас."""
    _auth(x_agent_token, token)
    items = _parallel_models()
    return {"active": _model_label(_active_model()),
            "url": UPSTREAM,
            "extra": items,
            "count": len(items)}


@app.post("/ask/multi")
async def ask_multi(req: ChatRequest, x_agent_token: str = Header(default=""),
                    token: str = ""):
    """Один вопрос — несколько моделей одновременно, ответы рядом.

    Честно о цене: ядра у рантайма те же самые, поэтому СУММАРНАЯ скорость
    не растёт — каждый ответ просто становится в N раз дольше. Польза не в
    скорости, а в сравнении: видно, где сильная модель ошибается, а где
    слабая справляется, и ответ можно выбрать руками.

    Дополнительные движки поднимает ячейка D (THINKING_PARALLEL) и пишет
    манифест; если его нет — отвечает только основная модель.
    """
    _auth(x_agent_token, token)
    t0 = time.time()
    targets: list[tuple[str, str]] = [(_model_label(_active_model()), UPSTREAM)]
    targets += [(str(d.get("label") or d.get("path")), str(d["url"]))
                for d in _parallel_models()]
    max_tokens = min(int(os.environ.get("THINKING_CHAT_MAX_TOKENS",
                                      str(max(MAX_TOKENS, 700)))), _ctx_limit())
    await emit("thought", f"Параллельный вопрос {len(targets)} моделям: "
                          f"{req.message[:100]}")

    async def one(label: str, url: str) -> dict:
        started = time.time()
        buf: list[str] = []
        try:
            async for piece in chat_stream(SYSTEM_CHAT, build_chat_prompt(req),
                                           max_tokens, url=url):
                buf.append(piece)
            return {"model": label, "ok": True,
                    "answer": "".join(buf).strip()[:4000],
                    "seconds": round(time.time() - started, 1)}
        except Exception as exc:             # noqa: BLE001
            S.stats["errors"] += 1
            return {"model": label, "ok": False,
                    "error": f"{type(exc).__name__}: {exc}"[:200],
                    "seconds": round(time.time() - started, 1)}

    answers = await asyncio.gather(*(one(lbl, u) for lbl, u in targets))
    S.stats["chats"] += 1
    return {"answers": list(answers), "count": len(answers),
            "wall_seconds": round(time.time() - t0, 1),
            "note": "суммарная скорость не растёт: ядра те же, ответы идут параллельно"}


@app.post("/chat/stream")
async def chat_stream_ep(req: ChatRequest, x_agent_token: str = Header(default=""),
                         token: str = ""):
    """Чат потоком: токены приходят по мере генерации.

    Нужен потому, что бесплатный туннель Cloudflare рвёт молчащий запрос на
    120-й секунде: развёрнутый ответ на 700 токенов на CPU занимает минуты.
    Поток держит соединение живым, поэтому длина ответа больше не ограничена
    временем ожидания первого байта.
    """
    _auth(x_agent_token, token)
    t0 = time.time()
    memory = req.memory if isinstance(req.memory, dict) else {}
    facts = [str(f) for f in (memory.get("facts") or []) if str(f).strip()]
    used = bool(facts or req.profile or req.history)
    await emit("thought", f"Чат (поток): {req.message[:120]}")
    max_tokens = min(int(os.environ.get("THINKING_CHAT_MAX_TOKENS",
                                      str(max(MAX_TOKENS, 700)))),
                      _ctx_limit())      # бюджет не должен превышать окно

    async def gen():
        buf: list[str] = []
        try:
            async with sem_slot():
                async for piece in chat_stream(SYSTEM_CHAT,
                                               build_chat_prompt(req), max_tokens):
                    buf.append(piece)
                    yield ("data: " + json.dumps({"type": "token", "text": piece},
                                                 ensure_ascii=False) + "\n\n")
            raw = "".join(buf)
            data = await guard_reply(raw)
            reply = str(data.get("reply") or "").strip()
            yield ("data: " + json.dumps({
                "type": "done",
                "reply": reply,
                "rationale": str(data.get("rationale") or "")[:600],
                "source": "colab", "fallback": False, "memory_used": used,
                "duration_ms": int((time.time() - t0) * 1000),
            }, ensure_ascii=False) + "\n\n")
            await emit("final", f"[чат-поток] {reply[:200]}")
            S.stats["chats"] += 1
        except Exception as exc:             # noqa: BLE001
            S.stats["errors"] += 1
            yield ("data: " + json.dumps({"type": "error", "text": str(exc)[:300]},
                                         ensure_ascii=False) + "\n\n")
    return StreamingResponse(with_heartbeat(gen()), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


@app.post("/dev")
async def dev(req: DevRequest, x_agent_token: str = Header(default=""),
              token: str = ""):
    """Режим «Разработка» (без потока): модель возвращает предложение
    {action, filename, code, comment}. Файлы НЕ изменяются — применение
    остаётся за человеком на ПК (контроль сохраняется)."""
    _auth(x_agent_token, token)
    t0 = time.time()
    await emit("thought", f"Разработка: {req.message[:120]}")
    max_tokens = min(int(os.environ.get("THINKING_DEV_MAX_TOKENS",
                                      str(max(MAX_TOKENS, 1600)))),
                      _ctx_limit())
    try:
        raw, usage = await chat_json_usage(SYSTEM_DEV, build_dev_prompt(req),
                                           max_tokens)
        data = await dev_data(raw)
    except HTTPException:
        raise
    except Exception as exc:             # noqa: BLE001
        S.stats["errors"] += 1
        await emit("error", f"разработка: {str(exc)[:200]}")
        raise HTTPException(500, str(exc)[:300])
    S.stats["chats"] += 1
    await emit("final", f"[{data['action']}] {data['filename'] or 'без файла'}: "
                        f"{data['comment'][:200]}")
    data["tokens_in"] = int(usage.get("prompt_tokens") or 0)
    data["tokens_out"] = int(usage.get("completion_tokens") or 0)
    data["tokens_estimate"] = not bool(usage)
    data["duration_ms"] = int((time.time() - t0) * 1000)
    data["at"] = utcnow()
    return data


@app.post("/dev/stream")
async def dev_stream(req: DevRequest, x_agent_token: str = Header(default=""),
                     token: str = ""):
    """Тот же запрос, но потоком: JSON приходит токенами, поэтому длинный
    код не обрывается 120-секундным лимитом бесплатного туннеля."""
    _auth(x_agent_token, token)
    t0 = time.time()
    await emit("thought", f"Разработка (поток): {req.message[:120]}")
    # бюджет, как и у соседних маршрутов (/dev, /chat/stream, /ask/multi),
    # прижимается к окну модели: иначе промт с кодом активного файла уезжает
    # вместе с 1600 токенами и llama.cpp молча режет его (аудит B-4)
    max_tokens = min(int(os.environ.get("THINKING_DEV_MAX_TOKENS",
                                        str(max(MAX_TOKENS, 1600)))),
                     _ctx_limit())

    async def gen():
        buf: list[str] = []
        try:
            async with sem_slot():
                async for piece in chat_stream(SYSTEM_DEV, build_dev_prompt(req),
                                               max_tokens):
                    buf.append(piece)
                    yield ("data: " + json.dumps({"type": "token", "text": piece},
                                                 ensure_ascii=False) + "\n\n")
            data = await dev_data("".join(buf))
            yield ("data: " + json.dumps({
                "type": "done", **data,
                "tokens_estimate": True,
                "duration_ms": int((time.time() - t0) * 1000),
                "at": utcnow(),
            }, ensure_ascii=False) + "\n\n")
            await emit("final", f"[{data['action']}] "
                                f"{data['filename'] or 'без файла'}: "
                                f"{data['comment'][:200]}")
            S.stats["chats"] += 1
        except Exception as exc:         # noqa: BLE001
            S.stats["errors"] += 1
            yield ("data: " + json.dumps({"type": "error", "text": str(exc)[:300]},
                                         ensure_ascii=False) + "\n\n")
    return StreamingResponse(with_heartbeat(gen()), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


def _read_dump(path: str) -> bytes:
    """Чтение дампа в рабочем потоке.

    Файл может быть многомегабайтным, а синхронный read() в обработчике
    блокировал весь цикл событий — вместе с пингами живых потоков и
    соседними запросами (аудит B, 03-colab).
    """
    with open(path, "rb") as fh:
        return fh.read()


@app.get("/dump/{name}")
async def dump(name: str, x_agent_token: str = Header(default=""), token: str = ""):
    """Отдаёт серверный файл из белого списка — чтобы ПК мог снять логи
    в архив без ручного копирования. Требует тот же токен, что и остальное."""
    _auth(x_agent_token, token)
    path = DUMP_FILES.get(name)
    if not path:
        raise HTTPException(404, f"файл «{name}» не в списке: {', '.join(DUMP_FILES)}")
    try:
        data = await asyncio.to_thread(_read_dump, path)
    except OSError as exc:
        raise HTTPException(404, f"нет файла: {exc}")
    await emit("thought", f"выгружен файл {name} ({len(data)} байт)")
    return StreamingResponse(iter([data]), media_type="application/octet-stream",
                             headers={"Content-Disposition":
                                      f'attachment; filename="{name}"'})


PANEL_HTML = """<!doctype html><html lang="ru"><head><meta charset="utf-8">
<title>Мышление</title><style>
body{margin:0;background:#0d1117;color:#c9d1d9;font:13px/1.45 ui-monospace,Consolas,monospace}
header{display:flex;gap:12px;align-items:center;padding:9px 13px;background:#161b22;border-bottom:1px solid #30363d}
#b{padding:2px 9px;border-radius:11px;background:#3d1418;color:#ff7b72}
#b.on{background:#12351f;color:#7ee787}
main{display:grid;grid-template-columns:1fr 1fr;gap:11px;padding:11px 13px}
section{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:9px;height:84vh;overflow-y:auto}
h2{font-size:11px;margin:0 0 7px;color:#8b949e;letter-spacing:.08em}
.ev{padding:3px 0;border-bottom:1px dashed #21262d;font-size:12px;word-break:break-word}
.t{color:#8b949e;margin-right:6px}
.thought .x{color:#9ecbff}.rationale .x{color:#e6edf3}.plan_step .x{color:#7ee787}
.contradiction .x{color:#ffa657}.final .x{color:#fff;font-weight:600}
.error .x{color:#ff7b72}.token .x{color:#8b949e}
</style></head><body>
<header><b>Субагент «Мышление»</b><span id="b">…</span><span id="m" style="color:#8b949e"></span></header>
<main><section><h2>МЫСЛИ (LIVE)</h2><div id="a"></div></section>
<section><h2>ПОСЛЕДНИЕ СОБЫТИЯ</h2><div id="c"></div></section></main>
<script>
const TOK=__TOKEN__;
function H(){return TOK?{"X-Agent-Token":TOK}:{}}
function add(id,ev){const b=document.getElementById(id);if(b.children.length>300)b.removeChild(b.firstChild);
const d=document.createElement('div');d.className='ev '+(ev.type||'');
const s=document.createElement('span');s.className='t';s.textContent=(ev.ts||'').slice(11,19);
const x=document.createElement('span');x.className='x';x.textContent=ev.text||'';
d.append(s,x);b.appendChild(d);b.scrollTop=b.scrollHeight}
let es=new EventSource("/events/stream?token="+encodeURIComponent(TOK));
es.onopen=()=>{document.getElementById('b').className='on';document.getElementById('b').textContent='ПОДКЛЮЧЕНО'};
es.onerror=()=>{document.getElementById('b').className='';document.getElementById('b').textContent='НЕТ СВЯЗИ'};
es.onmessage=e=>{try{const ev=JSON.parse(e.data);add('c',ev);if(ev.type!=='final')add('a',ev)}catch(x){}};
async function st(){try{const r=await fetch('/health?token='+encodeURIComponent(TOK),{headers:H()});
const h=await r.json();document.getElementById('m').textContent=h.model+' · '+h.gpu+' · uptime '+h.uptime_s+' с'
+' · планов '+h.plans}catch(x){}}
st();setInterval(st,4000);
</script></body></html>"""
