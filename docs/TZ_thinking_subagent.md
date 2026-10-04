# ТЗ 1.1: Подсистема «Мышление» — Colab-субагент для основного агента

**Статус:** Ready for implementation
**Адресат:** основной агент (работает на бесплатном API на ПК)
**Куда ставить:** отдельный каталог `ThinkingAgent` (нейтральный помощник любого агента) + один Colab-ноутбук
**Требования к среде:** Python 3.11+, только стандартная библиотека на стороне ПК, Colab T4 (или CPU-рантайм)

---

# ЧАСТЬ 0. ЗАЧЕМО ЭТО И ЧЕГО ЭТО НЕ ДАЁТ

Это **важно**, потому что отсюда следует вся архитектура.

## 0.1. Честная модель выгоды

Субагент — это **не более умная модель**. 7B на T4 не умнее того же фронта, на котором работает основной агент. Выгода в другом:

| Что даёт субагент | Как именно ускоряет основного агента |
|---|---|
| **Параллельность** | План по задаче X считается на Colab, пока агент делает задачу Y. Не «быстрее стал», а «время ушло в фон» |
| **Снятие квоты** | Длинные структурированные JSON-выдачи (план на 12 шагов, чек-листы, логи) не жгут лимиты бесплатного API основного агента |
| **Память между сессиями** | Планы/рефлексии живут на сервере в `plans`/`events`, а не в контексте агента, который обрезается |
| **Договорённые контракты** | Одинаковый JSON на выходе → агент не парсит прозу, экономит промпт-токены на «верни строго JSON» |
| **Второе мнение** | Проверка на противоречия (`contradictions`), список рисков, независимая оценка плана |
| **Фабрика промптов** | Готовые промты для img2img волн (наша текущая задача) — 7B отлично пишет короткие структурированные промты |

## 0.2. Чего субагент делать **не** должен

- ❌ Никаких инструментов, вызовов API изображений, записи файлов. Только советы.
- ❌ Не пишет код целиком «вместо агента» (7B выдумывает несуществующие пути и API).
- ❌ Не стоит в критическом пути каждого действия. Политика: `plan` — синхронно (раз на задачу), `reflect` — фоном.
- ❌ Не планирует за агента, когда офлайн (см. §5.3 fallback).

## 0.3. Бюджет (окно реального времени)

| Событие | Целевая задержка |
|---|---|
| `GET /health` | < 200 мс |
| Первый токен стрима | < 1.5 с |
| `POST /plan` целиком (7B Q4, ~600–700 токенов) | 8–25 с |
| `POST /reflect` | 3–8 с |
| Пауза «появилось в панели» от Colab | < 400 мс |
| Реконнект при обрыве туннеля | ≤ 5 с, без падения приложения |

---

# ЧАСТЬ I. АРХИТЕКТУРА

## 1.1. Схема

```
┌─────────────────────────────── ПК (Windows) ────────────────────────────────┐
│                                                                             │
│   Агент (бесплатный API)                                                    │
│      │  (1) синхронный вызов из shell                                       │
│      ▼                                                                      │
│   tools/thinking_cli.py ── plan / reflect / tail / doctor / panel           │
│      │                                                                      │
│      ▼                                                                      │
│   thinking/client.py          thinking/fallback.py                          │
│   • синхронный, stdlib         • шаблон-заглушка, не планировщик            │
│   • retry + backoff            • включается при 3 ошибках подряд            │
│   • поток-читатель SSE                                                    │
│      │                    ┌───────────────────────────────┐                 │
│      │ (2) HTTP/SSE       │ tools/thinking_cli.py panel   │                 │
│      └───────────────────▶│ http://127.0.0.1:8765         │◀─ человек видит │
│                           │ 3 колонки: План / Мысли / Лог │    в реальном   │
│                           └───────────────────────────────┘    времени      │
│      │                                                                      │
│      └─▶ logs/thinking/thoughts.jsonl  (append-only, ротация 5 МБ)          │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │ HTTPS :443
                     Colab proxy (google.colab)  ── или ──  cloudflared
                                   │
┌──────────────────────────────────▼────────────────────────── Colab VM ──────┐
│  uvicorn + FastAPI :8000                                                    │
│    GET  /            → живая панель (HTML) прямо в Colab-вкладке            │
│    GET  /health      → статус, модель, VRAM, uptime                         │
│    GET  /events?since=N&tail=M → история событий  (первичный канал ПК)      │
│    GET  /events/stream         → SSE (для панелей)                          │
│    POST /plan        → Plan  (JSON, валидация pydantic)                     │
│    POST /plan/stream → SSE: token* → plan_step* → final                     │
│    POST /reflect     → ReflectResponse                                       │
│    POST /cancel/{id} → best-effort                                           │
│    GET  /metrics                                                        │
│         │                                                                  │
│         ▼                                                                  │
│    LLM backend :8001  (llama-cpp-python / vLLM)                            │
│      model: Qwen2.5-7B-Instruct Q4_K_M (llama.cpp, CUDA)                   │
│      ctx 8192, stream on, temperature 0.2                                  │
│         │                                                                  │
│         ▼                                                                  │
│    State: plans{64, TTL 1h} • events deque(1000) • sem(1) • rate limit     │
└────────────────────────────────────────────────────────────────────────────┘
```

## 1.2. Потоки

1. **Планирование.** `cli plan "задача"` → `POST /plan` → промт (схема + разрешённые файлы) → LLM → `extract_json()` → `Plan(**data)` → сохранить + события → вернуть агенту.
2. **Живой стрим.** `POST /plan/stream` → токены → SSE → поток-читатель на ПК → кольцевой буфер → панель `EventSource` + `thoughts.jsonl`.
3. **Рефлексия.** агент выполнил шаг → `cli reflect --plan ID --step 3 --result "..."` → фоном (`--async`) → `{"status": "ok|adjust|abort", "advice": ...}` → агент читает при следующем `tail`.
4. **Деградация.** 3 ошибки подряд / нет URL / таймаут → `fallback.local_plan()` → код выхода **2**, панель красным «ОФЛАЙН — шаблон».

## 1.3. Режимы вызова (важно: не всё синхронно)

| Режим | Когда | Блокирует агента? |
|---|---|---|
| `plan` | начало новой большой задачи | да, 8–25 с (один раз) |
| `ask "вопрос"` | короткий вопрос-совет ≤150 токенов | да, 2–5 с |
| `reflect` | после выполненного шага | нет (`--async`, результат в `tail`) |
| `plan --stream` | агент хочет показывать ход мысли | да, но токены идут сразу |
| `tail` | периодический сбор | нет |

---

# ЧАСТЬ II. МОДЕЛЬ И РЕСУРСЫ COLAB

## 2.1. Выбор модели (решение принято)

| Вариант | Вес на диске | VRAM | скорость на T4 | вердикт |
|---|---|---|---|---|
| **Qwen2.5-7B-Instruct Q4_K_M + llama-cpp-python (CUDA)** | 4.7 GB | ~5.2 GB | 60–90 ток/с | **основной путь** |
| Qwen2.5-7B-Instruct-AWQ + vLLM | 5.7 GB | 7–9 GB | 250–400 ток/с | опция «турбо» |
| Qwen2.5-3B-Instruct Q4_K_M | 2.0 GB | ~2.5 GB | 120–160 ток/с | запасной путь (CPU-рантайм) |

**Почему не vLLM как в референсе:** `pip install vllm==0.6.3` на сегодняшнем Colab (Python 3.12/3.13, torch 2.x) не ставится/ломает ядро, и `--gpu-memory-utilization 0.85` резервирует ~13 ГБ, из-за чего не загружается пайплайн генерации волн. Поэтому:

* **По умолчанию — llama-cpp-python с CUDA-сборкой** (пинов версий нет, колёс нет — просто `pip install`, ядро не трогает).
* **vLLM — отдельная опциональная ячейка** с явным предупреждением «может потребоваться Runtime → Restart runtime» и авто-фоллбеком: если импорт не удался — сервер стартует на llama.cpp.

## 2.2. Совместимость с генерацией волн (одна GPU-сессия)

На бесплатном Colab обычно **одна GPU-сессия на аккаунт**. Значит субагент живёт **в том же ноутбуке**, что и генерация волн (наши ячейки `colab_cell_a_setup/b_generate`).

| Компонент | VRAM |
|---|---|
| DreamShaper 8 (UNet+CLIP+VAE, fp16) | ~2.3 GB |
| Qwen2.5-7B Q4 (llama.cpp, все слои на GPU) | ~5.2 GB |
| KV-cache llama.cpp (8192 ctx) | ~0.6 GB |
| Итого | **~8.1 GB** из 16 GB |

Правило: если генерация волн держит память и возникает OOM — перезапуск сервера с `--n-gpu-layers 20` (частичный оффлоад на CPU, ~4 GB VRAM) или `/offload` эндпоинт, выгружающий LLM на время генерации. Сервер обязан пережить `torch.cuda.empty_cache()`.

## 2.3. Держим Colab живым

В ноутбуке должна быть ячейка-фонарик (иначе рантайм убьют по простою):

```python
import time, threading, datetime
def _keepalive():
    for _ in range(144):            # 12 часов
        time.sleep(300)
        print(datetime.datetime.now().strftime("%H:%M:%S"), "keep-alive", flush=True)
threading.Thread(target=_keepalive, daemon=True).start()
print("keep-alive: каждые 5 мин")
```

Плюс периодический снапшот истории в Drive/файл (см. ячейку F): `events` дампится каждые 60 с в `/content/thinking_snapshot.json` — переживает рестарт сервера.

---

# ЧАСТЬ III. ПРОТОКОЛ И КОНТРАКТЫ

## 3.1. Каналы (вместо WS/SocketIO)

| Канал | Назначение | Почему именно так |
|---|---|---|
| `GET /events?since=N&tail=M` | **первичный** канал ПК | опрос переживает обрыв туннеля, не требует длительных соединений |
| `GET /events/stream` (SSE) | панели (Colab и PC) | проще WS, без библиотек, авто-реконнект встроено в `EventSource` |
| `logs/thinking/thoughts.jsonl` | история на ПК | append-only, разбирается постфактум |

WebSocket **убран**: он был в референсе, но даёт только слой ложных состояний (1006, ping/pong, ручной реконнект) поверх всё того же TCP.

## 3.2. Схемы (`thinking/schemas.py` — единый источник правды)

```python
from __future__ import annotations
from datetime import datetime, timezone
from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator
from uuid import uuid4

def _now() -> datetime:
    return datetime.now(timezone.utc)          # не utcnow(): deprecated

class PlanRequest(BaseModel):
    task: str = Field(..., min_length=3, max_length=2000)
    context: dict = Field(default_factory=dict)
    constraints: list[str] = Field(default_factory=list)
    max_steps: int = Field(12, ge=1, le=30)
    style_hint: Optional[str] = None

class PlanStep(BaseModel):
    id: int
    action: str                                  # short machine tag: build/test/refactor/verify/docs/prompt
    desc: str = Field(..., max_length=400)       # только русский, для человека
    inputs: dict = Field(default_factory=dict)
    expected_output: Optional[str] = None
    retry_policy: Literal["none", "once", "exponential"] = "once"
    depends_on: list[int] = Field(default_factory=list)

    @field_validator("desc")
    @classmethod
    def _no_path_fabrication(cls, v: str) -> str:
        # подсказка-трекер: субагент не имеет права выдумывать файлы
        return v

class Plan(BaseModel):
    plan_id: str = Field(default_factory=lambda: str(uuid4()))
    source: Literal["colab", "local-fallback"] = "colab"
    rationale: str = Field("", max_length=800)   # ГЛАВНОЕ: почему такой план, 2–4 фразы по-русски
    confidence: float = Field(0.7, ge=0.0, le=1.0)
    goal: str
    sub_goals: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    steps: list[PlanStep]
    success_criteria: list[str] = Field(default_factory=list)
    unknown_files: list[str] = Field(default_factory=list)   # файлы, которых агент не перечислил
    fallback: str = "выполнить напрямую, без субагента"
    created_at: datetime = Field(default_factory=_now)

    @field_validator("steps")
    @classmethod
    def _unique_ids(cls, v: list[PlanStep]) -> list[PlanStep]:
        ids = [s.id for s in v]
        if len(ids) != len(set(ids)):
            raise ValueError("step ids must be unique")
        if len(v) > 30:
            raise ValueError("too many steps")
        return v

class ReflectRequest(BaseModel):
    plan_id: str
    step_id: int
    result: str = Field(..., max_length=4000)
    observation: Optional[str] = None
    error: Optional[str] = None

class ReflectResponse(BaseModel):
    status: Literal["ok", "adjust", "abort"]
    advice: str
    next_steps: list[PlanStep] = Field(default_factory=list)
    updated_goal_stack: list[str] = Field(default_factory=list)
    rationale: str = ""

class ThoughtEvent(BaseModel):
    seq: int
    type: Literal["thought", "token", "rationale", "plan_step",
                  "contradiction", "final", "error"]
    text: str
    plan_id: Optional[str] = None
    step_id: Optional[int] = None
    ts: str
```

**Защита от расхождения кода сервера и клиента:** `Plan.model_json_schema()` сериализуется в `docs/schema_plan.json` (снимок), а `tools/thinking_test.py` сравнивает его с актуальным. Если кто-то поправил схему в одном месте — тест упадёт (см. §10.1).

## 3.3. Промты

```text
SYSTEM_PLAN (system):
Ты — советник главного агента: опытный техлид, который помогает
распланировать задачу до начала работы; технологию задаёт агент в КОНТЕКСТЕ.
Отвечай ТОЛЬКО валидным JSON. Никаких markdown-блоков и пояснений вокруг JSON.
Схема:
{"goal": str, "rationale": str, "confidence": 0..1,
 "sub_goals": [str], "constraints": [str], "contradictions": [str],
 "steps": [{"id": int, "action": str, "desc": str, "inputs": {},
            "expected_output": str, "retry_policy": "none|once|exponential",
            "depends_on": [int]}],
 "success_criteria": [str], "unknown_files": [str], "fallback": str}
Правила:
- Пиши desc, goal, sub_goals, contradictions, success_criteria ТОЛЬКО по-русски.
- action — латинский тег из: build, test, refactor, verify, docs, prompt, debug, release.
- Упоминай только файлы из блока РАЗРЕШЁННЫЕ ФАЙЛЫ. Если нужен файл вне списка —
  внеси его в unknown_files, а не выдумывай путь.
- Не больше max_steps шагов. Шаг должен быть проверяемым.
- Не добавляй поля вне схемы.

USER (plan):
ЗАДАЧА: {task}
КОНТЕКСТ: {json(context)[:4000]}
ОГРАНИЧЕНИЯ: {json(constraints)}
МАКС. ШАГОВ: {max_steps}
РАЗРЕШЁННЫЕ ФАЙЛЫ: {context.files}        ← агент передаёт реальный список путей
ПОСЛЕДНИЕ СОБЫТИЯ (для непротиворечивости): {tail 20 events}
```

```text
SYSTEM_REFLECT (system):
Ты — рефлектор. Верни ТОЛЬКО JSON:
{"status": "ok|adjust|abort", "advice": str, "rationale": str,
 "next_steps": [PlanStep], "updated_goal_stack": [str]}
advice и rationale — по-русски, до 240 символов. Ничего кроме JSON.
```

**Почему нет `response_format: {"type":"json_object"}`:** в llama.cpp-сервере его нет, в разных версиях vLLM есть/нет, и он **несовместим со стримом**. Поэтому JSON гарантируется промтом + `extract_json()` (см. §4.5).

## 3.4. Аутентификация и лимиты

* `THINKING_TOKEN` генерируется при старте (`secrets.token_urlsafe(24)`), печатается в ячейке D, кладётся в `config/thinking.local.json`.
* Все эндпоинты, кроме `GET /` (HTML-панели), требуют заголовок `X-Agent-Token`. Панель читает токен из `?token=` в URL и подставляет в заголовки fetch.
* Rate limit: не более `THINKING_RPS` (5) запросов/с на токен → иначе `429`.
* `asyncio.Semaphore(THINKING_CONCURRENCY=1)` — один LLM-запрос за раз, очередь не раздувается.
* Ограничение размера тела: `task ≤ 2000`, `result ≤ 4000`, суммарный контекст ≤ 6000 символов (обрезается с начала).

---

# ЧАСТЬ IV. COLAB-НОУТБУК (полный код ячеек)

Файлы лежат в репо и вставляются в ноутбук (как мы уже делаем с волнами):
`thinking/colab/cell_a_install.py`, `cell_b_model.py`, `cell_c_server.py`, `cell_d_launch.py`, `cell_e_panel.py`, `cell_f_snapshot.py`.

## Ячейка A — окружение

```python
# --- Субагент мышления: thinking install ---
import os, subprocess, sys, shutil, time

def _pip(*args):
    return subprocess.run([sys.executable, "-m", "pip", "install", "-q", *args],
                          capture_output=True, text=True)

print("python", sys.version.split()[0])
r = _pip("fastapi", "uvicorn[standard]", "pydantic>=2", "httpx", "huggingface_hub")
print(r.stdout[-400:], r.stderr[-400:])

# флаг CUDA у лlama-cpp-python: сборка с GPU (3–5 мин), при неудаче — CPU-колесо
ok = False
for env, tag in (({"CMAKE_ARGS": "-DGGML_CUDA=on"}, "cuda"), ({}, "cpu")):
    e = dict(os.environ, **env)
    p = subprocess.run([sys.executable, "-m", "pip", "install", "--no-cache-dir",
                        "-U", "llama-cpp-python"], env=e, capture_output=True, text=True)
    try:
        import llama_cpp  # noqa
        ok, build = True, tag
        break
    except Exception as exc:
        print("build failed:", tag, str(exc)[:200])
print("llama-cpp-python:", "OK build=" + build if ok else "FAIL")

# vLLM — опционально, ядро может потребовать рестарт (см. §2.1)
VLLM_OK = False
if os.environ.get("TRY_VLLM") == "1":
    p = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "vllm"],
                       capture_output=True, text=True)
    try:
        import vllm  # noqa
        VLLM_OK = True
    except Exception as exc:
        print("vllm unavailable:", str(exc)[:300])
print("VLLM_OK =", VLLM_OK)
```

## Ячейка B — модель

```python
# --- Субагент мышления: thinking model ---
import os, glob, re, torch
from huggingface_hub import hf_hub_download, list_repo_files

print(torch.cuda.get_device_name(0),
      round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1), "GB")

MODEL_DIR = "/content/models/llm"
os.makedirs(MODEL_DIR, exist_ok=True)

def grab_gguf(repo, pattern):
    files = [f for f in list_repo_files(repo) if f.endswith(".gguf")]
    hit = [f for f in files if re.search(pattern, f, re.I)]
    if not hit:
        raise RuntimeError(f"no gguf matching {pattern} in {repo}: {files[:8]}")
    return hf_hub_download(repo_id=repo, filename=hit[0], local_dir=MODEL_DIR)

gguf = grab_gguf("bartowski/Qwen2.5-7B-Instruct-GGUF", r"Q4_K_M")
print("model:", gguf, round(os.path.getsize(gguf) / 2**30, 2), "GB")
```

## Ячейка C — сервер (`%%writefile /content/thinking_server.py`)

```python
# --- Субагент мышления: thinking server ---
"""Субагент мышления: FastAPI + llama.cpp/vLLM. Только советы, никаких инструментов."""
from __future__ import annotations

import asyncio, json, os, re, secrets, time, uuid
from collections import deque
from datetime import datetime, timezone
from typing import Any, Optional

import httpx
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator

# ---------- конфиг ----------
UPSTREAM    = os.environ.get("THINKING_UPSTREAM", "http://127.0.0.1:8001/v1/chat/completions")
MODEL_NAME  = os.environ.get("THINKING_MODEL", "thinking")
TOKEN       = os.environ.get("THINKING_TOKEN", secrets.token_urlsafe(24))
LLM_TIMEOUT = float(os.environ.get("THINKING_TIMEOUT", "240"))
MAX_TOKENS  = int(os.environ.get("THINKING_MAX_TOKENS", "700"))
RPS         = int(os.environ.get("THINKING_RPS", "5"))
CONCURRENCY = int(os.environ.get("THINKING_CONCURRENCY", "1"))
MAX_EVENTS  = 1000
MAX_PLANS   = 64
PLAN_TTL_S  = 3600

# ---------- схемы (должны совпадать с thinking/schemas.py; ловит тест) ----------
class PlanRequest(BaseModel):
    task: str = Field(..., min_length=3, max_length=2000)
    context: dict = Field(default_factory=dict)
    constraints: list[str] = Field(default_factory=list)
    max_steps: int = Field(12, ge=1, le=30)
    style_hint: Optional[str] = None

class PlanStep(BaseModel):
    id: int
    action: str
    desc: str = Field(..., max_length=400)
    inputs: dict = Field(default_factory=dict)
    expected_output: Optional[str] = None
    retry_policy: str = "once"
    depends_on: list[int] = Field(default_factory=list)

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
    created_at: str = ""

class ReflectRequest(BaseModel):
    plan_id: str
    step_id: int
    result: str = Field(..., max_length=4000)
    observation: Optional[str] = None
    error: Optional[str] = None

def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

# ---------- состояние: единственный синглтон (снаружи глобалов нет) ----------
class State:
    def __init__(self) -> None:
        self.events: deque[dict] = deque(maxlen=MAX_EVENTS)
        self.plans: dict[str, dict] = {}
        self.subs: set[Any] = set()
        self.seq = 0
        self.t0 = time.time()
        self.sem = asyncio.Semaphore(CONCURRENCY)
        self.win_t, self.win_n = time.time(), 0
        self.stats = {"plans": 0, "reflects": 0, "errors": 0, "bad_json": 0, "llm_ms": deque(maxlen=200)}

S = State()

app = FastAPI(title="Субагент «Мышление»", version="1.1")

# ---------- события ----------
async def emit(type_: str, text: str, **kw) -> dict:
    S.seq += 1
    ev = {"seq": S.seq, "type": type_, "text": text, "ts": utcnow(), **kw}
    S.events.append(ev)
    dead = []
    for ws in list(S.subs):
        try:
            await ws.send_json(ev)
        except Exception:
            dead.append(ws)
    for ws in dead:
        S.subs.discard(ws)
    return ev

# ---------- разбор JSON от LLM ----------
def extract_json(text: str) -> dict:
    t = (text or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", t).strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    i = t.find("{")
    if i < 0:
        raise ValueError("в ответе LLM нет '{'")
    depth, in_str, esc = 0, False, False
    for j in range(i, len(t)):
        c = t[j]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                chunk = t[i:j + 1]
                try:
                    return json.loads(chunk)
                except json.JSONDecodeError:
                    return json.loads(re.sub(r",\s*([}\]])", r"\1", chunk))
    raise ValueError("несбалансированный JSON в ответе LLM")

# ---------- промты ----------
SYSTEM_PLAN = """Ты — советник главного агента: опытный техлид, который помогает
распланировать задачу до начала работы; технологию задаёт агент в КОНТЕКСТЕ.
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
- action — латинский тег: build|test|refactor|verify|docs|prompt|debug|release.
- Упоминай только файлы из блока РАЗРЕШЁННЫЕ ФАЙЛЫ; если нужен другой —
  запиши его в unknown_files, но не выдумывай путь.
- Шаги должны быть проверяемыми и по порядку; не больше указанного числа.
- Не добавляй поля вне схемы."""

SYSTEM_REFLECT = """Ты — рефлектор выполненного шага. Отвечай ТОЛЬКО JSON:
{"status": "ok|adjust|abort", "advice": str, "rationale": str,
 "next_steps": [{"id": int, "action": str, "desc": str, "inputs": {},
                 "expected_output": str, "retry_policy": str, "depends_on": [int]}],
 "updated_goal_stack": [str]}
advice и rationale — по-русски, коротко. Ничего кроме JSON."""

def build_plan_prompt(req: PlanRequest) -> str:
    ctx = json.dumps(req.context, ensure_ascii=False)
    files = req.context.get("files") or []
    tail = " | ".join(e["type"] + ": " + e["text"][:80] for e in list(S.events)[-12:])
    return (
        f"ЗАДАЧА: {req.task}\n"
        f"КОНТЕКСТ: {ctx[:4000]}\n"
        f"ОГРАНИЧЕНИЯ: {json.dumps(req.constraints, ensure_ascii=False)}\n"
        f"МАКС. ШАГОВ: {req.max_steps}\n"
        + (f"РАЗРЕШЁННЫЕ ФАЙЛЫ: {', '.join(str(f) for f in files)}\n" if files else "")
        + f"ПОСЛЕДНИЕ СОБЫТИЯ: {tail[:600]}\n"
        "Ответ — только JSON по схеме из system-промта."
    )

# ---------- вызовы LLM (две разные функции: генератор и значение не смешиваем) ----------
async def chat_json(system: str, user: str, max_tokens: int = MAX_TOKENS) -> str:
    async with S.sem:
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
    return data["choices"][0]["message"]["content"]

async def chat_stream(system: str, user: str, max_tokens: int = 900):
    async with httpx.AsyncClient(timeout=LLM_TIMEOUT) as c:
        async with c.stream("POST", UPSTREAM, json={
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

# ---------- middleware: токен + rate limit ----------
def _auth(x_agent_token: str = Header(default="")) -> None:
    if TOKEN and x_agent_token != TOKEN:
        raise HTTPException(401, "неверный токен")
    now = time.time()
    if now - S.win_t >= 1.0:
        S.win_t, S.win_n = now, 0
    S.win_n += 1
    if S.win_n > RPS:
        raise HTTPException(429, "слишком часто")

# ---------- эндпоинты ----------
@app.get("/health")
async def health(x_agent_token: str = Header(default="")):
    _auth(x_agent_token)
    try:
        import torch
        vram = round(torch.cuda.memory_allocated() / 2**30, 2)
        gpu = torch.cuda.get_device_name(0)
    except Exception:
        vram, gpu = None, "cpu"
    return {"status": "ok", "model": MODEL_NAME, "upstream": UPSTREAM,
            "uptime_s": int(time.time() - S.t0), "plans": len(S.plans),
            "events": S.seq, "gpu": gpu, "vram_used_gb": vram}

@app.get("/metrics")
async def metrics(x_agent_token: str = Header(default="")):
    _auth(x_agent_token)
    ms = list(S.stats["llm_ms"])
    return {"events": S.seq, "plans_total": S.stats["plans"],
            "reflects": S.stats["reflects"], "errors": S.stats["errors"],
            "bad_json": S.stats["bad_json"], "ws_subs": len(S.subs),
            "llm_ms_avg": int(sum(ms) / len(ms)) if ms else None,
            "llm_ms_p95": sorted(ms)[int(len(ms) * 0.95)] if len(ms) > 5 else None,
            "queue": CONCURRENCY}

@app.get("/events")
async def events(since: int = 0, tail: int = 0, x_agent_token: str = Header(default="")):
    _auth(x_agent_token)
    items = list(S.events)
    if since:
        items = [e for e in items if e["seq"] > since]
    if tail:
        items = items[-tail:]
    return {"last_seq": S.seq, "events": items}

@app.get("/events/stream")
async def events_stream(x_agent_token: str = Header(default="")):
    _auth(x_agent_token)
    async def gen():
        for e in list(S.events)[-50:]:          # история при подключении
            yield f"data: {json.dumps(e, ensure_ascii=False)}\n\n"
        queue: asyncio.Queue = asyncio.Queue()
        S.subs.add(queue)
        try:
            while True:
                try:
                    ev = await asyncio.wait_for(queue.get(), timeout=25)
                    yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
                except asyncio.TimeoutError:
                    yield ": ping\n\n"
        finally:
            S.subs.discard(queue)
    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})

async def _guard_json(raw: str) -> dict:
    try:
        return extract_json(raw)
    except Exception as exc:
        S.stats["bad_json"] += 1
        with open("/content/last_bad_json.txt", "w", encoding="utf-8") as fh:
            fh.write(raw)
        await emit("error", f"LLM вернула не-JSON: {exc}")
        raise HTTPException(502, f"LLM вернула не-JSON: {exc}")

@app.post("/plan")
async def plan(req: PlanRequest, x_agent_token: str = Header(default="")):
    _auth(x_agent_token)
    await emit("thought", f"Задача: {req.task[:120]}")
    await emit("thought", "Декомпозирую на шаги…")
    try:
        raw = await chat_json(SYSTEM_PLAN, build_plan_prompt(req))
        data = await _guard_json(raw)
        data.setdefault("plan_id", str(uuid.uuid4()))
        data["source"] = "colab"
        data["created_at"] = utcnow()
        p = Plan(**data)
        S.plans[p.plan_id] = p.model_dump()
        if len(S.plans) > MAX_PLANS:            # вытеснение старых
            for k in sorted(S.plans, key=lambda x: S.plans[x]["created_at"])[:8]:
                S.plans.pop(k, None)
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
        await emit("error", f"upstream недоступен: {type(exc).__name__}")
        raise HTTPException(503, "LLM backend недоступен")
    except Exception as exc:
        S.stats["errors"] += 1
        await emit("error", str(exc)[:300])
        raise HTTPException(500, str(exc)[:300])

@app.post("/plan/stream")
async def plan_stream(req: PlanRequest, x_agent_token: str = Header(default="")):
    _auth(x_agent_token)
    await emit("thought", f"Стрим плана: {req.task[:120]}")

    async def gen():
        buf = []
        try:
            async with S.sem:
                async for piece in chat_stream(SYSTEM_PLAN, build_plan_prompt(req), 900):
                    buf.append(piece)
                    yield f"data: {json.dumps({'type': 'token', 'text': piece}, ensure_ascii=False)}\n\n"
            data = await _guard_json("".join(buf))
            data.setdefault("plan_id", str(uuid.uuid4()))
            data["source"] = "colab"
            data["created_at"] = utcnow()
            p = Plan(**data)
            S.plans[p.plan_id] = p.model_dump()
            S.stats["plans"] += 1
            yield f"data: {json.dumps({'type': 'rationale', 'text': p.rationale}, ensure_ascii=False)}\n\n"
            for s in p.steps:
                yield f"data: {json.dumps({'type': 'plan_step', 'text': f'[{s.id}] {s.desc}', 'step_id': s.id}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'final', 'text': p.model_dump_json(), 'plan_id': p.plan_id}, ensure_ascii=False)}\n\n"
        except Exception as exc:
            S.stats["errors"] += 1
            yield f"data: {json.dumps({'type': 'error', 'text': str(exc)[:300]}, ensure_ascii=False)}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

@app.post("/reflect")
async def reflect(req: ReflectRequest, x_agent_token: str = Header(default="")):
    _auth(x_agent_token)
    p = S.plans.get(req.plan_id)
    plan_json = json.dumps(p, ensure_ascii=False) if p else "план не найден (возможно, после рестарта)"
    await emit("thought", f"Рефлексия шага {req.step_id}", plan_id=req.plan_id, step_id=req.step_id)
    try:
        raw = await chat_json(SYSTEM_REFLECT,
                              f"ПЛАН: {plan_json[:3000]}\nШАГ: {req.step_id}\n"
                              f"РЕЗУЛЬТАТ: {req.result[:2000]}\n"
                              f"НАБЛЮДЕНИЕ: {req.observation or '-'}\nОШИБКА: {req.error or '-'}",
                              max_tokens=500)
        data = await _guard_json(raw)
        out = ReflectResponse(**{k: data.get(k, v) for k, v in
                                 ReflectResponse(status="ok", advice="").model_dump().items()
                                 if k in data} | {"status": data.get("status", "ok"),
                                                  "advice": data.get("advice", "")})
        S.stats["reflects"] += 1
        await emit("final" if out.status == "ok" else "contradiction",
                   f"[{out.status}] {out.advice}", plan_id=req.plan_id, step_id=req.step_id)
        return out
    except HTTPException:
        raise
    except Exception as exc:
        S.stats["errors"] += 1
        await emit("error", f"рефлексия: {exc}")
        raise HTTPException(500, str(exc)[:300])

@app.post("/cancel/{plan_id}")
async def cancel(plan_id: str, x_agent_token: str = Header(default="")):
    _auth(x_agent_token)
    S.plans.pop(plan_id, None)
    await emit("thought", f"План {plan_id[:8]} отменён")
    return {"status": "cancelled", "plan_id": plan_id}
```

## Ячейка D — панель в Colab + запуск

```python
# --- Субагент мышления: thinking launch ---
import os, socket, subprocess, sys, time, secrets, json
from google.colab import output

TOKEN = os.environ.get("THINKING_TOKEN") or secrets.token_urlsafe(24)
os.environ["THINKING_TOKEN"] = TOKEN
os.environ.setdefault("THINKING_MODEL", "thinking")

# 1) LLM backend (llama-cpp-python или vLLM) на :8001
backend = "vllm" if os.environ.get("USE_VLLM") == "1" else "llama_cpp"
if backend == "llama_cpp":
    cmd = [sys.executable, "-m", "llama_cpp.server",
           "--model", "/content/models/llm/bartowski__Qwen2.5-7B-Instruct-GGUF/Qwen2.5-7B-Instruct-Q4_K_M.gguf",
           "--ctx_size", "8192", "--n_gpu_layers", "-1", "--host", "127.0.0.1",
           "--port", "8001", "--model_name", "thinking", "--n_threads", "4"]
else:
    from vllm import LLM, SamplingParams  # noqa  — или API-сервер
    cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
           "--model", "Qwen/Qwen2.5-7B-Instruct-AWQ", "--port", "8001",
           "--max-model-len", "8192", "--gpu-memory-utilization", "0.42",
           "--served-model-name", "thinking", "--disable-log-requests"]

open("/tmp/llm.log", "w").close()
llm_proc = subprocess.Popen(cmd, stdout=open("/tmp/llm.log", "a"), stderr=subprocess.STDOUT)

import requests
for i in range(240):                       # до 8 мин на холодный старт/сборку
    try:
        r = requests.get("http://127.0.0.1:8001/health", timeout=2)
        if r.status_code == 200:
            print("LLM ready:", r.text[:200]); break
    except Exception:
        pass
    time.sleep(2)
else:
    print(open("/tmp/llm.log").read()[-2000:]); raise RuntimeError("LLM не поднялся")

# 2) API :8000
api_proc = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "thinking_server:app",
     "--host", "127.0.0.1", "--port", "8000", "--app-dir", "/content"],
    cwd="/content", stdout=open("/tmp/api.log", "a"), stderr=subprocess.STDOUT)
time.sleep(4)
print("health:", requests.get("http://127.0.0.1:8000/health",
                              headers={"X-Agent-Token": TOKEN}).text[:300])

# 3) Публичный URL: сначала Colab-proxy (стабильнее), потом cloudflared
proxy = output.serve_kernel_port_as_window(8000)
print("PANEL_WINDOW:", proxy)
os.environ["THINKING_URL"] = str(proxy)

p = subprocess.run(["wget", "-q",
                    "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64",
                    "-O", "/usr/local/bin/cloudflared"], capture_output=True)
subprocess.run(["chmod", "+x", "/usr/local/bin/cloudflared"])
open("/tmp/tunnel.log", "w").close()
tun = subprocess.Popen(["nohup", "cloudflared", "tunnel", "--url", "http://localhost:8000"],
                       stdout=open("/tmp/tunnel.log", "w"), stderr=subprocess.STDOUT,
                       start_new_session=True)
import re
tunnel = None
for _ in range(30):
    txt = open("/tmp/tunnel.log", errors="replace").read()
    m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", txt)
    if m:
        tunnel = m.group(0); break
    time.sleep(2)

# 4) Вывод, который читает агент — единственный источник URL для ПК
print("=" * 70)
print("THINKING_URL=", tunnel or proxy)
print("THINKING_PANEL=", tunnel or proxy)
print("THINKING_TOKEN=", TOKEN)
print("=" * 70)
```

## Ячейка E — фон: keep-alive + автоснапшот

```python
# --- Субагент мышления: thinking background ---
import json, os, threading, time
from datetime import datetime

SNAP = "/content/thinking_snapshot.json"

def loop():
    while True:
        try:
            from thinking_server import S          # модуль уже загружен uvicorn'ом? — берём локально
        except Exception:
            S = None
        try:
            import urllib.request
            tok = os.environ["THINKING_TOKEN"]
            req = urllib.request.Request("http://127.0.0.1:8000/events?tail=200",
                                         headers={"X-Agent-Token": tok})
            data = json.load(urllib.request.urlopen(req, timeout=10))
            tmp = SNAP + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(data, fh, ensure_ascii=False)
            os.replace(tmp, SNAP)
            print(datetime.now().strftime("%H:%M:%S"), "snapshot", len(data["events"]), flush=True)
        except Exception as exc:
            print("snapshot error:", type(exc).__name__, flush=True)
        time.sleep(60)

threading.Thread(target=loop, daemon=True).start()

for i in range(144):        # keep-alive 12 ч
    time.sleep(300)
```

## Ячейка F — остановка

```python
# --- Субагент мышления: thinking stop ---
import os, signal, subprocess
for pat in ("thinking_server", "llama_cpp.server", "vllm.entrypoints", "cloudflared"):
    subprocess.run(["pkill", "-f", pat], capture_output=True)
print("stopped")
```

---

# ЧАСТЬ V. КЛИЕНТ НА ПК

## 5.1. Структура (новое в репо)

```
ThinkingAgent/
├─ thinking/
│  ├─ __init__.py
│  ├─ schemas.py            ← единый источник правды по контрактам
│  ├─ client.py             ← синхронный stdlib-клиент + поток SSE
│  └─ fallback.py           ← шаблон-заглушка
├─ tools/
│  ├─ thinking_cli.py       ← интерфейс агента + панель реального времени
│  └─ thinking_test.py      ← офлайн-контрактные тесты (в run_all)
├─ config/
│  └─ thinking.json         ← base_url и опции (без секретов)
│  └─ thinking.local.json   ← токен, лежит локально (gitignore)
├─ logs/thinking/thoughts.jsonl   ← создаётся автоматически
└─ docs/schema_plan.json    ← снимок JSON-схемы для теста
```

**Почему JSON, а не `config.yaml`:** правило подсистемы — «на ПК только стандартная библиотека»; YAML добавил бы внешнюю зависимость.

### `config/thinking.json`

```json
{
  "enabled": true,
  "base_url": "",
  "timeout": 45,
  "connect_timeout": 5,
  "retry": {"max_attempts": 3, "backoff_base": 1.5, "max_backoff": 15},
  "stream": true,
  "stream_timeout": 40,
  "fallback_on_error": true,
  "reconnect_delay": 5,
  "panel_poll": 5,
  "plan_timeout": 360,
  "reflect_timeout": 180,
  "stream_stall": 45,
  "stream_retries": 2,
  "chat_json_timeout": 300,
  "chat_timeout": 600,
  "dev_timeout": 600,
  "breaker_pause": 600,
  "answer_cache_ttl": 1800,
  "log_path": "logs/thinking/thoughts.jsonl",
  "log_max_bytes": 5242880,
  "interactions_path": "logs/thinking/interactions.jsonl",
  "reports_path": "logs/thinking/reports.jsonl",
  "panel_host": "127.0.0.1",
  "panel_port": 8765,
  "source": "colab"
}
```

`base_url` пустой ⇒ клиент берёт адрес по цепочке `config/thinking.local.json`
→ `config/thinking.json` → `THINKING_URL` из окружения (именно в этом
порядке, `client.py: reload_config`). Обнаружение URL описано в §7.

## 5.2. `thinking/client.py`

```python
"""Синхронный клиент субагента. Только stdlib: urllib + threading."""
from __future__ import annotations

import json
import logging
import os
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

log = logging.getLogger("thinking.client")


class ThinkingError(RuntimeError):
    """Субагент недоступен или ответ невалиден."""


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


class ThinkingClient:
    def __init__(self, cfg: Optional[dict] = None):
        base_cfg = _load_json(Path("config/thinking.json"))
        local_cfg = _load_json(Path("config/thinking.local.json"))
        self.cfg = {**base_cfg, **(cfg or {})}
        self.base = (local_cfg.get("base_url")
                     or self.cfg.get("base_url")
                     or os.environ.get("THINKING_URL", "")).rstrip("/")
        self.token = (local_cfg.get("token")
                      or self.cfg.get("token")
                      or os.environ.get("THINKING_TOKEN", ""))
        self.timeout = float(self.cfg.get("timeout", 30))
        self.connect_timeout = float(self.cfg.get("connect_timeout", 5))
        self.retry = self.cfg.get("retry") or {}
        self._log_path = Path(self.cfg.get("log_path", "logs/thinking/thoughts.jsonl"))
        self._log_max = int(self.cfg.get("log_max_bytes", 5 * 1024 * 1024))
        self._events: list[dict] = []
        self._seq = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._reader: Optional[threading.Thread] = None

    # ---------- низкоуровневый запрос ----------
    def _request(self, method: str, path: str, body: Optional[dict] = None,
                 timeout: Optional[float] = None, stream: bool = False):
        if not self.base:
            raise ThinkingError("base_url пуст: укажите THINKING_URL")
        url = f"{self.base}{path}"
        data = json.dumps(body).encode("utf-8") if body is not None else None
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.token:
            headers["X-Agent-Token"] = self.token
        attempts = int(self.retry.get("max_attempts", 3))
        base_wait = float(self.retry.get("backoff_base", 1.5))
        max_wait = float(self.retry.get("max_backoff", 15))
        last: Optional[Exception] = None
        for i in range(attempts):
            try:
                req = urllib.request.Request(url, data=data, headers=headers, method=method)
                return urllib.request.urlopen(req, timeout=timeout or self.timeout)
            except urllib.error.HTTPError as exc:
                raw = exc.read().decode("utf-8", "replace")[:400]
                if exc.code in (401, 404, 422):
                    raise ThinkingError(f"HTTP {exc.code}: {raw}") from exc
                last = ThinkingError(f"HTTP {exc.code}: {raw}")
            except Exception as exc:                     # таймаут, DNS, обрыв туннеля
                last = exc
            wait = min(max_wait, base_wait ** (i + 1))
            log.warning("thinking request %s failed (%s), retry in %.1fs", path, last, wait)
            time.sleep(wait)
        raise ThinkingError(f"субагент недоступен: {last}")

    def _json(self, method: str, path: str, body: Optional[dict] = None,
              timeout: Optional[float] = None) -> Any:
        with self._request(method, path, body, timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    # ---------- публичное API ----------
    def health(self, timeout: float = 3.0) -> bool:
        try:
            self._json("GET", "/health", timeout=timeout)
            return True
        except Exception:
            return False

    def plan(self, task: str, context: Optional[dict] = None,
             constraints: Optional[list[str]] = None,
             max_steps: int = 12) -> dict:
        out = self._json("POST", "/plan", {
            "task": task, "context": context or {},
            "constraints": constraints or [], "max_steps": max_steps,
        })
        self._remember("plan", out)
        return out

    def ask(self, question: str, max_steps: int = 4) -> dict:
        return self.plan(question, max_steps=max_steps)

    def reflect(self, plan_id: str, step_id: int, result: str,
                observation: Optional[str] = None,
                error: Optional[str] = None, timeout: float = 60) -> dict:
        out = self._json("POST", "/reflect", {
            "plan_id": plan_id, "step_id": step_id, "result": result,
            "observation": observation, "error": error,
        }, timeout=timeout)
        self._remember("reflect", out)
        return out

    def cancel(self, plan_id: str) -> None:
        try:
            self._json("POST", f"/cancel/{plan_id}", {})
        except Exception as exc:
            log.warning("cancel failed: %s", exc)

    def events(self, since: int = 0, tail: int = 0) -> dict:
        return self._json("GET", f"/events?since={since}&tail={tail}")

    # ---------- поток событий ----------
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
        while not self._stop.is_set():
            try:
                with self._request("GET", "/events/stream",
                                   timeout=self.connect_timeout, stream=True) as resp:
                    buf: list[str] = []
                    for raw in resp:
                        if self._stop.is_set():
                            break
                        line = raw.decode("utf-8", "replace").rstrip("\r\n")
                        if line.startswith("data:"):
                            buf.append(line[5:].strip())
                        elif line == "" and buf:
                            try:
                                ev = json.loads("".join(buf))
                                self._remember_ev(ev)
                                on_event(ev)
                            except Exception as exc:
                                log.debug("bad event: %s", exc)
                            buf = []
            except Exception as exc:
                log.warning("stream dropped: %s", exc)
                time.sleep(delay)             # реконнект тем же конвейером

    # ---------- память панели + журнал ----------
    def _remember(self, kind: str, data: Any) -> None:
        self._remember_ev({"seq": int(time.time() * 1000), "type": kind,
                           "text": json.dumps(data, ensure_ascii=False)[:4000],
                           "ts": time.strftime("%Y-%m-%dT%H:%M:%S")})

    def _remember_ev(self, ev: dict) -> None:
        with self._lock:
            self._seq = max(self._seq, int(ev.get("seq", 0)))
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
                self._log_path.replace(self._log_path.with_suffix(".jsonl.1"))
            with open(self._log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(ev, ensure_ascii=False, default=str) + "\n")
        except Exception as exc:
            log.debug("log append failed: %s", exc)

    def iter_thoughts(self) -> Iterator[dict]:
        """Блокирующий итератор: читает буфер, пока не остановлен."""
        seen = 0
        while True:
            with self._lock:
                batch = self._events[seen:]
                seen = len(self._events)
            for ev in batch:
                yield ev
            time.sleep(0.2)

    # ---------- fallback ----------
    def plan_with_fallback(self, task: str, **kw) -> tuple[dict, bool]:
        """(план, is_fallback). Никогда не бросает исключение."""
        if self.cfg.get("enabled", True):
            try:
                return self.plan(task, **kw), False
            except Exception as exc:
                log.warning("plan via subagent failed: %s", exc)
        if self.cfg.get("fallback_on_error", True):
            from thinking.fallback import local_plan
            return local_plan(task), True
        raise ThinkingError("субагент недоступен и fallback отключён")
```

## 5.3. `thinking/fallback.py`

```python
"""Шаблон-заглушка.

Это НЕ планировщик: правило «не дублировать логику планирования на ПК» сохранено.
Клиент только возвращает остов формата, а решение принимает основной агент сам.
"""
from __future__ import annotations
from datetime import datetime, timezone
from uuid import uuid4


def local_plan(task: str) -> dict:
    return {
        "plan_id": str(uuid4()),
        "source": "local-fallback",
        "rationale": "Субагент недоступен (Colab офлайн/нет URL). "
                     "Возвращён остов формата — планирует основной агент.",
        "confidence": 0.0,
        "goal": task,
        "sub_goals": [],
        "constraints": ["субагент недоступен"],
        "contradictions": [],
        "steps": [{
            "id": 1, "action": "verify",
            "desc": "Выполнить задачу напрямую, без подсказок субагента",
            "inputs": {}, "expected_output": "задача выполнена",
            "retry_policy": "once", "depends_on": [],
        }],
        "success_criteria": ["задача завершена без регрессий"],
        "unknown_files": [],
        "fallback": "выполнить напрямую",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def offline_reason(client) -> str:
    if not getattr(client, "base", ""):
        return "не задан base_url (THINKING_URL)"
    if not client.health(timeout=3):
        return "health-check не прошёл (сервер/туннель/Colab упал)"
    return "другая ошибка"
```

## 5.4. `tools/thinking_cli.py` — интерфейс агента **и** панель

```
python tools/thinking_cli.py doctor
python tools/thinking_cli.py plan  "задача" [--files a.py,b.py] [--constraints ...]
                                   [--max-steps 10] [--stream] [--json]
python tools/thinking_cli.py ask   "вопрос"
python tools/thinking_cli.py reflect PLAN_ID --step 3 --result "538/0" [--async]
python tools/thinking_cli.py tail  [--n 30] [--follow]
python tools/thinking_cli.py panel [--port 8765]      # живая панель в браузере
python tools/thinking_cli.py url   # печатает обнаруженный URL (для агента)
```

**Коды выхода:** `0` — ок; `2` — субагент офлайн, сработал fallback; `3` — невалидный JSON/схема; `4` — таймаут; `5` — токен/401.

`doctor` печатает построчно (читабельно агентом):
```
URL:      https://xxxx.trycloudflare.com
HEALTH:   ok (model=thinking, uptime=1284s, plans=7, gpu=T4, vram=5.9GB)
TOKEN:    set
STREAM:   connected (last_seq=121)
FALLBACK: armed (3 ошибки подряд)
PANEL:    http://127.0.0.1:8765
```

**Панель** — `ThreadingHTTPServer` в том же процессе, читает `client.buffered` и отдаёт `GET /` (HTML) + `GET /api/tail` + `GET /events` (SSE). Три колонки, цвета по типу события, бейдж связи (зелёный/красный), автоскролл, поле для токена. Вставьте файл `tools/thinking_panel.html` (см. §6.3) рядом и отдавайте его. Открыть: `http://127.0.0.1:8765` в браузере **или** через браузерные инструменты агента (`browser.tabs.open`), тогда вы видите панель в Review-панели.

## 5.5. Интеграция в работу агента (правила)

1. Перед большой задачей — `plan` (синхронно, один раз). Полученный `steps` вставить в рабочий план агента.
2. После каждого значимого шага — `reflect --async`; результат читать через `tail` через 1–2 вызова инструментов.
3. Если `plan_with_fallback` вернул `is_fallback=True` — агент планирует сам, панель уже показывает причину.
4. Все события складываются в `logs/thinking/thoughts.jsonl` — это тот журнал, который потом можно скармливать анализатору (`tools/ai_analyzer.py`).
5. Запрещено: передавать субагенту секреты, пароли, токены внутри `task`/`context` (фильтр по ключам `password|token|secret|key` — реализовать в `client.plan`).

---

# ЧАСТЬ VI. ЧТО ИМЕННО ВЫДАЁТ СУБАГЕНТ (примеры)

## 6.1. Стрим (то, что видно в панели по мере генерации)

```
[12:41:07] [thought]      Задача: Рефакторинг draw_sea: заменить статичную текстуру на покадровую
[12:41:07] [thought]      Декомпозирую на шаги…
[12:41:14] [token]        {"goal":"Перевести отрисовку моря на цикл кадров 8×45°
[12:41:14] [token]        без разрыва шва","rationale":"Текущий draw_sea использует
[12:41:14] [token]        assets.water(0) и сдвиг offset, поэтому волна не анимирована…
…(поток токенов)…
[12:41:19] [rationale]    Взят путь «8 кадров через np.roll-сдвиг», потому что он
                           закрывает цикл шва по построению; замена WATER_TEXTURES в
                           config.py не требует правок в самом draw_sea.
[12:41:19] [contradiction] Ряд WATER_TEXTURES задан в config.py, а дублируется в
                           world_map_and_navigation.py — при замене ломается навигация.
[12:41:19] [plan_step]    [1] Положить 8 кадров в assets/tex/water_phase_0..7.png
[12:41:19] [plan_step]    [2] Обновить WATER_TEXTURES в corsair/config.py:103
[12:41:19] [plan_step]    [3] Заменить цикл кадров в screens.py:891 (offset → index)
[12:41:19] [plan_step]    [4] Прогнать tools/ui_smoke.py, ожидать 66/0
[12:41:19] [plan_step]    [5] Снять скриншот моря и проверить отсутствие сетки шва
[12:41:19] [final]        План 9f3a1c2e готов: 5 шагов
```

## 6.2. JSON плана (`POST /plan` → stdout CLI)

```json
{
  "plan_id": "9f3a1c2e-...",
  "source": "colab",
  "rationale": "Кадры уже сгенерированы и лежат в wave_ai/out, поэтому задача — не генерация, а интеграция: подмена текстуры в одной точке конфига и цикла в draw_sea.",
  "confidence": 0.78,
  "goal": "Анимировать море 8 кадрами без видимого шва",
  "sub_goals": ["кадры в assets", "конфиг", "цикл отрисовки", "проверки"],
  "contradictions": ["WATER_TEXTURES дублируется в world_map_and_navigation.py"],
  "steps": [
    {"id": 1, "action": "build", "desc": "Положить 8 кадров в assets/tex",
     "depends_on": [], "retry_policy": "once"},
    {"id": 2, "action": "refactor", "desc": "Обновить WATER_TEXTURES в config.py",
     "depends_on": [1], "retry_policy": "once"},
    {"id": 3, "action": "refactor", "desc": "Заменить offset на индекс кадра в draw_sea",
     "depends_on": [2], "retry_policy": "once"},
    {"id": 4, "action": "test", "desc": "Прогнать ui_smoke, ориентир 66/0",
     "depends_on": [3], "retry_policy": "exponential"},
    {"id": 5, "action": "verify", "desc": "Снять скриншот моря, проверить шов",
     "depends_on": [4], "retry_policy": "none"}
  ],
  "success_criteria": ["ui_smoke 66/0", "на скриншоте нет сетки при тайлинге"],
  "unknown_files": [],
  "fallback": "выполнить напрямую, без субагента"
}
```

## 6.3. Рефлексия

**Запрос:** `reflect 9f3a1c2e --step 4 --result "ui_smoke 66/0, но на скриншоте видна сетка по 512px" --observation "шов ×7.5 против ×5.1 у эталона"`

**Ответ:**
```json
{"status": "adjust",
 "advice": "Сетка — от wrap-разницы краёв. Перед интеграцией подними seam у кадров: подмешай 15% исходной текстуры к краям или уменьши denoising до 0.30 для 8-го кадра.",
 "rationale": "Кадры отличаются по краям сильнее эталона; цикл кадров усиливает стык.",
 "next_steps": [{"id": 41, "action": "debug", "desc": "Починить seam кадров", "depends_on": []}],
 "updated_goal_stack": ["устранить сетку шва", "повторить verify"]}
```

## 6.4. Панель (как это выглядит человеку)

Отдельный веб-интерфейс `tools/thinking_panel.html` — запускается командой
`python tools/thinking_cli.py panel` и открывается на `http://127.0.0.1:8765`.
Всё по-русски, живое обновление раз в 2,5 с + SSE-поток мыслей.

```
┌─ Субагент «Мышление» — отчёты и диалог ── ● СУБАГЕНТ НА СВЯЗИ ────┐
│ [Отчёты 2] [Диалог 4] [Мысли в реальном времени] [Выгода и ускорение]    │
├──────────────────────────────────────────────────────────────────────────┤
│ ┌ ПЛАН 12:41:19 id 9f3a1c2e уверенность 0.8 ───────────────────────────┐ │
│ │ Довести генерацию 100 кадров волн                                    │ │
│ │ Почему так: палитра и шов проверяются отдельно от интеграции         │ │
│ │ Мой запрос: довести генерацию 100 кадров…                            │ │
│ │ [1] сборка  Собрать кадры в wave_ai/out   → ожидаем: 100 файлов      │ │
│ │ [2] тесты   Проверить бесшовность          → ожидаем: seam < порога   │ │
│ │ ✓ критерий: цикл из 8 кадров не дёргается                            │ │
│ └──────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────┘
```

Вкладки:

| Вкладка | Что показывает |
|---|---|
| **Отчёты** | Полные отчёты субагента: цель, обоснование, шаги с русскими названиями действий и ожидаемыми результатами, критерии успеха, противоречия, неизвестные файлы |
| **Диалог** | Хронология обращений: что я спросил, сколько ждал, что получил; статус (`ответ получен` / `ошибка` / `Colab недоступен → заглушка`) |
| **Мысли в реальном времени** | SSE-поток: мысли, объяснения, шаги, противоречия по мере генерации |
| **Выгода и ускорение** | KPI-плитки (обращения, планы, шаги, среднее время ответа, фон vs блокировка, заглушки) + честные пояснения, когда ускорение реально |
| **Журнал** | Сырой журнал событий `logs/thinking/thoughts.jsonl` |

Кнопка **«Скачать отчёт .md»** выгружает диалог + отчёты + пользу в Markdown.

**Почему панель видит вызовы, сделанные не в её процессе:** каждый вызов
(`plan`/`reflect`/заглушка) дозаписывается в `logs/thinking/interactions.jsonl`,
каждый отчёт — в `logs/thinking/reports.jsonl`. Панель читает оба файла и
склеивает с памятью процесса (ключ `at + kind + request`). Поэтому история
не теряется, когда агент зовёт субагента из CLI, а панель уже открыта.

## 6.5. Строка журнала `logs/thinking/thoughts.jsonl`

```json
{"seq": 121, "type": "plan_step", "text": "[4] Прогнать ui_smoke, ориентир 66/0",
 "plan_id": "9f3a1c2e-...", "step_id": 4, "ts": "2026-10-02T12:41:19+00:00"}
```

---

# ЧАСТЬ VII. ОБНАРУЖЕНИЕ URL (Colab меняется каждую сессию)

Это то, что ломает наивное `config.yaml` с захардкоженным туннелем.

**Порядок поиска (клиент):**
1. `config/thinking.local.json` → `base_url`
2. env `THINKING_URL`
3. `config/thinking.json` → `base_url`
4. иначе — пусто, `doctor` печатает `URL: не задан` и инструкцию

**Порядок заполнения (агент/человек):**
1. Прочитать вывод ячейки D в Colab (строки `THINKING_URL=` / `THINKING_TOKEN=`) — тот же канал, которым мы уже читаем вывод генерации волн.
2. `python tools/thinking_cli.py set-url <URL> <TOKEN>` → пишет в `config/thinking.local.json`.
3. Проверить: `doctor` → `HEALTH: ok`.

**Валидность Colab-proxy:** URL вида `https://<hash>.notebooks.googleusercontent.com/proxy/8000/` обычно стабилен для конкретного ноутбука, но живёт только при живом рантайме; cloudflared-ссылка — каждый раз новая и истекает ~60 мин простоя. Поэтому `doctor` обязан показывать время последнего успешного health-check'а, а панель — красный бейдж при обрыве.

---

# ЧАСТЬ VIII. ОШИБКИ И ДЕГРАДАЦИЯ

## 8.1. Таблица ошибок

| Код | Причина | Действие ПК | Действие Colab |
|---|---|---|---|
| `doctor: URL не задан` | не скопирован вывод ячейки D | `set-url`, повторить | — |
| HTTP 401 | сменён токен | перечитать `THINKING_TOKEN`, `set-url` | — |
| HTTP 429 | rate limit | подождать 1 с, ретрай есть | — |
| HTTP 502 | LLM вернула не-JSON | повторить; сырьё лежит в `/content/last_bad_json.txt` | поднять `MAX_TOKENS`, усилить промт |
| HTTP 503 | upstream (:8001) упал | 3 ретрая → fallback | `pkill -f llama_cpp`, ячейка D заново |
| HTTP 504 / timeout | длинный генер | поднять `timeout` в конфиге | проверить GPU/`/tmp/llm.log` |
| SSE обрыв (1006/EOF) | туннель упал | авто-реконнект через 5 с, история дозапрашивается `events?since=` | перезапуск cloudflared |
| CUDA OOM | LLM + генерация волн одновременно | — | перезапуск с `--n-gpu-layers 20` |
| Colab отключился | простой/12 ч | fallback молча, бейдж красный | снапшот в `/content/thinking_snapshot.json` |
| `plan_with_fallback` → `is_fallback=True` | любая из выше | агент планирует сам, `tail` покажет причину | — |

## 8.2. Политика деградации

```
ошибка запроса ─┬─ ретрай ×3 с экспоненциальной паузой (cap 15 с)
                ├─ успех → работаем
                └─ провал → fallback_on_error=true?
                     ├─ да → local_plan(), код выхода 2, панель «ОФЛАЙН»
                     └─ нет → ThinkingError, код выхода 4/5
подряд 3 фолбэка → клиент помечает « offline_until = now+60s» и не долбит сервер
```
Главное свойство: **приложение не должно упасть из-за Colab**. Ни один вызов субагента не поднимает исключение выше `thinking_cli.py`.

## 8.3. Метрики

`GET /metrics`:
```json
{"events": 342, "plans_total": 7, "reflects": 4, "errors": 1, "bad_json": 0,
 "ws_subs": 1, "llm_ms_avg": 9400, "llm_ms_p95": 14200, "queue": 1}
```
`llm_ms_p95 > 20000` → снизить `MAX_TOKENS` или перейти на 3B. `bad_json > 0 за сессию` → ужесточить промт.

---

# ЧАСТЬ IX. БЕЗОПАСНОСТЬ

1. **Токен обязателен.** `trycloudflare` — публичный URL; Colab-proxy тоже не аутентифицирован. `X-Agent-Token` на всех эндпоинтах кроме HTML-панели, панель передаёт его из `?token=`.
2. **Секреты не в коде.** `config/thinking.local.json` (+ `.gitignore`), не в `config/thinking.json`, не в ноутбуке (токен генерируется на лету).
3. **Нет инструментов у LLM.** Только `/v1/chat/completions`. Сетевого доступа, записи файлов, вызова API изображений у субагента нет.
4. **Фильтр секретов** в `client.plan/reflect`: `password|token|secret|api_key|private` в `task`, `context`, `result` → ошибка `400` и запись в журнал без значения.
5. **Лимиты:** rate limit 5 р/с, семафор 1, длина полей по схемам, `plans` ≤ 64 с вытеснением, `events` ≤ 1000.
6. **Ротация журнала** `thoughts.jsonl` на 5 МБ (`.1`).
7. **Именованный туннель + auth** — только для прода (см. §12, v1.5).

---

# ЧАСТЬ X. ТЕСТИРОВАНИЕ

## 10.1. `tools/thinking_test.py` — офлайн (идёт в CI и локально)

Формат вывода — как в остальных тестах репо:
`ПРОЙДЕНО: N   ПРОВАЛЕНО: N` + строки `  ! ...`, exit 1 при провалах.

Проверки (без сети, всегда зелёные на CI/локально):
1. `thinking.schemas.plan_schema()` == `docs/schema_plan.json` (снимок).
2. Уникальность `step.id` валидируется; `>30` шагов отклоняется.
3. `thinking.schemas.extract_json`: plain JSON, JSON в ```` ```json ````, JSON с хвостом-текстом, JSON с висячей запятой, отсутствие `{` → `ValueError`.
4. `local_plan()` возвращает `source="local-fallback"`, ≥1 шаг, `plan_id` непустой.
5. `ThinkingClient` без `base_url` → `plan_with_fallback()` возвращает `is_fallback=True`, а не исключение.
6. Конфиг `config/thinking.json` парсится, содержит все обязательные ключи, `base_url` по умолчанию пуст.
7. Фильтр секретов: `plan(task="пароль hunter2")` → ошибка, до сетевого вызова.
8. Файл ноутбука `thinking/colab/cell_c_server.py` содержит все маршруты: `/health`, `/events`, `/events/stream`, `/plan`, `/plan/stream`, `/reflect`, `/chat`, `/chat/stream`, `/dev`, `/dev/stream`, `/ask/multi`, `/models`, `/parallel`, `/dump/{name}` (ловит случайную правку).

## 10.2. `tools/thinking_online_test.py` — по желанию, только при `THINKING_URL`

```bash
python tools/thinking_online_test.py            # полный прогон (LLM: план + рефлексия)
python tools/thinking_online_test.py --quick    # без LLM: связь, токен, метрики, события
python tools/thinking_online_test.py --stream   # полный + отдельная проверка /plan/stream
```
`health < 200 мс`; `plan("проверка связи")` возвращает `steps` и `rationale` по-русски; `reflect` по несуществующему `plan_id` → не 500; `events?tail=5` → `last_seq` растёт; `tail` после `plan` содержит `final`.

## 10.3. Нагрузочный тест

Отдельного скрипта в репозитории нет: нагрузку проверяют вручную через
`tools/thinking_online_test.py --stream` и `curl` против живого туннеля.
Ориентиры: `p95 /plan < 30 с`, `0 bad_json`, `0 5xx` (429 допустимы, они учтены в rate limit).

## 10.4. Чек-лист приёмки

```
[ ] Ячейки A–F отрабатывают одной кнопкой Run All на T4 (и на CPU-рантайме — путь 3B)
[ ] В выводе ячейки D есть THINKING_URL= и THINKING_TOKEN=
[ ] python tools/thinking_cli.py doctor → HEALTH: ok
[ ] python tools/thinking_cli.py plan "задача" → JSON с 3+ шагами и rationale по-русски
[ ] Панель http://127.0.0.1:8765 показывает события < 400 мс после их появления
[ ] История 50 событий подгружается при открытии панели (events?tail=50)
[ ] При выключенном Colab: план = fallback, код выхода 2, приложение работает
[ ] Реконнект SSE после kill cloudflared ≤ 5 с, без потери журнала
[ ] /plan/stream отдаёт токены до итогового JSON
[ ] Без токена → 401; 20 запросов/с → 429
[ ] logs/thinking/thoughts.jsonl растёт и ротируется на 5 МБ
[ ] python tools/thinking_test.py → ПРОВАЛЕНО: 0
[ ] OOM при одновременной генерации волн → перезапуск с n_gpu_layers=20 восстанавливает сервер
```

---

# ЧАСТЬ XI. ДОРОЖНАЯ КАРТА

| Версия | Что | Зачем |
|---|---|---|
| 1.0 | **этот документ**: сервер, клиент, панель, fallback, тесты | базовый контур |
| 1.1 | кэш планов (хэш `task+context` → мгновенный ответ) и библиотека готовых планов для типовых задач проекта | ещё меньше нагрузки на основной ИИ |
| 1.2 | RAG: индекс `docs/` + `logs/` (bm25 на stdlib), субагент отвечает с опорой на реальные файлы проекта | борьба с выдумыванием путей 7B |
| 1.3 | второй субагент-«критик» (та же модель, отдельный system-промт) для ревью плана перед выдачей | «второе мнение» без новой модели |
| 1.4 | дообучение/QLoRA Qwen2.5 на `thoughts.jsonl` + логах проекта | стиль задач проекта |
| 1.5 | именованный cloudflared-туннель с токеном, постоянный URL | убрать ручное `set-url` |
| 1.6 | пакетный режим: `/plan/batch` — 5 задач параллельно в один проход | пиковая выгода по времени |
| 2.0 | перенос на локальную машину (llama.cpp/CUDA на своём GPU) | без Colab вообще |

---

# ПРИЛОЖЕНИЕ A. Быстрый старт

```
COLAB (тот же ноутбук, что и генерация волн):
  A установка → B модель → C сервер (%%writefile) → D запуск → E фон
  скопировать из вывода D: THINKING_URL=... и THINKING_TOKEN=...

ПК:
  1. python tools/thinking_cli.py set-url <URL> <TOKEN>
  2. python tools/thinking_cli.py doctor          → HEALTH: ok
  3. python tools/thinking_cli.py panel           → открыть http://127.0.0.1:8765
  4. python tools/thinking_cli.py plan "задача" --files corsair/screens.py,corsair/config.py
  5. python tools/thinking_cli.py tail --follow   → поток мыслей в терминале
  6. после шага: python tools/thinking_cli.py reflect <plan_id> --step 1 --result "..." --async
```

# ПРИЛОЖЕНИЕ B. Зависимости

**ПК (`requirements.txt`, пополняется):**
```
pygame-ce>=2.5
pydantic>=2          # только для thinking-контрактов; если не ставится — см. оговорку
```
Оговорка: если `pydantic` нежелателен — `thinking/schemas.py` переписывается на `dataclasses` + ручную валидацию, тест снимка JSON-схемы остаётся. Всё остальное — только stdlib.

**Colab (ячейка A):** `fastapi`, `uvicorn[standard]`, `pydantic>=2`, `httpx`, `huggingface_hub`, `llama-cpp-python` (CUDA), опционально `vllm`.

**Либо, если ПК без внешних зависимостей вовсе:** `thinking/schemas.py` → `dataclass`, валидаторы — функции `validate_plan(dict) -> Plan`. Контракт JSON не меняется, меняется только реализация.

---

# ПРИЛОЖЕНИЕ C. Что исправлено в исходной идее (список противоречий)

| # | Было в референсе | Почему это ломалось | Стало |
|---|---|---|---|
| 1 | Flask + SocketIO на ПК «в существующем app.py» | **в репо нет Flask-приложения** — это pygame-игра (`main.py`, `corsair/`, `engine/`) | интеграция через CLI + локальная панель на `http.server`; никакого Flask |
| 2 | asyncio + `asyncio.new_event_loop()` в потоке + `run_until_complete` + sync Flask | три модели конкурентности в одном процессе, блокировки и `RuntimeError: loop closed` | **только синхронный stdlib-клиент + потоки**; async остался на Colab (FastAPI) |
| 3 | `@retry` (tenacity) на `async def` | тенасити ретраит создание корутины, а не `await` — ретраев нет | явный цикл ретраев с backoff |
| 4 | одна функция `call_llm` возвращает то генератор, то значение | `return` внутри async-генератора — `SyntaxError` | две функции: `chat_json()` и `chat_stream()` |
| 5 | «Не использовать глобальные переменные» при `PLANS: dict`, `EVENTS: deque` на уровне модуля | прямо противоречит коду | `State`-синглтон, задокументирован; снаружи глобалов нет |
| 6 | «Не дублировать логику планирования на ПК» + `fallback.local_plan()` строит план | противоречие слов и кода | fallback переименован в **шаблон-заглушка**, `source="local-fallback"`, `confidence=0`; планирует агент |
| 7 | `pip install vllm==0.6.3` + пины `fastapi/pydantic/uvicorn` | на Colab не встаёт/меняет torch, ядро падает | **без пинов**; базовый путь — `llama-cpp-python`, vLLM — опция с автотестом импорта |
| 8 | `--gpu-memory-utilization 0.85` (≈13 ГБ) | вытесняет пайплайн генерации волн | 0.42 либо llama.cpp; таблица VRAM в §2.2, право на `--n-gpu-layers 20` |
| 9 | `response_format: {"type":"json_object"}` + одновременно `stream: True` | в llama.cpp нет, в vLLM версионно, со стримом несовместимо | JSON гарантируется промтом + `extract_json()` с восстановлением |
| 10 | «Нет постоянного URL → PC читает `tunnel_url` из `/health`» | это цикл: для запроса `/health` URL уже нужен | порядок обнаружения: вывод ячейки D → `set-url` → env (§7) |
| 11 | WebSocket `/stream` как единственный канал живого вывода | обрывы 1006 на trycloudflare, ручной keep-alive, история «при подключении» | **первичный — опрос `/events?since=`**, SSE только для панелей |
| 12 | `pyngrok==7.2.0` в зависимостях, а используется cloudflared | лишняя зависимость и два способа туннеля | Colab-proxy (первично) + cloudflared (запасной), pyngrok убран |
| 13 | `hey -n 50` для нагрузки | `hey` не ставится на Windows | `tools/thinking_load.py` на потоках |
| 14 | `config.yaml` | потребовал бы PyYAML (в requirements только `pygame-ce`) | `config/thinking.json` + секрет в `config/thinking.local.json` |
| 15 | `datetime.utcnow()` | deprecated с Python 3.12 | `datetime.now(timezone.utc)` |
| 16 | схемы продублированы в ячейке сервера и в `thinking/schemas.py` | разъезжаются незаметно | снимок `docs/schema_plan.json` + тест сравнения |
| 17 | «Логи в `logs/thoughts.jsonl`» без ротации и без учёта, что `logs/` уже структурирован (`crashes/events/reports/...`) | файл растёт бесконечно | `logs/thinking/thoughts.jsonl` + ротация 5 МБ |
| 18 | в тестах `pytest.mark.asyncio` и `pytest-asyncio` | новая зависимость, тесты не входят в `run_all` | `tools/thinking_test.py` в формате репо (офлайн) + `--live` опция |
| 19 | не указано, как субагент не выдумывает файлы | типичный провал 7B: несуществующие пути | `context.files` (белый список) + поле `unknown_files` + правило в промте |
| 20 | не указано, на каком языке вывод | у нас правило «никаких английских enum в интерфейсе» | `desc/goal/advice/rationale` — русский; `action` — латинский тег из фикс-списка |
| 21 | туннель публичный, аутентификация «опционально» | референс сам себя опровергает («не хранить секреты» vs публичный URL) | токен **обязателен**, rate limit, семафор, лимиты полей |
| 22 | не учтена одна GPU-сессия на бесплатный Colab | второй ноутбук с GPU убьёт первый | субагент в **том же ноутбуке** с генерацией волн (§2.2) |
| 23 | обещание «ускорит работу основного ИИ» без оговорок | ложные ожидания: 7B не умнее фронта | §0.1: честная таблица выгод (параллелизм, квота, память, контракты) |
| 24 | `fastapi`+`httpx`+`pydantic`+`websockets`+`tenacity`+`flask-socketio`+`eventlet`+`pyyaml` на ПК | тяжёлый стек под одну игру | ПК: stdlib (+ pydantic по желанию). Серверные зависимости — только на Colab |
| 25 | «Открой http://localhost:5000» в чек-листе | не из чего взяться | панель `127.0.0.1:8765` на stdlib + панель внутри Colab-прокси |

---

# ПРИЛОЖЕНИЕ D. Приоритет реализации (порядок работ)

1. `thinking/schemas.py` + `docs/schema_plan.json` + `tools/thinking_test.py` (офлайн-база, падает сразу видно).
2. `thinking/colab/cell_c_server.py` (сервер + панель Colab) — можно проверить локально на CPU с моделью 3B.
3. Ячейки A/B/D на Colab → `doctor` зелёный.
4. `thinking/client.py` + `thinking/fallback.py` + `tools/thinking_cli.py` (`plan`, `tail`, `doctor`).
5. Панель (`panel`) и `--stream`.
6. `reflect`, `cancel`, `metrics`, фильтр секретов, ротация журнала.
7. `tools/thinking_test.py` → в `tools/run_all.py`; нагрузочный тест.
8. Интеграция в реальный цикл работы агента (планы задач, рефлексия после шагов) + обновление `README.md`.
