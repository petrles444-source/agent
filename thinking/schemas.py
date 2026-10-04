"""Контракты подсистемы «Мышление».

Единственный источник правды по полям, которые ходят между Colab и ПК.
Реализация — только стандартная библиотека (dataclasses + ручная валидация):
зависимостей нет вовсе, тянуть pydantic ради пары структур незачем.

Снимок схемы лежит в docs/schema_plan.json и сверяется tools/thinking_test.py —
правка полей без правки снимка валит тест.

Валидация делится на две категории:
  * структурная (нет шагов, дубли id, больше 30 шагов) -> SchemaError, план отбраковывается;
  * косметическая (неизвестный тег action, кривой retry_policy) -> нормализуется,
    чтобы план не выбрасывался из-за одного выдуманного слова.
"""
from __future__ import annotations

import json
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

ACTIONS = ("build", "test", "refactor", "verify", "docs", "prompt", "debug", "release")
RETRY_POLICIES = ("none", "once", "exponential")
EVENT_TYPES = ("thought", "token", "rationale", "plan_step",
               "contradiction", "final", "error")
SOURCES = ("colab", "local-fallback")
STATUSES = ("ok", "adjust", "abort")
CHAT_ROLES = ("user", "subagent")
MAX_STEPS = 30
# Русский текст ~= 2 символа на токен, английский/JSON ~= 4. Для честной оценки
# берём консервативное 3: завышать расход безопаснее, чем занижать.
CHARS_PER_TOKEN = 3
MAX_MEMORY_TURNS = 12
MAX_MEMORY_FACTS = 40

# ключи, значения которых нельзя писать в журнал и слать в промт.
# Кавычки вокруг ключа и значения (JSON/словарь) пропускаются: раньше
# `"password": "hunter2"` проходил мимо редьюсера, и секрет уезжал в Colab
# и оседал в журналах (аудит B-1).
SECRET_RE = re.compile(
    r"\b(password|passwd|secret|api[_-]?key|access[_-]?token|private[_-]?key|token)\b"
    r"""["']?\s*[=:]\s*["']?[^\s"',}]+""",
    re.I,
)


class SchemaError(ValueError):
    """Ответ субагента не прошёл контракт."""


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _secret_rhs(match: "re.Match[str]") -> str:
    """Правая часть «ключ = значение» из совпадения SECRET_RE."""
    full, key = match.group(0), match.group(1) or ""
    tail = full.split(key, 1)[-1] if key in full else full
    # кавычки значения в JSON — не часть секрета, их надо снять, чтобы
    # «token = os.getenv(…)» в кавычках осталось ссылкой, а не значением
    return tail.lstrip("=: \t\"'")


# Значения-ссылки, а не секреты: ключ присваивается вызову/переменной
# окружения или пуст/логичен. Раньше «token = os.getenv(…)» в сообщении
# или в коде блокировал запрос ложным срабатыванием (аудит AUD-13),
# при этом настоящие литералы (пароли, ключи) блокируются как раньше.
_SECRET_REF = ("os.getenv(", "os.environ[", "environ[", "getenv(",
               "self.", "cfg[", "config.", "settings.", "args.",
               '""', "''", "None", "True", "False")


def _is_secret_value(match: "re.Match[str]") -> bool:
    return not _secret_rhs(match).startswith(_SECRET_REF)


def redact_secrets(text: str) -> str:
    """Заменяет значения секретов на [скрыто] перед записью/отправкой.

    Ссылки на переменные окружения (os.getenv(...)) не трогаем — это код,
    а не значение."""
    def _sub(m: "re.Match[str]") -> str:
        return m.group(0) if not _is_secret_value(m) else f"{m.group(1)}=[скрыто]"
    return SECRET_RE.sub(_sub, text or "")


def has_secret(text: str) -> bool:
    return any(_is_secret_value(m) for m in SECRET_RE.finditer(text or ""))


def _str_list(value: Any, limit: int = 40) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)):
        return []
    return [str(v)[:500] for v in value if str(v).strip()][:limit]


def _redact(value: Any, limit: int = 800) -> str:
    """Редактирование любого текстового поля плана.

    Аудит B: редактировался только `rationale`, а `goal`, `fallback`,
    `sub_goals`, `constraints`, `contradictions`, `success_criteria`,
    `unknown_files` и `steps[].desc` уходили в журнал и на панель сырыми —
    секрет в любом из них утекал в logs/thinking/*.jsonl.
    """
    return redact_secrets(str(value or ""))[:limit]


def _redact_list(value: Any, limit: int = 40) -> list[str]:
    return [redact_secrets(str(v))[:500] for v in _str_list(value, limit)]


def _norm_action(value: Any) -> str:
    """Неизвестный тег -> verify (косметика, не ошибка контракта)."""
    tag = str(value or "").strip().lower()
    return tag if tag in ACTIONS else "verify"


def _norm_retry(value: Any) -> str:
    tag = str(value or "").strip().lower()
    return tag if tag in RETRY_POLICIES else "once"


# --------------------------------------------------------------------------- #
#  Шаг плана
# --------------------------------------------------------------------------- #
@dataclass
class PlanStep:
    id: int
    action: str
    desc: str
    inputs: dict = field(default_factory=dict)
    expected_output: Optional[str] = None
    retry_policy: str = "once"
    depends_on: list[int] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Any) -> "PlanStep":
        if not isinstance(data, dict):
            raise SchemaError(f"шаг должен быть объектом, получено {type(data).__name__}")
        try:
            sid = int(data.get("id"))
        except Exception:
            raise SchemaError(f"шаг без числового id: {data.get('id')!r}")
        desc = str(data.get("desc") or "").strip()
        if not desc:
            raise SchemaError(f"шаг {sid}: пустой desc")
        inputs = data.get("inputs")
        depends = data.get("depends_on") or []
        if not isinstance(depends, (list, tuple)):
            depends = []
        try:
            depends = [int(d) for d in depends]
        except Exception:
            depends = []
        exp = data.get("expected_output")
        return cls(
            id=sid,
            action=_norm_action(data.get("action")),
            desc=_redact(data.get("desc"), 400),
            inputs=inputs if isinstance(inputs, dict) else {},
            expected_output=_redact(exp, 400) if exp else None,
            retry_policy=_norm_retry(data.get("retry_policy")),
            depends_on=[d for d in depends if d != sid],
        )

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
#  План
# --------------------------------------------------------------------------- #
@dataclass
class Plan:
    plan_id: str
    goal: str
    steps: list[PlanStep]
    source: str = "colab"
    rationale: str = ""
    confidence: float = 0.7
    sub_goals: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    contradictions: list[str] = field(default_factory=list)
    success_criteria: list[str] = field(default_factory=list)
    unknown_files: list[str] = field(default_factory=list)
    fallback: str = "выполнить напрямую, без субагента"
    created_at: str = ""

    @classmethod
    def from_dict(cls, data: Any) -> "Plan":
        if not isinstance(data, dict):
            raise SchemaError(f"план должен быть объектом, получено {type(data).__name__}")
        raw = data.get("steps")
        if not isinstance(raw, list) or not raw:
            raise SchemaError("план без шагов (steps)")
        if len(raw) > MAX_STEPS:
            raise SchemaError(f"слишком много шагов: {len(raw)} > {MAX_STEPS}")
        steps = [PlanStep.from_dict(s) for s in raw]
        ids = [s.id for s in steps]
        if len(ids) != len(set(ids)):
            raise SchemaError(f"дублирующиеся id шагов: {ids}")
        unknown = set(ids)
        for step in steps:
            unknown -= set(step.depends_on)
        if unknown and len(steps) > 1 and any(s.depends_on for s in steps):
            # depends_on, указывающий на несуществующий шаг, — не фатально, но чистим
            for step in steps:
                step.depends_on = [d for d in step.depends_on if d in set(ids)]
        try:
            conf = float(data.get("confidence", 0.7))
        except Exception:
            conf = 0.7
        source = str(data.get("source") or "colab")
        if source not in SOURCES:
            source = "colab"
        return cls(
            plan_id=str(data.get("plan_id") or uuid.uuid4()),
            goal=_redact(data.get("goal"), 500) or "(без названия)",
            steps=steps,
            source=source,
            rationale=_redact(data.get("rationale"), 800),
            confidence=max(0.0, min(1.0, conf)),
            sub_goals=_redact_list(data.get("sub_goals"), 20),
            constraints=_redact_list(data.get("constraints")),
            contradictions=_redact_list(data.get("contradictions")),
            success_criteria=_redact_list(data.get("success_criteria"), 20),
            unknown_files=_redact_list(data.get("unknown_files"), 40),
            fallback=_redact(data.get("fallback"), 400),
            created_at=str(data.get("created_at") or utcnow()),
        )

    @classmethod
    def from_json(cls, text: str) -> "Plan":
        return cls.from_dict(extract_json(text))

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, **kw: Any) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, **kw)


# --------------------------------------------------------------------------- #
#  Рефлексия
# --------------------------------------------------------------------------- #
@dataclass
class ReflectResponse:
    status: str = "ok"
    advice: str = ""
    rationale: str = ""
    next_steps: list[PlanStep] = field(default_factory=list)
    updated_goal_stack: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Any) -> "ReflectResponse":
        if not isinstance(data, dict):
            raise SchemaError("рефлексия должна быть объектом")
        status = str(data.get("status") or "ok").strip().lower()
        if status not in STATUSES:
            status = "ok"
        steps: list[PlanStep] = []
        raw = data.get("next_steps")
        if isinstance(raw, list):
            for item in raw[:8]:
                try:
                    steps.append(PlanStep.from_dict(item))
                except SchemaError:
                    continue
        return cls(
            status=status,
            advice=redact_secrets(str(data.get("advice") or ""))[:800],
            rationale=redact_secrets(str(data.get("rationale") or ""))[:400],
            next_steps=steps,
            updated_goal_stack=_str_list(data.get("updated_goal_stack"), 12),
        )

    @classmethod
    def from_json(cls, text: str) -> "ReflectResponse":
        return cls.from_dict(extract_json(text))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["next_steps"] = [s.to_dict() for s in self.next_steps]
        return d


# --------------------------------------------------------------------------- #
#  Чат: человек ↔ субагент
# --------------------------------------------------------------------------- #
def estimate_tokens(text: Any) -> int:
    """Оценка токенов по длине текста — когда сервер не отдал usage.

    Точная цифра доступна только от LLM (usage в ответе v1/chat/completions).
    Пока сервер её не присылает, считаем по CHARS_PER_TOKEN и честно помечаем
    такую цифру как оценку (estimate=True), чтобы её не спутать с точной.
    """
    return max(0, (len(str(text or "")) + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN)


@dataclass
class ChatTurn:
    """Одна реплика в чате: кто говорил, что и когда."""
    role: str = "user"          # user | subagent
    text: str = ""
    at: str = ""
    plan_id: str = ""
    fallback: bool = False

    @classmethod
    def from_dict(cls, data: Any) -> "ChatTurn":
        if not isinstance(data, dict):
            raise SchemaError("реплика чата должна быть объектом")
        role = str(data.get("role") or "user").strip().lower()
        if role not in CHAT_ROLES:
            role = "user"
        return cls(
            role=role,
            text=redact_secrets(str(data.get("text") or ""))[:4000],
            at=str(data.get("at") or utcnow()),
            plan_id=str(data.get("plan_id") or "")[:64],
            fallback=bool(data.get("fallback")),
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ChatReply:
    """Ответ субагента в чате: текст + мысли + честный учёт токенов."""
    reply: str = ""
    rationale: str = ""
    steps: list[str] = field(default_factory=list)
    source: str = "colab"
    plan_id: str = ""
    fallback: bool = False
    memory_used: bool = False
    tokens_in: int = 0
    tokens_out: int = 0
    tokens_estimate: bool = True
    duration_ms: int = 0
    at: str = field(default_factory=utcnow)

    @classmethod
    def from_dict(cls, data: Any) -> "ChatReply":
        if not isinstance(data, dict):
            raise SchemaError("ответ чата должен быть объектом")
        source = str(data.get("source") or "colab")
        if source not in SOURCES:
            source = "colab"
        steps: list[str] = []
        for item in (data.get("steps") or [])[:8]:
            if isinstance(item, dict):
                desc = str(item.get("desc") or "").strip()
            else:
                desc = str(item).strip()
            if desc:
                steps.append(desc[:400])
        try:
            dur = max(0, int(data.get("duration_ms") or 0))
        except (TypeError, ValueError):
            dur = 0
        return cls(
            reply=redact_secrets(str(data.get("reply") or ""))[:4000],
            rationale=redact_secrets(str(data.get("rationale") or ""))[:800],
            steps=steps,
            source=source,
            plan_id=str(data.get("plan_id") or "")[:64],
            fallback=bool(data.get("fallback")) or source == "local-fallback",
            memory_used=bool(data.get("memory_used")),
            tokens_in=max(0, int(data.get("tokens_in") or 0)),
            tokens_out=max(0, int(data.get("tokens_out") or 0)),
            tokens_estimate=bool(data.get("tokens_estimate", True)),
            duration_ms=dur,
            at=str(data.get("at") or utcnow()),
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def tokens_total(self) -> int:
        return int(self.tokens_in) + int(self.tokens_out)


def chat_schema() -> dict:
    """Снимок контракта чата — сверяется тестом (docs/schema_chat.json)."""
    return {
        "turn": {
            "role": list(CHAT_ROLES),
            "text": "str, до 4000 символов",
            "at": "str, ISO-8601 UTC",
            "plan_id": "str, до 64",
            "fallback": "bool",
        },
        "reply": {
            "reply": "str, до 4000 — ответ человеку",
            "rationale": "str, до 800 — почему такой ответ",
            "steps": "list[str], до 8 — краткий план, если он уместен",
            "source": list(SOURCES),
            "plan_id": "str, до 64",
            "fallback": "bool — true, если это шаблон, а не LLM",
            "memory_used": "bool — подтянулась ли память",
            "tokens_in": "int — токены в промт (0 = неизвестно)",
            "tokens_out": "int — токены ответа (0 = неизвестно)",
            "tokens_estimate": "bool — цифры посчитаны по длине текста",
            "duration_ms": "int",
            "at": "str, ISO-8601 UTC",
        },
        "memory": {
            "facts": f"list[str], до {MAX_MEMORY_FACTS}",
            "turns": f"list[ChatTurn], до {MAX_MEMORY_TURNS}",
            "updated": "str, ISO-8601 UTC",
        },
        "constants": {
            "CHARS_PER_TOKEN": CHARS_PER_TOKEN,
            "MAX_MEMORY_TURNS": MAX_MEMORY_TURNS,
            "MAX_MEMORY_FACTS": MAX_MEMORY_FACTS,
        },
    }


# --------------------------------------------------------------------------- #
#  Разбор JSON от LLM
# --------------------------------------------------------------------------- #
def extract_json(text: str) -> dict:
    """Достаёт первый сбалансированный JSON из ответа LLM.

    Понимает markdown-обёртки, хвостовой текст и висячие запятые.
    Бросает ValueError, если JSON достать не удалось.
    """
    t = (text or "").strip()
    if not t:
        raise ValueError("пустой ответ LLM")
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", t).strip()
    try:
        out = json.loads(t)
        if isinstance(out, dict):
            return out
        raise ValueError("ответ LLM — не объект JSON")
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
                    fixed = re.sub(r",\s*([}\]])", r"\1", chunk)
                    out = json.loads(fixed)
                    if not isinstance(out, dict):
                        raise ValueError("ответ LLM — не объект JSON")
                    return out
    raise ValueError("несбалансированный JSON в ответе LLM")


# --------------------------------------------------------------------------- #
#  Снимок схемы (docs/schema_plan.json)
# --------------------------------------------------------------------------- #
def plan_schema() -> dict:
    return {
        "title": "Plan",
        "type": "object",
        "required": ["plan_id", "goal", "steps"],
        "properties": {
            "plan_id": {"type": "string"},
            "source": {"type": "string", "enum": list(SOURCES)},
            "rationale": {"type": "string", "maxLength": 800},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "goal": {"type": "string", "minLength": 1, "maxLength": 500},
            "sub_goals": {"type": "array", "items": {"type": "string"}, "maxItems": 20},
            "constraints": {"type": "array", "items": {"type": "string"}},
            "contradictions": {"type": "array", "items": {"type": "string"}},
            "steps": {
                "type": "array",
                "minItems": 1,
                "maxItems": MAX_STEPS,
                "items": {"$ref": "#/definitions/PlanStep"},
            },
            "success_criteria": {"type": "array", "items": {"type": "string"}, "maxItems": 20},
            "unknown_files": {"type": "array", "items": {"type": "string"}},
            "fallback": {"type": "string", "maxLength": 400},
            "created_at": {"type": "string", "format": "date-time"},
        },
        "definitions": {
            "PlanStep": {
                "type": "object",
                "required": ["id", "action", "desc"],
                "properties": {
                    "id": {"type": "integer"},
                    "action": {"type": "string", "enum": list(ACTIONS)},
                    "desc": {"type": "string", "minLength": 1, "maxLength": 400},
                    "inputs": {"type": "object"},
                    "expected_output": {"type": ["string", "null"], "maxLength": 400},
                    "retry_policy": {"type": "string", "enum": list(RETRY_POLICIES)},
                    "depends_on": {"type": "array", "items": {"type": "integer"}},
                },
            },
        },
    }


def reflect_schema() -> dict:
    return {
        "title": "ReflectResponse",
        "type": "object",
        "required": ["status", "advice"],
        "properties": {
            "status": {"type": "string", "enum": list(STATUSES)},
            "advice": {"type": "string", "maxLength": 800},
            "rationale": {"type": "string", "maxLength": 400},
            "next_steps": {"type": "array", "items": {"$ref": "#/definitions/PlanStep"}},
            "updated_goal_stack": {"type": "array", "items": {"type": "string"}},
        },
        "definitions": plan_schema()["definitions"],
    }
