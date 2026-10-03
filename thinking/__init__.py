"""Подсистема «Мышление» — Colab-субагент-советник для основного агента.

Только советы: никаких инструментов, вызовов API изображений и записи файлов
у субагента нет (см. docs/TZ_thinking_subagent.md, §0.2).
"""
from __future__ import annotations

from thinking.fallback import local_plan, offline_reason
from thinking.schemas import (
    ACTIONS,
    EVENT_TYPES,
    RETRY_POLICIES,
    Plan,
    PlanStep,
    ReflectResponse,
    SchemaError,
    extract_json,
    plan_schema,
)

__all__ = [
    "ACTIONS", "EVENT_TYPES", "RETRY_POLICIES",
    "Plan", "PlanStep", "ReflectResponse", "SchemaError",
    "extract_json", "plan_schema", "local_plan", "offline_reason",
]
