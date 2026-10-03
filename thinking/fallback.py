"""Шаблон-заглушка на случай, когда Colab недоступен.

Это НЕ планировщик. Правило «не дублировать логику планирования на ПК»
сохранено: клиент возвращает только остов формата, а решение принимает
основной агент сам. Отличить заглушку можно по полю source="local-fallback"
и confidence=0.0.
"""
from __future__ import annotations

from typing import Any
from uuid import uuid4

from thinking.schemas import utcnow


def local_plan(task: str) -> dict:
    return {
        "plan_id": str(uuid4()),
        "source": "local-fallback",
        "rationale": "Субагент недоступен (Colab офлайн / нет URL / обвалился туннель). "
                     "Возвращён остов формата — планирует основной агент.",
        "confidence": 0.0,
        "goal": str(task)[:500] or "(без названия)",
        "sub_goals": [],
        "constraints": ["субагент недоступен"],
        "contradictions": [],
        "steps": [{
            "id": 1,
            "action": "verify",
            "desc": "Выполнить задачу напрямую, без подсказок субагента",
            "inputs": {},
            "expected_output": "задача выполнена",
            "retry_policy": "once",
            "depends_on": [],
        }],
        "success_criteria": ["задача завершена без регрессий"],
        "unknown_files": [],
        "fallback": "выполнить напрямую, без субагента",
        "created_at": utcnow(),
    }


def offline_reason(client: Any) -> str:
    """Человекочитаемая причина, почему субагент не ответил."""
    if not getattr(client, "base", ""):
        return "не задан base_url (выполните: python tools/thinking_cli.py set-url <URL> <TOKEN>)"
    try:
        if client.health(timeout=3):
            return ""
    except Exception as exc:  # pragma: no cover - зависит от среды
        return f"health-check не прошёл: {type(exc).__name__}"
    return "сервер/туннель/Colab не отвечает (проверьте ячейку запуска в Colab)"
