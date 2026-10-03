"""Ручная проверка чата/памяти/токенов/удаления. Запуск: python scripts/smoke_chat.py"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from thinking.client import ThinkingClient  # noqa: E402

c = ThinkingClient()
print("память:", json.dumps(c.memory(), ensure_ascii=False)[:110])
c.remember("проект: ThinkingAgent, субагент на Colab")
r = c.chat("привет, что ты умеешь?")
print("reply:", r["reply"][:110].replace("\n", " "))
print("fallback:", r["fallback"], "| память подтянута:", r["memory_used"],
      "| токены:", r["tokens_in"], "+", r["tokens_out"], "оценка:", r["tokens_estimate"])
t = c.tokens()
print("итого токенов:", t["tokens_total"], "| вызовов:", t["calls"],
      "| только оценка:", t["estimate_only"])
print("удаление несуществующего:", c.delete_history("report", n=999))
mem = c.forget()
print("память после forget():", len(mem["facts"]), "фактов,",
      len(mem["turns"]), "реплик")
