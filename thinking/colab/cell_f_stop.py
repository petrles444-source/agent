# === Субагент «Мышление» — остановка ===
# Останавливает только субагента (LLM, API, туннель). Ячейки генерации волн
# в других ноутбуках не затронуты.
#
# ВАЖНО: эта ячейка входит в «Run all», а раньше из-за неё «Run all» поднимал
# субагента и тут же его гасил (план возвращался, но сервисы уже мертвы).
# Поэтому остановка требует явного разрешения: выполните ячейку, когда
#   THINKING_STOP = "1"
# задана в «Переменных» Colab (или в той же ячейке выше), — тогда она сработает.
import os
import subprocess

if os.environ.get("THINKING_STOP", "").strip() not in ("1", "yes", "true"):
    print("субагент НЕ остановлен: остановка при Run all отключена.\n"
          "  Чтобы остановить его по-настоящему, выполни эту ячейку при\n"
          '  THINKING_STOP = "1" (Переменные Colab) — либо в терминале:\n'
          "  pkill -f thinking_server; pkill -f llama_cpp.server; pkill -f cloudflared")
    raise SystemExit(0)

for pat in ("thinking_server", "llama_cpp.server", "uvicorn thinking_server", "cloudflared"):
    subprocess.run(["pkill", "-f", pat], capture_output=True)
print("субагент остановлен (ячейки генерации волн не затронуты)")
