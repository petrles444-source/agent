# === Субагент «Мышление», ячейка E (фон) ===
# Снапшот событий в файл + keep-alive. Всё в фоновых потоках:
# ячейка НЕ блокирует ядро, остальные ячейки остаются работоспособными.
import json
import os
import threading
import time
import urllib.request
from datetime import datetime

SNAP = "/content/thinking_snapshot.json"
TOKEN = os.environ.get("THINKING_TOKEN", "")


def _snapshot_loop():
    """Раз в минуту думпает события — переживает рестарт uvicorn."""
    while True:
        try:
            req = urllib.request.Request(
                "http://127.0.0.1:8000/events?tail=300",
                headers={"X-Agent-Token": TOKEN})
            data = json.load(urllib.request.urlopen(req, timeout=10))
            tmp = SNAP + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(data, fh, ensure_ascii=False)
            os.replace(tmp, SNAP)
            print(datetime.now().strftime("%H:%M:%S"), "snapshot:",
                  len(data.get("events", [])), flush=True)
        except Exception as exc:
            print("snapshot error:", type(exc).__name__, flush=True)
        time.sleep(60)


def _keepalive():
    for _ in range(144):          # 12 часов, печатает раз в 5 минут
        time.sleep(300)
        print(datetime.now().strftime("%H:%M:%S"), "keep-alive", flush=True)


if TOKEN:
    threading.Thread(target=_snapshot_loop, daemon=True).start()
    print("фон запущен: снапшот событий каждые 60 с ->", SNAP)
else:
    print("токен не задан (сначала ячейка запуска) — снапшот не нужен")
threading.Thread(target=_keepalive, daemon=True).start()
print("keep-alive: 12 ч в фоне (ядро не блокируется)")

# --- опционально: URL/токен на Google Drive, чтобы новая сессия Colab
#     подхватила их сама (примонтируйте Drive один раз в ячейке выше) -------
try:
    drive_dir = "/content/drive/MyDrive"
    if os.path.isdir(drive_dir):
        import shutil
        shutil.copy("/content/thinking_url.txt", os.path.join(drive_dir, "thinking_url.txt"))
        print("URL/токен сохранены на Google Drive -> thinking_url.txt")
    else:
        print("Drive не примонтирован — URL/токен остались только в /content")
except Exception as exc:
    print("Drive не сохранил:", type(exc).__name__, flush=True)

# --- подсказка: восстановить адрес/токен в новой сессии --------------------
print("\nЕсли завтра откроешь ноутбук заново: выполни ячейки A, C, D.")
print("Перед ячейкой D можно задать токен, чтобы адрес ПК не менялся:")
print('    THINKING_TOKEN = "<токен из thinking_url.txt>"')
print("Файл с URL и токеном: /content/thinking_url.txt (и Google Drive, если смонтирован)")
