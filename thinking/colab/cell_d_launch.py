# === Субагент «Мышление», ячейка D (запуск LLM + API + туннель) ===
# Идемпотентна: повторный запуск убивает старые процессы и поднимает заново.
import glob
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import time

import requests
from google.colab import output


def _token_from_file() -> str:
    """Токен прошлого запуска — чтобы URL/токен не менялись без нужды."""
    try:
        with open(URL_FILE, encoding="utf-8") as fh:
            m = re.search(r"THINKING_TOKEN=(\S+)", fh.read())
            return m.group(1) if m else ""
    except OSError:
        return ""


def _remember(url: str, token: str) -> None:
    try:
        with open(URL_FILE, "w", encoding="utf-8") as fh:
            fh.write(f"THINKING_URL={url}\nTHINKING_TOKEN={token}\n"
                     f"THINKING_PANEL={url}\nupdated={time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    except OSError:
        pass

LOG_LLM = "/content/llm.log"
LOG_API = "/content/thinking_api.log"
URL_FILE = "/content/thinking_url.txt"     # чтобы завтра не искать токен в выводе

# ---- 0) токен и модель -----------------------------------------------------
# Токен можно задать заранее: THINKING_TOKEN = "..." в этой ячейке (или в
# «Переменных» Colab). Тогда ПК не требует нового set-url после перезапуска.
TOKEN = (os.environ.get("THINKING_TOKEN") or _token_from_file() or secrets.token_urlsafe(24))
os.environ["THINKING_TOKEN"] = TOKEN
os.environ.setdefault("THINKING_MODEL", "thinking")

model_file = "/content/thinking_model_path.txt"
models_file = "/content/thinking_models.txt"
env_file = "/content/thinking_env.sh"
ACTIVE_FILE = "/content/thinking_active_model.txt"

# ячейка A могла оставить путь к библиотекам CUDA (если был починен импорт)
if os.path.exists(env_file):
    for line in open(env_file, encoding="utf-8"):
        if line.startswith("export "):
            key, _, val = line[7:].strip().partition("=")
            os.environ[key] = val

assert os.path.exists(model_file), "сначала выполните ячейку A (установка)"
MODEL = open(model_file, encoding="utf-8").read().strip()
assert os.path.exists(MODEL), f"модель не найдена: {MODEL}"
MODELS = [m for m in (open(models_file, encoding="utf-8").read().split("\n")
                      if os.path.exists(models_file) else []) if m] or [MODEL]
print("MODEL:", MODEL, "| запасные:", [os.path.basename(m) for m in MODELS[1:]])

# Длина ответа подбирается под скорость модели на этом рантайме.
# Замерено на Colab CPU (2 ядра): Qwen2.5-1.5B ≈ 6 ток/с, 3B ≈ 2.5 ток/с,
# 7B ≈ 1.3 ток/с. Режим «сильнее и дольше» (выбор пользователя): под 7B
# бюджеты подняты 500/800; не-потоковый ответ на 124.8 с через туннель
# прошёл (живая проба 03.10), старый страх обрыва на 120-й секунде
# себя не оправдал. С видеокартой можно больше: THINKING_MAX_TOKENS = "700".
MAX_TOKENS_BY_MODEL = {"1.5b": "400", "3b": "260", "7b": "500"}
_key = next((k for k in MAX_TOKENS_BY_MODEL
             if k in os.path.basename(MODEL).lower()), "3b")
os.environ.setdefault("THINKING_MAX_TOKENS", MAX_TOKENS_BY_MODEL[_key])
# Для потокового чата (/chat/stream) можно больше: токены идут по мере генерации,
# туннель Cloudflare не рвёт соединение, а человек видит текст сразу.
CHAT_TOKENS_BY_MODEL = {"1.5b": "700", "3b": "500", "7b": "800"}
os.environ.setdefault("THINKING_CHAT_MAX_TOKENS",
                      CHAT_TOKENS_BY_MODEL[_key])
# Активная модель — в файле: его читает и сервер (вкладка «Модели» в панели),
# и сторож, чтобы переключение не слетело после перезапуска движка.
with open(ACTIVE_FILE, "w", encoding="utf-8") as fh:
    fh.write(MODEL)
print("ЛИМИТ ОТВЕТА:", os.environ["THINKING_MAX_TOKENS"],
      "токенов / чат-поток:", os.environ["THINKING_CHAT_MAX_TOKENS"],
      "(модель", _key + ")")

# ---- 1) остановка прошлых экземпляров -------------------------------------
subprocess.run(["pkill", "-f", "llama_cpp.server"], capture_output=True)
subprocess.run(["pkill", "-f", "uvicorn thinking_server"], capture_output=True)
subprocess.run(["pkill", "-f", "cloudflared"], capture_output=True)
time.sleep(2)


def wait_http(url, tries=150, gap=2, payload=None):
    for _ in range(tries):
        try:
            r = requests.get(url, timeout=4) if payload is None else None
            if r is not None and r.status_code == 200:
                return r
        except Exception:
            pass
        time.sleep(gap)
    return None


# ---- 2) LLM backend на :8001 ----------------------------------------------
# Контекст 4096 на CPU (быстрее считать) и 8192 на GPU — качество не страдает.
GPU = os.path.exists("/dev/nvidia0") or bool(shutil.which("nvidia-smi"))
CTX = "8192" if GPU else "4096"
llm_cmd = [sys.executable, "-m", "llama_cpp.server",
           "--model", MODEL, "--n_ctx", CTX, "--n_gpu_layers", "-1",
           "--host", "127.0.0.1", "--port", "8001", "--model_alias", "thinking",
           "--n_threads", "4", "--n_batch", "512", "--verbose", "False"]
open(LOG_LLM, "a", encoding="utf-8").write(f"\n--- {time.strftime('%H:%M:%S')} {llm_cmd}\n")

# Модель могла не подойти — тогда берём следующую из списка (их до трёх).
llm_proc = None
health = None
for attempt, cand in enumerate(MODELS, 1):
    cmd = list(llm_cmd)
    cmd[cmd.index("--model") + 1] = cand
    llm_proc = subprocess.Popen(cmd, stdout=open(LOG_LLM, "a", encoding="utf-8"),
                                stderr=subprocess.STDOUT)
    print(f"llm pid {llm_proc.pid} ({attempt}/{len(MODELS)}: {os.path.basename(cand)})")
    health = wait_http("http://127.0.0.1:8001/v1/models", tries=45, gap=2)
    if health is not None:
        MODEL = cand
        break
    print(f"  {os.path.basename(cand)} не поднялась, пробуем следующую")
    try:
        llm_proc.kill()
    except Exception:
        pass
    time.sleep(3)          # порт 8001 должен освободиться до следующей попытки

if health is None:
    print("LLM НЕ ПОДНЯЛСЯ. Последние 60 строк лога:")
    print("\n".join(open(LOG_LLM, encoding="utf-8", errors="replace").readlines()[-60:]))
    raise RuntimeError("llama_cpp.server не стартовал")
print("MODEL в работе:", os.path.basename(MODEL))
# сторож в конце ячейки перезапускает llm_cmd — значит в нём должна быть
# та модель, которая реально поднялась, а не первая из списка
llm_cmd[llm_cmd.index("--model") + 1] = MODEL
print("LLM ready:", health.text[:200])

# ---- 3) смоук-тест генерации ----------------------------------------------
t0 = time.time()
smoke = requests.post(
    "http://127.0.0.1:8001/v1/chat/completions",
    json={"model": "thinking", "temperature": 0,
          "max_tokens": 24, "messages": [{"role": "user",
                                          "content": "Ответь ровно одним словом: готов"}]},
    timeout=180)
dt = time.time() - t0
txt = smoke.json()["choices"][0]["message"]["content"]
print(f"смоук: {dt:.1f} с -> {txt[:80]!r}")

# ---- 4) API на :8000 -------------------------------------------------------
assert os.path.exists("/content/thinking_server.py"), "сначала ячейка C (сервер)"
os.environ["THINKING_UPSTREAM"] = "http://127.0.0.1:8001/v1/chat/completions"
api_cmd = [sys.executable, "-m", "uvicorn", "thinking_server:app",
           "--host", "127.0.0.1", "--port", "8000", "--app-dir", "/content"]
open(LOG_API, "a", encoding="utf-8").write(f"\n--- {time.strftime('%H:%M:%S')} {api_cmd}\n")
api_proc = subprocess.Popen(api_cmd, cwd="/content",
                            stdout=open(LOG_API, "a", encoding="utf-8"),
                            stderr=subprocess.STDOUT)
print("api pid", api_proc.pid)

hdr = {"X-Agent-Token": TOKEN}
health_api = wait_http("http://127.0.0.1:8000/health", tries=60, gap=1)
# /health требует токен, поэтому просто бьём до тех пор, пока не получим 200
for _ in range(60):
    try:
        r = requests.get("http://127.0.0.1:8000/health", headers=hdr, timeout=4)
        if r.status_code == 200:
            health_api = r
            break
    except Exception:
        pass
    time.sleep(1)
if health_api is None:
    print("\n".join(open(LOG_API, encoding="utf-8", errors="replace").readlines()[-40:]))
    raise RuntimeError("uvicorn thinking_server не стартовал")
print("API ready:", health_api.text[:300])

# ---- 5) публичные URL ------------------------------------------------------
try:
    panel_url = output.serve_kernel_port_as_window(8000)
except Exception as exc:
    panel_url = None
    print("proxy window:", exc)

tunnel = None
tunnel_kind = "нет"
try:
    if not os.path.exists("/usr/local/bin/cloudflared"):
        subprocess.run(["wget", "-q",
                        "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64",
                        "-O", "/usr/local/bin/cloudflared"], check=True)
        os.chmod("/usr/local/bin/cloudflared", 0o755)
    open("/content/tunnel.log", "w", encoding="utf-8").close()

    # 1) ИМЕНОВАННЫЙ туннель Cloudflare — постоянный адрес на сутки и дольше.
    #    Задайте переменную CLOUDFLARE_TUNNEL_TOKEN (dashboard → tunnels → token)
    #    или вставьте токен ниже: CF_TOKEN = "eyJhIjoi..."
    CF_TOKEN = os.environ.get("CLOUDFLARE_TUNNEL_TOKEN", "")
    if CF_TOKEN:
        subprocess.Popen(["nohup", "cloudflared", "tunnel", "--no-autoupdate", "run",
                          "--token", CF_TOKEN],
                         stdout=open("/content/tunnel.log", "w", encoding="utf-8"),
                         stderr=subprocess.STDOUT, start_new_session=True)
        time.sleep(12)
        txt = open("/content/tunnel.log", encoding="utf-8", errors="replace").read()
        m = re.search(r"https://[a-z0-9.-]+\.(?:trycloudflare\.com|"
                      r"[a-z0-9-]+\.[a-z0-9-]+\.workers\.dev|[a-z0-9.-]+\.cfargotunnel\.com)", txt)
        if m:
            tunnel, tunnel_kind = m.group(0), "именованный (постоянный)"
        else:
            print("ИМЕНОВАННЫЙ туннель: адрес не распознан в логе, смотрите /content/tunnel.log")
    # 2) Быстрый туннель — бесплатно и без аккаунта, но адрес случайный на каждый запуск
    if not tunnel:
        # --protocol http2: quick-туннель по умолчанию ходит по QUIC/UDP, а на
        # Colab ядра дают крошечный UDP-буфер (quic-go пишет «wanted 7168 kiB,
        # got 416 kiB») — SSE-потоки посреди передачи замирали, done не доходил.
        # HTTP/2 идёт по TCP и этих потерь не видит.
        subprocess.Popen(["nohup", "cloudflared", "tunnel", "--no-autoupdate",
                          "--protocol", "http2",
                          "--url", "http://localhost:8000"],
                         stdout=open("/content/tunnel.log", "a", encoding="utf-8"),
                         stderr=subprocess.STDOUT, start_new_session=True)
        for _ in range(40):
            txt = open("/content/tunnel.log", encoding="utf-8", errors="replace").read()
            m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", txt)
            if m:
                tunnel, tunnel_kind = m.group(0), "быстрый (случайный адрес)"
                break
            time.sleep(2)
except Exception as exc:
    print("tunnel error:", exc)

url = tunnel or (str(panel_url) if panel_url else "")
_remember(url, TOKEN)

# ---- 6) проверка через публичный URL --------------------------------------
if url:
    for _ in range(20):
        try:
            r = requests.get(url.rstrip("/") + "/health",
                             headers={"X-Agent-Token": TOKEN}, timeout=8)
            if r.status_code == 200:
                print("публичный health:", r.text[:200])
                break
        except Exception:
            pass
        time.sleep(3)

print("=" * 72)
print("THINKING_URL=", url)
print("THINKING_TOKEN=", TOKEN)
print("THINKING_PANEL=", url)
print("ТУННЕЛЬ:", tunnel_kind)
print("=" * 72)

# ---- 7) сторож: если LLM/API умрёт — поднимем сами (ноутбук не ждёт вас) ----
import threading                                          # noqa: E402


def _alive(url_: str) -> bool:
    try:
        return requests.get(url_.rstrip("/") + "/health",
                            headers={"X-Agent-Token": TOKEN}, timeout=6).status_code == 200
    except Exception:
        return False


def _watch() -> None:
    api_fail = llm_fail = 0
    while True:
        time.sleep(60)
        try:
            if _alive("http://127.0.0.1:8000"):
                api_fail = llm_fail = 0
                continue
            api_fail += 1
            print(time.strftime("%H:%M:%S"), "API не отвечает,", api_fail, "/3")
            if api_fail >= 3:
                try:
                    r = requests.get("http://127.0.0.1:8001/v1/models", timeout=6)
                    llm_fail = 0
                except Exception:
                    llm_fail += 1
                if llm_fail >= 2:
                    print(time.strftime("%H:%M:%S"), "поднимаю LLM заново")
                    # ту модель, которую выбрали в панели, а не первоначальную
                    active = MODEL
                    try:
                        cur = open(ACTIVE_FILE, encoding="utf-8").read().strip()
                        if cur and os.path.exists(cur):
                            active = cur
                    except OSError:
                        pass
                    cmd = list(llm_cmd)
                    cmd[cmd.index("--model") + 1] = active
                    subprocess.Popen(cmd, stdout=open(LOG_LLM, "a", encoding="utf-8"),
                                     stderr=subprocess.STDOUT)
                    time.sleep(90)
                subprocess.Popen(api_cmd, cwd="/content",
                                 stdout=open(LOG_API, "a", encoding="utf-8"),
                                 stderr=subprocess.STDOUT)
                print(time.strftime("%H:%M:%S"), "API перезапущен")
                api_fail = llm_fail = 0
        except Exception as exc:
            print("watchdog:", type(exc).__name__)


threading.Thread(target=_watch, daemon=True).start()
print("сторож запущен: раз в 60 с проверяет /health и перезапускает упавшее")

print("\nЗАВТРА (когда откроешь этот ноутбук заново):")
print("  1) Runtime → Change runtime type → T4 GPU")
print("  2) выполни ячейки A, C, D подряд (модель обычно уже лежит в Drive —")
print("     тогда загрузка пропускается, весь запуск занимает пару минут)")
print("  3) скопируй THINKING_URL и THINKING_TOKEN из рамки выше")
print("  4) на ПК:  python tools/thinking_cli.py set-url <URL> <TOKEN>")
print("  5) проверка: python tools/thinking_cli.py doctor")
print("  Если задал CLOUDFLARE_TUNNEL_TOKEN — адрес будет тот же, set-url не нужен.")
