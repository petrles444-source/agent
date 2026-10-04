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

# Длина ответа — единственное, для чего нужны бюджеты: рантайм бесплатный,
# экономить нечего, а короткий ответ ломает контракт (обрезанный JSON не
# разбирается и план превращается в мусор — такое и случилось на 3B с
# бюджетом 260). Поэтому бюджет задаёт ДЛИЗИНУ, а не «боязнь долгого
# ответа»: сервер всё равно прижимает его к n_ctx (THINKING_CTX).
# Проверка идёт СВЕРХУ ВНИЗ, от самого длинного признака к короткому: иначе
# «a3b» внутри Qwen3-30B-A3B поймал бы трёхбайтовый бюджет, а «7b» внутри
# «27b» — семёрку.
MODEL_BUDGETS = [
    # (признак в имени, ответ, поток чата, поток разработки)
    ("30b-a3b", "1200", "2400", "3000"),
    ("32b", "1200", "2400", "3000"),
    ("27b", "1200", "2400", "3000"),
    ("14b", "1000", "2000", "2600"),
    ("7b", "900", "1800", "2400"),
    ("3b", "700", "1400", "2000"),
    ("1.5b", "500", "1000", "1600"),
]
_name = os.path.basename(MODEL).lower()
_match = next((b for b in MODEL_BUDGETS if b[0] in _name),
              ("3b", "700", "1400", "2000"))      # запасной вариант
os.environ.setdefault("THINKING_MAX_TOKENS", _match[1])
# Для потокового чата (/chat/stream) можно больше: токены идут по мере генерации,
# туннель Cloudflare не рвёт соединение, а человек видит текст сразу.
os.environ.setdefault("THINKING_CHAT_MAX_TOKENS", _match[2])
# Режим разработчика возвращает код целиком — тут обрезать нельзя.
os.environ.setdefault("THINKING_DEV_MAX_TOKENS", _match[3])
# ВНИМАНИЕ: это предварительный расчёт по ПЕРВОЙ кандидатуре. Если она не
# поднимется и ячейка возьмёт запасную — бюджет пересчитается ниже, уже по
# той модели, что реально работает.
# Активная модель — в файле: его читает и сервер (вкладка «Модели» в панели),
# и сторож, чтобы переключение не слетело после перезапуска движка.
with open(ACTIVE_FILE, "w", encoding="utf-8") as fh:
    fh.write(MODEL)
print("Бюджет по первой кандидатуре:", os.environ["THINKING_MAX_TOKENS"],
      "/", os.environ["THINKING_CHAT_MAX_TOKENS"], "/",
      os.environ["THINKING_DEV_MAX_TOKENS"],
      "(признак:", _match[0] + ") — уточним после выбора модели")

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


def _avail_gb() -> float:
    """Сколько памяти реально свободно (ГБ) — по факту, а не по памятке.

    Colab-железо меняется, и модель, которая влезла вчера, может не влезть
    сегодня. Смотрим MemAvailable, потому что после загрузки модели сами
    страницы весов и есть «занятая» память.
    """
    try:
        for line in open("/proc/meminfo", encoding="utf-8"):
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024 ** 2
    except Exception:
        pass
    return 0.0


# ---- 2) LLM backend на :8001 ----------------------------------------------
# Контекст 4096 на CPU (быстрее считать) и 8192 на GPU — качество не страдает.
GPU = os.path.exists("/dev/nvidia0") or bool(shutil.which("nvidia-smi"))
CTX = "8192" if GPU else "4096"
# Пишем контекст в env-файл: его читает _llm_cmd при сторожевом перезапуске
# (ячейка C), чтобы n_ctx не разъехался между запуском и перезапуском (AUD-08)
with open(env_file, "a", encoding="utf-8") as fh:
    fh.write(f'export THINKING_CTX="{CTX}"\n')
llm_cmd = [sys.executable, "-m", "llama_cpp.server",
           "--model", MODEL, "--n_ctx", CTX, "--n_gpu_layers", "-1",
           "--host", "127.0.0.1", "--port", "8001", "--model_alias", "thinking",
           "--n_threads", "4", "--n_batch", "512", "--verbose", "False"]
with open(LOG_LLM, "a", encoding="utf-8") as _fh:
    _fh.write(f"\n--- {time.strftime('%H:%M:%S')} {llm_cmd}\n")

# Модель могла не подойти — тогда берём следующую из списка (их до трёх).
llm_proc = None
health = None
for attempt, cand in enumerate(MODELS, 1):
    size_gb = os.path.getsize(cand) / 1024 ** 3
    cmd = list(llm_cmd)
    cmd[cmd.index("--model") + 1] = cand
    # with закрывает родительскую копию хендлера — процессу достаётся своя,
    # а лишних открытых файлов в долгоживущей ячейке не копится (аудит C-5)
    with open(LOG_LLM, "a", encoding="utf-8") as _fh:
        llm_proc = subprocess.Popen(cmd, stdout=_fh, stderr=subprocess.STDOUT)
    print(f"llm pid {llm_proc.pid} ({attempt}/{len(MODELS)}: "
          f"{os.path.basename(cand)}, {size_gb:.1f} ГБ)")
    # Большая модель читается с диска долго: 17 ГБ успевает не всегда за 90 с,
    # и раньше мы отвергали её как «не поднялась», не дав дописать веса.
    # Время ожидания растёт от размера файла.
    tries = 45 + int(size_gb * 12)
    # Показываем требуемую память ДО ожидания. Примерная оценка: веса плюс
    # KV-кэш (около 10% весов при n_ctx 4096). Иначе приходилось ждать
    # несколько минут, чтобы потом увидеть отказ.
    kv_gb = size_gb * 0.10 * (int(CTX) / 4096)
    need_gb = size_gb + kv_gb
    print(f"  ждём подъёма до {tries * 2 // 60} мин "
          f"(размер {size_gb:.1f} ГБ, веса+KV ≈ {need_gb:.1f} ГБ при RAM ~12 ГБ)")
    # Память проверяем по факту, а не по памятке: Colab даёт разное железо.
    # Модель, которая не помещается, не «не поднимется» — она поднимется и
    # будет работать в подкачке страниц, то есть в десятки раз медленнее.
    # Прогон 04.10: 14B (8.4 ГБ) при свободных 3.0 ГБ дала 0.05 ток/с,
    # то есть 20 раз медленнее 3B. Молча мириться с этим нельзя.
    if _avail_gb() and need_gb > _avail_gb() * 1.1:
        print(f"  (!) ВНИМАНИЕ: весам нужно ≈{need_gb:.1f} ГБ, а свободно только "
              f"{_avail_gb():.1f} ГБ. Будет подкачка страниц с диска — скорость "
              "упадёт в десятки раз (замерено 0.05 ток/с на 14B).")
        print("      Профиль gpu (7B, 4.7 ГБ) или light (3B, 1.8 ГБ) влезут "
              "свободно. Модель всё равно попробуем — но ответ будет долгим.")
    health = wait_http("http://127.0.0.1:8001/v1/models", tries=tries, gap=2)
    if health is not None:
        MODEL = cand
        break
    # Не молчим: показываем ПОЧЕМУ модель не поднялась. Раньше здесь
    # печаталось только «не поднялась», и причину приходилось угадывать.
    _why = open(LOG_LLM, encoding="utf-8", errors="replace").readlines()[-25:]
    print(f"  {os.path.basename(cand)} не поднялась. Последние строки лога:")
    for _line in _why:
        print("   |", _line.rstrip())
    # Код возврата — единственный источник правды, когда лог ПУСТ. Именно так
    # выглядит прогон 04.10: лог содержал только строку команды, ни слова об
    # ошибке, поэтому пришлось выбирать между «не хватило памяти» и «движок не
    # знает архитектуру». poll() < 0 значит «убит сигналом»:
    #   -9  OOM-киллер: ядро не дало памяти — крупная модель в RAM не влезет;
    #   -11 segfault: чаще всего старая сборка llama.cpp не знает Qwen3.
    # poll() > 0 — процесс вышел сам, None — жив, то есть просто не успел.
    rc = llm_proc.poll()
    if rc is None:
        print("   | процесс ещё ЖИВ: не упал, а не успел поднять порт за отведённое время")
    elif rc < 0:
        import signal as _sig
        try:
            _name_sig = _sig.Signals(-rc).name
        except Exception:
            _name_sig = f"сигнал {-rc}"
        print(f"   | ПРОЦЕСС УБИТ: {_name_sig} (код {rc})")
        if rc == -9:
            print("   | это OOM-киллер: памяти не хватило. Модель крупнее RAM "
                  "(~12 ГБ) так не запустится — бери профиль strong (14B, 8 ГБ)")
        elif rc == -11:
            print("   | это segfault: скорее всего движок не знает архитектуру "
                  "модели (Qwen3 требует свежей сборки llama.cpp). Профиль "
                  "strong (Qwen2.5-14B) поддерживается любой сборкой")
    else:
        print(f"   | процесс завершился сам, код возврата {rc}")
    if size_gb > 12:
        print(f"  (!) модель {size_gb:.1f} ГБ при RAM ~12 ГБ: может не хватить "
              "памяти. Возьми профиль strong (14B, 8 ГБ) или gpu.")
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
# Бюджет пересчитываем ПОСЛЕ выбора модели: раньше он считался по первой
# кандидатуре, и если поднималась запасная — работала с чужими лимитами
# (прогон 04.10: поднялась 3B, а бюджеты остались от 30B — 1200/2400/3000).
_name = os.path.basename(MODEL).lower()
_match = next((b for b in MODEL_BUDGETS if b[0] in _name),
              ("3b", "700", "1400", "2000"))
os.environ["THINKING_MAX_TOKENS"] = _match[1]
os.environ["THINKING_CHAT_MAX_TOKENS"] = _match[2]
os.environ["THINKING_DEV_MAX_TOKENS"] = _match[3]
print("ЛИМИТ ОТВЕТА (по модели в работе):", os.environ["THINKING_MAX_TOKENS"],
      "/", os.environ["THINKING_CHAT_MAX_TOKENS"], "/",
      os.environ["THINKING_DEV_MAX_TOKENS"], "признак:", _match[0])
# сторож в конце ячейки перезапускает llm_cmd — значит в нём должна быть
# та модель, которая реально поднялась, а не первая из списка
llm_cmd[llm_cmd.index("--model") + 1] = MODEL
# хвосты процессов для сторожа: он отличает «процесс жив» от «процесса нет»
PROCS: dict = {"llm": llm_proc}
print("LLM ready:", health.text[:200])

# ---- 2b) дополнительные движки: несколько моделей одновременно -------------
# Смысл: держать несколько моделей поднятыми, чтобы один вопрос получить сразу
# от нескольких (сверка качества) и не тратить время на переключение.
#
# Честно о цене. Ядра у рантайма одни и те же, поэтому СУММАРНАЯ скорость не
# растёт: каждый из N ответов идёт примерно в N раз дольше. Выигрыш здесь не в
# скорости, а в сравнении ответов и в доступности без перезапуска. Включается
# только если явно задан список путей в THINKING_PARALLEL — по умолчанию
# работает одна модель, как раньше.
#
# Память: сумма весов всех дополнительных моделей должна помещаться в RAM
# вместе с основной, иначе начнётся подкачка страниц и всё встанет колом.
PARALLEL_FILE = "/content/parallel_models.json"
_extra = [p.strip() for p in os.environ.get("THINKING_PARALLEL", "").split(",")
          if p.strip()]
_manifest: list[dict] = []
if _extra:
    print(f"дополнительных моделей: {len(_extra)} (каждой — свой порт)")
for _i, _path in enumerate(_extra, start=2):
    _name = os.path.basename(_path)
    if not os.path.exists(_path):
        print(f"  {_name}: файла нет — пропускаю")
        continue
    _gb = os.path.getsize(_path) / 1024 ** 3
    _avail = _avail_gb()
    if _avail and _gb > _avail * 0.8:
        print(f"  {_name}: {_gb:.1f} ГБ при свободных {_avail:.1f} ГБ — "
              "не поместится, будет подкачка страниц. Пропускаю.")
        continue
    _cmd = list(llm_cmd)
    _cmd[_cmd.index("--model") + 1] = _path
    _cmd[_cmd.index("--port") + 1] = str(8000 + _i)
    _cmd[_cmd.index("--model_alias") + 1] = f"thinking{_i}"
    with open(LOG_LLM, "a", encoding="utf-8") as _fh:
        _proc = subprocess.Popen(_cmd, stdout=_fh, stderr=subprocess.STDOUT)
    print(f"  {_name} ({_gb:.1f} ГБ) на порту {8000 + _i}, pid {_proc.pid}")
    if wait_http(f"http://127.0.0.1:{8000 + _i}/v1/models",
                 tries=45 + int(_gb * 12), gap=2) is None:
        print(f"  {_name}: не поднялась — пропускаю")
        try:
            _proc.kill()
        except Exception:
            pass
        continue
    PROCS[f"llm{_i}"] = _proc
    # key/port — чтобы сторож мог поднять именно этот движок, если он умрёт:
    # индекс в манифесте не совпадает с номером, когда часть моделей пропущена
    _manifest.append({"path": _path, "label": _name, "key": f"llm{_i}",
                      "port": 8000 + _i,
                      "url": f"http://127.0.0.1:{8000 + _i}",
                      "size_gb": round(_gb, 2), "pid": _proc.pid})
with open(PARALLEL_FILE, "w", encoding="utf-8") as _fh:
    json.dump(_manifest, _fh, ensure_ascii=False, indent=1)
if _manifest:
    print(f"параллельных моделей поднято: {len(_manifest)} → "
          + ", ".join(m["label"] for m in _manifest))
else:
    print("параллельные модели не запрошены (THINKING_PARALLEL пуст) — работает одна")

# ---- 3) смоук-тест генерации ----------------------------------------------
# На CPU модель может отвечать дольше 180 с или упасть на старте — это не
# повод ронять ячейку целиком: API и туннель поднимаются дальше, а проверить
# генерацию можно руками (AUD-24).
t0 = time.time()
try:
    smoke = requests.post(
        "http://127.0.0.1:8001/v1/chat/completions",
        json={"model": "thinking", "temperature": 0,
              "max_tokens": 24, "messages": [{"role": "user",
                                              "content": "Ответь ровно одним словом: готов"}]},
        timeout=180)
    dt = time.time() - t0
    txt = smoke.json()["choices"][0]["message"]["content"]
    print(f"смоук: {dt:.1f} с -> {txt[:80]!r}")
except Exception as _smoke_exc:
    print(f"! смоук не прошёл: {_smoke_exc}")
    print("  модель отвечает медленно или ещё грузится — проверь руками:")
    print("  curl -s http://127.0.0.1:8001/v1/chat/completions -X POST "
          "-H 'Content-Type: application/json' "
          "-d '{\"model\":\"thinking\",\"max_tokens\":24,"
          "\"messages\":[{\"role\":\"user\",\"content\":\"привет\"}]}'")

# ---- 4) API на :8000 -------------------------------------------------------
assert os.path.exists("/content/thinking_server.py"), "сначала ячейка C (сервер)"
os.environ["THINKING_UPSTREAM"] = "http://127.0.0.1:8001/v1/chat/completions"
api_cmd = [sys.executable, "-m", "uvicorn", "thinking_server:app",
           "--host", "127.0.0.1", "--port", "8000", "--app-dir", "/content"]
with open(LOG_API, "a", encoding="utf-8") as _fh:
    _fh.write(f"\n--- {time.strftime('%H:%M:%S')} {api_cmd}\n")
# with закрывает родительскую копию хендлера (процессу достаётся своя) —
# открытые файлы в долгоживущей ячейке не копятся (аудит C-5)
with open(LOG_API, "a", encoding="utf-8") as _fh:
    api_proc = subprocess.Popen(api_cmd, cwd="/content",
                                stdout=_fh, stderr=subprocess.STDOUT)
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
        with open("/content/tunnel.log", "w", encoding="utf-8") as _fh:
            subprocess.Popen(["nohup", "cloudflared", "tunnel", "--no-autoupdate", "run",
                              "--token", CF_TOKEN"],
                             stdout=_fh, stderr=subprocess.STDOUT,
                             start_new_session=True)
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
        with open("/content/tunnel.log", "a", encoding="utf-8") as _fh:
            subprocess.Popen(["nohup", "cloudflared", "tunnel", "--no-autoupdate",
                              "--protocol", "http2",
                              "--url", "http://localhost:8000"],
                             stdout=_fh, stderr=subprocess.STDOUT,
                             start_new_session=True)
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


def _restart_extra() -> None:
    """Поднимает умершие параллельные движки (llm2, llm3…).

    Иначе /ask/multi молча отвечает меньшим числом моделей, чем просили, и
    ни панель, ни клиент этого не показывают (аудит C-5). Команды
    собираем заново из манифеста — там путь, порт и ключ процесса.
    """
    try:
        with open(PARALLEL_FILE, encoding="utf-8") as fh:
            manifest = json.load(fh)
    except Exception:
        return                      # параллельные модели не запускались
    if not isinstance(manifest, list):
        return
    for idx, item in enumerate(manifest, start=2):
        if not isinstance(item, dict) or not str(item.get("path") or ""):
            continue
        key = str(item.get("key") or f"llm{idx}")
        proc = PROCS.get(key)
        if proc is not None and proc.poll() is None:
            continue
        cmd = list(llm_cmd)
        cmd[cmd.index("--model") + 1] = str(item["path"])
        cmd[cmd.index("--port") + 1] = str(item.get("port") or (8000 + idx))
        cmd[cmd.index("--model_alias") + 1] = f"thinking{idx}"
        try:
            with open(LOG_LLM, "a", encoding="utf-8") as fh:
                PROCS[key] = subprocess.Popen(cmd, stdout=fh,
                                              stderr=subprocess.STDOUT)
            print(time.strftime("%H:%M:%S"),
                  f"{key} ({item.get('label') or item['path']}) поднимаю заново")
        except Exception as exc:
            print(time.strftime("%H:%M:%S"),
                  f"{key}: не поднялся: {type(exc).__name__} {exc}")


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
                # Жив ли движок — вопрос не к HTTP, а к ПРОЦЕССУ. Запрос к
                # /v1/models с таймаутом 6 с врёт: занятая генерацией модель
                # (или модель, которая подгружает страницы весов с диска) за
                # 6 с не ответит, и сторож решил бы, что движок умер, и
                # перезапустил его. На медленной модели ответ так и не доехал
                # бы никогда — сторож убивал бы его каждые пару минут.
                # poll() отвечает мгновенно и однозначно.
                proc = PROCS.get("llm")
                if proc is not None and proc.poll() is None:
                    llm_fail = 0
                    print(time.strftime("%H:%M:%S"),
                          "LLM занят, но процесс жив — не трогаю")
                else:
                    try:
                        requests.get("http://127.0.0.1:8001/v1/models", timeout=20)
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
                    with open(LOG_LLM, "a", encoding="utf-8") as _fh:
                        PROCS["llm"] = subprocess.Popen(
                            cmd, stdout=_fh, stderr=subprocess.STDOUT)
                    time.sleep(90)
                    # доп. движки могли упасть вместе с основным
                    _restart_extra()
                with open(LOG_API, "a", encoding="utf-8") as _fh:
                    subprocess.Popen(api_cmd, cwd="/content",
                                     stdout=_fh, stderr=subprocess.STDOUT)
                print(time.strftime("%H:%M:%S"), "API перезапущен")
                api_fail = llm_fail = 0
        except Exception as exc:
            # текст ошибки обязателен: «watchdog: RuntimeError» ничего не
            # говорит, а причина обычно в строке (аудит C-5)
            print("watchdog:", type(exc).__name__, str(exc)[:200])


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
