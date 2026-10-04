# === Субагент «Мышление», ячейка A (окружение + модель) ===
# Отличия от оригинала (только ускорение, страховка и выбор моделей):
#   * если модель уже лежит на диске — загрузка пропускается целиком;
#   * качаем НЕ всё, а выбранный профиль (THINKING_PROFILE): «light»,
#     «gpu», «strong» (14B), «big» (Qwen3-30B-A3B, 17 ГБ, MoE),
#     «coder» (32B для разработки), «uncensored», «all»;
#   * профиль задаётся переменной окружения — её ставит ячейка ноутбука
#     (одна строка PROFILE = "…" прямо перед запуском);
#   * при провале импорта CUDA-колеса чиним это одной командой с PyPI,
#     но только если в системе есть видеокарта (иначе не тратим время).
import glob
import importlib
import os
import subprocess
import sys
import time

START = time.time()
CUDA_INDEX = "https://abetlen.github.io/llama-cpp-python/whl/cu124/"
CPU_INDEX = "https://abetlen.github.io/llama-cpp-python/whl/cpu/"


def say(msg: str) -> None:
    print(msg, flush=True)


def pip(args, env=None, timeout=900):
    return subprocess.run([sys.executable, "-m", "pip", "install", "-q", *args],
                          env=env, capture_output=True, text=True, timeout=timeout)


def has_gpu() -> bool:
    if os.path.exists("/dev/nvidia0"):
        return True
    try:
        import torch                                  # noqa: PLC0415
        return bool(torch.cuda.is_available())
    except Exception:
        pass
    return False


def try_import() -> tuple[bool, str]:
    try:
        import llama_cpp
        importlib.reload(llama_cpp)
        from llama_cpp import server as _srv          # noqa: F401
        return True, ""
    except Exception as exc:
        return False, str(exc)[:200]


GPU = has_gpu()
say(f"python {sys.version.split()[0]}  GPU={GPU}")

# --- 1) зависимости ( idempotent: если стоят, pip отвечает за секунды) ---
r = pip(["fastapi", "uvicorn[standard]", "httpx", "huggingface_hub", "requests"])
say(f"deps rc={r.returncode}")

# --- 2) движок: готовое колесо, без компиляции ---
ok, why = try_import()
if not ok:
    if "libcudart" in why and GPU:            # есть GPU, но нет библиотек CUDA
        say("чиню CUDA-библиотеки с PyPI…")
        pip(["nvidia-cuda-runtime-cu12", "nvidia-cublas-cu12"])
        libdirs = []
        for mod in ("nvidia.cuda_runtime", "nvidia.cublas"):
            try:
                m = importlib.import_module(mod)
                libdirs.append(os.path.join(os.path.dirname(m.__file__), "lib"))
            except Exception:
                pass
        libdirs = [d for d in libdirs if os.path.isdir(d)]
        if libdirs:
            cur = os.environ.get("LD_LIBRARY_PATH", "")
            os.environ["LD_LIBRARY_PATH"] = ":".join(libdirs + ([cur] if cur else []))
            with open("/content/thinking_env.sh", "w", encoding="utf-8") as fh:
                fh.write("export LD_LIBRARY_PATH=" + os.environ["LD_LIBRARY_PATH"] + "\n")
            ok, why = try_import()

if not ok:
    # без GPU берём CPU-колесо: одна попытка, ~10 секунд вместо компиляции
    index = CUDA_INDEX if GPU else CPU_INDEX
    tag = "wheel-cu124" if GPU else "wheel-cpu"
    say(f"{tag}…")
    t0 = time.time()
    pip(["--no-cache-dir", "--extra-index-url", index, "llama-cpp-python[server]"])
    ok, why = try_import()
    say(f"  импорт: {'ОК' if ok else 'НЕ ОК — ' + why}  ({time.time()-t0:.0f}с)")

if not ok and GPU:                             # на CPU-wheel не вышло — пробуем CUDA
    say("fallback: wheel-cu124…")
    pip(["--no-cache-dir", "--extra-index-url", CUDA_INDEX, "llama-cpp-python[server]"])
    ok, why = try_import()
    if ok and "libcudart" in why:              # страховка от «битой» строки
        pass

assert ok, "не поставился llama-cpp-python: " + why
# Версию движка печатаем отдельно и всегда: от неё зависит, умеет ли llama.cpp
# архитектуру модели. Qwen3 появился в llama.cpp в 2025 году, и старая сборка
# на нём падает молча — в логе нет ни строки (прогон 04.10: так пропал
# Qwen3-30B-A3B, и пришлось гадать, память это или архитектура).
try:
    import llama_cpp
    say(f"движок: llama-cpp-python {getattr(llama_cpp, '__version__', 'версия неизвестна')}")
except Exception:
    say("движок: версию определить не удалось")

# --- 3) модель: сначала проверяем диск, потом качаем ---
MODEL_DIR = "/content/models/llm"
os.makedirs(MODEL_DIR, exist_ok=True)

have = [p for p in glob.glob(os.path.join(MODEL_DIR, "**", "*.gguf"), recursive=True)
        if os.path.getsize(p) > 600_000_000]

# Каталог: тег → (репозиторий, минимальный размер, подпись, ранг).
# Тег — точная подстрока имени файла: по нему узнаём свою модель и никогда
# не путаем соседние по имени (старые широкие теги «3b»/«uncensored» приняли
# uncensored-модель за обычную). Ранг — чем выше, тем сильнее модель.
from huggingface_hub import hf_hub_download, list_repo_files     # noqa: E402

CATALOG: dict[str, tuple[str, int, str, int]] = {
    "1.5b-instruct-q4": ("Qwen/Qwen2.5-1.5B-Instruct-GGUF", 600_000_000, "1.5B", 10),
    "3b-instruct-q4": ("bartowski/Qwen2.5-3B-Instruct-GGUF", 1_000_000_000, "3B", 20),
    "7b-instruct-q4": ("bartowski/Qwen2.5-7B-Instruct-GGUF", 3_000_000_000, "7B", 30),
    # Теги у больших моделей — строго те, что реально лежат в имени файла.
    # «14b-instruct-q4» не годится: он есть и в имени Coder-14B
    # (qwen2.5-coder-14b-instruct-q4_k_m.gguf) — и обычная 14B приняла бы
    # файл кодера. Поэтому «qwen2.5-14b-instruct» против «coder-14b».
    "qwen2.5-14b-instruct": ("bartowski/Qwen2.5-14B-Instruct-GGUF", 5_000_000_000,
                             "14B", 40),
    "coder-14b": ("bartowski/Qwen2.5-Coder-14B-Instruct-GGUF", 5_000_000_000,
                  "14B-CODE", 45),
    # MoE: 30B параметров, но активных только 3B — качество большой модели
    # при скорости небольшой. Единственный «сильный» вариант, который
    # осмысленно работает на бесплатном Colab (17.3 ГБ, RAM 12.6 ГБ — по
    # умолчанию mmap с диска, поэтому медленнее, но отвечает).
    "30b-a3b-q4": ("Qwen/Qwen3-30B-A3B-GGUF", 10_000_000_000, "30B-A3B", 60),
    "qwen2.5-32b-instruct": ("bartowski/Qwen2.5-32B-Instruct-GGUF", 10_000_000_000,
                             "32B", 70),
    "coder-32b": ("bartowski/Qwen2.5-Coder-32B-Instruct-GGUF", 10_000_000_000,
                  "32B-CODE", 80),
    "7b-instruct-uncensored": ("QuantFactory/Qwen2.5-7B-Instruct-Uncensored-GGUF",
                               3_000_000_000, "7B-UNC", 31),
    "3b-instruct-uncensored": ("mradermacher/Qwen2.5-3B-Instruct-Uncensored-GGUF",
                               1_000_000_000, "3B-UNC", 21),
    "1.5b-instruct-uncensored": ("mradermacher/Qwen2.5-1.5B-Instruct-uncensored-GGUF",
                                 600_000_000, "1.5B-UNC", 11),
}

# Профили: что качать. Первый элемент — активная модель профиля.
PROFILES: dict[str, list[str]] = {
    "light": ["3b-instruct-q4", "1.5b-instruct-q4"],
    "gpu": ["7b-instruct-q4", "3b-instruct-q4", "1.5b-instruct-q4"],
    "strong": ["qwen2.5-14b-instruct", "coder-14b", "3b-instruct-q4"],
    "big": ["30b-a3b-q4", "3b-instruct-q4"],
    "coder": ["coder-32b", "coder-14b", "3b-instruct-q4"],
    "uncensored": ["3b-instruct-uncensored", "3b-instruct-q4", "1.5b-instruct-q4"],
    "all": sorted(CATALOG, key=lambda t: -CATALOG[t][3]),
}
PROFILE = (os.environ.get("THINKING_PROFILE") or "auto").strip().lower()
if PROFILE in ("", "auto"):
    PROFILE = "gpu" if GPU else "light"
WANTED_TAGS = PROFILES.get(PROFILE) or PROFILES["light"]
say(f"профиль моделей: {PROFILE} → {', '.join(WANTED_TAGS)}")
if PROFILE not in PROFILES:
    say(f"  ⚠ неизвестный профиль, взят light (доступно: {', '.join(PROFILES)})")


def _has(tag: str) -> str:
    """Путь к уже скачанной модели по точному тегу (3b-instruct-q4, …)."""
    low = tag.lower()
    floor = CATALOG[tag][1]
    for p in have:
        name = os.path.basename(p).lower()
        if low in name and os.path.getsize(p) >= floor:
            return p
    return ""


def _fetch(repo: str, min_size: int) -> str:
    files = [f for f in list_repo_files(repo)
             if f.endswith(".gguf") and "Q4_K_M" in f.upper()
             and "IQ" not in f.upper()]
    if not files:
        say(f"  {repo}: подходящего .gguf нет")
        return ""
    t0 = time.time()
    got = hf_hub_download(repo_id=repo, filename=files[0], local_dir=MODEL_DIR)
    size = os.path.getsize(got)
    say(f"  {repo}: {size / 2**30:.2f} ГБ за {time.time() - t0:.0f}с")
    return got if size >= min_size else ""


MODELS: list[str] = []
tag_of: dict[str, str] = {}                  # путь → тег (для выбора активной)
for tag in WANTED_TAGS:
    repo, min_size, label, _rank = CATALOG[tag]
    path = _has(tag)
    if path:
        say(f"{tag}: уже на диске ({os.path.getsize(path) / 2**30:.2f} ГБ) — пропускаем")
        MODELS.append(path)
        tag_of[path] = tag
        continue
    say(f"{tag}: качаю {repo}…")
    try:
        path = _fetch(repo, min_size)
    except Exception as exc:                 # сеть отвалилась — не молчим
        say(f"  {repo}: {type(exc).__name__} {str(exc)[:100]}")
        path = ""
    if path:
        MODELS.append(path)
        tag_of[path] = tag
    else:
        say(f"  {tag}: не скачалась, пропускаем (работаем на остальных)")

if not MODELS:                               # аварийный план: что уже есть
    MODELS = have
    tag_of = {}
assert MODELS, "ни одна модель не нашлась"

# Активная — первая из профиля, что реально скачалась; если профиль не дал
# ничего, берём самую сильную из того, что лежит на диске.
active_tag = next((t for t in WANTED_TAGS if t in tag_of), "")
if active_tag:
    MODEL = tag_of[active_tag]
else:
    MODEL = max(MODELS, key=lambda p: os.path.getsize(p))
    say("  профиль пуст — взял самую большую модель с диска")


def _hint(p: str) -> str:
    """Подсказка режима (CPU/T4) — печатается при сборке и идёт в панель."""
    name = os.path.basename(p).lower()
    unc = " (без цензуры)" if "uncensored" in name else ""
    code = " · для кода" if "coder" in name else ""
    if "a3b" in name or "30b" in name:
        return (f"30B-A3B{code}{unc}: MoE — 30B параметров, активны 3B, "
                f"поэтому на CPU считает почти как 3B, а отвечает как большая")
    if "32b" in name:
        return (f"32B{code}{unc}: очень сильная, но на CPU медленная — "
                f"смени Runtime на T4 GPU")
    if "14b" in name:
        return (f"14B{code}{unc}: влезает в RAM целиком, на CPU ~2–4 ток/с")
    if "7b" in name:
        return (f"7B{unc}: смени Runtime → Change runtime type → T4 GPU "
                f"(на CPU ~1 ток/с)")
    if "3b" in name:
        return f"3B{unc}: работает на CPU (~2–3 ток/с), на T4 — быстрее"
    return f"1.5B{unc}: самая быстрая на CPU (~6 ток/с)"


say(f"ГОТОВО за {time.time()-START:.0f}с. Модели: "
    + ", ".join(f"{os.path.basename(p)} ({os.path.getsize(p) / 2**30:.2f} ГБ)"
                for p in MODELS))
for p in MODELS:
    say(f"  · {_hint(p)}")
if not GPU:
    say("  ⚠ Рантайм на CPU: большие модели качаются заранее, но считаются "
        "медленно — переключи Runtime на T4 GPU, если нужна скорость.")
say(f"активная: {MODEL}  ·  профиль: {PROFILE}  ·  "
    "переключение — в панели (вкладка «Модели»)")
with open("/content/thinking_model_path.txt", "w", encoding="utf-8") as fh:
    fh.write(MODEL)
with open("/content/thinking_models.txt", "w", encoding="utf-8") as fh:
    fh.write("\n".join(MODELS) + "\n")
with open("/content/thinking_backend.txt", "w", encoding="utf-8") as fh:
    fh.write("llama_cpp\n")
print("OK-A")