# === Субагент «Мышление», ячейка A (окружение + модель) — БЫСТРАЯ ВЕРСИЯ ===
# Основа — та ячейка, с которой субагент впервые поднялся за ~3 минуты.
# Отличия от оригинала (только ускорение и страховка, ничего лишнего):
#   * если модель уже лежит на диске — загрузка пропускается целиком;
#   * без GPU берём лёгкую модель (3B), иначе 7B — так быстрее;
#   * три модели-кандидата: если первая не скачалась, берём следующую;
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
say("движок готов")

# --- 3) модель: сначала проверяем диск, потом качаем ---
MODEL_DIR = "/content/models/llm"
os.makedirs(MODEL_DIR, exist_ok=True)

have = [p for p in glob.glob(os.path.join(MODEL_DIR, "**", "*.gguf"), recursive=True)
        if os.path.getsize(p) > 600_000_000]

# Качаем ВСЕ нужные модели (их переключают прямо в панели), но не качаем заново
# то, что уже лежит: повторный запуск остаётся быстрым. Порядок важен: на GPU
# активной становится первая (7B), без GPU — 3B (см. ниже).
from huggingface_hub import hf_hub_download, list_repo_files     # noqa: E402

WANTED = [("bartowski/Qwen2.5-3B-Instruct-GGUF", 1_000_000_000, "3b"),
          ("Qwen/Qwen2.5-1.5B-Instruct-GGUF", 600_000_000, "1.5b"),
          ("bartowski/Qwen2.5-7B-Instruct-GGUF", 3_000_000_000,
           "7b-instruct-q4"),                      # обычная 7B
          ("QuantFactory/Qwen2.5-7B-Instruct-Uncensored-GGUF", 3_000_000_000,
           "uncensored")]                          # та же 7B без цензуры
if GPU:                                   # с видеокартой 7B — первая (активна)
    WANTED.insert(0, ("bartowski/Qwen2.5-7B-Instruct-GGUF", 3_000_000_000,
                      "7b-instruct-q4"))
    WANTED = list(dict.fromkeys(WANTED))   # дубликат строки не нужен


def _has(tag: str) -> str:
    """Путь к уже скачанной модели по её размеру (1.5b / 3b / 7b)."""
    low = tag.lower()
    for p in have:
        name = os.path.basename(p).lower()
        if low in name and os.path.getsize(p) > 800_000_000:
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


MODELS: list[str] = []                       # 3b, 1.5b, 7b — в этом порядке
for repo, min_size, tag in WANTED:
    path = _has(tag)
    if path:
        say(f"{tag}: уже на диске ({os.path.getsize(path) / 2**30:.2f} ГБ) — пропускаем")
        MODELS.append(path)
        continue
    try:
        path = _fetch(repo, min_size)
    except Exception as exc:                # сеть отвалилась — не молчим
        say(f"  {repo}: {type(exc).__name__} {str(exc)[:100]}")
        path = ""
    if path:
        MODELS.append(path)
    else:
        say(f"  {tag}: не скачалась, пропускаем (работаем на остальных)")

if not MODELS:                              # аварийный план: что уже есть
    MODELS = have
assert MODELS, "ни одна модель не нашлась"

# Без видеокарты активной делаем 3B — на CPU 7B считается ~1 ток/с и мешает
# работать; 1.5B остаётся запасной (быстрее, но отвечает короче и проще).
if GPU:
    MODEL = MODELS[0]
else:
    MODEL = next((p for p in MODELS
                  if "3b" in os.path.basename(p).lower()), MODELS[0])


def _hint(p: str) -> str:
    """Подсказка режима (CPU/T4) — печатается при сборке и идёт в панель."""
    name = os.path.basename(p).lower()
    unc = " (без цензуры)" if "uncensored" in name else ""
    if any(t in name for t in ("7b", "8b", "14b")):
        return (f"7B{unc}: для лучшей работы смени Runtime → Change runtime "
                f"type → T4 GPU (на CPU ~1 ток/с)")
    if "3b" in name:
        return f"3B{unc}: работает на CPU (~2–3 ток/с), на T4 — быстрее"
    return f"1.5B{unc}: самая быстрая на CPU (~6 ток/с)"


say(f"ГОТОВО за {time.time()-START:.0f}с. Модели: "
    + ", ".join(f"{os.path.basename(p)} ({os.path.getsize(p) / 2**30:.2f} ГБ)"
                for p in MODELS))
for p in MODELS:
    say(f"  · {_hint(p)}")
if not GPU:
    say("  ⚠ Рантайм на CPU: 7B качается заранее, но считается медленно — "
        "для неё переключи Runtime на T4 GPU.")
say(f"активная: {MODEL}  ·  переключение — в панели (вкладка «Модели»)")
with open("/content/thinking_model_path.txt", "w", encoding="utf-8") as fh:
    fh.write(MODEL)
with open("/content/thinking_models.txt", "w", encoding="utf-8") as fh:
    fh.write("\n".join(MODELS) + "\n")
with open("/content/thinking_backend.txt", "w", encoding="utf-8") as fh:
    fh.write("llama_cpp\n")
print("OK-A")