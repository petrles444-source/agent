"""CLI подсистемы «Мышление» + панель реального времени.

Интерфейс основного агента (и человека):

    python tools/thinking_cli.py doctor
    python tools/thinking_cli.py set-url <URL> [TOKEN]
    python tools/thinking_cli.py plan "задача" --files a.py,b.py --max-steps 8
    python tools/thinking_cli.py plan "задача" --stream        # токены в реальном времени
    python tools/thinking_cli.py ask  "короткий вопрос"
    python tools/thinking_cli.py reflect PLAN_ID --step 1 --result "538/0" [--async]
    python tools/thinking_cli.py tail [--n 30] [--follow]
    python tools/thinking_cli.py panel [--port 8765]
    python tools/thinking_cli.py metrics
    python tools/thinking_cli.py reflect-metrics [--engine local] [--loop]

Режим разработчика живёт в панели (вкладка «Разработка»): файлы
dev_sandbox/, журнал logs/thinking/devlog.jsonl, дампы в dumps/.

Коды выхода:
    0 — ок
    2 — субагент офлайн, сработал fallback (работать можно дальше)
    3 — ответ не прошёл контракт (SchemaError)
    4 — субагент недоступен / таймаут
    5 — отказано в доступе (401/403: устарел токен)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.parse
import zipfile
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from thinking.client import ThinkingClient, ThinkingError  # noqa: E402
from thinking.schemas import ChatReply, SchemaError  # noqa: E402
from thinking.fallback import offline_reason  # noqa: E402

# русские названия тегов — в интерфейсе латиница не показывается
ACTION_RU = {
    "build": "сборка", "test": "тесты", "refactor": "рефакторинг",
    "verify": "проверка", "docs": "документация", "prompt": "промт",
    "debug": "отладка", "release": "релиз",
}
EVENT_RU = {
    "thought": "мысль", "token": "токен", "rationale": "объяснение",
    "plan_step": "шаг плана", "contradiction": "противоречие",
    "final": "итог", "error": "ошибка", "plan": "план", "reflect": "рефлексия",
    "interaction": "обращение", "dev": "разработка",
}
# Что можно снять с Colab одной командой `dump --all` (белый список сервера)
DUMP_FILES = ("llm.log", "api.log", "tunnel.log", "server.py",
              "snapshot.json", "last_bad_json.txt", "model_path.txt")


def _out(text: str) -> None:
    sys.stdout.write(text + "\n")
    sys.stdout.flush()


def _port_busy(host: str, port: int) -> bool:
    """Занят ли порт кем-то ещё.

    На Windows SO_REUSEADDR позволяет второму обработчику молча сесть на уже
    занятый порт, и пользователь видит чужую (устаревшую) панель. Поэтому
    проверяем порт ДО привязки и честно отказываем, а не подменяем панель.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.4)
        return sock.connect_ex((host, port)) == 0


def cmd_doctor(client: ThinkingClient, args: argparse.Namespace) -> int:
    ok = client.health(timeout=4)
    reason = "" if ok else offline_reason(client)
    _out(f"URL:      {client.base or 'не задан'}")
    _out(f"HEALTH:   {'ok' if ok else 'НЕТ'}" + (f" ({reason})" if reason else ""))
    if ok:
        h = getattr(client, "health_data", {}) or {}
        _out(f"MODEL:    {h.get('model')}  upstream={h.get('upstream')}")
        _out(f"GPU:      {h.get('gpu')}  vram={h.get('vram_used_gb')} GB  uptime={h.get('uptime_s')} с")
        _out(f"PLANS:    {h.get('plans')}   событий: {h.get('events')}")
        # Бесплатный Colab без чётких квот: GPU-сессии обычно 2–4 ч, остаток
        # CPU (бывает ~28 ч) — только ориентир. Не пугаем конкретным числом,
        # а напоминаем проверить лимиты, когда всё встало.
        up = h.get("uptime_s")
        gpu = str(h.get("gpu") or "cpu")
        if isinstance(up, (int, float)) and gpu != "cpu" and up >= 7200:
            _out(f"! ЛИМИТЫ: GPU-сессия идёт уже {up / 60:.0f} мин — бесплатный "
                 f"GPU обычно живёт 2–4 ч. Сохрани ноутбук заранее; если всё "
                 f"встало — Colab: Сессия → Лимиты (цифры меняются и не "
                 f"гарантированы). Дальше: CPU, пауза ~сутки или смена аккаунта.")
        if isinstance(up, (int, float)) and up >= 93600 and gpu == "cpu":
            _out(f"! ЛИМИТЫ: CPU-сессия работает {up / 3600:.0f} ч — остаток "
                 f"в Colab лишь ориентир; при остановке проверь Сессия → "
                 f"Лимиты (обычно возвращаются через сутки).")
        if gpu == "cpu":
            _out("! БЕЗ GPU: бесплатные GPU-часы, видимо, закончились — можно "
                 "продолжить на CPU (медленно), подождать ~сутки или сменить "
                 "Google-аккаунт. На будущее: Kaggle (~30 ч GPU/нед), "
                 "Paperspace, локальный runtime.")
    _out(f"TOKEN:    {'задан' if client.token else 'НЕ задан'}")
    if client.cfg.get("enabled", True):
        client.sync_history(tail=10)
        _out(f"STREAM:   last_seq={client.last_seq}  в буфере {len(client.buffered)}"
             + (f"  поток: {client.stream_error}" if client.stream_error else ""))
    _out(f"FALLBACK: {'включён' if client.cfg.get('fallback_on_error', True) else 'выключен'}")
    panel = f"http://{client.cfg.get('panel_host', '127.0.0.1')}:{client.cfg.get('panel_port', 8765)}"
    _out(f"PANEL:    {panel}")
    _out(f"JOURNAL:  {client.cfg.get('log_path')}")
    if not client.base:
        _out("")
        _out("Дальше:  python tools/thinking_cli.py set-url <THINKING_URL> <THINKING_TOKEN>")
        return 4
    return 0 if ok else 4


def cmd_set_url(client: ThinkingClient, args: argparse.Namespace) -> int:
    path = ThinkingClient.set_url(args.url, args.token or "")
    _out(f"сохранено в {path}")
    fresh = ThinkingClient()
    return 0 if fresh.health(timeout=6) else 4


def cmd_url(client: ThinkingClient, args: argparse.Namespace) -> int:
    _out(client.base or "")
    return 0 if client.base else 4


def _print_plan(plan: dict) -> None:
    _out("")
    _out(f"ПЛАН {str(plan.get('plan_id'))[:8]}  [{plan.get('source')}]  "
         f"уверенность {plan.get('confidence')}")
    _out(f"Цель: {plan.get('goal')}")
    if plan.get("rationale"):
        _out(f"Почему так: {plan['rationale']}")
    for s in plan.get("steps", []):
        tag = ACTION_RU.get(str(s.get("action")), "шаг")
        dep = f"  (ждёт {s.get('depends_on')})" if s.get("depends_on") else ""
        _out(f"  {s.get('id')}. [{tag}] {s.get('desc')}{dep}")
        if s.get("expected_output"):
            _out(f"     → ожидаем: {s['expected_output']}")
    for c in plan.get("contradictions", []) or []:
        _out(f"  ⚠ противоречие: {c}")
    for c in plan.get("success_criteria", []) or []:
        _out(f"  ✓ критерий: {c}")
    for f in plan.get("unknown_files", []) or []:
        _out(f"  ? файл вне списка: {f}")
    _out("")


def cmd_plan(client: ThinkingClient, args: argparse.Namespace) -> int:
    files = [f.strip() for f in (args.files or "").split(",") if f.strip()]
    context = {"files": files, "cwd": str(ROOT)}
    if args.context:
        context["extra"] = args.context
    constraints = list(args.constraint or [])
    stream = getattr(args, "stream", False)

    if stream:
        plan = _plan_streamed(client, args.task, context, constraints, args.max_steps)
        _print_plan(plan)
        return 0

    plan, is_fallback = client.plan_with_fallback(
        args.task, context=context, constraints=constraints, max_steps=args.max_steps)
    if is_fallback:
        _out(f"[субагент недоступен: {offline_reason(client)}] → шаблон-заглушка")
    if args.json:
        _out(json.dumps(plan, ensure_ascii=False, indent=2))
    else:
        _print_plan(plan)
    return 2 if is_fallback else 0


def _plan_streamed(client: ThinkingClient, task: str, context: dict,
                   constraints: list[str], max_steps: int) -> dict:
    """Стримит токены в терминал и возвращает итоговый план."""
    def on_event(ev: dict) -> None:
        kind = ev.get("type", "")
        text = str(ev.get("text", ""))
        if kind == "token":
            sys.stdout.write(text)
            sys.stdout.flush()
            return
        label = EVENT_RU.get(kind, kind)
        _out(f"\n[{label}] {text}")

    if not client.cfg.get("enabled", True):
        plan, _ = client.plan_with_fallback(task, context=context,
                                            constraints=constraints, max_steps=max_steps)
        return plan
    _out("--- поток мыслей субагента ---")
    plan = client.plan_stream(task, context=context, constraints=constraints,
                              max_steps=max_steps, on_event=on_event)
    _out("--- конец потока ---")
    return plan


def cmd_ask(client: ThinkingClient, args: argparse.Namespace) -> int:
    args.files = ""
    args.constraint = []
    args.context = None
    args.json = False
    args.stream = getattr(args, "stream", False)
    args.max_steps = min(args.max_steps, 5)
    return cmd_plan(client, args)


def cmd_reflect(client: ThinkingClient, args: argparse.Namespace) -> int:
    if args.async_mode:
        th = threading.Thread(
            target=lambda: _do_reflect(client, args, echo=True),
            daemon=True)
        th.start()
        _out("рефлексия запущена в фоне — результат: python tools/thinking_cli.py tail")
        return 0
    return _do_reflect(client, args, echo=True)


def _do_reflect(client: ThinkingClient, args: argparse.Namespace, echo: bool = True) -> int:
    try:
        out = client.reflect(args.plan_id, args.step, args.result,
                             observation=args.observation, error=args.error)
    except SchemaError as exc:
        _out(f"! контракт: {exc}")
        return 3
    except ThinkingError as exc:
        _out(f"! {exc}")
        return 5 if "401" in str(exc) or "403" in str(exc) else 4
    if echo:
        _out(f"статус: {out.get('status')}")
        _out(f"совет:  {out.get('advice')}")
        if out.get("rationale"):
            _out(f"почему: {out['rationale']}")
        for s in out.get("next_steps", []) or []:
            _out(f"  далее [{ACTION_RU.get(str(s.get('action')), 'шаг')}] {s.get('desc')}")
    return 0


def cmd_tail(client: ThinkingClient, args: argparse.Namespace) -> int:
    if not args.follow:
        try:
            data = client.events(tail=args.n)
        except Exception as exc:
            _out(f"! {exc}")
            return 4
        for ev in data.get("events", []):
            _out(f"{ev.get('ts', '')}  [{EVENT_RU.get(ev.get('type', ''), ev.get('type'))}] "
                 f"{ev.get('text', '')[:300]}")
        _out(f"-- last_seq={data.get('last_seq')} --")
        return 0

    _out("--- следю за мыслями (Ctrl+C для остановки) ---")
    client.sync_history(tail=args.n)
    client.start_stream(lambda ev: _out(
        f"{ev.get('ts', '')}  [{EVENT_RU.get(ev.get('type', ''), ev.get('type'))}] "
        f"{str(ev.get('text', ''))[:400]}"))
    try:
        seen = len(client.buffered)
        while True:
            time.sleep(0.5)
            if not client.online:
                _out(f"[нет связи] {client.last_error or 'реконнект…'}")
    except KeyboardInterrupt:
        client.stop_stream()
        _out("остановлено")
    return 0


def cmd_metrics(client: ThinkingClient, args: argparse.Namespace) -> int:
    try:
        data = client.metrics()
    except Exception as exc:
        _out(f"! {exc}")
        return 4
    _out(json.dumps(data, ensure_ascii=False, indent=2))
    return 0


def cmd_models(client: ThinkingClient, args: argparse.Namespace) -> int:
    """Показывает модели, скачанные в Colab, и какая из них активна."""
    data = client.models()
    if not data:
        _out("! Colab-сервер не знает /models — перезапусти ноутбук")
        return 4
    _out("GPU: " + ("есть (T4) — модели считаются быстро"
                    if data.get("gpu") else
                    "нет — бесплатные GPU-часы кончились: можно сменить "
                    "Google-аккаунт или продолжить на CPU"))
    for m in data.get("models") or []:
        mark = "*" if m.get("active") else " "
        _out(f"{mark} {m.get('label'):<8} {m.get('size_gb')} ГБ  {m.get('path')}")
        if m.get("hint"):
            _out(f"      {m['hint']}")
    if not data.get("models"):
        _out("(в рантайме нет моделей — выполни ячейку A)")
    return 0


def cmd_use_model(client: ThinkingClient, args: argparse.Namespace) -> int:
    """Переключает модель в Colab: сервер перезапускает движок сам."""
    _out(f"переключаю модель на {args.model}… (20–60 с)")
    out = client.set_model(args.model)
    _out(f"готово: активна {out.get('label') or out.get('active')}")
    if out.get("note"):
        _out(f"({out['note']})")
    return 0


def cmd_dump(client: ThinkingClient, args: argparse.Namespace) -> int:
    """Скачивает логи/исходник с Colab в локальную папку (для архива)."""
    names = DUMP_FILES if args.all else [args.name]
    out = Path(args.out) if args.out else ROOT / "colab_downloads"
    out.mkdir(parents=True, exist_ok=True)
    ok = 0
    for name in names:
        try:
            data = client.dump(name, timeout=float(args.timeout))
        except Exception as exc:                                # noqa: BLE001
            _out(f"! {name}: {exc}")
            continue
        target = out / name
        target.write_bytes(data)
        ok += 1
        _out(f"сохранено {target} ({len(data)} байт)")
    _out(f"готово: {ok} из {len(names)}")
    return 0 if ok else 4


# --------------------------------------------------------------------------- #
#  Режим разработчика: файлы на ПК, запуск, откаты, журнал и дамп
# --------------------------------------------------------------------------- #
def _dev_root(client: ThinkingClient) -> Path:
    root = Path(client.cfg.get("dev_path", "dev_sandbox"))
    root.mkdir(parents=True, exist_ok=True)
    return root


def _dev_log_path(client: ThinkingClient) -> Path:
    return Path(client.cfg.get("devlog_path", "logs/thinking/devlog.jsonl"))


def _safe_name(name: str) -> str:
    """Имя файла без путей: ../ и слэши не пропускаем (контроль доступа)."""
    clean = os.path.basename(str(name or "").replace("\\", "/").strip())
    if not clean or clean.startswith("."):
        raise ValueError(f"недопустимое имя файла: {name!r}")
    if len(clean) > 120 or any(c in clean for c in '<>:"|?*'):
        raise ValueError(f"подозрительное имя файла: {clean!r}")
    return clean


def _dev_log(client: ThinkingClient, action: str, who: str, file: str = "",
             detail: str = "", ok: bool = True) -> None:
    """Каждое действие режима «Разработка» пишется в devlog.jsonl."""
    rec = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "action": action, "who": who, "file": file,
           "detail": str(detail)[:500], "ok": bool(ok)}
    try:
        path = _dev_log_path(client)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except OSError as exc:
        _out(f"! журнал разработки не дописался: {exc}")


def _dev_files(client: ThinkingClient) -> list[dict]:
    root = _dev_root(client)
    out: list[dict] = []
    for p in sorted(root.iterdir()):
        if not p.is_file() or p.name.startswith("."):
            continue
        try:
            st = p.stat()
        except OSError:
            continue
        out.append({"name": p.name, "size": st.st_size,
                    "mtime": datetime.fromtimestamp(
                        st.st_mtime, timezone.utc).isoformat(timespec="seconds")})
    return out


def _dev_read(client: ThinkingClient, name: str) -> str:
    target = _dev_root(client) / _safe_name(name)
    if not target.is_file():
        raise ValueError(f"нет файла {name}")
    return target.read_text(encoding="utf-8", errors="replace")


def _dev_write(client: ThinkingClient, name: str, content: str,
               source: str = "user") -> dict:
    """Запись файла с откатом: перед правкой старая версия копируется в
    .rollback (хранится 20 последних откатов на файл)."""
    target = _dev_root(client) / _safe_name(name)
    existed = target.is_file()
    if existed:
        rb = _dev_root(client) / ".rollback"
        rb.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        shutil.copy2(target, rb / f"{target.name}.{ts}.bak")
        extra = sorted(rb.glob(target.name + ".*.bak"))
        for old in extra[:-20]:
            try:
                old.unlink()
            except OSError:
                pass
    text = str(content if content is not None else "")
    target.write_text(text, encoding="utf-8")
    _dev_log(client, "edit" if existed else "create", source, target.name,
             f"записано байт: {len(text.encode('utf-8'))}")
    return {"ok": True, "name": target.name, "existed": existed,
            "bytes": len(text.encode("utf-8"))}


def _dev_rollback(client: ThinkingClient, name: str) -> dict:
    root = _dev_root(client)
    target = root / _safe_name(name)
    rb = root / ".rollback"
    backups = sorted(rb.glob(target.name + ".*.bak")) if rb.is_dir() else []
    if not backups:
        raise ValueError(f"для {target.name} нет сохранённых откатов")
    shutil.copy2(backups[-1], target)
    _dev_log(client, "rollback", "user", target.name,
             f"возврат на {backups[-1].name}")
    return {"ok": True, "name": target.name, "restored": backups[-1].name,
            "left": len(backups) - 1}


def _dev_run(client: ThinkingClient, name: str) -> dict:
    """Запуск .py из рабочей папки: stdout/stderr идут в окно интерпретатора."""
    target = _dev_root(client) / _safe_name(name)
    if not target.is_file():
        raise ValueError(f"нет файла {name}")
    if target.suffix != ".py":
        raise ValueError("запускать можно только .py-файлы")
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    try:
        # target относительный (dev_sandbox/x.py), а cwd уже dev_sandbox —
        # без resolve() путь склеивался в dev_sandbox/dev_sandbox/…
        r = subprocess.run([sys.executable, str(target.resolve())],
                           cwd=str(_dev_root(client)), env=env,
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=30)
        code, out_, err_ = r.returncode, r.stdout or "", r.stderr or ""
    except subprocess.TimeoutExpired as exc:
        code = -1
        raw = exc.stdout or ""
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", "replace")
        out_ = raw
        err_ = "интерпретатор остановлен: превышен таймаут 30 с"
    except OSError as exc:
        raise ValueError(f"не удалось запустить интерпретатор: {exc}")
    _dev_log(client, "run", "user", target.name, f"код выхода {code}",
             ok=code == 0)
    return {"code": code, "stdout": (out_ or "")[-20000:],
            "stderr": (err_ or "")[-8000:]}


def _dev_dump(client: ThinkingClient) -> Path:
    """Дамп: рабочая папка + все журналы программы в один zip."""
    when = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(client.cfg.get("dump_path", "dumps"))
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"thinking_dev_{when}.zip"
    root = _dev_root(client)
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(root.rglob("*")):
            if p.is_file():
                zf.write(p, arcname="dev_sandbox/"
                                    + str(p.relative_to(root)).replace("\\", "/"))
        logs = Path(client.cfg.get("devlog_path", "logs/thinking/devlog.jsonl")).parent
        if logs.is_dir():
            for p in sorted(logs.rglob("*")):
                if p.is_file() and p.stat().st_size < 20 * 1024 * 1024:
                    zf.write(p, arcname=str(p).replace("\\", "/"))
    _dev_log(client, "dump", "user", "", str(target))
    return target


def cmd_reflect_metrics(client: ThinkingClient, args: argparse.Namespace) -> int:
    """Рефлексия по метрикам выгоды: разовая либо зацикленная (--loop)."""
    interval = max(30.0, float(getattr(args, "interval", 600) or 600))
    while True:
        res = client.reflect_metrics(engine=getattr(args, "engine", "subagent"),
                                     since=getattr(args, "since", "") or "")
        _out(f"рефлексия [{res['status']}] за {res['period']}")
        _out(f"совет:   {res['advice']}")
        if res.get("rationale"):
            _out(f"почему:  {res['rationale']}")
        _out(f"метрики: {res['metrics']}")
        _out(f"сохранено: {res['saved_to']}  (+ отчёт во вкладке «Отчёты»)")
        if not getattr(args, "loop", False):
            return 0
        _out(f"следующая итерация через {interval:.0f} с (Ctrl+C — остановить)")
        try:
            time.sleep(interval)
        except KeyboardInterrupt:
            _out("цикл рефлексии остановлен")
            return 0


# --------------------------------------------------------------------------- #
#  Кэш ответов: повторный тот же вопрос не жжёт Colab. При обрывах потока
#  человек нередко шлёт одно и то же сообщение несколько раз подряд — теперь
#  повтор отдаётся мгновенно. TTL — answer_cache_ttl (по умолчанию 30 минут),
#  живёт и в памяти процесса, и в постоянном журнале диалога.
# --------------------------------------------------------------------------- #
_ANSWER_CACHE: dict[str, tuple[float, dict]] = {}
_CACHE_MAX = 120


def _cache_norm(message: str) -> str:
    """Нормализация ключа: регистр и повторные пробелы не должны делить кэш."""
    return " ".join(str(message).split()).lower()


def _cache_ttl(client: ThinkingClient) -> float:
    try:
        return max(0.0, float(client.cfg.get("answer_cache_ttl", 1800)))
    except (TypeError, ValueError):
        return 1800.0


def _cache_get(client: ThinkingClient, kind: str, message: str) -> dict | None:
    ttl, norm = _cache_ttl(client), _cache_norm(message)
    if not norm or ttl <= 0:
        return None
    hit = _ANSWER_CACHE.get(f"{kind}\x00{norm}")
    if hit and time.time() - hit[0] <= ttl and not hit[1].get("fallback"):
        return dict(hit[1], cached=True)
    # перезапуск панели: такой же вопрос мог остаться в журнале диалога
    path = getattr(client, "_chat_path", None)
    if not path or not Path(path).exists():
        return None
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()[-60:]
    except OSError:
        return None
    for line in reversed(lines):
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if _cache_norm(str(rec.get("question") or "")) != norm or rec.get("fallback"):
            continue
        try:
            age = (datetime.now(timezone.utc)
                   - datetime.fromisoformat(str(rec.get("at")))).total_seconds()
        except ValueError:
            continue
        if 0 <= age <= ttl:
            return {"reply": rec.get("reply"), "at": rec.get("at"),
                    "duration_ms": rec.get("duration_ms"),
                    "tokens_in": rec.get("tokens_in"),
                    "tokens_out": rec.get("tokens_out"),
                    "tokens_estimate": rec.get("tokens_estimate"),
                    "cached": True}
    return None


def _cache_put(client: ThinkingClient, kind: str, message: str,
               reply: dict) -> None:
    ttl, norm = _cache_ttl(client), _cache_norm(message)
    if not norm or ttl <= 0 or reply.get("fallback"):
        return
    _ANSWER_CACHE[f"{kind}\x00{norm}"] = (time.time(), dict(reply))
    while len(_ANSWER_CACHE) > _CACHE_MAX:
        _ANSWER_CACHE.pop(next(iter(_ANSWER_CACHE)))


def _cache_journal(client: ThinkingClient, message: str, reply: dict) -> None:
    """Ответ из кэша тоже пишем в журнал диалога.

    Иначе перерисовка истории (она строится из журнала) стирает обмен,
    которого в журнале нет, — человек видит ответ секунду и он исчезает.
    """
    try:
        client._remember_chat(                              # noqa: SLF001
            message, ChatReply.from_dict(
                {k: v for k, v in reply.items() if k != "cached"}),
            cached=True)
    except Exception as exc:                                # noqa: BLE001
        _out(f"! кэш: не записал в журнал диалога: {exc}")


# --------------------------------------------------------------------------- #
#  Панель реального времени
# --------------------------------------------------------------------------- #
def cmd_panel(client: ThinkingClient, args: argparse.Namespace) -> int:
    html_path = ROOT / "tools" / "thinking_panel.html"
    if not html_path.exists():
        _out(f"! нет {html_path}")
        return 3
    # HTML читается по mtime: правка веб-панели видна сразу, без рестарта.
    html_cache: dict = {"mtime": -1.0, "data": b""}

    def _panel_html() -> bytes:
        try:
            st = html_path.stat()
            if st.st_mtime != html_cache["mtime"]:
                html_cache["mtime"] = st.st_mtime
                html_cache["data"] = html_path.read_bytes()
        except OSError:
            pass
        return html_cache["data"]

    client.sync_history(tail=200)
    client.start_stream(lambda ev: None)

    # Первичный канал ПК по ТЗ — опрос GET /events: SSE через туннель может
    # молчать (ping-строки не доходят), поэтому фоновый опрос страхует поток.
    stop_poll = threading.Event()

    def _poll_loop() -> None:
        every = max(2.0, float(client.cfg.get("panel_poll", 5)))
        # Раз в ~25 минут — явный прогрев /health: Colab между рабочими
        # сессиями не засыпает, туннель считается живым.
        warm_every = max(1, int(1500 / every))
        n = 0
        while not stop_poll.wait(every):
            n += 1
            try:
                client.sync_history(tail=50)
                if n % warm_every == 0:
                    try:
                        ok = client.health(timeout=15)
                    except Exception:                          # noqa: BLE001
                        ok = False
                    _out(f"[прогрев] /health {'ok' if ok else 'не прошёл'}")
            except Exception as exc:                           # noqa: BLE001
                _out(f"! опрос событий не прошёл: {exc}")

    threading.Thread(target=_poll_loop, daemon=True).start()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # тишина в консоли
            pass

        def _send(self, code: int, ctype: str, body: bytes) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
                pass

        def _json_out(self, code: int, data: dict) -> None:
            self._send(code, "application/json; charset=utf-8",
                       json.dumps(data, ensure_ascii=False, default=str).encode("utf-8"))

        def _body(self) -> dict:
            """Читает JSON тела запроса; плохой JSON — не ошибка сервера."""
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except (TypeError, ValueError):
                length = 0
            if length <= 0:
                return {}
            try:
                raw = self.rfile.read(length).decode("utf-8", "replace")
                data = json.loads(raw)
                return data if isinstance(data, dict) else {}
            except (ValueError, OSError):
                return {}

        def do_POST(self):  # noqa: N802
            """Действия панели: чат, память, удаление записей."""
            path = self.path.split("?")[0]
            data = self._body()
            if path == "/api/chat":
                message = str(data.get("message") or "").strip()
                if not message:
                    self._json_out(400, {"error": "пустое сообщение"})
                    return
                kind = "chat:%d" % int(bool(data.get("memory", True)))
                hit = _cache_get(client, kind, message)
                if hit:
                    _out(f"[чат] из кэша: {message[:80]}")
                    _cache_journal(client, message, hit)
                    self._json_out(200, {"reply": hit})
                    return
                _out(f"[чат] {message[:120]}")
                try:
                    reply = client.chat(message,
                                        use_memory=bool(data.get("memory", True)),
                                        max_steps=int(data.get("max_steps") or 4))
                except SchemaError as exc:
                    self._json_out(400, {"error": str(exc)})
                    return
                except ThinkingError as exc:
                    self._json_out(502, {"error": str(exc)[:300]})
                    return
                _cache_put(client, kind, message, reply)
                self._json_out(200, {"reply": reply})
                return
            if path == "/api/model":
                # Переключение модели: сервер Colab перезапускает движок сам.
                want = str(data.get("model") or "").strip()
                if not want:
                    self._json_out(400, {"error": "не указана модель"})
                    return
                try:
                    out = client.set_model(want)
                except SchemaError as exc:
                    self._json_out(400, {"error": str(exc)})
                    return
                except ThinkingError as exc:
                    self._json_out(502, {"error": str(exc)[:300]})
                    return
                _out(f"[модель] переключаюсь на {want}")
                self._json_out(200, out)
                return
            if path == "/api/chat/stream":
                # Поток ответа: панель показывает текст по мере генерации.
                message = str(data.get("message") or "").strip()
                if not message:
                    self._json_out(400, {"error": "пустое сообщение"})
                    return
                kind = "chat:%d" % int(bool(data.get("memory", True)))
                cached = _cache_get(client, kind, message)
                if cached:
                    _out(f"[чат] из кэша: {message[:80]}")
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                # После done соединение обязано закрыться: HTTP/1.0-сервер с
                # keep-alive держит сокет открытым, браузер ждёт EOF, промис
                # apiStream виснет — и кнопка «Отправить» умирает навсегда.
                self.send_header("Connection", "close")
                self.close_connection = True
                self.end_headers()
                done: dict = {}

                def _emit(ev: dict) -> None:
                    self.wfile.write(
                        ("data: " + json.dumps(ev, ensure_ascii=False,
                                                default=str) + "\n\n").encode("utf-8"))
                    self.wfile.flush()

                try:
                    if cached:
                        reply = cached
                        _cache_journal(client, message, reply)
                        _emit({"type": "token",
                               "text": str(reply.get("reply") or "")})
                    else:
                        reply = client.chat_stream(
                            message, on_token=lambda piece: _emit({"type": "token",
                                                                   "text": piece}),
                            on_retry=lambda n, total: _emit(
                                {"type": "retry", "attempt": n, "of": total}),
                            use_memory=bool(data.get("memory", True)))
                        _cache_put(client, kind, message, reply)
                    done = {"type": "done", **reply}
                    _emit(done)
                except SchemaError as exc:
                    _emit({"type": "error", "text": str(exc)})
                except ThinkingError as exc:
                    _emit({"type": "error", "text": str(exc)[:300]})
                except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
                    pass                        # вкладку закрыли — это не ошибка
                return
            if path == "/api/memory":
                try:
                    if data.get("action") == "forget":
                        mem = client.forget(facts=bool(data.get("facts", True)),
                                            turns=bool(data.get("turns", True)))
                    elif data.get("action") == "remove" and data.get("fact"):
                        mem = client.memory()
                        mem["facts"] = [f for f in mem["facts"]
                                        if f != str(data["fact"])]
                        client._write_memory(mem)  # noqa: SLF001 — свой же метод
                        mem = client.memory()
                    else:
                        mem = client.remember(fact=str(data.get("fact") or ""),
                                              profile=str(data.get("profile") or ""))
                except (SchemaError, ThinkingError) as exc:
                    self._json_out(400, {"error": str(exc)[:200]})
                    return
                self._json_out(200, {"memory": mem})
                return
            if path == "/api/delete":
                kind = str(data.get("kind") or "report")
                n = data.get("n")
                before = str(data.get("before") or "")
                try:
                    removed = client.delete_history(
                        kind=kind,
                        rid=str(data.get("rid") or ""),
                        n=int(n) if str(n or "").strip() else None,
                        at=str(data.get("at") or ""),
                        before=before)
                except (SchemaError, ThinkingError) as exc:
                    self._json_out(400, {"error": str(exc)[:200]})
                    return
                _out(f"[удалено] {kind}: {removed} записей")
                self._json_out(200, {"removed": removed})
                return
            if path == "/api/tokens":
                self._json_out(200, client.tokens())
                return
            # ---- токен: панель подставляет его сама, здесь только сохранение
            if path == "/api/token":
                tok = str(data.get("token") or "")
                try:
                    cfg_path = client.set_token(tok)
                except OSError as exc:
                    self._json_out(500, {"error": f"не записался конфиг: {exc}"})
                    return
                _out(f"[токен] сохранён в {cfg_path}")
                self._json_out(200, {"ok": True, "path": str(cfg_path)})
                return
            # ---- рефлексия по метрикам выгоды (вкладка «Выгода»)
            if path == "/api/reflect-metrics":
                engine = str(data.get("engine") or "subagent")
                if engine not in ("subagent", "local"):
                    engine = "subagent"
                try:
                    res = client.reflect_metrics(
                        engine=engine, since=str(data.get("since") or ""))
                except ThinkingError as exc:
                    self._json_out(502, {"error": str(exc)[:300]})
                    return
                _out(f"[рефлексия] {engine}: {res.get('status')} — "
                     f"{str(res.get('advice'))[:120]}")
                self._json_out(200, {"result": res})
                return
            # ---- режим «Разработка»: чат с моделью по коду
            if path == "/api/dev/chat":
                message = str(data.get("message") or "").strip()
                if not message:
                    self._json_out(400, {"error": "пустое сообщение"})
                    return
                active = str(data.get("active") or "")
                try:
                    code = _dev_read(client, active) if active else ""
                except (ValueError, OSError):
                    code = ""
                _out(f"[разработка] {message[:120]}")
                try:
                    prop = client.dev(
                        message,
                        files=[str(f.get("name") or "")
                               for f in _dev_files(client)],
                        active_name=active if code else "",
                        active_code=code,
                        on_token=None,
                        on_retry=lambda n, total: _out(
                            f"[разработка] поток оборвался, повтор {n}/{total}"))
                except SchemaError as exc:
                    self._json_out(400, {"error": str(exc)})
                    return
                except ThinkingError as exc:
                    self._json_out(502, {"error": str(exc)[:300]})
                    return
                _dev_log(client, "chat", "user", active,
                         f"предложение модели: {prop.get('action')} "
                         f"{prop.get('filename')}")
                self._json_out(200, {"proposal": prop})
                return
            # ---- файлы рабочей папки: запись (с откатом) и запуск
            if path == "/api/dev/file":
                try:
                    out = _dev_write(client, str(data.get("name") or ""),
                                     str(data.get("content") or ""),
                                     source=str(data.get("source") or "user"))
                except ValueError as exc:
                    self._json_out(400, {"error": str(exc)})
                    return
                self._json_out(200, out)
                return
            if path == "/api/dev/run":
                try:
                    out = _dev_run(client, str(data.get("name") or ""))
                except ValueError as exc:
                    self._json_out(400, {"error": str(exc)})
                    return
                self._json_out(200, out)
                return
            if path == "/api/dev/rollback":
                try:
                    out = _dev_rollback(client, str(data.get("name") or ""))
                except ValueError as exc:
                    self._json_out(400, {"error": str(exc)})
                    return
                _out(f"[откат] {out['name']} ← {out['restored']}")
                self._json_out(200, out)
                return
            if path == "/api/devump":
                try:
                    target = _dev_dump(client)
                except OSError as exc:
                    self._json_out(500, {"error": str(exc)})
                    return
                self._json_out(200, {"ok": True, "path": str(target)})
                return
            self._send(404, "text/plain; charset=utf-8", b"not found")

        def do_GET(self):  # noqa: N802
            path = self.path.split("?")[0]
            query = urllib.parse.parse_qs(self.path.partition("?")[2])
            if path in ("/", "/index.html"):
                self._send(200, "text/html; charset=utf-8", _panel_html())
            elif path == "/api/state":
                # ?since= — период «за сегодня»: панель передаёт локальную
                # полночь в UTC, остальное фильтрует клиент.
                since = str(query.get("since", [""])[0] or "")
                self._send(200, "application/json; charset=utf-8",
                           json.dumps(client.status(since=since),
                                      ensure_ascii=False).encode("utf-8"))
            elif path == "/api/events":
                body = json.dumps({"events": client.buffered[-200:]},
                                  ensure_ascii=False).encode("utf-8")
                self._send(200, "application/json; charset=utf-8", body)
            elif path == "/api/models":
                # Список моделей в рантайме: панель рисует их и переключает.
                try:
                    data = client.models()
                except SchemaError as exc:
                    data = {"error": str(exc)}
                if not data:
                    data = {"error": "Colab-сервер не знает /models (старый ноутбук)",
                            "models": []}
                self._send(200, "application/json; charset=utf-8",
                           json.dumps(data, ensure_ascii=False).encode("utf-8"))
            elif path == "/api/dev/files":
                body = json.dumps({"files": _dev_files(client),
                                   "root": str(_dev_root(client))},
                                  ensure_ascii=False).encode("utf-8")
                self._send(200, "application/json; charset=utf-8", body)
            elif path == "/api/dev/file":
                name = str(query.get("name", [""])[0] or "")
                try:
                    content = _dev_read(client, name)
                except (ValueError, OSError) as exc:
                    self._json_out(400, {"error": str(exc)})
                    return
                body = json.dumps({"name": name, "content": content},
                                  ensure_ascii=False).encode("utf-8")
                self._send(200, "application/json; charset=utf-8", body)
            elif path == "/api/dev/log":
                lines: list[dict] = []
                p = _dev_log_path(client)
                if p.exists():
                    for row in p.read_text(encoding="utf-8",
                                           errors="replace").splitlines()[-200:]:
                        try:
                            lines.append(json.loads(row))
                        except ValueError:
                            continue
                body = json.dumps({"log": lines}, ensure_ascii=False).encode("utf-8")
                self._send(200, "application/json; charset=utf-8", body)
            elif path == "/events":
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.end_headers()
                seen = 0
                try:
                    while True:
                        for ev in client.buffered:
                            seq = int(ev.get("seq") or 0)
                            if seq and seq > seen:
                                seen = seq
                                payload = json.dumps(ev, ensure_ascii=False, default=str)
                                self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                        self.wfile.write(b": ping\n\n")
                        self.wfile.flush()
                        time.sleep(0.3)
                except (BrokenPipeError, ConnectionResetError,
                        ConnectionAbortedError, TimeoutError, OSError):
                    # Вкладку закрыли/перезагрузили — это обычное дело, не ошибка
                    pass
            else:
                self._send(404, "text/plain; charset=utf-8", b"not found")

    class PanelServer(ThreadingHTTPServer):
        """Панель, которая не пугает трассировками из-за ушедшего браузера."""

        daemon_threads = True

        def handle_error(self, request, client_address):       # noqa: D102
            exc = sys.exc_info()[1]
            if isinstance(exc, (ConnectionAbortedError, ConnectionResetError,
                                BrokenPipeError, TimeoutError)):
                return       # вкладку перезагрузили / EventSource переподключился
            super().handle_error(request, client_address)

    host = client.cfg.get("panel_host", "127.0.0.1")
    port = int(args.port or client.cfg.get("panel_port", 8765))
    if _port_busy(host, port):
        _out(f"! порт {host}:{port} уже занят — скорее всего, панель уже запущена.")
        _out(f"  Откройте http://{host}:{port} — или возьмите другой порт: "
             f"panel --port {port + 1}")
        return 4
    try:
        httpd = PanelServer((host, port), Handler)
    except OSError as exc:
        _out(f"! не удалось занять {host}:{port}: {exc}")
        return 4
    _out(f"панель:  http://{host}:{port}")
    _out(f"субагент: {client.base or 'URL не задан'}")
    _out(f"поток:    SSE {'ждём первый поток' if not client.stream_ok else 'ok'}"
         + (f" ({client.stream_error})" if client.stream_error else "")
         + " ·  опрос /events каждые "
         + f"{max(2.0, float(client.cfg.get('panel_poll', 5))):.0f} с")
    _out("Ctrl+C — остановить")

    if getattr(args, "open", False):
        # Браузер открываем сами, когда порт уже слушает: иначе страница
        # успевает загрузиться раньше сервера и показывает ошибку.
        def _open_browser() -> None:
            for _ in range(40):
                if _port_busy(host, port):
                    break
                time.sleep(0.25)
            time.sleep(0.7)
            url = f"http://{host}:{port}"
            try:
                import webbrowser
                webbrowser.open(url, new=2)
                _out(f"браузер открыт: {url}")
            except Exception as exc:                          # noqa: BLE001
                _out(f"! не смог открыть браузер ({exc}) — открой вручную {url}")
        threading.Thread(target=_open_browser, daemon=True).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        stop_poll.set()
        client.stop_stream()
        httpd.shutdown()
        _out("остановлено")
    return 0


# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="thinking_cli", description="Субагент «Мышление»")
    sub = p.add_subparsers(dest="cmd")

    sub.add_parser("doctor")
    sub.add_parser("url")
    sub.add_parser("metrics")

    s = sub.add_parser("set-url")
    s.add_argument("url")
    s.add_argument("token", nargs="?")

    s = sub.add_parser("plan")
    s.add_argument("task")
    s.add_argument("--files", default="", help="список путей через запятую")
    s.add_argument("--constraint", action="append")
    s.add_argument("--context", help="дополнительный контекст строкой")
    s.add_argument("--max-steps", type=int, default=12)
    s.add_argument("--json", action="store_true")
    s.add_argument("--stream", action="store_true")

    s = sub.add_parser("ask")
    s.add_argument("task")
    s.add_argument("--files", default="")
    s.add_argument("--constraint", action="append")
    s.add_argument("--context", default=None)
    s.add_argument("--max-steps", type=int, default=5)
    s.add_argument("--json", action="store_true")
    s.add_argument("--stream", action="store_true")

    s = sub.add_parser("reflect")
    s.add_argument("plan_id")
    s.add_argument("--step", type=int, required=True)
    s.add_argument("--result", required=True)
    s.add_argument("--observation")
    s.add_argument("--error")
    s.add_argument("--async", dest="async_mode", action="store_true")

    s = sub.add_parser("tail")
    s.add_argument("--n", type=int, default=30)
    s.add_argument("--follow", action="store_true")

    s = sub.add_parser("panel")
    s.add_argument("--port", type=int, default=0)
    s.add_argument("--open", action="store_true",
                   help="открыть панель в браузере сразу после старта")

    s = sub.add_parser("dump", help="скачать логи Colab в локальную папку")
    s.add_argument("name", nargs="?", default="llm.log",
                   help=f"файл из списка: {', '.join(DUMP_FILES)}")
    s.add_argument("--all", action="store_true", help="снять весь белый список")
    s.add_argument("--out", default="", help="куда класть (по умолчанию colab_downloads/)")
    s.add_argument("--timeout", type=float, default=30.0)

    sub.add_parser("models", help="показать модели в Colab и активную из них")
    s = sub.add_parser("use-model", help="переключить модель в Colab")
    s.add_argument("model", help="короткое имя (3b, 1.5B) или путь к .gguf")

    s = sub.add_parser("reflect-metrics",
                       help="рефлексия по метрикам выгоды (отчёт в «Отчёты»)")
    s.add_argument("--engine", choices=("subagent", "local"), default="subagent",
                   help="субагент (модель Colab) или локальные правила")
    s.add_argument("--since", default="",
                   help="начало периода ISO; по умолчанию — сегодня")
    s.add_argument("--loop", action="store_true",
                   help="зациклить: повторять рефлексию")
    s.add_argument("--interval", type=int, default=600,
                   help="пауза между итерациями, секунд (по умолчанию 600)")

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.cmd:
        build_parser().print_help()
        return 0
    client = ThinkingClient()
    try:
        if args.cmd == "doctor":
            return cmd_doctor(client, args)
        if args.cmd == "set-url":
            return cmd_set_url(client, args)
        if args.cmd == "url":
            return cmd_url(client, args)
        if args.cmd == "plan":
            return cmd_plan(client, args)
        if args.cmd == "ask":
            return cmd_ask(client, args)
        if args.cmd == "reflect":
            return cmd_reflect(client, args)
        if args.cmd == "tail":
            return cmd_tail(client, args)
        if args.cmd == "panel":
            return cmd_panel(client, args)
        if args.cmd == "metrics":
            return cmd_metrics(client, args)
        if args.cmd == "dump":
            return cmd_dump(client, args)
        if args.cmd == "models":
            return cmd_models(client, args)
        if args.cmd == "use-model":
            return cmd_use_model(client, args)
        if args.cmd == "reflect-metrics":
            return cmd_reflect_metrics(client, args)
    except SchemaError as exc:
        _out(f"! контракт нарушен: {exc}")
        return 3
    except ThinkingError as exc:
        _out(f"! {exc}")
        msg = str(exc)
        if "401" in msg or "403" in msg:
            return 5
        return 4
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
