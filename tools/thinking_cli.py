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
from thinking.schemas import (ChatReply, MAX_MEMORY_FACTS, MAX_MEMORY_TURNS,  # noqa: E402
                              SchemaError, utcnow)
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


def _out(text: str, file=None) -> None:
    """Печать в консоль, которая не умеет в unicode.

    На Windows консоль по умолчанию cp1251, и любой символ вне неё (стрелка,
    галочка, значок) ронял команду с UnicodeEncodeError — пользователь видел
    трейсбек вместо результата (04.10: `models` падал на стрелке «→»).
    Раньше каждая команда должна была обойти это вручную; теперь печать
    безопасна везде, а символы, которые консоль не тянет, заменяются на
    похожие: → ⇒ >, ✓ v, ✗ x, ⚠ !

    `file` — куда печатать: по умолчанию stdout, а в режиме `--json`
    диагностика уходит в stderr, иначе контракт JSON ломается (аудит B:
    `plan --json` при fallback печатал `[субагент недоступен: …]` прямо
    перед JSON, и парсер получал мусор).
    """
    dst = file or sys.stdout
    text = text.translate(_ASCII_FALLBACK)
    try:
        dst.write(text + "\n")
        dst.flush()
    except UnicodeEncodeError:
        enc = dst.encoding or "ascii"
        dst.write(text.encode(enc, "replace").decode(enc, "replace") + "\n")
        dst.flush()


# Символы, которых нет в cp1251, но которые мы печатаем регулярно.
# str.translate оставляет на месте всё, чего в таблице нет, — добирает
# try/except выше на случай экзотики.
_ASCII_FALLBACK = str.maketrans({
    "→": "=>", "⇒": "=>", "←": "<-", "✓": "v", "✔": "v", "✗": "x", "✘": "x",
    "⚠": "!", "★": "*", "•": "-", "—": "-", "–": "-", "…": "...",
    "«": '"', "»": '"', "№": "N", "≈": "~", "≥": ">=", "≤": "<=",
})


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
    if getattr(args, "back", False):
        path = ThinkingClient.set_url("", "", back=True)
        fresh = ThinkingClient()
        _out(f"возврат на {fresh.base or '(пусто)'} → {path}")
        _out("панель подхватит новый адрес сама, рестарт не нужен")
        return 0 if fresh.health(timeout=6) else 4
    if not args.url:
        _out("! не указан адрес: set-url <URL> [TOKEN] либо set-url --back")
        return 3
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
    # --json: в stdout только сам JSON, вся диагностика — в stderr, иначе
    # контракт ломается ровно в самом частом сценарии (fallback, exit 2)
    diag = sys.stderr if getattr(args, "json", False) else None

    if stream:
        plan, is_fallback = _plan_streamed(client, args.task, context,
                                           constraints, args.max_steps,
                                           diag=diag)
        if is_fallback:
            _out(f"[субагент недоступен: {offline_reason(client)}] → шаблон-заглушка",
                 file=diag)
        if args.json:
            _out(json.dumps(plan, ensure_ascii=False, indent=2))
        else:
            _print_plan(plan)
        return 2 if is_fallback else 0

    plan, is_fallback = client.plan_with_fallback(
        args.task, context=context, constraints=constraints, max_steps=args.max_steps)
    if is_fallback:
        _out(f"[субагент недоступен: {offline_reason(client)}] → шаблон-заглушка",
             file=diag)
    if args.json:
        _out(json.dumps(plan, ensure_ascii=False, indent=2))
    else:
        _print_plan(plan)
    return 2 if is_fallback else 0


def _plan_streamed(client: ThinkingClient, task: str, context: dict,
                   constraints: list[str], max_steps: int,
                   diag=None) -> tuple[dict, bool]:
    """Стримит токены в терминал и возвращает итоговый план.

    `diag` — поток для диагностики и токенов (stderr в режиме `--json`);
    по умолчанию всё уходит в stdout, как раньше.
    """
    def on_event(ev: dict) -> None:
        kind = ev.get("type", "")
        text = str(ev.get("text", ""))
        if kind == "token":
            dst = diag or sys.stdout
            dst.write(text)
            dst.flush()
            return
        label = EVENT_RU.get(kind, kind)
        _out(f"\n[{label}] {text}", file=diag)

    if not client.cfg.get("enabled", True):
        plan, is_fb = client.plan_with_fallback(task, context=context,
                                                constraints=constraints,
                                                max_steps=max_steps)
        return plan, is_fb
    _out("--- поток мыслей субагента ---", file=diag)
    try:
        plan = client.plan_stream(task, context=context, constraints=constraints,
                                  max_steps=max_steps, on_event=on_event)
    except ThinkingError as exc:
        # Обрыв туннеля, 524 или ошибка модели на середине потока (живой
        # прогон 04.10: сервер додумал план за 5 с, а байты до клиента не
        # доехали). Раньше здесь падал трейсбек прямо из http.client.
        # Передаём причину: при обрыве транспорта обычный /plan съел бы
        # те же 120 с, поэтому дальше сразу шаблон-план.
        _out("", file=diag)
        _out(f"! поток не прошёл ({exc}) → пробую обычный запрос", file=diag)
        plan, is_fallback = client.plan_with_fallback(
            task, stream_error=exc, context=context, constraints=constraints,
            max_steps=max_steps)
        _out("--- конец потока ---", file=diag)
        return plan, is_fallback
    _out("--- конец потока ---", file=diag)
    return plan, False


def cmd_ask_multi(client: ThinkingClient, args: argparse.Namespace) -> int:
    """Один вопрос — всем моделям сразу, ответы рядом.

    Честно о смысле: ядра у рантайма одни, поэтому СУММАРНАЯ скорость не
    растёт — каждый ответ дольше примерно в N раз. Польза в сравнении: видно,
    где сильная модель спотыкается, а где справляется слабая, и ответ можно
    выбрать руками.
    """
    try:
        out = client.multi_chat(args.task)
    except ThinkingError as exc:
        _out(f"! параллельный вопрос не прошёл: {exc}")
        return 1
    if getattr(args, "json", False):
        _out(json.dumps(out, ensure_ascii=False, indent=2))
        return 0
    answers = out.get("answers") or []
    if not answers:
        _out("! ни одна модель не ответила")
        return 1
    _out(f"моделей: {out.get('count')}  ·  стены: {out.get('wall_seconds')} с")
    if out.get("note"):
        _out(f"({out['note']})")
    ok = [a for a in answers if a.get("ok")]
    for i, a in enumerate(answers, 1):
        mark = "v" if a.get("ok") else "x"
        _out("")
        _out(f"--- {mark} {i}. {a.get('model')}  ({a.get('seconds')} с) ---")
        if a.get("ok"):
            _out(str(a.get("answer") or "").strip() or "(пусто)")
        else:
            _out(f"ошибка: {a.get('error')}")
    if len(ok) > 1:
        _out("")
        _out(f"ответов пришло: {len(ok)} из {len(answers)} · "
              f"самый быстрый: {min(ok, key=lambda a: float(a.get('seconds') or 0)).get('model')}")
    return 0


def cmd_ask(client: ThinkingClient, args: argparse.Namespace) -> int:
    # Аудит B: флаги ask молча обнулялись — `ask --json` печатал человекочитаемый
    # текст, а `--files/--constraint/--context` просто игнорировались, хотя
    # парсер их принимает. Единственное отличие ask от plan — короткий ответ,
    # поэтому ограничиваем только max_steps, остальное пробрасываем как есть.
    args.max_steps = min(args.max_steps, 5)
    args.stream = getattr(args, "stream", False)
    return cmd_plan(client, args)


def cmd_reflect(client: ThinkingClient, args: argparse.Namespace) -> int:
    if args.async_mode:
        # Фон — только отсоединённым процессом: daemon-поток умирал вместе
        # с CLI на середине HTTP-запроса, и результат рефлексии терялся
        # (баг аудита AUD-03). Дочерний процесс сам дописывает журнал и
        # отчёт, родитель сразу освобождает терминал.
        cmd = [sys.executable, os.path.abspath(__file__), "reflect",
               args.plan_id, "--step", str(args.step), "--result", args.result,
               # дочерний процесс должен пометить запись фоновой — иначе
               # метрика background_ms остаётся нулевой навсегда (аудит B)
               "--background"]
        if args.observation:
            cmd += ["--observation", args.observation]
        if args.error:
            cmd += ["--error", args.error]
        kw: dict = {}
        if os.name == "posix":
            kw["start_new_session"] = True   # переживает закрытие терминала
        try:
            proc = subprocess.Popen(cmd, cwd=str(ROOT), **kw)
        except OSError as exc:
            _out(f"! фон не запустился ({exc}) — выполняю синхронно")
            return _do_reflect(client, args, echo=True)
        _out(f"рефлексия запущена в фоне (pid {proc.pid}) — "
             f"результат: python tools/thinking_cli.py tail")
        return 0
    return _do_reflect(client, args, echo=True)


def _do_reflect(client: ThinkingClient, args: argparse.Namespace, echo: bool = True) -> int:
    try:
        out = client.reflect(args.plan_id, args.step, args.result,
                             observation=args.observation, error=args.error,
                             # дочерний процесс от reflect --async помечает
                             # запись фоновой, иначе метрика background_ms
                             # не заполняется никогда (аудит B)
                             background=bool(getattr(args, "background", False)))
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
    was_online, last_warn = True, 0.0
    try:
        while True:
            time.sleep(0.5)
            if client.online:
                if not was_online:
                    _out("[связь восстановлена]")
                was_online, last_warn = True, 0.0
                continue
            # Печатаем при изменении состояния и не чаще раза в 15 с —
            # иначе при длинном офлайне хвост спамит «[нет связи]» каждые
            # 0,5 с (аудит AUD-23).
            now = time.time()
            if was_online or now - last_warn >= 15:
                _out(f"[нет связи] {client.last_error or 'реконнект…'}")
                last_warn = now
            was_online = False
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


def _dev_run(client: ThinkingClient, name: str, stdin_text: str = "") -> dict:
    """Запуск .py из рабочей папки: stdout/stderr идут в окно интерпретатора.

    stdin_text — что получит программа на ввод. Если его не задать, ввод
    закрывается (DEVNULL): иначе input() в калькуляторе висел бы все 30 с
    молча, и человек считал бы, что «Запустить» сломан.
    """
    target = _dev_root(client) / _safe_name(name)
    if not target.is_file():
        raise ValueError(f"нет файла {name}")
    if target.suffix != ".py":
        raise ValueError("запускать можно только .py-файлы")
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    data = str(stdin_text or "")
    try:
        # target относительный (dev_sandbox/x.py), а cwd уже dev_sandbox —
        # без resolve() путь склеивался в dev_sandbox/dev_sandbox/…
        r = subprocess.run([sys.executable, str(target.resolve())],
                           cwd=str(_dev_root(client)), env=env,
                           input=data if data else None,
                           stdin=None if data else subprocess.DEVNULL,
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=30)
        code, out_, err_ = r.returncode, r.stdout or "", r.stderr or ""
    except subprocess.TimeoutExpired as exc:
        code = -1
        raw = exc.stdout or ""
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", "replace")
        out_ = raw
        # stderr тоже читаем (аудит C-8): причина зависания обычно там,
        # а без неё человек видит только «превышен таймаут» и не знает, что
        # делать
        raw_err = exc.stderr or ""
        if isinstance(raw_err, bytes):
            raw_err = raw_err.decode("utf-8", "replace")
        err_ = (str(raw_err).rstrip() + "\n" if str(raw_err).strip() else "")
        err_ += "интерпретатор остановлен: превышен таймаут 30 с"
    except OSError as exc:
        raise ValueError(f"не удалось запустить интерпретатор: {exc}")
    # input() без переданного stdin даёт EOFError — объясняем по-человечески,
    # иначе человек видит только трассировку и думает, что сломан запуск
    if "EOFError" in (err_ or "") and not data:
        err_ += ("\n[подсказка] программа ждёт ввода (input()). Впиши его в поле "
                 "«ввод для программы» и запусти снова — например: 5\n7")
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
                    # относительный путь: абсолютный Windows-arcname
                    # раскидывает файлы по дереву диска при распаковке
                    # (аудит C-7)
                    arc = "logs/" + str(p.relative_to(logs)).replace("\\", "/")
                    zf.write(p, arcname=arc)
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


def _tail_lines(path: Any, count: int = 60, kb: int = 512) -> list[str]:
    """Последние `count` строк файла.

    Читается только хвост (до `kb` КБ): полное чтение всего журнала на
    каждом промахе кэша было заметно при живом чате (аудит AUD-25).
    Обрезанная первая строка не распарсится — она и не нужна: окно и так
    хвостовые строки.
    """
    p = Path(path) if path else None
    if not p or not p.exists():
        return []
    try:
        with open(p, "rb") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            fh.seek(max(0, size - kb * 1024))
            tail = fh.read().decode("utf-8", "replace")
    except OSError:
        return []
    return tail.splitlines()[-count:]


def _cache_get(client: ThinkingClient, kind: str, message: str) -> dict | None:
    ttl, norm = _cache_ttl(client), _cache_norm(message)
    if not norm or ttl <= 0:
        return None
    hit = _ANSWER_CACHE.get(f"{kind}\x00{norm}")
    if hit and time.time() - hit[0] <= ttl and not hit[1].get("fallback"):
        return dict(hit[1], cached=True)
    # перезапуск панели: такой же вопрос мог остаться в журнале диалога
    path = getattr(client, "_chat_path", None)
    if not path:
        return None
    for line in reversed(_tail_lines(path, 60)):
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
    # служебный флаг «ответ без потока» в кэш не кладём: кэшированный
    # повтор — это уже не про обрыв, а про скорость
    clean = {k: v for k, v in reply.items() if k != "stream_fallback"}
    _ANSWER_CACHE[f"{kind}\x00{norm}"] = (time.time(), clean)
    while len(_ANSWER_CACHE) > _CACHE_MAX:
        _ANSWER_CACHE.pop(next(iter(_ANSWER_CACHE)))


def _cache_seen(client: ThinkingClient, question: str, reply: str) -> bool:
    """Такой обмен уже лежит в журнале диалога?

    Смотрится и память процесса, и хвост файла: после рестарта панели
    `chat_log` пуст, и без чтения файла дубль возвращался рядом с
    оригиналом — в «Истории» он появлялся дважды (аудит C-3).
    """
    want_q, want_r = str(question or "")[:2000], str(reply or "")[:4000]
    for rec in reversed(list(client.chat_log[-10:])):
        if rec.get("question") == want_q and rec.get("reply") == want_r:
            return True
    for line in _tail_lines(getattr(client, "_chat_path", None), 40):
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("question") == want_q and rec.get("reply") == want_r:
            return True
    return False


def _cache_journal(client: ThinkingClient, message: str, reply: dict) -> None:
    """Ответ из кэша тоже пишем в журнал диалога.

    Иначе перерисовка истории (она строится из журнала) стирает обмен,
    которого в журнале нет, — человек видит ответ секунду и он исчезает.
    """
    try:
        # тот же обмен уже лежит в журнале — вторая копия рядом с оригиналом
        # попадает в историю дважды и второй раз в хвостовой скан кэша
        # (аудит C-3). Перерисовка, ради которой писался дубль, находит
        # оригинал: он и есть первая запись.
        if _cache_seen(client, message, str(reply.get("reply") or "")):
            return
        client._remember_chat(                              # noqa: SLF001
            message, ChatReply.from_dict(
                {k: v for k, v in reply.items() if k != "cached"}),
            cached=True)
    except Exception as exc:                                # noqa: BLE001
        _out(f"! кэш: не записал в журнал диалога: {exc}")


def _local_name(name: str) -> bool:
    """Хост, который имеет право трогать локальную панель."""
    return str(name or "").strip("[]").lower() in ("127.0.0.1", "localhost", "::1")


def host_allowed(host: str, origin: str = "", port: object = "") -> bool:
    """AUD-17: запрос к панели пришёл от своего адреса?

    Панель отдаёт память, журналы, состояние и умеет менять настройки, при
    этом слушает 127.0.0.1. DNS-подмена или запись в hosts превращает
    «localhost» в чужой хост — тогда панель обслуживает чужого клиента.
    Чужой `Host` → 403. Чужой `Origin` у POST → 403 (страховка от форм
    на чужих страницах: на no-cors-запрос заголовок Origin всё равно идёт).
    """
    host = str(host or "").strip().lower()
    # IPv6 надо разбирать до обычного «хост:порт»: у ::1 двоеточий больше
    # одного, и rpartition(":") даёт имя "::" — localhost считался бы чужим
    if host.startswith("["):
        end = host.find("]")                 # [::1]:8765 или [::1]
        if end < 0:
            return False
        name, rest = host[1:end], host[end + 1:]
        p = rest[1:] if rest.startswith(":") else rest
    elif host.count(":") > 1:
        name, p = host, ""                   # ::1 без скобок и без порта
    elif ":" in host:
        name, _, p = host.rpartition(":")
    else:
        name, p = host, ""
    if not _local_name(name):
        return False
    if p and str(port) and p != str(port):
        return False
    origin = str(origin or "").strip()
    if not origin:
        return True                       # curl/свой скрипт — Origin не шлёт
    if origin.lower() in ("null", "undefined"):
        # Аудит A-1: браузер у POST Origin ВСЕГДА шлёт, поэтому пустой Origin —
        # это curl. А вот литерал "null" — sandboxed-iframe, data:/file:-страница
        # чужого сайта: раньше она считалась «своей», и кросс-запрос доезжал до
        # /api/delete и стирал журналы. Свои страницы шлют http://127.0.0.1:…
        return False
    try:
        u = urllib.parse.urlparse(origin)
    except ValueError:
        return False
    if u.scheme not in ("http", "https") or not _local_name(u.hostname or ""):
        return False
    if u.port and str(port) and str(u.port) != str(port):
        return False
    return True


def memory_action(client: ThinkingClient, data: dict) -> tuple[int, dict]:
    """Действия вкладки «Память» — вынесены из обработчика панели, чтобы
    тесты звали их напрямую как поведение (раньше в ветках export/import
    падал NameError из-за неимпортированного utcnow — аудит AUD-01,
    баг с первого релиза панели)."""
    act = str(data.get("action") or "")
    if act == "export":
        # «Поделиться памятью»: обычный JSON с профилем и фактами
        return 200, {"memory": client.memory(), "exported": utcnow()}
    if act == "import":
        raw = data.get("memory")
        if not isinstance(raw, dict):
            return 400, {"error": "импорт: нужен объект memory"}
        # ввозим только известные поля, с лимитами схемы
        turns = [t for t in (raw.get("turns") or [])
                 if isinstance(t, dict) and str(t.get("text") or "").strip()]
        clean = {"profile": str(raw.get("profile") or "")[:800],
                 "facts": [str(f)[:300] for f in (raw.get("facts") or [])
                           if str(f).strip()][-MAX_MEMORY_FACTS:],
                 "turns": turns[-MAX_MEMORY_TURNS:],
                 "updated": utcnow()}
        client._write_memory(clean)  # noqa: SLF001 — свой же метод
        _out(f"[память] импорт: фактов {len(clean['facts'])}, "
             f"реплик {len(clean['turns'])}")
        return 200, {"memory": client.memory()}
    if act == "forget":
        mem = client.forget(facts=bool(data.get("facts", True)),
                            turns=bool(data.get("turns", True)))
    elif act == "remove" and data.get("fact"):
        mem = client.memory()
        mem["facts"] = [f for f in mem["facts"]
                        if f != str(data["fact"])]
        client._write_memory(mem)  # noqa: SLF001 — свой же метод
        mem = client.memory()
    else:
        mem = client.remember(fact=str(data.get("fact") or ""),
                              profile=str(data.get("profile") or ""))
    return 200, {"memory": mem}


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

    def _sync_target() -> None:
        """Панель подхватывает новый адрес без рестарта.

        set-url меняет config/thinking.local.json; раньше панель держала
        адрес, созданный при старте, и после каждой смены (заглушка ↔ Colab)
        приходилось её перезапускать. Теперь сверка идёт на каждом запросе.
        """
        try:
            if client.reload_config():
                _out(f"[конфиг] новый адрес: {client.base}")
        except Exception:                                       # noqa: BLE001
            pass

    # Первичный канал ПК по ТЗ — опрос GET /events: SSE через туннель может
    # молчать (ping-строки не доходят), поэтому фоновый опрос страхует поток.
    stop_poll = threading.Event()

    def _poll_loop() -> None:
        every = max(2.0, float(client.cfg.get("panel_poll", 5)))
        # Раз в ~25 минут — явный прогрев /health: Colab между рабочими
        # сессиями не засыпает, туннель считается живым.
        warm_every = max(1, int(1500 / every))
        n = 0
        quiet_until = 0.0
        last_note = ""
        while not stop_poll.wait(every):
            n += 1
            if time.time() < quiet_until:
                continue          # молчим: тот же адрес, тот же провал
            try:
                _sync_target()
                client.sync_history(tail=50)
                last_note, quiet_until = "", 0.0
                if n % warm_every == 0:
                    try:
                        ok = client.health(timeout=15)
                    except Exception:                          # noqa: BLE001
                        ok = False
                    _out(f"[прогрев] /health {'ok' if ok else 'не прошёл'}")
            except Exception as exc:                           # noqa: BLE001
                # Живой случай 04.10: туннель умер (502 → 530/1033 → адрес
                # перестал резолвиться), и опрос писал одно и то же каждые
                # 5 с — около 600 строк «предохранитель: пауза ещё N с» за
                # паузу, и настоящий сигнал тонул в шуме. Теперь причина
                # называется один раз, дальше молчим, а на смену адреса
                # реагируем сразу (его перезагрузит reload_config).
                note = str(exc)
                if note == last_note:
                    quiet_until = time.time() + 300
                    continue
                last_note = note
                if any(m in note for m in ("502", "530", "1033", "522",
                                          "524", "getaddrinfo")):
                    # туннель/адрес мёртв: стучаться бессмысленно
                    quiet_until = time.time() + 120
                    _out(f"! опрос событий: туннель недоступен — {note[:160]}")
                    _out("  Нужен новый адрес (Colab перезапущен или ноутбук "
                         "закрыт). Панель продолжит опрос сама.")
                elif "предохранитель" in note:
                    quiet_until = time.time() + 120   # пауза и так длинная
                    _out(f"! опрос событий: {note[:200]}")
                else:
                    _out(f"! опрос событий не прошёл: {note[:200]}")

    threading.Thread(target=_poll_loop, daemon=True).start()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # тишина в консоли
            pass

        def _send(self, code: int, ctype: str, body: bytes) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            # Аудит A-1: панель умеет менять настройки и стирать журналы —
            # не даём встроить её в чужую страницу (клик по прозрачному iframe
            # шлёт уже «свой» Origin, и проверку Origin он обходит силами
            # пользователя). nosniff — чтобы JSON не читался как скрипт.
            self.send_header("X-Frame-Options", "DENY")
            # только frame-ancestors — остальные ресурсы (шрифты, иконки с CDN)
            # этот заголовок не трогает
            self.send_header("Content-Security-Policy", "frame-ancestors 'none'")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
                pass

        def _json_out(self, code: int, data: dict) -> None:
            self._send(code, "application/json; charset=utf-8",
                       json.dumps(data, ensure_ascii=False, default=str).encode("utf-8"))

        def _body(self) -> dict | None:
            """Читает JSON тела запроса. `None` — тело принять нельзя.

            Аудит A-1: раньше и битый JSON, и не-JSON тело молча превращались
            в `{}` — кросс-запрос формой (`enctype="text/plain"`) получал
            «пустой запрос», которого в `/api/delete` хватало на полную
            очистку журнала. Теперь не-JSON — это 400, а не пустой словарь.
            Пустое тело (Content-Length = 0) остаётся `{}`: маршруты сами
            проверяют свои поля.
            """
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except (TypeError, ValueError):
                length = 0
            if length <= 0:
                return {}
            ctype = str(self.headers.get("Content-Type") or "").lower()
            if "application/json" not in ctype:
                # тело всё равно съедаем, иначе соединение рассинхронизируется
                try:
                    self.rfile.read(length)
                except OSError:
                    pass
                return None
            try:
                raw = self.rfile.read(length).decode("utf-8", "replace")
                data = json.loads(raw)
            except (ValueError, OSError):
                return None
            return data if isinstance(data, dict) else None

        def _deny_host(self, with_origin: bool = False) -> bool:
            """AUD-17: отсекаем чужой Host, а у POST — и чужой Origin."""
            port = self.server.server_address[1]
            origin = str(self.headers.get("Origin") or "") if with_origin else ""
            if host_allowed(str(self.headers.get("Host") or ""), origin, port):
                return False
            _out(f"[403] чужой Host/Origin отклонён: "
                 f"{self.headers.get('Host')!r} origin={origin!r}")
            self._json_out(403, {"error": "панель принимает только 127.0.0.1"})
            return True

        def do_POST(self):  # noqa: N802
            _sync_target()
            if self._deny_host(with_origin=True):
                return
            """Действия панели: чат, память, удаление записей."""
            path = self.path.split("?")[0]
            data = self._body()
            if data is None:
                # Аудит A-1: молчаливое превращение не-JSON тела в {}
                # заканчивалось чужим «пустым запросом» вместо ошибки
                self._json_out(400, {"error": "нужен JSON "
                                              "(Content-Type: application/json)"})
                return
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
                                        max_steps=int(data.get("max_steps") or 4),
                                        author="human")   # запрос из панели, не из CLI
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
                            use_memory=bool(data.get("memory", True)),
                            author="human")   # запрос из панели, не из CLI
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
                # одна логика на двоих (аудит B-6): раньше обработчик панели
                # дублировал memory_action строка в строку, и правка лимитов
                # в одной ветке не доходила до второй
                try:
                    code, payload = memory_action(client, data)
                except (SchemaError, ThinkingError) as exc:
                    self._json_out(400, {"error": str(exc)[:200]})
                    return
                self._json_out(code, payload)
                return
            if path == "/api/delete":
                kind = str(data.get("kind") or "report")
                n = data.get("n")
                before = str(data.get("before") or "")
                rid = str(data.get("rid") or "")
                at = str(data.get("at") or "")
                wipe = bool(data.get("wipe"))
                if not rid and not str(n or "").strip() and not before and not wipe:
                    # Аудит A-1: пустое тело раньше читалось как «удали всё»
                    self._json_out(400, {"error": "нужен rid, n+at, before "
                                                  "или wipe=true"})
                    return
                try:
                    removed = client.delete_history(
                        kind=kind,
                        rid=rid,
                        n=int(n) if str(n or "").strip() else None,
                        at=at,
                        before=before,
                        wipe=wipe)
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
                if not tok:
                    # пустой токен не затирает сохранённый (аудит B-8):
                    # поле в панели теперь не подставляется автоматически
                    self._json_out(400, {"error": "пустой токен не сохраняется"})
                    return
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
                            f"[разработка] поток оборвался, повтор {n}/{total}"),
                        author="human")   # запрос из панели, не из CLI
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
                    out = _dev_run(client, str(data.get("name") or ""),
                                   str(data.get("stdin") or ""))
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
            if path == "/api/dev/dump":
                try:
                    target = _dev_dump(client)
                except OSError as exc:
                    self._json_out(500, {"error": str(exc)})
                    return
                self._json_out(200, {"ok": True, "path": str(target)})
                return
            self._send(404, "text/plain; charset=utf-8", b"not found")

        def do_GET(self):  # noqa: N802
            _sync_target()
            if self._deny_host():
                return
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
                # параллельные движки — какие модели отвечают на /ask/multi
                # прямо сейчас (маршрут /parallel есть, вкладка его показывает)
                data["parallel"] = client.parallel_models()
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
                last_ping = 0.0
                try:
                    while True:
                        for ev in client.buffered:
                            seq = int(ev.get("seq") or 0)
                            if seq and seq > seen:
                                seen = seq
                                payload = json.dumps(ev, ensure_ascii=False, default=str)
                                self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                        # ping — раз в 5 с, а не каждый тик: частые
                        # комментарии шумят в логах прокси (аудит AUD-25)
                        now = time.time()
                        if now - last_ping >= 5:
                            self.wfile.write(b": ping\n\n")
                            last_ping = now
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

    # Адрес на один запуск, без записи в конфиг: так локальную заглушку
    # можно прогнать, не трогая настройки настоящего Colab-адреса.
    once = argparse.ArgumentParser(add_help=False)
    once.add_argument("--url", default="",
                      help="адрес субагента только на этот запуск")
    once.add_argument("--token", default="",
                      help="токен только на этот запуск")

    def add(name: str, **kw):
        kw.setdefault("parents", [once])
        return sub.add_parser(name, **kw)

    add("doctor")
    add("url")
    add("metrics")

    s = sub.add_parser("set-url")
    s.add_argument("url", nargs="?")
    s.add_argument("token", nargs="?")
    s.add_argument("--back", action="store_true",
                   help="вернуть предыдущий адрес (заглушка ↔ Colab)")

    s = add("plan")
    s.add_argument("task")
    s.add_argument("--files", default="", help="список путей через запятую")
    s.add_argument("--constraint", action="append")
    s.add_argument("--context", help="дополнительный контекст строкой")
    s.add_argument("--max-steps", type=int, default=12)
    s.add_argument("--json", action="store_true")
    s.add_argument("--stream", action="store_true")

    s = add("ask")
    s.add_argument("task")
    s.add_argument("--files", default="")
    s.add_argument("--constraint", action="append")
    s.add_argument("--context", default=None)
    s.add_argument("--max-steps", type=int, default=5)
    s.add_argument("--json", action="store_true")
    s.add_argument("--stream", action="store_true")

    s = add("ask-multi")
    s.add_argument("task")
    s.add_argument("--json", action="store_true")

    s = add("reflect")
    s.add_argument("plan_id")
    s.add_argument("--step", type=int, required=True)
    s.add_argument("--result", required=True)
    s.add_argument("--observation")
    s.add_argument("--error")
    s.add_argument("--async", dest="async_mode", action="store_true")
    s.add_argument("--background", action="store_true",
                   help="рефлексия выполнена отсоединённым процессом "
                        "(ставится командой --async автоматически)")

    s = add("tail")
    s.add_argument("--n", type=int, default=30)
    s.add_argument("--follow", action="store_true")

    s = add("panel")
    s.add_argument("--port", type=int, default=0)
    s.add_argument("--open", action="store_true",
                   help="открыть панель в браузере сразу после старта")

    s = add("dump", help="скачать логи Colab в локальную папку")
    s.add_argument("name", nargs="?", default="llm.log",
                   help=f"файл из списка: {', '.join(DUMP_FILES)}")
    s.add_argument("--all", action="store_true", help="снять весь белый список")
    s.add_argument("--out", default="", help="куда класть (по умолчанию colab_downloads/)")
    s.add_argument("--timeout", type=float, default=30.0)

    add("models", help="показать модели в Colab и активную из них")
    s = add("use-model", help="переключить модель в Colab")
    s.add_argument("model", help="короткое имя (3b, 1.5B) или путь к .gguf")

    s = add("reflect-metrics",
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
    # --url/--token: адрес на один запуск, файл конфигурации не трогаем
    if args.cmd != "set-url" and getattr(args, "url", ""):
        client.base = str(args.url).rstrip("/")
        client.base_locked = True
        if getattr(args, "token", ""):
            client.token = str(args.token)
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
        if args.cmd == "ask-multi":
            return cmd_ask_multi(client, args)
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
