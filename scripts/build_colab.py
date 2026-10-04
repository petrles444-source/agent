#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Быстрый сбор Colab для субагента «Мышление».

Что делает одной командой:
  1. соберает zip с исходниками ячеек (thinking/colab/cell_*.py);
  2. заливает его на tmpfiles и получает прямую ссылку (~60 минут жизни);
  3. генерирует colab/thinking_colab.ipynb с уже вшитой ссылкой;
  4. печатает чек-лист запуска в Colab.

Запуск:
    python scripts/build_colab.py                # zip + загрузка + ноутбук
    python scripts/build_colab.py --no-upload    # только zip (без интернета)
    python scripts/build_colab.py --url URL      # ноутбук под готовую ссылку
"""
from __future__ import annotations

import argparse
import io
import json
import secrets
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CELLS_DIR = ROOT / "thinking" / "colab"
ZIP_PATH = ROOT / "_thinking_colab.zip"
IPYNB_PATH = ROOT / "colab" / "thinking_colab.ipynb"
# Личный ноутбук: в нём токен зафиксирован, поэтому в git он не попадает
# (иначе публичный репозиторий раздавал бы доступ к модели).
PERSONAL_NB_NAME = "thinking_agent_personal.ipynb"
LOCAL_CFG = ROOT / "config" / "thinking.local.json"

CELL_FILES = [
    "cell_a_setup.py",
    "cell_c_server.py",
    "cell_d_launch.py",
    "cell_e_background.py",
    "cell_f_stop.py",
]

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
TMPFILES_UPLOAD = "https://tmpfiles.org/api/v1/upload"


def say(msg: str) -> None:
    # Консоль Windows может быть cp866/cp1251: не падаем на «→» и кириллице
    try:
        sys.stdout.reconfigure(errors="replace")       # noqa: PYI
    except Exception:
        pass
    print(msg, flush=True)


# --------------------------------------------------------------------------- #
def build_zip() -> bytes:
    missing = [n for n in CELL_FILES if not (CELLS_DIR / n).exists()]
    if missing:
        raise SystemExit(f"нет файлов ячеек: {missing} (ищи в {CELLS_DIR})")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in CELL_FILES:
            zf.write(CELLS_DIR / name, arcname=name)
    data = buf.getvalue()
    ZIP_PATH.write_bytes(data)
    say(f"  zip   : {ZIP_PATH} ({len(data)} байт, файлов {len(CELL_FILES)})")
    return data


def _multipart(fields: dict, filename: str, payload: bytes) -> tuple[bytes, str]:
    boundary = f"----thinking{int(time.time())}"
    out = io.BytesIO()
    for key, val in fields.items():
        out.write(f"--{boundary}\r\nContent-Disposition: form-data; "
                  f"name=\"{key}\"\r\n\r\n{val}\r\n".encode())
    out.write(f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
              f"filename=\"{filename}\"\r\n"
              f"Content-Type: application/zip\r\n\r\n".encode())
    out.write(payload)
    out.write(f"\r\n--{boundary}--\r\n".encode())
    return out.getvalue(), boundary


def upload(data: bytes) -> str:
    """Заливает zip на tmpfiles и возвращает ПРЯМУЮ ссылку на скачивание."""
    body, boundary = _multipart({}, ZIP_PATH.name, data)
    req = urllib.request.Request(
        TMPFILES_UPLOAD, data=body, method="POST",
        headers={"User-Agent": UA, "Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    if payload.get("status") != "success":
        raise SystemExit(f"загрузка не удалась: {payload}")
    page = payload["data"]["url"]                      # https://tmpfiles.org/NNN/file.zip
    direct = page.replace("tmpfiles.org/", "tmpfiles.org/dl/", 1)
    say(f"  ссылка: {direct}")
    say("          (живёт ~60 минут — выполняйте ячейку 1 ноутбука сразу)")
    return direct


# --------------------------------------------------------------------------- #
def _read_cell(name: str) -> str:
    """Исходник ячейки без магической первой строки (%%writefile / %%bash)."""
    text = (CELLS_DIR / name).read_text(encoding="utf-8")
    lines = text.splitlines()
    if lines and lines[0].startswith("%%"):
        lines = lines[1:]
    return "\n".join(lines)


def _nb_cell(kind: str, source: str) -> dict:
    cell = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        cell.update({"outputs": [], "execution_count": None})
    else:
        cell["metadata"] = {}
    return cell


INTRO = """# Субагент «Мышление» — автозапуск{ver}

Запускайте ячейки **по порядку** (или `Runtime → Run all`):

| Ячейка | Что делает | Ожидаемый итог |
|---|---|---|
| {row1} |
| 2 (A) | зависимости, сборка llama-cpp, все модели Qwen2.5 (1.5B/3B/7B и те же три без цензуры) | `OK-A`, подсказки режимов CPU/T4 |
| 3 (C) | собирает API-сервер | `записан … строк: N` |
| 4 (D) | LLM + API + публичный туннель + сторож | `смоук: X.X с`, `THINKING_URL=`, `THINKING_TOKEN=` |
| 5 (E) | фон: снапшот событий + keep-alive 12 ч | `фон запущен: …` |
| 6 (F) | остановка (по необходимости) | `субагент остановлен` |
| 7 | **печатает адрес и готовую команду для ПК** | `АДРЕС ТУННЕЛЯ: https://…` |

После ячейки 4 (или 7 — там всё готово копированием) выполните на ПК:

```bash
python tools/thinking_cli.py set-url <THINKING_URL> <THINKING_TOKEN>
python tools/thinking_cli.py doctor
python tools/thinking_cli.py panel     # → http://127.0.0.1:8765
```

**Чтобы адрес не менялся изо дня в день**, задайте в ячейке D переменную
`CLOUDFLARE_TUNNEL_TOKEN` (Cloudflare Dashboard → Zero Trust → Tunnels →
Create tunnel → token): адрес станет постоянным, и `set-url` больше не нужен.

**Лимиты бесплатного Colab** (ориентиры, чётких квот нет): GPU-сессии обычно
2–4 часа, CPU-остаток бывает и ~28 часов — но это не гарантия. Если работа
встала — Colab: Сессия → Лимиты; лимиты часто возвращаются примерно через
сутки (или смените Google-аккаунт). Сначала работайте на GPU, потом переходите
на CPU (в панели это видно и подсказывается).

---

## ⚡ Как поднять (пошагово)

**Открывайте в своём обычном браузере** — в том, где вы уже залогинены в
Google. В браузере код-агента вход в Google-аккаунт блокируется («Couldn’t
sign you in: браузер не поддерживает JavaScript») — это защита Google от
автоматизированных браузеров. Пароль туда вводить не нужно и не нужно.

1. Откройте <https://colab.research.google.com> и загрузите
   `colab/thinking_agent_ver3.ipynb` (**Upload notebook**).
2. **Runtime → Run all** (ячейки A → C → D).
3. В **ячейке 2 (A)** по умолчанию стоит `PROFILE = "dev"` — рекомендуемый
   профиль: качает **3B + 1.5B + 7B (≈6,7 ГБ)**, активна 3B, она и отвечает
   живьём на бесплатном Colab (≈2,5 ток/с). 7B в цепочку отката не входит и
   нужна только для переключения во вкладке «Модели» (на T4 она в разы
   быстрее). Ячейка A печатает, что именно качает и сколько мегабайт.
   Другие профили — на свой страх и риск: `strong` (14B) даёт 0,05 ток/с,
   `big` (30B, 17 ГБ) на бесплатном Colab вообще не поднялась.
4. В конце выполните **ячейку 7/7** — она напечатает готовый адрес туннеля
   и готовую команду для ПК. Искать адрес в выводе ячейки D больше не нужно.

**Как передать результат агенту**

Скопируйте из вывода ячейки 7/7 строку `АДРЕС ТУННЕЛЯ` и токен и вставьте
в чат — агент настроит ПК и прогонит полный тест субагента.

Либо вставьте вывод ячейки D (там строки `THINKING_URL=` и
`THINKING_TOKEN=`) — этого достаточно.
"""

SHOW_URL = """#@title 7/7 · Как открыть веб-панель (адрес туннеля + команда для ПК)
import pathlib, re
f = pathlib.Path("/content/thinking_url.txt")
data = f.read_text(encoding="utf-8") if f.exists() else ""
url = (re.search(r"THINKING_URL=(\\S+)", data) or [None, ""])[1]
token = (re.search(r"THINKING_TOKEN=(\\S+)", data) or [None, ""])[1]
err = (re.search(r"THINKING_TUNNEL_ERROR=(.*)", data) or [None, ""])[1]
print("=" * 66)
print("  АДРЕС ТУННЕЛЯ:", url or "НЕ ПОДНЯЛСЯ")
print("=" * 66)
print()
if url:
    print("ШАГ 1. На ПК скопируй и выполни одну команду:")
    print()
    print(f'    python tools/thinking_cli.py set-url "{url}" "{token}"')
    print()
    print("ШАГ 2. Открой панель (в этом же окне PowerShell):")
    print()
    print("    python tools/thinking_cli.py panel")
    print()
    print("    -> откроется http://127.0.0.1:8765  (панель на 9 вкладок)")
    print()
    print("ШАГ 3. Проверь, что субагент отвечает:")
    print()
    print("    python tools/thinking_cli.py doctor")
    print()
    print("ШАГ 4. Если субагентом пользуется ещё и код-агент — просто вставь")
    print("       ему строку «АДРЕС ТУННЕЛЯ» и токен выше из этого вывода:")
    print("       он выполнит set-url и прогонит полный тест (план, рефлексия,")
    print("       отчёт, память). Отдельно ничего настраивать не нужно.")
else:
    print("ТУННЕЛЬ НЕ ПОДНЯЛСЯ — адреса для ПК нет.")
    print()
    print("ПРИЧИНА:", err or "не записана — выполни ячейку 4/7 (D) заново")
    print()
    print("Последние строки /content/tunnel.log:")
    try:
        for line in pathlib.Path("/content/tunnel.log").read_text(
                encoding="utf-8", errors="replace").splitlines()[-25:]:
            print("   |", line)
    except Exception as exc:
        print("   | лог недоступен:", exc)
    print()
    print("Что делать:")
    print("  1) подождать 5-15 минут — Cloudflare режет частые quick-туннели")
    print("     с одного адреса рантайма;")
    print("  2) выполнить ячейку 4/7 (D) заново: теперь она делает три попытки")
    print("     (http2 и quic) сама и печатает причину, если не вышло;")
    print("  3) если адрес нужен надёжно — задайте CLOUDFLARE_TUNNEL_TOKEN,")
    print("     тогда адрес будет постоянным и не зависит от лимитов.")
"""

FINISH = """---

## Где взять адрес (чтобы не искать его каждый раз)

Выполните **ячейку 7/7** в самом конце ноутбука — она напечатает всё готовое
копированием:

```
АДРЕС ТУННЕЛЯ: https://xxxx-xxxx.trycloudflare.com
```

**Веб-панель на ПК** (самая частая задача):

```bash
python tools/thinking_cli.py panel
```

→ откроется **http://127.0.0.1:8765** — панель на 9 вкладок (Отчёты,
Чат с субагентом, Разработка, Модели, Диалог, Токены, Связь агентов,
Выгода и ускорение, Журнал). Токен подставится
сам из `config/thinking.local.json`.

**Связь не работает / адрес не подходит:**

```bash
python tools/thinking_cli.py set-url <АДРЕС> <ТОКЕН>   # адрес и токен — из ячейки 7/7
python tools/thinking_cli.py doctor                     # что сломано и где смотреть
python tools/thinking_cli.py panel                     # открыть панель
```

`doctor` печатает причину по-человечески: не отвечает Colab, отвечает, но
молчит, или токен не тот.

**Панель молчит при открытии** — это нормально: интерфейс и история лежат
на ПК. Colab нужен только чтобы отвечать на новые запросы.
"""


def make_notebook(url: str, inline: bool = True, ver: str = "",
                 token: str = "") -> dict:
    """Самодостаточный ноутбук (код внутри, ссылки не нужны) либо zip-вариант."""
    row1 = ("| — | этот ноутбук самодостаточен: весь код внутри | — |"
            if inline else
            f"| 1 | скачивает исходники в `/content/tc` | `OK: исходники на месте` |")
    intro = INTRO.format(row1=row1, ver=(f" ({ver})" if ver else ""))

    cells = [_nb_cell("markdown", intro)]
    if inline:
        # %%writefile обязан быть первой строкой ячейки, поэтому папки создаёт
        # первая ячейка (os.makedirs) — иначе запись падает с FileNotFoundError.
        write = lambda path, name: _nb_cell(
            "code", f"%%writefile {path}\n{_read_cell(name)}")
        cells += [
            _nb_cell("code",
                     "#@title 1/7 · Исходники внутри ноутбука — загрузка не нужна\n"
                     "import os\n"
                     "os.makedirs('/content/tc', exist_ok=True)\n"
                     "print('ноутбук самодостаточен: каждая ячейка содержит свой код')"),
            write("/content/tc/cell_a_setup.py", "cell_a_setup.py"),
            write("/content/thinking_server.py", "cell_c_server.py"),
            write("/content/tc/cell_d_launch.py", "cell_d_launch.py"),
            write("/content/tc/cell_e_background.py", "cell_e_background.py"),
            write("/content/tc/cell_f_stop.py", "cell_f_stop.py"),
            _nb_cell("code", "#@title 2/7 · Ячейка A — окружение, сборка, модель\n"
                             "import os, runpy\n"
                             "# ── Что скачивать (поменяй строку и перезапусти) ──\n"
                             "# dev        — РЕКОМЕНДУЕТСЯ. Качает 3B + 1.5B + 7B\n"
                             "#               (��6,7 ГБ), активна 3B — на CPU ≈2,5 ток/с.\n"
                             "#               7B в цепочку отката НЕ входит: лежит только\n"
                             "#               для переключения во вкладке «Модели».\n"
                             "# light      — только 3B + 1.5B (~3 ГБ), ничего лишнего\n"
                             "# gpu        — 7B + 3B + 1.5B (~7 ГБ), для T4\n"
                             "# strong     — Qwen2.5-14B + Coder-14B (17 ГБ): на бесплатном\n"
                             "#               Colab 0,05 ток/с (только подкачка), нужен GPU\n"
                             "# big        — Qwen3-30B-A3B (17 ГБ): на бесплатном Colab\n"
                             "#               не поднялась — RAM 12,7 ГБ не хватает\n"
                             "# coder      — Qwen2.5-Coder-32B (18 ГБ), лучший для кода\n"
                             "# uncensored — сборки без цензуры + обычная 3B\n"
                             "# all        — всё сразу (~60 ГБ, НЕ советуем)\n"
                             "PROFILE = \"dev\"\n"
                             "os.environ[\"THINKING_PROFILE\"] = PROFILE\n"
                             "runpy.run_path('/content/tc/cell_a_setup.py', run_name='__main__')"),
            _nb_cell("code", "#@title 3/7 · Ячейка C — сборка API-сервера\n"
                             "import pathlib\n"
                             "src = pathlib.Path('/content/thinking_server.py')\n"
                             "print('записан /content/thinking_server.py:', len(src.read_text(encoding='utf-8').splitlines()), 'строк')"),
            _nb_cell("code", "#@title 4/7 · Ячейка D — LLM + API + туннель + сторож\n"
                             "import os, runpy\n"
                             "os.environ.setdefault('THINKING_TOKEN', "
                             + (repr(token) if token else "''")
                             + ")   # токен зафиксирован: адрес меняется, доступ — нет\n"
                             "runpy.run_path('/content/tc/cell_d_launch.py', run_name='__main__')"),
            _nb_cell("code", "#@title 5/7 · Ячейка E — фон (снапшот + keep-alive)\n"
                             "import runpy; runpy.run_path('/content/tc/cell_e_background.py', run_name='__main__')"),
            _nb_cell("code", "#@title 6/7 · Ячейка F — остановка (по необходимости)\n"
                             "import runpy; runpy.run_path('/content/tc/cell_f_stop.py', run_name='__main__')"),
            _nb_cell("code", SHOW_URL),
            _nb_cell("markdown", FINISH),
        ]
        return _wrap(cells)

    fetch = f'''#@title 1/7 · Скачивание исходников субагента
import io, pathlib, shutil, urllib.request, zipfile
URL = "{url}"  #@param {{type:"string"}}
req = urllib.request.Request(URL, headers={{"User-Agent": "{UA}"}})
data = urllib.request.urlopen(req, timeout=180).read()
assert data[:2] == b"PK", f"ожидался zip, получено: {{data[:60]!r}}"
shutil.rmtree("/content/tc", ignore_errors=True)
zipfile.ZipFile(io.BytesIO(data)).extractall("/content/tc")
print("OK: исходники на месте:", sorted(p.name for p in pathlib.Path("/content/tc").iterdir()))
'''
    build_server = '''#@title 3/7 · Ячейка C — сборка API-сервера
import pathlib
src = pathlib.Path("/content/tc/cell_c_server.py").read_text(encoding="utf-8")
lines = src.splitlines()
if lines and lines[0].startswith("%%"):        # %%writefile — магия ячейки, не Python
    lines = lines[1:]
code = "\\n".join(lines)
pathlib.Path("/content/thinking_server.py").write_text(code, encoding="utf-8")
print("записан /content/thinking_server.py:", len(lines), "строк")
'''
    run = lambda title, name: f'#@title {title}\n!python /content/tc/{name}\n'
    cells += [
        _nb_cell("code", fetch),
        _nb_cell("code", run("2/7 · Ячейка A — окружение, сборка, модель", "cell_a_setup.py")),
        _nb_cell("code", build_server),
        _nb_cell("code", run("4/7 · Ячейка D — LLM + API + туннель", "cell_d_launch.py")),
        _nb_cell("code", run("5/7 · Ячейка E — фон (снапшот + keep-alive)", "cell_e_background.py")),
        _nb_cell("code", run("6/7 · Ячейка F — остановка (по необходимости)", "cell_f_stop.py")),
        _nb_cell("code", SHOW_URL),
        _nb_cell("markdown", FINISH),
    ]
    return _wrap(cells)


def _wrap(cells: list[dict], name: str = "thinking_colab.ipynb") -> dict:
    return {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {
            "colab": {"provenance": [], "name": name},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
        },
        "cells": cells,
    }


def write_notebook(url: str, inline: bool = True, name: str = "",
                   ver: str = "", token: str = "") -> Path:
    nb_name = name or (IPYNB_PATH.name if inline
                       else "thinking_colab_zip.ipynb")
    target = IPYNB_PATH.with_name(nb_name)
    target.parent.mkdir(parents=True, exist_ok=True)
    nb = make_notebook(url, inline=inline, ver=ver, token=token)
    nb = _retitle(nb, nb_name)
    target.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    say(f"  ноутбук: {target}")
    return target


def _personal_token() -> str:
    """Токен для личного ноутбука: берём из локального конфига, иначе создаём.

    Токен живёт только в config/thinking.local.json (в git его нет) и
    вшивается в ЛИЧНЫЙ ноутбук, который тоже не коммитится. Из-за этого
    перезапуск Colab больше не ломает доступ: адрес меняется, токен — нет,
    и на ПК не нужно ничего вводить руками.
    """
    cfg = json.loads(LOCAL_CFG.read_text(encoding="utf-8")) \
        if LOCAL_CFG.exists() else {}
    token = str(cfg.get("token") or "").strip()
    if not token:
        token = secrets.token_urlsafe(24)
        cfg["token"] = token
        LOCAL_CFG.parent.mkdir(parents=True, exist_ok=True)
        LOCAL_CFG.write_text(json.dumps(cfg, ensure_ascii=False, indent=2),
                             encoding="utf-8")
        say(f"  создан новый токен в {LOCAL_CFG.name} (в git его нет)")
    return token


def _retitle(nb: dict, name: str) -> dict:
    """Имя ноутбука в metadata.colab.name должно совпадать с файлом —
    иначе Colab в Drive перепутывает копии."""
    nb["metadata"]["colab"]["name"] = name
    return nb


# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Сборка Colab для субагента «Мышление»")
    ap.add_argument("--no-upload", action="store_true", help="не заливать zip в сеть")
    ap.add_argument("--url", default="", help="готовая прямая ссылка на zip")
    ap.add_argument("--zip", action="store_true",
                    help="старая схема: ноутбук качает zip по ссылке (живёт ~60 мин)")
    ap.add_argument("--name", default="",
                    help="имя выходного ноутбука (по умолчанию thinking_colab.ipynb)")
    ap.add_argument("--ver", default="",
                    help="подпись версии в заголовке, например ver3")
    ap.add_argument("--personal", action="store_true",
                    help="личный ноутбук с ЗАФИКСИРОВАННЫМ токеном: адрес "
                         "туннеля меняется при каждом запуске, а доступ — нет, "
                         "и на ПК больше не нужно вводить токен руками. "
                         "Такой ноутбук НЕ коммитится (в нём токен)")
    args = ap.parse_args(argv)
    inline = not args.zip
    name, token = args.name, ""
    if args.personal:
        token = _personal_token()
        name = args.name or PERSONAL_NB_NAME

    say("[1/3] собираю zip с исходниками ячеек (уходит в архив проекта)…")
    data = build_zip()

    url = args.url
    if inline:
        say("[2/3] загрузка не нужна: ноутбук самодостаточный, весь код внутри")
    elif not url and not args.no_upload:
        say("[2/3] заливаю на tmpfiles…")
        try:
            url = upload(data)
        except Exception as exc:                       # noqa: BLE001
            say(f"  ! загрузка не удалась ({exc}); делаю ноутбук без ссылки")
    elif args.no_upload:
        say("[2/3] загрузка пропущена (--no-upload)")

    say("[3/3] генерирую ноутбук…")
    nb_path = write_notebook(url or "ВСТАВЬТЕ_СЮДА_ССЫЛКУ_НА_ZIP", inline=inline,
                             name=name, ver=args.ver, token=token)

    say("")
    say("Чек-лист:")
    say("  1) открыть https://colab.research.google.com → Upload notebook")
    say(f"     → выбрать {nb_path.name}")
    say("  2) Runtime → Change runtime type → T4 GPU (можно и CPU)")
    say("  3) Runtime → Run all")
    if token:
        say("  4) из вывода ячейки 7/7 скопировать ТОЛЬКО адрес "
            "(строка «АДРЕС ТУННЕЛЯ»)")
        say("  5) на ПК выполнить одну команду с этим адресом — токен уже "
            "зафиксирован в ноутбуке и в конфиге ПК:")
        say("             python tools/thinking_cli.py set-url <АДРЕС>")
        say("             python tools/thinking_cli.py doctor")
        say("  ВНИМАНИЕ: этот ноутбук содержит токен — не выкладывайте его "
            "в открытый репозиторий.")
    else:
        say("  4) из вывода ячейки 4 скопировать THINKING_URL / THINKING_TOKEN")
        say("  5) на ПК:  python tools/thinking_cli.py set-url <URL> <TOKEN>")
        say("             python tools/thinking_cli.py doctor")
    if url and not inline:
        say(f"  Ссылка на zip действует ~60 минут: {url}")
    if inline:
        say("")
        say("  Ноутбук самодостаточный: ссылка на zip не нужна и не истекает —")
        say("  сохраните его в Google Drive и открывайте в любой день.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
