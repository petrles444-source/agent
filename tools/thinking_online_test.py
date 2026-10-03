"""Онлайн-проверки подсистемы «Мышление» — против ЖИВОГО сервера Colab.

В отличие от tools/thinking_test.py (офлайн: контракты, фолбэк, панель),
здесь всё ходит по реальной сети: связь, токен, метрики, события, ответ
настоящей LLM (план) и рефлексия шага. Секреты берутся из
config/thinking.local.json (или THINKING_URL / THINKING_TOKEN), в код не
вписываются.

Запуск:
    python tools/thinking_online_test.py            # полный прогон (LLM: план + рефлексия)
    python tools/thinking_online_test.py --quick    # без LLM: связь, токен, метрики, события
    python tools/thinking_online_test.py --stream   # полный + отдельная проверка /plan/stream

Коды выхода:
    0   все проверки прошли
    1   есть проваленные проверки
    2   URL не задан или сервер недоступен — проверять нечего
    130 прервано пользователем
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from thinking.client import ThinkingClient, ThinkingError            # noqa: E402
from thinking.schemas import (ACTIONS, EVENT_TYPES, Plan,            # noqa: E402
                              ReflectResponse, SchemaError)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")       # Windows cp1251

PASSED = FAILED = 0
FAILS: list[str] = []


def check(cond: bool, name: str) -> bool:
    global PASSED, FAILED
    if cond:
        PASSED += 1
    else:
        FAILED += 1
        FAILS.append(name)
        print(f"  ! {name}")
    return bool(cond)


def has_rus(text: object) -> bool:
    return any(ord(c) > 0x400 for c in str(text or ""))


def section(title: str) -> None:
    print(f"\n== {title} ==")


def count_lines(path: object) -> int:
    try:
        return sum(1 for _ in open(path, encoding="utf-8"))
    except OSError:
        return 0


# --------------------------------------------------------------------------- #
#  Быстрые проверки (без LLM)
# --------------------------------------------------------------------------- #
def probe(base: str, token: str, path: str = "/health") -> object:
    """Один сырой GET. Возвращает код HTTP, либо 'ERR:…' при сетевой ошибке."""
    req = urllib.request.Request(f"{base}{path}",
                                 headers={"X-Agent-Token": token or ""},
                                 method="GET")
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            return resp.status
    except urllib.error.HTTPError as exc:
        return exc.code
    except Exception as exc:                                    # noqa: BLE001
        return f"ERR:{type(exc).__name__}"


def test_connection(client: ThinkingClient) -> bool:
    section("Связь")
    t0 = time.time()
    ok = client.health(timeout=8)
    ms = int((time.time() - t0) * 1000)
    check(ok, f"GET /health отвечает ({ms} мс, ошибка: {client.last_error or '-'})")
    if not ok:
        return False
    h = client.health_data or {}
    check(h.get("status") == "ok", "health: status == ok")
    check(bool(h.get("model")), "health: модель названа")
    check(str(h.get("gpu", "")).lower() != "" or "vram_used_gb" in h,
          "health: поля gpu/vram присутствуют")
    check(isinstance(h.get("uptime_s"), int) and h["uptime_s"] > 0,
          "health: uptime > 0")
    check(ms < 3000, f"бюджет health < 3 с (факт {ms} мс)")
    print(f"  · модель={h.get('model')} gpu={h.get('gpu')} "
          f"uptime={h.get('uptime_s')} с, ответ за {ms} мс")
    return True


def test_auth(client: ThinkingClient) -> None:
    section("Аутентификация")
    check(bool(client.token), "токен задан (set-url / THINKING_TOKEN)")
    wrong = probe(client.base, "wrong-" + "token-123")
    check(wrong == 401, f"неверный токен отклоняется (получено {wrong})")
    empty = probe(client.base, "")
    check(empty == 401, f"запрос без токена отклоняется (получено {empty})")
    good = probe(client.base, client.token)
    check(good == 200, f"верный токен принят (получено {good})")


def test_metrics(client: ThinkingClient) -> None:
    section("Метрики сервера")
    try:
        m = client.metrics()
    except Exception as exc:                                    # noqa: BLE001
        check(False, f"GET /metrics отвечает: {exc}")
        return
    check(isinstance(m, dict) and bool(m), "GET /metrics вернул объект")
    for key in ("events", "plans_total", "reflects", "errors", "bad_json"):
        check(key in m and isinstance(m[key], int), f"метрика {key} есть и целочисленна")
    check(m.get("events", 0) > 0, "счётчик событий > 0")
    print(f"  · plans={m.get('plans_total')} reflects={m.get('reflects')} "
          f"errors={m.get('errors')} bad_json={m.get('bad_json')} "
          f"llm_ms_avg={m.get('llm_ms_avg')}")


def test_events(client: ThinkingClient) -> None:
    section("События (журнал мыслей)")
    try:
        data = client.events(tail=25)
    except Exception as exc:                                    # noqa: BLE001
        check(False, f"GET /events отвечает: {exc}")
        return
    check(isinstance(data.get("last_seq"), int), "last_seq целочисленный")
    events = data.get("events")
    check(isinstance(events, list) and events, "события возвращаются списком")
    if not isinstance(events, list) or not events:
        return
    bad_types = [e.get("type") for e in events
                 if not isinstance(e, dict) or e.get("type") not in EVENT_TYPES]
    check(not bad_types, f"все типы событий из схемы (лишние: {bad_types[:3]})")
    seqs = [e.get("seq") for e in events if isinstance(e.get("seq"), int)]
    check(seqs == sorted(seqs) and len(seqs) == len(events),
          "seq событий монотонно возрастает")
    check(all(e.get("ts") for e in events), "у каждого события есть ts")
    tail = events[-1]
    print(f"  · last_seq={data.get('last_seq')}, хвост: [{tail.get('type')}] "
          f"{str(tail.get('text'))[:70]}")


# --------------------------------------------------------------------------- #
#  LLM: настоящий план и рефлексия
# --------------------------------------------------------------------------- #
PLAN_TASK = ("Онлайн-тесты подсистемы «Мышление»: проверить по живому серверу "
             "связь, токен, план LLM и рефлексию")
PLAN_FILES = ["tools/thinking_online_test.py", "tools/thinking_test.py",
              "thinking/client.py", "thinking/schemas.py"]


def test_plan(client: ThinkingClient) -> tuple[dict | None, dict]:
    """Реальный POST /plan. Возвращает (сырой ответ, счётчики истории до)."""
    section("План от настоящей LLM (POST /plan)")
    before = {
        "interactions": count_lines(client.cfg.get("interactions_path")),
        "reports": count_lines(client.cfg.get("reports_path")),
    }
    t0 = time.time()
    try:
        out = client.plan(PLAN_TASK,
                          context={"files": PLAN_FILES, "cwd": str(ROOT)},
                          constraints=["только стандартная библиотека на ПК",
                                       "не менять контракты thinking/schemas.py"],
                          max_steps=5)
    except Exception as exc:                                    # noqa: BLE001
        check(False, f"план получен без ошибки: {exc}")
        return None, before
    ms = int((time.time() - t0) * 1000)
    print(f"  · ответ за {ms} мс ({ms / 1000:.1f} с)")

    check(isinstance(out, dict), "ответ — JSON-объект")
    if not isinstance(out, dict):
        return None, before
    check(out.get("source") == "colab",
          f"план от настоящей LLM, а не заглушки (source={out.get('source')})")
    check(bool(out.get("plan_id")), "plan_id присутствует")

    raw_steps = out.get("steps")
    check(isinstance(raw_steps, list) and bool(raw_steps), "шаги непустым списком")
    if not isinstance(raw_steps, list) or not raw_steps:
        return out, before
    check(len(raw_steps) <= 5, f"не больше 5 запрошенных шагов (факт {len(raw_steps)})")

    ids = [s.get("id") for s in raw_steps if isinstance(s, dict)]
    check(len(ids) == len(set(ids)) and all(isinstance(i, int) for i in ids),
          "id шагов — целые и уникальные (сырая проверка)")
    bad_action = [s.get("action") for s in raw_steps
                  if isinstance(s, dict) and s.get("action") not in ACTIONS]
    check(not bad_action, f"action каждого шага из ALLOWED (лишние: {bad_action})")
    check(all(isinstance(s, dict) and str(s.get("desc", "")).strip()
              for s in raw_steps), "у каждого шага есть desc")
    check(all(has_rus(s.get("desc")) for s in raw_steps if isinstance(s, dict)),
          "desc шагов по-русски")
    check(all(len(str(s.get("desc", ""))) <= 400 for s in raw_steps if isinstance(s, dict)),
          "desc укладывается в 400 символов")
    check(all("id" in s and "action" in s for s in raw_steps if isinstance(s, dict)),
          "шаги содержат id и action")

    goal = out.get("goal")
    check(isinstance(goal, str) and bool(goal.strip()), "goal непустой")
    check(has_rus(goal), "goal по-русски")
    rationale = str(out.get("rationale") or "")
    check(not rationale or has_rus(rationale), "rationale (если есть) по-русски")
    try:
        conf = float(out.get("confidence"))
        check(0.0 <= conf <= 1.0, "confidence в пределах 0..1")
    except (TypeError, ValueError):
        check(False, f"confidence — число (получено {out.get('confidence')!r})")
    for key in ("sub_goals", "constraints", "contradictions",
                "success_criteria", "unknown_files"):
        check(isinstance(out.get(key, []), list), f"{key} — список")

    try:
        Plan.from_dict(out)
        check(True, "контракт Plan.from_dict проходит")
    except SchemaError as exc:
        check(False, f"контракт Plan.from_dict проходит: {exc}")

    criteria = out.get("success_criteria") or []
    print(f"  · шагов: {len(raw_steps)}, критериев успеха: {len(criteria)}, "
          f"противоречий: {len(out.get('contradictions') or [])}")
    print(f"  · цель: {str(goal)[:100]}")
    for s in raw_steps:
        print(f"      {s.get('id')}. [{s.get('action')}] {str(s.get('desc'))[:90]}")
    return out, before


def test_reflect(client: ThinkingClient, plan: dict) -> None:
    section("Рефлексия шага (POST /reflect)")
    plan_id = str(plan.get("plan_id") or "")
    try:
        out = client.reflect(plan_id, 1,
                              result="Онлайн-тесты написаны и запущены, связь с Colab жива",
                              observation="health 200, токен 401/200 сходится")
    except Exception as exc:                                    # noqa: BLE001
        check(False, f"рефлексия получена без ошибки: {exc}")
        return
    check(isinstance(out, dict), "ответ рефлексии — JSON-объект")
    if not isinstance(out, dict):
        return
    check(out.get("status") in ("ok", "adjust", "abort"),
          f"status из схемы (получено {out.get('status')!r})")
    advice = str(out.get("advice") or "")
    check(bool(advice.strip()), "advice непустой")
    check(has_rus(advice), "advice по-русски")
    check(isinstance(out.get("next_steps", []), list), "next_steps — список")
    try:
        ReflectResponse.from_dict(out)
        check(True, "контракт ReflectResponse.from_dict проходит")
    except SchemaError as exc:
        check(False, f"контракт ReflectResponse.from_dict проходит: {exc}")
    print(f"  · [{out.get('status')}] {advice[:110]}")


def test_history_grew(before: dict, client: ThinkingClient) -> None:
    section("История на диске")
    inter = count_lines(client.cfg.get("interactions_path"))
    rep = count_lines(client.cfg.get("reports_path"))
    check(inter > before.get("interactions", 0),
          f"interactions.jsonl пополнился ({before.get('interactions')} → {inter})")
    check(rep > before.get("reports", 0),
          f"reports.jsonl пополнился ({before.get('reports')} → {rep})")
    check(count_lines(client.cfg.get("log_path")) > 0,
          "thoughts.jsonl непуст — панель покажет поток")


def test_stream(client: ThinkingClient) -> None:
    section("Стрим плана (POST /plan/stream)")
    tokens: list[str] = []
    t0 = time.time()
    try:
        out = client.plan_stream(
            "Короткий стрим-тест: план из 3 шагов для проверки потока токенов",
            context={"files": ["tools/thinking_online_test.py"]},
            max_steps=3,
            on_event=lambda ev: tokens.append(ev.get("type", "")))
    except Exception as exc:                                    # noqa: BLE001
        check(False, f"стрим вернул план: {exc}")
        return
    ms = int((time.time() - t0) * 1000)
    check(isinstance(out, dict) and bool(out.get("steps")), "стрим вернул план")
    check(any(t == "token" for t in tokens),
          f"токены стримились по мере генерации (события: {sorted(set(tokens))})")
    try:
        Plan.from_dict(out)
        check(True, "план из стрима проходит контракт")
    except SchemaError as exc:
        check(False, f"план из стрима проходит контракт: {exc}")
    print(f"  · стрим за {ms} мс, событий клиента: {len(tokens)}")


# --------------------------------------------------------------------------- #
#  Запуск
# --------------------------------------------------------------------------- #
def main() -> int:
    global PASSED, FAILED
    ap = argparse.ArgumentParser(
        description="Онлайн-проверки субагента «Мышление» против живого Colab")
    ap.add_argument("--quick", action="store_true",
                    help="без LLM-вызовов: только связь, токен, метрики, события")
    ap.add_argument("--stream", action="store_true",
                    help="дополнительно проверить /plan/stream (ещё один LLM-вызов)")
    args = ap.parse_args()

    client = ThinkingClient()
    print(f"Онлайн-тесты · URL: {client.base or 'НЕ ЗАДАН'} · "
          f"токен: {'задан' if client.token else 'НЕ ЗАДАН'} · "
          f"режим: {'quick' if args.quick else 'полный'}")

    if not client.base:
        print("\nURL не задан — выполните: "
              "python tools/thinking_cli.py set-url <URL> <TOKEN>")
        return 2

    if not test_connection(client):
        print(f"\nСЕРВЕР НЕДОСТУПЕН: {client.last_error}")
        print("Colab уснул или туннель умер — см. README §11, ячейка D.")
        return 2

    test_auth(client)
    test_metrics(client)
    test_events(client)

    if not args.quick:
        plan, before = test_plan(client)
        if plan:
            test_reflect(client, plan)
            test_history_grew(before, client)
        if args.stream:
            test_stream(client)

    section("Итог")
    print(f"ПРОЙДЕНО: {PASSED}")
    print(f"ПРОВАЛЕНО: {FAILED}")
    for name in FAILS:
        print(f"  - {name}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nпрервано пользователем")
        sys.exit(130)
