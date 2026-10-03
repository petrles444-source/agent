# -*- coding: utf-8 -*-
"""Пересборка .bat-запускалок: UTF-8 без BOM + chcp 65001 + CRLF.

Старая ОСТАНОВИТЬ была в cp1251 без chcp — в консоли это «??????».
Здесь фиксируется канонический вид обоих файлов.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

START = r'''@echo off
chcp 65001 >nul
rem ============================================================
rem  Субагент «Мышление» - быстрый запуск на ПК: doctor + панель
rem  Проверки: doctor показывает связь с Colab, токен и здоровье.
rem
rem  Если Colab упал или закончились лимиты (Colab: Сессия → Лимиты):
rem  открой colab\thinking_agent_ver3.ipynb и выполни Runtime → Run all,
rem  затем при новом токене:
rem      python tools\thinking_cli.py set-url <URL> <TOKEN>
rem ============================================================
cd /d "%~dp0"
title Субагент Мышление - запуск панели

where python >nul 2>nul
if errorlevel 1 (
    echo.
    echo  Python не найден в PATH. Установи Python 3.11+ и добавь в PATH.
    echo.
    pause
    exit /b 1
)

echo.
echo  Проверка связи с Colab...
python tools\thinking_cli.py doctor

echo.
echo  Запускаю панель управления (порт 8765, браузер откроется сам)...
python tools\thinking_cli.py panel --open

echo.
echo  Панель остановится вместе с этим окном. Ручная остановка:
echo  ОСТАНОВИТЬ_МЫШЛЕНИЕ.bat
pause
'''

STOP = r'''@echo off
chcp 65001 >nul
rem ============================================================
rem  Остановка панели субагента на ПК (порт 8765)
rem  Colab на этом не останавливается - там ячейка F.
rem ============================================================
cd /d "%~dp0"
title Остановка панели Мышление

set "FOUND="
for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":8765" ^| findstr "LISTENING"') do (
    echo  Останавливаю процесс %%p на порту 8765...
    taskkill /PID %%p /F >nul 2>nul
    set "FOUND=1"
)
if not defined FOUND echo  Панель не запущена - порт 8765 свободен.

echo.
echo  Готово. Остановка выполнена.
timeout /t 2 >nul
'''


def write_bat(name: str, text: str) -> None:
    path = ROOT / name
    data = text.replace("\r\n", "\n").replace("\n", "\r\n").encode("utf-8")
    assert not data.startswith(b"\xef\xbb\xbf"), "BOM не нужен"
    assert b"chcp 65001" in data, "обязателен chcp 65001"
    path.write_bytes(data)
    print(f"  {name}: {len(data)} bytes, utf-8 no BOM, CRLF")


if __name__ == "__main__":
    write_bat("ЗАПУСТИТЬ_МЫШЛЕНИЕ.bat", START)
    write_bat("ОСТАНОВИТЬ_МЫШЛЕНИЕ.bat", STOP)
    print("OK")
