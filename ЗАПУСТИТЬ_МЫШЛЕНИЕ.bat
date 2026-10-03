@echo off
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
