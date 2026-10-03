@echo off
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
