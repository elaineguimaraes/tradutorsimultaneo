@echo off
rem Inicia o Tradutor Simultaneo
cd /d "%~dp0"
set PYTHONPATH=%~dp0src
".venv\Scripts\python.exe" -m tradutor.main
pause
