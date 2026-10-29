@echo off
REM Prepara o ambiente (Windows): cria .venv e instala as dependencias. Pode rodar mais de uma vez.
cd /d "%~dp0"

if not exist .venv (
    python -m venv .venv || (echo Python 3.9+ nao encontrado. Instale em python.org e marque "Add to PATH". & exit /b 1)
)

.venv\Scripts\python -m pip install --quiet --upgrade pip
.venv\Scripts\python -m pip install --quiet -r requirements.txt || exit /b 1

echo Pronto! Use .venv\Scripts\python ^<script^> ...
