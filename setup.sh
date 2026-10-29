#!/usr/bin/env bash
# Prepara o ambiente (Linux/macOS): cria .venv e instala as dependências. Pode rodar mais de uma vez.
set -e
cd "$(dirname "$0")"

if ! command -v python3 >/dev/null; then
    echo "Python 3 não encontrado. Instale o Python 3.9+ e rode de novo." >&2
    exit 1
fi

if [ ! -d .venv ]; then
    python3 -m venv .venv || {
        echo "Falha ao criar o ambiente virtual. No Ubuntu/Debian, rode: sudo apt install python3-venv" >&2
        exit 1
    }
fi

.venv/bin/pip install --quiet --upgrade pip
.venv/bin/pip install --quiet -r requirements.txt

echo "Pronto! Use .venv/bin/python <script> ... ou ative com: source .venv/bin/activate"
