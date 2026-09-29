#!/usr/bin/env bash
# Налаштування середовища курсу на Linux та macOS.
#
#   ./scripts/setup.sh              # створити/оновити середовище cv-advanced
#   ./scripts/setup.sh --force      # перестворити з нуля
#   ./scripts/setup.sh --dry-run    # лише показати, що буде виконано
#
# Якщо файл не виконуваний: chmod +x scripts/setup.sh

set -euo pipefail

if ! command -v conda >/dev/null 2>&1; then
    echo "conda не знайдено." >&2
    echo "Постав Miniconda/Anaconda й перезапусти термінал," >&2
    echo "або виконай: eval \"\$(\$HOME/miniconda3/bin/conda shell.bash hook)\"" >&2
    exit 1
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
conda_base="$(conda info --base)"

# bin/python — Linux та macOS; python.exe — Windows-встановлення conda, якщо скрипт
# запускають із Git Bash
conda_python=""
for candidate in "$conda_base/bin/python" "$conda_base/python.exe"; do
    if [ -x "$candidate" ]; then
        conda_python="$candidate"
        break
    fi
done

if [ -z "$conda_python" ]; then
    echo "Не знайшов python у базовому середовищі conda: $conda_base" >&2
    exit 1
fi

exec "$conda_python" "$script_dir/setup.py" "$@"
