# Налаштування середовища курсу на Windows.
#
#   .\scripts\setup.ps1              # створити/оновити середовище cv-advanced
#   .\scripts\setup.ps1 --force      # перестворити з нуля
#   .\scripts\setup.ps1 --dry-run    # лише показати, що буде виконано
#
# Прапорці йдуть прямо в setup.py (argparse), тому пишуться з двома дефісами.
#
# Якщо PowerShell не дає запустити скрипт:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

$ErrorActionPreference = "Stop"

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    Write-Host "conda не знайдено." -ForegroundColor Red
    Write-Host "Постав Miniconda/Anaconda й відкрий Anaconda Prompt," -ForegroundColor Yellow
    Write-Host "або виконай `conda init powershell` і перезапусти термінал." -ForegroundColor Yellow
    exit 1
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$condaBase = (& conda info --base).Trim()
$condaPython = Join-Path $condaBase "python.exe"

if (-not (Test-Path $condaPython)) {
    Write-Host "Не знайшов python у базовому середовищі conda: $condaPython" -ForegroundColor Red
    exit 1
}

& $condaPython (Join-Path $scriptDir "setup.py") @args
exit $LASTEXITCODE
