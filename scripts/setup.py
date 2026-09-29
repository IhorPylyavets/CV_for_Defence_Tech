#!/usr/bin/env python
"""Створює conda-середовище курсу й ставить залежності під поточну платформу.

Запускати зручніше через лаунчери: scripts/setup.ps1 (Windows) або scripts/setup.sh (Linux/macOS).

Що робить:
  1. створює (або оновлює) середовище з environment.yml;
  2. ставить torch зі збіркою під платформу — CUDA, CPU або Apple MPS;
  3. ставить решту з requirements.txt;
  4. реєструє ядро Jupyter, щоб ноутбуки бачили саме це середовище;
  5. перевіряє, що все імпортується, і друкує версії.
"""

import argparse
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):  # інакше кирилиця падає на cp1251-консолі Windows
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = Path(__file__).resolve().parent.parent
ENV_FILE = REPO / "environment.yml"
REQUIREMENTS = REPO / "requirements.txt"
DEFAULT_ENV = "cv-advanced"
TORCH_INDEX = "https://download.pytorch.org/whl/{}"
# CUDA-збірки torch для Windows, новіші спершу. PyPI на Windows віддає лише CPU-колеса,
# тож збірку під GPU доводиться брати з індексу PyTorch. Беремо найновішу, яку тягне драйвер.
# Список перевіряти з виходом нових версій torch: https://pytorch.org/get-started/locally/
WINDOWS_CUDA_TAGS = ("cu132", "cu130", "cu126")

VERIFY_SNIPPET = """
import importlib, platform
print("python     ", platform.python_version())
for module in ("numpy", "cv2", "matplotlib", "torch", "torchvision", "ultralytics", "ipywidgets"):
    try:
        lib = importlib.import_module(module)
        print(f"{module:<11}", getattr(lib, "__version__", "?"))
    except Exception as exc:
        print(f"{module:<11} НЕ ІМПОРТУЄТЬСЯ: {exc}")
try:
    import torch
    if torch.cuda.is_available():
        print("прискорення  CUDA:", torch.cuda.get_device_name(0))
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        print("прискорення  Apple MPS")
    else:
        print("прискорення  немає, рахуємо на CPU")
except Exception:
    pass
"""


def display(command: list[str], limit: int = 120) -> str:
    """Команда у вигляді, придатному для копіювання: з лапками й без простирадла коду."""
    parts = []
    for arg in command:
        shown = arg if len(arg) <= limit else arg[:limit].replace("\n", " ") + "…"
        parts.append(f'"{shown}"' if any(ch in shown for ch in ' \n"') else shown)
    return " ".join(parts)


class Runner:
    """Запускає команди, а в режимі --dry-run лише друкує їх."""

    def __init__(self, dry_run: bool):
        self.dry_run = dry_run

    def __call__(self, command: list[str], check: bool = True) -> int:
        print("\n$", display(command), flush=True)
        if self.dry_run:
            return 0
        code = subprocess.call(command)
        if code != 0 and check:
            sys.exit(f"\nКоманда завершилась з кодом {code}. Далі не йдемо.")
        return code


def find_conda() -> str:
    conda = os.environ.get("CONDA_EXE") or shutil.which("conda")
    if not conda:
        sys.exit("conda не знайдено. Постав Miniconda/Anaconda й відкрий новий термінал "
                 "(на Windows — Anaconda Prompt або спершу `conda init powershell`).")
    return conda


def env_exists(conda: str, name: str) -> bool:
    try:
        output = subprocess.check_output([conda, "env", "list"], text=True)
    except subprocess.CalledProcessError:
        return False
    for line in output.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        if line.split()[0] == name:
            return True
    return False


def driver_cuda_version() -> tuple[int, int] | None:
    """Найвища версія CUDA, яку тягне драйвер NVIDIA; None — GPU немає.

    (0, 0) — GPU є, але версію з nvidia-smi прочитати не вдалося.
    """
    if not shutil.which("nvidia-smi"):
        return None
    try:
        output = subprocess.run(["nvidia-smi"], capture_output=True, text=True, errors="replace")
    except OSError:
        return None
    if output.returncode != 0:
        return None
    match = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", output.stdout)
    return (int(match[1]), int(match[2])) if match else (0, 0)


def tag_version(cuda_tag: str) -> tuple[int, int]:
    """cu132 → (13, 2), cu126 → (12, 6)."""
    digits = cuda_tag.removeprefix("cu")
    return int(digits[:-1]), int(digits[-1])


def torch_install_args(force_cpu: bool, cuda_tag: str | None) -> tuple[list[str], str, str | None]:
    """Аргументи pip для torch, людський опис вибору і очікувана мітка збірки.

    Мітка — те, що стоїть після «+» у torch.__version__ (cpu, cu132); None — PyPI-колесо без мітки.
    """
    if sys.platform == "darwin":
        flavour = "Apple Silicon (MPS)" if platform.machine() == "arm64" else "CPU (Intel Mac)"
        # CUDA-збірок під macOS не існує, тож прапорець тут можна лише проігнорувати — але вголос
        ignored = " | --cuda проігноровано: CUDA-збірок під macOS не буває" if cuda_tag else ""
        return [], f"macOS → стандартні колеса з PyPI, {flavour}{ignored}", None

    cpu = ["--index-url", TORCH_INDEX.format("cpu")]

    if cuda_tag:
        return ["--index-url", TORCH_INDEX.format(cuda_tag)], f"примусово CUDA-збірка {cuda_tag}", cuda_tag

    if force_cpu:
        return cpu, "примусово CPU-збірка", "cpu"

    driver = driver_cuda_version()
    if driver is None:
        return cpu, "NVIDIA GPU не знайдено → CPU-збірка (менша й швидше ставиться)", "cpu"

    if sys.platform != "win32":
        # на Linux типовий PyPI-індекс і так віддає CUDA-збірку поточної версії
        return [], "знайдено NVIDIA GPU → стандартні колеса з PyPI (CUDA-збірка)", None

    fitting = [tag for tag in WINDOWS_CUDA_TAGS if tag_version(tag) <= driver]
    if not fitting:
        if driver == (0, 0):
            return cpu, ("знайдено NVIDIA GPU, але версію CUDA драйвера не прочитано → CPU-збірка. "
                         "Задай збірку вручну: --cuda cu126"), "cpu"
        return cpu, (f"драйвер NVIDIA тягне лише CUDA {driver[0]}.{driver[1]} — застарий для "
                     f"{', '.join(WINDOWS_CUDA_TAGS)} → CPU-збірка. Онови драйвер і запусти ще раз"), "cpu"
    tag = fitting[0]
    return (["--index-url", TORCH_INDEX.format(tag)],
            f"знайдено NVIDIA GPU, драйвер CUDA {driver[0]}.{driver[1]} → CUDA-збірка {tag}", tag)


def installed_torch_label(in_env: list[str]) -> str | None:
    """Мітка вже встановленого torch (cpu, cu132, "" — без мітки); None — torch не встановлено."""
    try:
        version = subprocess.run(in_env + ["python", "-c", "import torch; print(torch.__version__)"],
                                 capture_output=True, text=True).stdout.strip().splitlines()
    except OSError:
        return None
    if not version:
        return None
    _, _, label = version[-1].partition("+")
    return label


def main() -> None:
    parser = argparse.ArgumentParser(description="Налаштування середовища курсу cv-advanced")
    parser.add_argument("--name", default=DEFAULT_ENV, help=f"назва середовища (типово {DEFAULT_ENV})")
    parser.add_argument("--force", action="store_true", help="перестворити середовище з нуля")
    parser.add_argument("--cpu", action="store_true", help="поставити CPU-збірку torch навіть за наявності GPU")
    parser.add_argument("--cuda", metavar="TAG", help="конкретна CUDA-збірка torch, напр. cu126")
    parser.add_argument("--no-kernel", action="store_true", help="не реєструвати ядро Jupyter")
    parser.add_argument("--dry-run", action="store_true", help="лише показати команди, нічого не робити")
    args = parser.parse_args()

    if not ENV_FILE.exists() or not REQUIREMENTS.exists():
        sys.exit(f"Не бачу {ENV_FILE.name} або {REQUIREMENTS.name} — запускай скрипт із репозиторію курсу.")

    conda = find_conda()
    run = Runner(args.dry_run)
    exists = env_exists(conda, args.name)

    print(f"Платформа:   {platform.system()} {platform.machine()}")
    print(f"conda:       {conda}")
    print(f"Середовище:  {args.name} ({'існує' if exists else 'буде створене'})")

    if exists and args.force:
        run([conda, "env", "remove", "-y", "-n", args.name])
        exists = False

    if exists:
        run([conda, "env", "update", "-n", args.name, "-f", str(ENV_FILE), "--prune"])
    else:
        run([conda, "env", "create", "-n", args.name, "-f", str(ENV_FILE)])

    in_env = [conda, "run", "--no-capture-output", "-n", args.name]
    pip = in_env + ["python", "-m", "pip"]

    run(pip + ["install", "--upgrade", "pip"])

    torch_args, explanation, wanted_label = torch_install_args(args.cpu, args.cuda)
    print(f"\nTorch:       {explanation}")

    # pip вважає torch 2.x+cpu таким, що задовольняє «torch», і CUDA-збірку поверх не поставить.
    # Тож якщо в середовищі стоїть інша збірка, спершу її знімаємо
    installed = installed_torch_label(in_env) if exists else None
    if installed is not None and wanted_label is not None and installed != wanted_label:
        print(f"Torch:       зараз стоїть збірка «{installed or 'PyPI'}», потрібна «{wanted_label}» → перевстановлюю")
        run(pip + ["uninstall", "-y", "torch", "torchvision"])
    run(pip + ["install", "torch", "torchvision"] + torch_args)

    # решта requirements ставиться після torch: він уже задоволений і не перевстановиться
    run(pip + ["install", "-r", str(REQUIREMENTS)])

    if not args.no_kernel:
        run(pip + ["install", "ipykernel"])
        run(in_env + ["python", "-m", "ipykernel", "install", "--user",
                      "--name", args.name, "--display-name", f"Python ({args.name})"])

    print("\n" + "=" * 62)
    print("ПЕРЕВІРКА")
    print("=" * 62)
    run(in_env + ["python", "-c", VERIFY_SNIPPET], check=False)

    print("\n" + "=" * 62)
    print("ГОТОВО")
    print("=" * 62)
    print(f"  conda activate {args.name}")
    print("  jupyter lab")
    print(f"\nУ ноутбуці обери ядро «Python ({args.name})» — інакше Jupyter візьме base-середовище,")
    print("де немає torch і ultralytics.")

    if args.dry_run:
        print("\n(це був --dry-run: жодної команди не виконано)")


if __name__ == "__main__":
    main()
