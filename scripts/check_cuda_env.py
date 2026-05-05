from __future__ import annotations

import importlib.util
import shutil
import subprocess
from textwrap import dedent


def _command_status(name: str) -> tuple[bool, str]:
    path = shutil.which(name)
    if not path:
        return False, "missing"
    try:
        completed = subprocess.run([name, "--version"], capture_output=True, text=True, timeout=5)
        line = (completed.stdout or completed.stderr).splitlines()
        return True, line[0] if line else path
    except Exception as exc:  # pragma: no cover
        return True, f"present but version check failed: {exc}"


def main() -> None:
    nvcc_ok, nvcc_info = _command_status("nvcc")
    smi_ok, smi_info = _command_status("nvidia-smi")
    torch_ok = importlib.util.find_spec("torch") is not None

    print(
        dedent(
            f"""
            CUDA environment check
            - nvcc: {'yes' if nvcc_ok else 'no'} ({nvcc_info})
            - nvidia-smi: {'yes' if smi_ok else 'no'} ({smi_info})
            - torch installed: {'yes' if torch_ok else 'no'}
            """
        ).strip()
    )

    if not (nvcc_ok and smi_ok and torch_ok):
        print("Result: CPU reference path is available. CUDA benchmark path is not ready on this machine.")
    else:
        print("Result: CUDA prerequisites look available.")


if __name__ == "__main__":
    main()
