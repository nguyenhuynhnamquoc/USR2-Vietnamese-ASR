import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def main():
    command = [
        sys.executable,
        str(PROJECT_ROOT / "train_asr.py"),
        "--epochs",
        "1",
        "--batch-size",
        "1",
        "--seed",
        "42",
        "--limit-train-batches",
        "100",
        "--limit-validation-batches",
        "20",
        "--output-dir",
        str(PROJECT_ROOT / "output" / "pilot_100"),
    ]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
