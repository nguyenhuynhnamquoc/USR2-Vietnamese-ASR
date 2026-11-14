import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    command = [
        sys.executable,
        str(PROJECT_ROOT / "train_asr.py"),
        "--epochs",
        "3",
        "--batch-size",
        "1",
        "--seed",
        "42",
        "--limit-train-batches",
        "100",
        "--limit-validation-batches",
        "20",
        "--output-dir",
        str(PROJECT_ROOT / "output" / "pilot_3epochs"),
    ]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
