
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

STAGES = [
    "app/ingest.py",
    "app/chunk.py",
    "app/embed.py",
]


def run_pipeline():
    for stage in STAGES:
        print(f"\n--- Running {stage} ---")

        result = subprocess.run(
            [sys.executable, str(BASE_DIR / stage)],
            cwd=BASE_DIR,
            check=False,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"Pipeline stopped: {stage} failed."
            )

    print("\nIndexing pipeline completed successfully.")


if __name__ == "__main__":
    try:
        run_pipeline()
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
