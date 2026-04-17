from __future__ import annotations

import subprocess
import sys


def main() -> None:
    task = "Open a text editor and type hello world"
    cmd = [sys.executable, "main.py", "--task", task]
    print("Starting demo run:", " ".join(cmd))
    subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()
