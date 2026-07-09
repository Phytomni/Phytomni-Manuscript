from __future__ import annotations

import sys
from pathlib import Path

from scripts.reproduce_lib import reproduce_main


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    raise SystemExit(reproduce_main(sys.argv[1:], root))


if __name__ == "__main__":
    main()
