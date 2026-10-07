# Allows `python -m alpaccaroo` with no install at all. MIT License.
from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
