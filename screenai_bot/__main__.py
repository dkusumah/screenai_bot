"""Allow ``python -m screenai_bot`` to launch the desktop GUI."""

from __future__ import annotations

from . import main

if __name__ == "__main__":
    raise SystemExit(main())
