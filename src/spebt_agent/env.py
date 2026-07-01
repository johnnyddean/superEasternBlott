from __future__ import annotations

import os
from pathlib import Path

from spebt_agent.paths import project_root


def load_dotenv_file(path: str | Path | None = None, *, override: bool = False) -> bool:
    dotenv_path = Path(path) if path is not None else project_root() / ".env"
    if not dotenv_path.exists():
        return False

    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if value and len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if override or not os.getenv(key):
            os.environ[key] = value
    return True
