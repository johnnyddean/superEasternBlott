from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(path: str | Path, root: Path | None = None) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (root or project_root()) / p
