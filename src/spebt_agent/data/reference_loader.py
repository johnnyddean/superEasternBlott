from pathlib import Path

import pandas as pd


def load_parent_sequence(processed_dir: str | Path, parent_name: str) -> str:
    refs = pd.read_csv(Path(processed_dir) / "reference_parents.csv")
    match = refs[refs["name"] == parent_name]
    if match.empty:
        raise ValueError(f"Parent sequence not found: {parent_name}")
    return str(match.iloc[0]["sequence"])
