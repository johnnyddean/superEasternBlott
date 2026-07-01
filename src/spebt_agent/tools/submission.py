from pathlib import Path

import pandas as pd

AA = set("ACDEFGHIKLMNPQRSTVWY")


def export_submission_csv(selected, team_name, out_csv):
    rows = [
        {"Team_Name": team_name, "Seq_ID": i, "Sequence": item["sequence"]}
        for i, item in enumerate(selected, start=1)
    ]
    df = pd.DataFrame(rows, columns=["Team_Name", "Seq_ID", "Sequence"])
    if len(df) > 6:
        raise ValueError("Submission cannot contain more than 6 sequences.")
    for seq in df["Sequence"]:
        if not seq.startswith("M"):
            raise ValueError("All sequences must start with M.")
        if not (220 <= len(seq) <= 250):
            raise ValueError(f"Invalid length: {len(seq)}")
        if set(seq) - AA:
            raise ValueError("Invalid amino acid symbol detected.")
    out = Path(out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return out
