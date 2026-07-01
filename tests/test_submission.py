import pandas as pd

from spebt_agent.tools.submission import export_submission_csv


def test_export_submission(tmp_path):
    seq = "M" + "A" * 219
    out = export_submission_csv([{"sequence": seq}], "Team", tmp_path / "submission.csv")
    df = pd.read_csv(out)
    assert list(df.columns) == ["Team_Name", "Seq_ID", "Sequence"]
    assert len(df) == 1
