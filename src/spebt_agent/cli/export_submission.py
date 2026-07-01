from __future__ import annotations

import argparse

import pandas as pd

from spebt_agent.tools.submission import export_submission_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected", default="outputs/selected_top6.csv")
    parser.add_argument("--team-name", required=True)
    parser.add_argument("--out", default="outputs/submission.csv")
    args = parser.parse_args()
    selected = pd.read_csv(args.selected).to_dict("records")
    print(export_submission_csv(selected, args.team_name, args.out))


if __name__ == "__main__":
    main()
