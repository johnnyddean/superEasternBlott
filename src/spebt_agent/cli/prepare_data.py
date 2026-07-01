from __future__ import annotations

import argparse

from spebt_agent.config import load_default_configs
from spebt_agent.data.prepare_competition_data import prepare_data
from spebt_agent.paths import project_root, resolve_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refs")
    parser.add_argument("--gfp-xlsx")
    parser.add_argument("--exclusion")
    parser.add_argument("--out", default="data/processed")
    args = parser.parse_args()
    cfg = load_default_configs()["competition"]["raw_data"]
    summary = prepare_data(
        resolve_path(args.refs or cfg["refs"]),
        resolve_path(args.gfp_xlsx or cfg["gfp_xlsx"]),
        resolve_path(args.exclusion or cfg["exclusion_csv"]),
        resolve_path(args.out, project_root()),
    )
    print(summary)


if __name__ == "__main__":
    main()
