from __future__ import annotations

import argparse
import json

from spebt_agent.graph import run_design


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--team-name", default=None)
    parser.add_argument("--full-search", action="store_true")
    parser.add_argument("--resume", dest="resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-stage2-candidates", type=int, default=None)
    parser.add_argument("--esmfold-top-k", type=int, default=None)
    args = parser.parse_args()
    state = run_design(
        team_name=args.team_name,
        profile="full_search" if args.full_search else "stable",
        resume=args.resume,
        run_id=args.run_id,
        output_dir=args.output_dir,
        max_stage2_candidates=args.max_stage2_candidates,
        esmfold_top_k=args.esmfold_top_k,
    )
    outputs = state.get("outputs", state.get("artifacts", {}))
    if not outputs:
        outputs = {
            "ok": state.get("ok", False),
            "status": state.get("status", "failed"),
            "run_dir": state.get("run_dir", ""),
            "failure_reason": state.get("failure_reason", ""),
            "warnings": state.get("warnings", []),
        }
    print(json.dumps(outputs, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
