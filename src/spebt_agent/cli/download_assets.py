from __future__ import annotations

import argparse

from spebt_agent.assets.downloader import download_assets_interactively
from spebt_agent.assets.registry import list_manual_assets, list_model_assets, list_tool_environments


def main() -> None:
    parser = argparse.ArgumentParser(description="List or download model assets with per-asset confirmation.")
    parser.add_argument("--module", default=None, help="Limit to one tool module.")
    parser.add_argument("--list", action="store_true", help="List configured assets without downloading.")
    parser.add_argument("--plan-only", action="store_true", help="Print planned downloads and sizes without prompting.")
    args = parser.parse_args()

    if args.list:
        print("Tool environments:")
        for env in list_tool_environments():
            print(f"- {env.module}: {env.env_name} requirements={list(env.requirements)}")
        print("\nDownloadable assets:")
        for asset in list_model_assets(tool_module=args.module):
            print(f"- {asset.asset_id} [{asset.tool_module}] {asset.kind} {asset.size_human} -> {asset.target}")
        print("\nManual assets:")
        for asset in list_manual_assets():
            print(f"- {asset['asset_id']} [{asset['tool_module']}] -> {asset['target_dir']}: {asset['description']}")
        return

    results = download_assets_interactively(tool_module=args.module, plan_only=args.plan_only)
    print(dict(results))


if __name__ == "__main__":
    main()
