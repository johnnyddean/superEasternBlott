from __future__ import annotations

import argparse

from spebt_agent.assets.envs import create_all_tool_environments, create_tool_environment
from spebt_agent.assets.registry import list_tool_environments


def main() -> None:
    parser = argparse.ArgumentParser(description="Create isolated virtual environments for spEBT tool modules.")
    parser.add_argument("--module", default=None, help="Create only one module environment.")
    parser.add_argument("--install-requirements", action="store_true", help="Install each module's configured dependencies.")
    args = parser.parse_args()

    envs = list_tool_environments()
    if args.module:
        envs = [env for env in envs if env.module == args.module]
        if not envs:
            raise SystemExit(f"Unknown module: {args.module}")
    paths = [create_tool_environment(env, install_requirements=args.install_requirements) for env in envs]
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
