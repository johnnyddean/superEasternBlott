from __future__ import annotations

import argparse
import json
import subprocess

from spebt_agent.assets.envs import env_dir, env_python
from spebt_agent.assets.registry import list_tool_environments
from spebt_agent.paths import project_root


DEFAULT_MODULES = ("brightness", "esmc", "inverse_folding", "stability")


def run_module_smoke(module: str) -> dict:
    env = next((item for item in list_tool_environments() if item.module == module), None)
    if env is None:
        return {"module": module, "status": "missing_env", "ok": False, "error": "Tool environment is not configured."}

    py = env_python(env_dir(env))
    if not py.exists():
        return {"module": module, "status": "missing_python", "ok": False, "error": f"Environment python not found: {py}"}

    proc = subprocess.run(
        [str(py), "scripts/smoke_tool.py", module],
        cwd=project_root(),
        capture_output=True,
        text=True,
    )
    payload = None
    if proc.stdout.strip():
        try:
            payload = json.loads(proc.stdout)
        except json.JSONDecodeError:
            payload = {"raw_stdout": proc.stdout.strip()}
    return {
        "module": module,
        "env_name": env.env_name,
        "python": str(py),
        "status": "pass" if proc.returncode == 0 else "fail",
        "ok": proc.returncode == 0,
        "result": payload,
        "stderr": proc.stderr.strip(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", action="append", dest="modules", help="Run health checks for one or more modules.")
    parser.add_argument("--all", action="store_true", help="Run health checks for every configured tool module.")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero if any module fails.")
    args = parser.parse_args()

    if args.all:
        modules = [env.module for env in list_tool_environments()]
    elif args.modules:
        modules = args.modules
    else:
        modules = list(DEFAULT_MODULES)

    report = {"modules": [run_module_smoke(module) for module in modules]}
    report["ok"] = all(item["ok"] for item in report["modules"])
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.strict and not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
