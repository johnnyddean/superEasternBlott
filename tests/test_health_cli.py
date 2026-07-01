import json
import os
import subprocess
import sys


def test_health_check_cli_returns_json():
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    result = subprocess.run(
        [sys.executable, "-m", "spebt_agent.cli.health_check", "--module", "esmc"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    payload = json.loads(result.stdout)
    assert "modules" in payload
    assert payload["modules"][0]["module"] == "esmc"
