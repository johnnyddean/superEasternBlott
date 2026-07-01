import subprocess
import sys


def test_smoke_tool_candidates_script():
    result = subprocess.run([sys.executable, "scripts/smoke_tool.py", "candidates"], check=True, capture_output=True, text=True)
    assert '"ok": true' in result.stdout
