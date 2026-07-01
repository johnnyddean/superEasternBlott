import json
import sys

from spebt_agent.cli import run_design as run_design_cli


def test_run_design_cli_passes_new_options(monkeypatch, capsys):
    captured = {}

    def fake_run_design(**kwargs):
        captured.update(kwargs)
        return {"outputs": {"submission": "outputs/latest/submission.csv"}}

    monkeypatch.setattr(run_design_cli, "run_design", fake_run_design)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_design.py",
            "--team-name",
            "TeamA",
            "--full-search",
            "--no-resume",
            "--run-id",
            "run-123",
            "--output-dir",
            "outputs",
            "--max-stage2-candidates",
            "1500",
            "--esmfold-top-k",
            "12",
        ],
    )

    run_design_cli.main()
    out = capsys.readouterr().out
    payload = json.loads(out)

    assert captured["team_name"] == "TeamA"
    assert captured["profile"] == "full_search"
    assert captured["resume"] is False
    assert captured["run_id"] == "run-123"
    assert captured["output_dir"] == "outputs"
    assert captured["max_stage2_candidates"] == 1500
    assert captured["esmfold_top_k"] == 12
    assert payload["submission"].endswith("submission.csv")


def test_run_design_cli_defaults_to_stable_resume(monkeypatch):
    captured = {}

    def fake_run_design(**kwargs):
        captured.update(kwargs)
        return {"outputs": {}}

    monkeypatch.setattr(run_design_cli, "run_design", fake_run_design)
    monkeypatch.setattr(sys, "argv", ["run_design.py"])

    run_design_cli.main()

    assert captured["profile"] == "stable"
    assert captured["resume"] is True
