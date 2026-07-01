from pathlib import Path


def test_readme_mentions_run_design_cli_options():
    readme = Path("README.md").read_text(encoding="utf-8")
    expected = [
        "--team-name",
        "--full-search",
        "--resume",
        "--run-id",
        "--max-stage2-candidates",
        "--esmfold-top-k",
        "outputs/runs/<run_id>/",
        "outputs/latest/",
        "python -m spebt_agent.cli.run_design --team-name YourTeamName",
    ]
    for token in expected:
        assert token in readme
