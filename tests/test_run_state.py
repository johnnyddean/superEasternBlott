import json

from spebt_agent.run_state import RunTracker, fingerprint_config


def test_run_tracker_resume_requires_matching_fingerprint(tmp_path):
    run_dir = tmp_path / "runs" / "run1"
    latest_dir = tmp_path / "latest"
    tracker1 = RunTracker(
        run_id="run1",
        run_dir=run_dir,
        latest_dir=latest_dir,
        profile="stable",
        team_name="TeamA",
        config_fingerprint=fingerprint_config({"a": 1}),
        allow_resume=True,
    )
    tracker1.stage_start("stage1")
    tracker1.stage_success("stage1", {"artifact_a": str(run_dir / "a.txt")})

    tracker2 = RunTracker(
        run_id="run1",
        run_dir=run_dir,
        latest_dir=latest_dir,
        profile="stable",
        team_name="TeamA",
        config_fingerprint=fingerprint_config({"a": 2}),
        allow_resume=True,
    )
    assert tracker2.resumed is False
    assert tracker2.state["stage_status"] == {}


def test_stage_can_resume_requires_existing_artifact(tmp_path):
    run_dir = tmp_path / "runs" / "run2"
    latest_dir = tmp_path / "latest"
    tracker = RunTracker(
        run_id="run2",
        run_dir=run_dir,
        latest_dir=latest_dir,
        profile="stable",
        team_name="TeamA",
        config_fingerprint=fingerprint_config({"a": 1}),
        allow_resume=True,
    )
    tracker.stage_start("stage1")
    tracker.stage_success("stage1", {"artifact_a": str(run_dir / "missing.csv")})
    assert tracker.stage_can_resume("stage1", ["artifact_a"]) is False


def test_fail_running_stage_marks_failure(tmp_path):
    run_dir = tmp_path / "runs" / "run3"
    latest_dir = tmp_path / "latest"
    tracker = RunTracker(
        run_id="run3",
        run_dir=run_dir,
        latest_dir=latest_dir,
        profile="stable",
        team_name="TeamA",
        config_fingerprint=fingerprint_config({"a": 1}),
        allow_resume=True,
    )
    tracker.stage_start("stage1")
    failed_stage = tracker.fail_running_stage("boom")
    state = json.loads(tracker.state_path.read_text(encoding="utf-8"))
    assert failed_stage == "stage1"
    assert state["stage_status"]["stage1"]["status"] == "failed"
    assert state["failure_reason"] == "boom"
