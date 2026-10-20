from types import SimpleNamespace

import pytest

from app.services import simulation_runner as runner_module
from app.services import zep_graph_memory_updater as updater_module
from app.services.simulation_runner import RunnerStatus, SimulationRunState, SimulationRunner
from app.services.zep_graph_memory_updater import ZepGraphMemoryManager, ZepGraphMemoryUpdater


def _stuck_updater(monkeypatch):
    client = SimpleNamespace(graph=SimpleNamespace(
        add=lambda **_kwargs: SimpleNamespace(uuid_="episode-1"),
        episode=SimpleNamespace(get=lambda **_kwargs: SimpleNamespace(processed=False)),
    ))
    monkeypatch.setattr(updater_module, "get_zep_client", lambda *_a, **_k: client)
    monkeypatch.setattr(updater_module.time, "sleep", lambda _seconds: None)
    updater = ZepGraphMemoryUpdater("graph-1", simulation_id="sim-stuck")
    updater._pending_episode_uuids = ["episode-1"]
    return updater


def test_stop_can_leave_slow_episodes_to_the_background(monkeypatch):
    updater = _stuck_updater(monkeypatch)
    updater.stop(max_wait=0, abandon_pending=True)
    assert updater._pending_episode_uuids == []


def test_stop_still_raises_on_timeout_without_abandon(monkeypatch):
    updater = _stuck_updater(monkeypatch)
    with pytest.raises(TimeoutError):
        updater.stop(max_wait=0)


@pytest.fixture
def run_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(SimulationRunner, "RUN_STATE_DIR", str(tmp_path))
    monkeypatch.setattr(SimulationRunner, "_sync_simulation_status", classmethod(lambda *a, **k: None))
    yield tmp_path
    SimulationRunner._run_states.pop("sim-orphan", None)


def _orphan(run_dir, status, finished):
    state = SimulationRunState(simulation_id="sim-orphan", runner_status=status, current_round=3)
    if finished:
        for platform in ("twitter", "reddit"):
            (run_dir / "sim-orphan" / platform).mkdir(parents=True)
            (run_dir / "sim-orphan" / platform / "actions.jsonl").write_text("")
        state.twitter_completed = state.reddit_completed = True
    SimulationRunner._save_run_state(state)
    return state


def test_finished_run_without_process_is_completed(run_dir):
    _orphan(run_dir, RunnerStatus.RUNNING, finished=True)
    state = SimulationRunner.reconcile_orphaned_run("sim-orphan")
    assert state.runner_status == RunnerStatus.COMPLETED
    assert state.completed_at


def test_interrupted_run_without_process_fails_with_reason(run_dir):
    (run_dir / "sim-orphan" / "twitter").mkdir(parents=True)
    (run_dir / "sim-orphan" / "twitter" / "actions.jsonl").write_text("")
    _orphan(run_dir, RunnerStatus.RUNNING, finished=False)
    state = SimulationRunner.reconcile_orphaned_run("sim-orphan")
    assert state.runner_status == RunnerStatus.FAILED
    assert "перезапуск" in state.error


def test_live_run_is_left_alone(run_dir, monkeypatch):
    _orphan(run_dir, RunnerStatus.RUNNING, finished=True)
    monkeypatch.setitem(SimulationRunner._processes, "sim-orphan", SimpleNamespace(poll=lambda: None))
    state = SimulationRunner.reconcile_orphaned_run("sim-orphan")
    assert state.runner_status == RunnerStatus.RUNNING


def test_run_with_pending_graph_writes_is_left_alone(run_dir, monkeypatch):
    _orphan(run_dir, RunnerStatus.STOPPING, finished=True)
    monkeypatch.setattr(ZepGraphMemoryManager, "get_updater", classmethod(lambda _c, _s: object()))
    state = SimulationRunner.reconcile_orphaned_run("sim-orphan")
    assert state.runner_status == RunnerStatus.STOPPING
