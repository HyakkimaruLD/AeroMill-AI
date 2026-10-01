"""T4 binding API, detached reads, one engine, priority stop and terminal grading."""

from dataclasses import fields, FrozenInstanceError
import importlib
import json
import numpy as np
import pytest


def api():
    from aeromill import session

    return session


def assert_public(value):
    if isinstance(value, dict):
        assert (
            not {
                "truth",
                "envelope",
                "engagement",
                "gain",
                "centers",
                "widths",
                "A",
                "h(x)",
            }
            & value.keys()
        )
        for child in value.values():
            assert_public(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            assert_public(child)


def test_public_contract_and_pure_lazy_reads(monkeypatch):
    m = api()
    calls = []
    from aeromill import training

    original = training.load_model

    def load(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(training, "load_model", load)
    assert "recover_a" in m.list_scenarios()
    assert m.list_modes() == [
        "no_adaptation",
        "baseline",
        "threshold_search",
        "ml_agent",
    ]
    s = m.RunSession("recover_a", 42, "ml_agent")
    expected = {
        "time_s",
        "state",
        "reason",
        "terminal",
        "x_mm",
        "path_length_mm",
        "rpm",
        "feed_mm_min",
        "target_rpm",
        "target_feed_mm_min",
        "zones",
        "vibration_tail",
        "spectrum",
        "tooth_hz",
        "series",
        "events",
        "evaluation",
    }
    snap = s.snapshot()
    assert {f.name for f in fields(snap)} == expected
    assert snap.time_s == 0 and not snap.terminal and snap.state == "WARMUP"
    assert snap.zones == [(80.0, 400.0, "material-zone")]
    assert snap.evaluation is None and calls == []
    with pytest.raises(FrozenInstanceError):
        snap.state = "HOLD"
    for _ in range(3):
        s.snapshot()
        s.export()
        s.advance(0)
    assert calls == []
    a = s.advance(5)
    assert a.time_s == 0.5 and len(calls) == 1
    assert a.vibration_tail.shape == (4096, 3)
    assert a.spectrum[0].shape == a.spectrum[1].shape == (513,)
    assert set(a.series) == {"t", "score", "rpm", "feed", "x"}
    assert all(isinstance(v, np.ndarray) and v.shape == (5,) for v in a.series.values())
    assert (
        np.isnan(a.series["score"][:2]).all()
        and np.isfinite(a.series["score"][2:]).all()
    )
    assert a.tooth_hz == pytest.approx(4 * a.rpm / 60)
    assert a.target_rpm == 3200 and a.target_feed_mm_min == 1200
    for name in (
        "time_s",
        "x_mm",
        "path_length_mm",
        "rpm",
        "feed_mm_min",
        "target_rpm",
        "target_feed_mm_min",
        "tooth_hz",
    ):
        assert type(getattr(a, name)) is float
    assert type(a.terminal) is bool and type(a.reason) is str and type(a.state) is str
    a.series["t"][:] = 999
    a.events.clear()
    a.zones.clear()
    a.vibration_tail[:] = 999
    b = s.snapshot()
    assert b.time_s == 0.5 and b.events and b.zones and b.series["t"][0] == 0.1
    assert np.max(b.vibration_tail) < 999
    assert_public(b.events)
    assert b.evaluation is None
    s.advance(5)
    assert len(calls) == 1


def test_terminal_idempotence_export_and_manual_stop():
    s = api().RunSession("recover_a", 42, "threshold_search")
    while s.snapshot().time_s < 5:
        s.advance(5)
    before = s.snapshot()
    s.request_stop()
    s.request_stop()
    assert s.snapshot().state == before.state
    after = s.advance(5)
    assert after.terminal and after.state == "HOLD" and after.reason == "MANUAL_STOP"
    assert (
        after.x_mm == before.x_mm
        and after.feed_mm_min == 0
        and after.target_feed_mm_min == 0
    )
    assert after.evaluation["run_success"] is False
    exports = s.export()
    for _ in range(4):
        s.advance(10)
        s.request_stop()
        s.snapshot()
    assert s.export() == exports
    csv_bytes, json_bytes = exports
    assert csv_bytes.decode().splitlines()[0] == "t,score,rpm,feed,x"
    exported = json.loads(json_bytes)
    assert (
        exported["evaluation"] == after.evaluation
        and exported["events"] == after.events
    )
    assert sum(e["event"] == "set_cutting_parameters" for e in after.events) == 1
    assert sum(e["event"] == "feed_hold" for e in after.events) == 1


def test_session_completion_evaluates_truth_and_compare():
    m = api()
    s = m.RunSession("recover_a", 42, "ml_agent")
    last = s.advance(650)
    assert last.terminal and last.evaluation["run_success"]
    assert last.vibration_tail.shape == (8192, 3)
    assert {
        "planning_record",
        "ack",
        "reached",
        "record_outcome",
        "recovery_claim",
        "memory_write",
        "transition",
    } <= {e["event"] for e in last.events}
    rows = m.compare("stable", 42, ["no_adaptation", "baseline"])
    assert [r["mode"] for r in rows] == ["no_adaptation", "baseline"]
    assert all(r["stable_without_intervention"] for r in rows)


@pytest.mark.parametrize("ticks", [-1, 0.5, True])
def test_invalid_advance_does_not_start(ticks):
    s = api().RunSession("stable", 42, "no_adaptation")
    with pytest.raises((ValueError, TypeError)):
        s.advance(ticks)
    assert s.snapshot().time_s == 0
