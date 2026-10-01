"""Memory must change real planning, yet never bypass verification or limits."""

import importlib
from dataclasses import replace
import pytest
from test_agent import observation, warm, settled, make_agent
from aeromill.config import ToolProfile


def memory():
    m = importlib.import_module("aeromill.memory")
    assert hasattr(m, "Memory"), "memory missing"
    return m.Memory()


def test_versions_noops_and_invalid_pairs():
    store = memory()
    key = ("four-tooth-v1", "nominal", "v1", "v1")
    store.remember(key, (2880.0, 960.0), "verify-1", 5.3)
    assert store.get(key).pair == (2880.0, 960.0)
    assert store.get(key).verification_id == "verify-1" and store.get(key).time_s == 5.3
    for i in range(4):
        wrong = list(key)
        wrong[i] = "different"
        assert store.get(tuple(wrong)) is None
    for pair, current, expected in [
        ((2880.0, 960.0), (3200.0, 1200.0), (2880.0, 960.0)),
        ((3200.0, 1200.0), (3200.0, 1200.0), (3520.0, 960.0)),
        ((3840.0, 720.0), (3840.0, 720.0), (3520.0, 960.0)),
        ((0.0, 960.0), (3200.0, 1200.0), (3520.0, 960.0)),
    ]:
        store.remember(key, pair, "v", 1.0)
        a = make_agent(memory=store, model_version="v1")
        a.now = 30
        a.zone_class = "nominal"
        cmd = a.plan(*current)
        assert (cmd.rpm, cmd.feed_mm_min) == expected
        assert a.state == "APPLY" and not any(
            e["event"] == "recovery_claim" for e in a.events
        )
        record = next(e for e in a.events if e["event"] == "planning_record")
        assert record["memory"] == "hit" and record["ranked"][0]["pair"] == list(pair)


def test_claim_writes_and_failed_reuse_deletes():
    store = memory()
    a = make_agent(memory=store, model_version="v1")
    settled(a)
    for k in range(39, 54):
        a.tick(observation(k, rpm=3520, feed=960))
    key = ("four-tooth-v1", "nominal", "v1", "v1")
    saved = store.get(key)
    assert saved.pair == (3520, 960) and saved.time_s == 5.3
    assert saved.verification_id == a.command.command_id
    b = make_agent(memory=store, model_version="v1")
    settled(b)
    for k in range(39, 54):
        b.tick(observation(k, score=0.5, rpm=3520, feed=960))
    assert store.get(key) is None
    assert any(e["event"] == "memory_delete" for e in b.events)


def test_default_memory_is_cold_and_version_mismatch_misses():
    store = memory()
    store.remember(("four-tooth-v1", "nominal", "old", "v1"), (2880, 960), "v", 1)
    a = make_agent(memory=store, model_version="new")
    a.now = 30
    a.plan(3200, 1200)
    assert (
        next(e for e in a.events if e["event"] == "planning_record")["memory"] == "miss"
    )
    a = make_agent()
    b = make_agent()
    assert a.memory is not b.memory
    key = ("cold-check", "zone", "model", "v1")
    a.memory.remember(key, (3520, 960), "v", 1.0)
    assert b.memory.get(key) is None


@pytest.mark.parametrize(
    "reason", ["ACK_TIMEOUT", "SETPOINT_TIMEOUT", "COMMAND_REJECTED"]
)
def test_failed_reuse_on_hold_is_deleted(reason):
    store = memory()
    key = ("four-tooth-v1", "nominal", "v1", "v1")
    store.remember(key, (2880, 960), "v", 1)
    a = make_agent(memory=store, model_version="v1")
    a.plan(3200, 1200)
    a.hold(reason)
    assert store.get(key) is None


def test_in_range_non_candidate_memory_is_excluded():
    store = memory()
    key = ("four-tooth-v1", "nominal", "v1", "v1")
    store.remember(key, (3500, 900), "v", 1)
    a = make_agent(memory=store, model_version="v1")
    cmd = a.plan(3200, 1200)
    assert (cmd.rpm, cmd.feed_mm_min) == (3520, 960)
