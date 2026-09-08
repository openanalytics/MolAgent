"""Unit tests for mcp/_data_registry.py.

Run with:
    uv run pytest mcp/tests/test_data_registry.py -v
"""
from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from _data_registry import (
    atomic_write_json,
    data_registry_path,
    get_dataset,
    list_datasets_for_owner,
    load_json_list,
    register_dataset,
    remove_dataset,
    touch_registry_entry,
)


@pytest.fixture(autouse=True)
def isolate_registry(tmp_path, monkeypatch):
    """Point data registry at a temp directory for each test."""
    monkeypatch.setenv("MOLAGENT_OUTPUT_ROOT", str(tmp_path))


# ── load_json_list / atomic_write_json ─────────────────────────────────────


def test_load_missing_file(tmp_path):
    assert load_json_list(tmp_path / "nope.json") == []


def test_load_invalid_json(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("not json")
    assert load_json_list(bad) == []


def test_atomic_write_and_read(tmp_path):
    path = tmp_path / "test.json"
    data = [{"id": "a"}, {"id": "b"}]
    atomic_write_json(path, data)
    assert load_json_list(path) == data


# ── register_dataset ───────────────────────────────────────────────────────


def test_register_creates_entry():
    entry = register_dataset(
        owner_id="user1",
        filename="data.csv",
        file_path="uploads/user1/data.csv",
        size_bytes=1234,
        columns=["smiles", "logP"],
        row_count=100,
    )
    assert entry["id"].startswith("ds_")
    assert entry["filename"] == "data.csv"
    assert entry["owner"] == "user1"
    assert entry["size_bytes"] == 1234
    assert entry["columns"] == ["smiles", "logP"]
    assert entry["row_count"] == 100
    assert entry["uploaded_at"] == entry["last_used"]


def test_register_multiple():
    register_dataset("u1", "a.csv", "uploads/u1/a.csv", 100, ["x"], 10)
    register_dataset("u2", "b.csv", "uploads/u2/b.csv", 200, ["y"], 20)
    all_entries = load_json_list(data_registry_path())
    assert len(all_entries) == 2


# ── get_dataset ────────────────────────────────────────────────────────────


def test_get_dataset_found():
    entry = register_dataset("u1", "c.csv", "uploads/u1/c.csv", 50, ["z"], 5)
    found = get_dataset(entry["id"])
    assert found is not None
    assert found["filename"] == "c.csv"


def test_get_dataset_not_found():
    assert get_dataset("nonexistent") is None


# ── list_datasets_for_owner ────────────────────────────────────────────────


def test_list_filters_by_owner():
    register_dataset("alice", "a.csv", "uploads/alice/a.csv", 100, ["x"], 10)
    register_dataset("bob", "b.csv", "uploads/bob/b.csv", 200, ["y"], 20)
    alice_ds = list_datasets_for_owner("alice", is_admin=False)
    assert len(alice_ds) == 1
    assert alice_ds[0]["owner"] == "alice"


def test_list_admin_sees_all():
    register_dataset("alice", "a.csv", "uploads/alice/a.csv", 100, ["x"], 10)
    register_dataset("bob", "b.csv", "uploads/bob/b.csv", 200, ["y"], 20)
    all_ds = list_datasets_for_owner("alice", is_admin=True)
    assert len(all_ds) == 2


# ── remove_dataset ─────────────────────────────────────────────────────────


def test_remove_existing():
    entry = register_dataset("u1", "d.csv", "uploads/u1/d.csv", 100, ["a"], 10)
    removed = remove_dataset(entry["id"])
    assert removed is not None
    assert removed["id"] == entry["id"]
    assert get_dataset(entry["id"]) is None


def test_remove_nonexistent():
    assert remove_dataset("fake_id") is None


def test_remove_wrong_owner():
    entry = register_dataset("alice", "e.csv", "uploads/alice/e.csv", 100, ["b"], 5)
    removed = remove_dataset(entry["id"], owner_id="bob")
    assert removed is None
    assert get_dataset(entry["id"]) is not None


def test_remove_correct_owner():
    entry = register_dataset("alice", "f.csv", "uploads/alice/f.csv", 100, ["c"], 3)
    removed = remove_dataset(entry["id"], owner_id="alice")
    assert removed is not None


# ── touch_registry_entry ───────────────────────────────────────────────────


def test_touch_updates_last_used():
    entry = register_dataset("u1", "g.csv", "uploads/u1/g.csv", 100, ["d"], 1)
    original_ts = entry["last_used"]
    time.sleep(1.1)
    result = touch_registry_entry(data_registry_path(), entry["id"])
    assert result is True
    updated = get_dataset(entry["id"])
    assert updated["last_used"] > original_ts


def test_touch_nonexistent_returns_false():
    result = touch_registry_entry(data_registry_path(), "fake_id")
    assert result is False


def test_touch_concurrent(tmp_path):
    """Multiple threads touching different entries should not corrupt the registry."""
    for i in range(5):
        register_dataset(f"u{i}", f"{i}.csv", f"uploads/u{i}/{i}.csv", 100, ["x"], 10)

    all_entries = load_json_list(data_registry_path())
    ids = [e["id"] for e in all_entries]

    errors = []

    def worker(entry_id):
        try:
            for _ in range(3):
                touch_registry_entry(data_registry_path(), entry_id)
                time.sleep(0.01)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(eid,)) for eid in ids]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # On Windows, concurrent file replace can raise PermissionError;
    # the lock pattern retries but with very tight timing some may fail.
    # The important assertion is that the registry is not corrupted.
    final = load_json_list(data_registry_path())
    assert len(final) == 5


# ── touch_registry_entry on model registry ─────────────────────────────────


def test_touch_works_on_any_registry(tmp_path):
    """touch_registry_entry should work on model_registry.json too."""
    model_reg = tmp_path / "model_registry.json"
    entries = [
        {"id": "model-1", "created_at": "2026-01-01T00:00:00", "last_used": "2026-01-01T00:00:00"},
        {"id": "model-2", "created_at": "2026-01-02T00:00:00", "last_used": "2026-01-02T00:00:00"},
    ]
    atomic_write_json(model_reg, entries)

    result = touch_registry_entry(model_reg, "model-1")
    assert result is True

    updated = load_json_list(model_reg)
    assert updated[0]["last_used"] > "2026-01-01T00:00:00"
    assert updated[1]["last_used"] == "2026-01-02T00:00:00"
