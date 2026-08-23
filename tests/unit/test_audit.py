"""Audit trail: chaining, verification and tamper detection."""

from __future__ import annotations

import itertools
import json

from feg_mlops.governance.audit import GENESIS_HASH, AuditTrail


def test_chain_verifies_and_sequences(tmp_path):
    trail = AuditTrail(tmp_path / "audit" / "trail.jsonl")
    trail.append("pipeline.start", {"mode": "test"})
    trail.append("data.generated", {"rows": 10}, actor="generator")
    trail.append("model.registered", {"version": 1})

    verification = trail.verify()
    assert verification.valid and verification.records == 3
    records = trail.records()
    assert [r["seq"] for r in records] == [1, 2, 3]
    assert records[0]["prev_hash"] == GENESIS_HASH
    for prev, cur in itertools.pairwise(records):
        assert cur["prev_hash"] == prev["hash"]


def test_tamper_detection_value_edit(tmp_path):
    trail = AuditTrail(tmp_path / "t" / "t.jsonl")
    trail.append("decision.made", {"decision": "approve"})
    trail.append("decision.made", {"decision": "approve"})

    path = tmp_path / "t" / "t.jsonl"
    lines = path.read_text().splitlines()
    record = json.loads(lines[0])
    record["payload"]["decision"] = "decline"  # retroactive edit
    lines[0] = json.dumps(record, sort_keys=True)
    path.write_text("\n".join(lines) + "\n")

    verification = AuditTrail(path).verify()
    assert not verification.valid
    assert "seq 1" in verification.reason


def test_tamper_detection_line_deletion(tmp_path):
    trail = AuditTrail(tmp_path / "t2" / "t.jsonl")
    for i in range(4):
        trail.append("event", {"i": i})
    path = tmp_path / "t2" / "t.jsonl"
    lines = path.read_text().splitlines()
    path.write_text("\n".join(lines[:2] + lines[3:]) + "\n")  # drop seq 3
    verification = AuditTrail(path).verify()
    assert not verification.valid


def test_tail_and_empty(tmp_path):
    trail = AuditTrail(tmp_path / "t3" / "t.jsonl")
    assert trail.verify().valid and trail.verify().records == 0
    for i in range(5):
        trail.append("e", {"i": i})
    assert [r["seq"] for r in trail.tail(2)] == [4, 5]
