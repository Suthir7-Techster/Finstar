"""Append-only, hash-chained audit trail.

Every material event in the system — data generation, quality gates, gate
evaluations, registrations, approvals, production decisions, drift alerts —
is appended here as one JSON record whose hash covers the previous record's
hash. Any retroactive edit or deletion breaks the chain and is detected by
: meth:`AuditTrail.verify` (the tamper-evidence property regulators expect;
paper Section 4.3 "audit trails").

Single-process implementation with a thread lock; multi-worker deployments
should front this with a database (see docs/architecture.md).
"""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

GENESIS_HASH = "0" * 64


@dataclass(frozen=True)
class AuditVerification:
    valid: bool
    records: int
    reason: str = ""

    def render(self) -> str:
        if self.valid:
            return f"audit trail OK: {self.records} records, hash chain intact"
        return f"audit trail TAMPERED at or before seq {self.records}: {self.reason}"


def _canonical(record: dict[str, Any]) -> str:
    return json.dumps(record, sort_keys=True, separators=(",", ":"), default=str)


class AuditTrail:
    """JSONL audit log with SHA-256 chaining across records."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            self._path.touch()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------

    def append(
        self, event_type: str, payload: dict[str, Any], actor: str = "system"
    ) -> dict[str, Any]:
        """Append one event; returns the record as written."""
        with self._lock:
            prev_hash = self._last_hash()
            record: dict[str, Any] = {
                "seq": self._count_records() + 1,
                "timestamp": datetime.now(UTC).isoformat(),
                "event_type": event_type,
                "actor": actor,
                "payload": payload,
                "prev_hash": prev_hash,
            }
            record["hash"] = hashlib.sha256(_canonical(record).encode("utf-8")).hexdigest()
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, sort_keys=True, default=str) + "\n")
            return record

    def records(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for line in self._path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                out.append(json.loads(line))
        return out

    def verify(self) -> AuditVerification:
        """Recompute the chain; any mismatch reports the first bad record."""
        prev = GENESIS_HASH
        expected_seq = 1
        for record in self.records():
            recomputed = dict(record)
            claimed_hash = recomputed.pop("hash", None)
            if record.get("seq") != expected_seq:
                return AuditVerification(
                    False, expected_seq, f"expected seq {expected_seq}, found {record.get('seq')}"
                )
            if record.get("prev_hash") != prev:
                return AuditVerification(
                    False, expected_seq, f"prev_hash mismatch at seq {expected_seq}"
                )
            actual = hashlib.sha256(_canonical(recomputed).encode("utf-8")).hexdigest()
            if claimed_hash != actual:
                return AuditVerification(
                    False, expected_seq, f"hash mismatch at seq {expected_seq}"
                )
            prev = actual
            expected_seq += 1
        return AuditVerification(True, expected_seq - 1)

    def tail(self, n: int = 20) -> list[dict[str, Any]]:
        return self.records()[-n:]

    # ------------------------------------------------------------------

    def _count_records(self) -> int:
        with self._path.open("rb") as fh:
            return sum(1 for _ in fh)

    def _last_hash(self) -> str:
        last_line = None
        with self._path.open("rb") as fh:
            for line in fh:
                if line.strip():
                    last_line = line
        if last_line is None:
            return GENESIS_HASH
        return str(json.loads(last_line)["hash"])
