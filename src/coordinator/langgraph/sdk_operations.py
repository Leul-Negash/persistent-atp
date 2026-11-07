"""Durable client request identity and observation, never a proof/operation authority.

The receiver owns admission, execution and its authoritative operation ledger.
This separate local SQLite cache only remembers exactly what the client sent
and observed. It does not grant missing receiver capabilities or enable the
SDKBackend mutation methods.
"""
from __future__ import annotations

import asyncio
from contextlib import contextmanager
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sqlite3
from types import SimpleNamespace


class IntentConflict(ValueError):
    """A durable identity cannot be reused for a different request or receiver."""


class OperationPending(RuntimeError):
    """Delivery or observation remains unresolved; this never means rollback."""

    def __init__(self, message, checkpoint_ref):
        super().__init__(message)
        self.checkpoint_ref = checkpoint_ref


@dataclass(frozen=True)
class IntentRecord:
    receiver_id: str
    workspace_id: str
    intent_id: str
    request_id: str
    proof_id: str
    fingerprint: str
    request_json: bytes
    state: str
    operation_json: str | None
    admission_json: str | None
    observation_json: bytes | None

    def checkpoint_ref(self):
        """Plain identities only; no client, credentials, leases or proof state."""
        return {"receiver_id": self.receiver_id, "workspace_id": self.workspace_id,
                "intent_id": self.intent_id, "request_id": self.request_id,
                "fingerprint": self.fingerprint,
                "operation_ref": json.loads(self.operation_json) if self.operation_json else None}


def _sdk():
    try:
        from persistent_atp_client import (
            ActionRequest, Admission, AsyncClient, OperationRef,
        )
        from persistent_atp_client.models import TERMINAL_LIFECYCLES
    except ImportError as exc:
        raise ImportError("DurableSDKOperations requires persistent-atp-client 0.2") from exc
    from persistent_atp_client.codecs import canonical_json, fingerprint
    from persistent_atp_client.errors import AdmissionRejected, ReceiverProtocolError, SubmissionUncertain, TransientUnavailable
    return SimpleNamespace(Request=ActionRequest, Admission=Admission,
        Client=AsyncClient, Ref=OperationRef,
        terminal=TERMINAL_LIFECYCLES, canonical=canonical_json, fingerprint=fingerprint,
        Rejected=AdmissionRejected, ProtocolError=ReceiverProtocolError,
        Uncertain=SubmissionUncertain, Unavailable=TransientUnavailable)


def _identity(value, name):
    if not isinstance(value, str) or not 1 <= len(value) <= 128 or any(ord(c) < 32 for c in value):
        raise ValueError(f"{name} must be a bounded nonempty identity")


class DurableSDKOperations:
    """Persist before sending; reattach before considering any explicit resend.

    ``receiver_id`` must name one stable receiver/principal binding. Use a
    separate cache file from the proof journal and receiver operation database.
    The caller owns the AsyncClient and chooses each intent explicitly. This
    transport helper does not enforce mathematical execution preconditions;
    coordinator mutation wiring stays blocked until the receiver supports them.
    """

    def __init__(self, client, path, *, receiver_id, workspace_id, proof_ids,
                 poll_interval=.1, max_observation_bytes=16 * 1024 * 1024):
        self._sdk = _sdk()
        if not isinstance(client, self._sdk.Client):
            raise TypeError("client must be persistent_atp_client.AsyncClient")
        _identity(receiver_id, "receiver_id")
        import re
        if not isinstance(workspace_id, str) or re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}", workspace_id) is None:
            raise ValueError("workspace_id must be a registered alias")
        if isinstance(proof_ids, (str, bytes)):
            raise ValueError("proof_ids must be a sequence")
        registered = tuple(proof_ids)
        for proof in registered:
            _identity(proof, "proof_id")
        if len(set(registered)) != len(registered):
            raise ValueError("register each proof ID once")
        if type(poll_interval) not in (int, float) or not math.isfinite(poll_interval) or poll_interval <= 0:
            raise ValueError("poll_interval must be finite and positive")
        if type(max_observation_bytes) is not int or max_observation_bytes < 1:
            raise ValueError("max_observation_bytes must be a positive integer")
        self.client, self.receiver_id, self.workspace_id = client, receiver_id, workspace_id
        self.proof_ids = registered
        if str(path) == ":memory:":
            raise ValueError("durable operations require a persistent file")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.poll_interval, self.max_observation_bytes = poll_interval, max_observation_bytes
        with self._db() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("CREATE TABLE IF NOT EXISTS coordinator_sdk_cache_meta (id INTEGER PRIMARY KEY CHECK(id=1), version INTEGER NOT NULL)")
            db.execute("INSERT OR IGNORE INTO coordinator_sdk_cache_meta VALUES (1,1)")
            if db.execute("SELECT version FROM coordinator_sdk_cache_meta WHERE id=1").fetchone()[0] != 1:
                raise ValueError("unsupported SDK client cache schema; no automatic rewrite")
            db.execute("""CREATE TABLE IF NOT EXISTS coordinator_sdk_intents (
                receiver_id TEXT NOT NULL, workspace_id TEXT NOT NULL, intent_id TEXT NOT NULL,
                request_id TEXT NOT NULL, proof_id TEXT NOT NULL, fingerprint TEXT NOT NULL,
                request_json BLOB NOT NULL, state TEXT NOT NULL,
                operation_json TEXT, admission_json TEXT, observation_json BLOB,
                PRIMARY KEY(receiver_id, workspace_id, intent_id),
                UNIQUE(receiver_id, workspace_id, request_id),
                CHECK(state IN ('prepared', 'uncertain', 'admitted', 'rejected', 'terminal'))
            )""")
            db.commit()

    @contextmanager
    def _db(self, *, write=False):
        db = sqlite3.connect(self.path, timeout=1)
        db.row_factory = sqlite3.Row
        try:
            db.execute("PRAGMA synchronous=FULL")
            if write:
                db.execute("BEGIN IMMEDIATE")
            yield db
            if write:
                db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def _key(self, intent_id):
        _identity(intent_id, "intent_id")
        return self.receiver_id, self.workspace_id, intent_id

    def _record(self, row):
        if row is None:
            raise KeyError("intent is not in this receiver/workspace cache")
        record = IntentRecord(**dict(row))
        try:
            request = self._sdk.Request.from_dict(json.loads(record.request_json))
        except (ValueError, TypeError, KeyError) as exc:
            raise IntentConflict("cached request cannot be decoded") from exc
        if (self._sdk.fingerprint(request) != record.fingerprint
                or self._sdk.canonical(request) != record.request_json
                or request.workspace_id != record.workspace_id or request.request_id != record.request_id
                or getattr(request.action, "proof_id", None) != record.proof_id):
            raise IntentConflict("cached request identity or bytes changed")
        if record.proof_id not in self.proof_ids:
            raise IntentConflict("cached proof is not registered")
        return record

    def _load(self, db, intent_id):
        return db.execute("SELECT * FROM coordinator_sdk_intents WHERE receiver_id=? AND workspace_id=? AND intent_id=?",
                          self._key(intent_id)).fetchone()

    def get(self, intent_id) -> IntentRecord:
        with self._db() as db:
            return self._record(self._load(db, intent_id))

    def prepare(self, intent_id, action, *, preconditions=None, request_id=None, deadline=None) -> IntentRecord:
        """Create an intent once; repeated preparation must preserve every byte."""
        key = self._key(intent_id)
        if request_id is not None:
            _identity(request_id, "request_id")
        if getattr(action, "proof_id", None) not in self.proof_ids:
            raise ValueError("action proof is not registered")
        with self._db(write=True) as db:
            row = self._load(db, intent_id)
            prior = self._record(row) if row else None
            request = self.client.prepare(action, workspace_id=self.workspace_id, preconditions=preconditions,
                request_id=request_id or (prior.request_id if prior else None), deadline=deadline)
            body = self._sdk.canonical(request)
            if len(body) > 65_536:
                raise ValueError("prepared request exceeds the receiver body bound")
            if prior:
                if body != prior.request_json:
                    raise IntentConflict("intent already names a different prepared request")
                return prior
            try:
                db.execute("""INSERT INTO coordinator_sdk_intents
                    (receiver_id,workspace_id,intent_id,request_id,proof_id,fingerprint,request_json,state)
                    VALUES (?,?,?,?,?,?,?,'prepared')""",
                    (*key, request.request_id, action.proof_id, self._sdk.fingerprint(request), body))
            except sqlite3.IntegrityError as exc:
                raise IntentConflict("request ID already belongs to another intent") from exc
            return self._record(self._load(db, intent_id))

    def _claim_first_send(self, intent_id):
        with self._db(write=True) as db:
            self._record(self._load(db, intent_id))
            return db.execute("""UPDATE coordinator_sdk_intents SET state='uncertain'
                WHERE receiver_id=? AND workspace_id=? AND intent_id=? AND state='prepared'""",
                self._key(intent_id)).rowcount == 1

    def _bind(self, intent_id, ref, *, admission=None):
        if ref is not None and ref.workspace_id != self.workspace_id:
            raise self._sdk.ProtocolError("operation belongs to another workspace")
        with self._db(write=True) as db:
            record = self._record(self._load(db, intent_id))
            encoded = json.dumps(ref.to_dict(), sort_keys=True) if ref else None
            if record.operation_json is not None and record.operation_json != encoded:
                raise self._sdk.ProtocolError("operation identity changed for the same prepared request")
            if record.state == "rejected" and ref is not None:
                raise self._sdk.ProtocolError("rejected admission changed to accepted")
            state = "admitted" if ref else "rejected"
            if record.state == "terminal":
                state = "terminal"
            db.execute("""UPDATE coordinator_sdk_intents SET state=?, operation_json=?, admission_json=?
                WHERE receiver_id=? AND workspace_id=? AND intent_id=?""",
                (state, encoded, self._sdk.canonical(admission).decode() if admission else record.admission_json,
                 *self._key(intent_id)))

    def _admission(self, record, receipt):
        if (receipt.request_id != record.request_id or receipt.fingerprint != record.fingerprint
                or any(ref is not None and ref.workspace_id != record.workspace_id
                       for ref in (receipt.operation_ref, receipt.coalesced_to))):
            raise self._sdk.ProtocolError("admission does not match the durable prepared request")
        self._bind(record.intent_id, receipt.operation_ref, admission=receipt)
        if not receipt.accepted:
            raise self._sdk.Rejected(receipt)
        return receipt.operation_ref

    def _observe(self, intent_id, snapshot):
        payload = self._sdk.canonical(snapshot)
        if len(payload) > self.max_observation_bytes:
            raise ValueError("operation observation exceeds the configured byte bound")
        with self._db(write=True) as db:
            record = self._record(self._load(db, intent_id))
            if (snapshot.ref.to_dict() != json.loads(record.operation_json)
                    or snapshot.request_id != record.request_id or snapshot.proof_id != record.proof_id):
                raise self._sdk.ProtocolError("operation observation belongs to another durable request or proof")
            if record.observation_json is not None:
                previous = json.loads(record.observation_json)
                # The receiver reports current project metadata even when the
                # operation itself is unchanged. Keep operation identity/results
                # immutable, and check the project chain independently below.
                current = json.loads(payload)
                live_fields = {"project_revision", "journal_hash", "projection_watermark"}
                operation_changed = ({k: v for k, v in current.items() if k not in live_fields}
                                     != {k: v for k, v in previous.items() if k not in live_fields})
                if (snapshot.observation_version < previous["observation_version"]
                        or ((snapshot.observation_version == previous["observation_version"]
                             or previous["lifecycle"] in self._sdk.terminal)
                            and operation_changed)):
                    raise self._sdk.ProtocolError("operation observation regressed or changed without a valid transition")
                if previous["project_revision"] is not None and snapshot.project_revision is not None:
                    if (snapshot.project_revision < previous["project_revision"]
                            or (snapshot.project_revision == previous["project_revision"]
                                and previous["journal_hash"] is not None and snapshot.journal_hash is not None
                                and snapshot.journal_hash != previous["journal_hash"])):
                        raise self._sdk.ProtocolError("operation journal revision or identity regressed")
            db.execute("""UPDATE coordinator_sdk_intents SET state=?, observation_json=?
                WHERE receiver_id=? AND workspace_id=? AND intent_id=?""",
                ("terminal" if snapshot.lifecycle in self._sdk.terminal else "admitted",
                 payload, *self._key(intent_id)))

    async def advance(self, intent_id, *, timeout=10, retry_same_request=False):
        """Send once or reconcile/observe; resending an uncertain body is opt-in.

        A new timeout bounds this observation, not receiver execution. Even a
        cached terminal operation is re-read from the receiver on reattachment.
        """
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        if type(retry_same_request) is not bool:
            raise ValueError("retry_same_request must be a boolean")
        record = self.get(intent_id)
        try:
            async with asyncio.timeout(timeout):
                ref = self._sdk.Ref.from_dict(json.loads(record.operation_json)) if record.operation_json else None
                if record.state == "rejected":
                    raw = json.loads(record.admission_json)
                    for name in ("operation_ref", "coalesced_to"):
                        if raw[name] is not None:
                            raw[name] = self._sdk.Ref.from_dict(raw[name])
                    self._admission(record, self._sdk.Admission(**raw))
                if ref is None:
                    send = self._claim_first_send(intent_id)
                    if not send:
                        receipt = await self.client.lookup_request(workspace_id=self.workspace_id,
                                                                   request_id=record.request_id)
                        if receipt is not None:
                            ref = self._admission(record, receipt)
                        elif not retry_same_request:
                            raise OperationPending("admission is unresolved; reconcile or explicitly retry the same request",
                                                   self.get(intent_id).checkpoint_ref())
                    if ref is None:
                        request = self._sdk.Request.from_dict(json.loads(record.request_json))
                        try:
                            handle = await self.client.submit(request)
                        except self._sdk.Rejected as exc:
                            self._admission(record, exc.admission)
                            raise
                        ref = handle.ref
                        self._bind(intent_id, ref)
                handle = self.client.attach(ref)
                while True:
                    snapshot = await handle.snapshot()
                    self._observe(intent_id, snapshot)
                    if snapshot.lifecycle in self._sdk.terminal:
                        return snapshot
                    await asyncio.sleep(self.poll_interval)
        except (TimeoutError, ConnectionError, self._sdk.Uncertain, self._sdk.Unavailable) as exc:
            raise OperationPending("delivery or observation remains unresolved; receiver work may continue",
                                   self.get(intent_id).checkpoint_ref()) from exc

    def restore(self, checkpoint_ref) -> IntentRecord:
        """Resolve plain checkpoint identities using the durable cache, not graph state."""
        expected = {"receiver_id", "workspace_id", "intent_id", "request_id", "fingerprint", "operation_ref"}
        if not isinstance(checkpoint_ref, dict) or set(checkpoint_ref) != expected:
            raise IntentConflict("invalid operation checkpoint reference")
        record = self.get(checkpoint_ref["intent_id"])
        current = record.checkpoint_ref()
        if (any(checkpoint_ref[k] != current[k] for k in expected - {"operation_ref"})
                or (checkpoint_ref["operation_ref"] is not None
                    and checkpoint_ref["operation_ref"] != current["operation_ref"])):
            raise IntentConflict("checkpoint belongs to another durable intent or receiver")
        return record
