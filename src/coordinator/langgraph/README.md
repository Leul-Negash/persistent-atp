# Coordinator

This directory contains the LangGraph proof-cycle coordinator: the outer loop that runs rounds of dispatch cycles over one or more proofs, tracks their progress from committed state, and applies the stagnation policy of the architecture (Sections 10.2–10.5, 12.1). It is a LangGraph prototype of the coordinator role, built on the scheduler, dispatcher and commit gate; it replaces none of them.

## What a round does

```
plan_round --Send per active proof--> work --> assess --> plan_round | END
```

1. **`plan_round`** opens the round and fans the active proofs out with LangGraph's `Send`.
2. **`work`** runs one proof's portfolio (Section 8.6): one `run_cycle` per slot that has work — *exploit* (formal search), *explore* (research moves, including the coordinator's escalations) and *critic* (provisional claims). Proofs run in parallel; the slots of one proof run in turn, because a proposal must be based on the head it commits onto.
3. **`assess`** joins the round. For each proof it re-reads committed state, updates the stall count, raises the escalations the stagnation policy calls for, commits them, and records milestone messages (Section 10.7).
4. The run ends when every proof is solved, when no proof can move without a replay or a human (`halted`), or when the round budget runs out. An unknown worker outcome sets that proof to `waiting-on-worker` and requires reconciliation before resuming.

## The stagnation policy

`policy.classify` applies the six rules of Section 10.4 in order:

| Rule | Read from committed state | Escalation |
|---|---|---|
| 1 | a critic refuted a claim that is still provisional | `repair`, at the critic's `earliest_gap` |
| 2 | `non_progress_rounds` rounds without verified gain | `five-mechanisms` (Section 10.5) |
| 3 | one obstruction kind repeated on a declaration's runs | `diversify-formal` |
| 4 | a stagnated run missing a lemma, or several runs at one residual | `bridge-lemma` (or a representation change) |
| 5 | a reviewed alignment or an obstruction doubting the statement | `alignment-review`, which holds rules 3 and 4 for that declaration |
| 6 | an empty frontier, or `global_stall_rounds` rounds stalled | `global-audit` |

Progress tracks settled claims, closed formal states and replayed declarations. A critic-accepted claim is a research milestone, not formal proof. A certificate waiting for replay pauses the stall rules instead of counting as stagnation (Section 10.3). Thresholds are `StagnationPolicy` configuration.

Completion requires at least one registered `HAS_TARGET` claim and every registered target at the gate-protected `lean-verified` status. Replayed declarations alone do not solve the proof. The gate is responsible for enforcing exact alignment and independent replay when a claim is promoted.

## Authority

The coordinator decides *that* something must happen, never the mathematics of it. Each escalation becomes a queued `ResearchMove` under the coordinator's own `ResearchState`, committed through the gate under a proof-wide lease; a worker takes the move in a later round like any other. The gate trusts the `coordinator` worker class with any label, so the limit is kept in `escalate.py` and checked by the tests: the coordinator writes no claim status, formal state, certificate or obstruction (Section 3.2, invariant 7).

`mechanisms.py` fixes what a worker's five-mechanism answer must state and checks it is complete and diverse (one technique tag per mechanism) before it becomes queued moves.

The portfolio applies this validator to actual `five-mechanisms` worker answers before building a proposal. Answers may use `mechanisms` or `moves`; each entry must supply the complete `MECHANISM_FIELDS` contract. Invalid answers produce `invalid-worker-result`, release that completed dispatch's lease, and commit nothing. Required prerequisite claims must belong to the leased proof.

## The journal is the authority

The graph state is a cache (Section 2.3, invariant 6). It holds round counters, stall counts, reports and messages — never proof state. Every decision is re-read from the journal, and an escalation already on record is not raised again (once per target for the local rules, once per progress level for the stall rules), so a coordinator restarted on the same journal with an empty view makes the same decisions. A LangGraph checkpointer may be passed to resume a run; losing its checkpoints loses no proof work.

## File Summary

- **`workspace.py`**: `ProofWorkspace` (a proof's store factory, view, dispatcher and adapters) and `project`, which catches the view up to the journal head. Each caller opens its own store connection.
- **`progress.py`**: Reads a proof's position from committed state: progress counts, residuals, misaligned declarations, refuted reviews and escalations on record.
- **`policy.py`**: The stagnation policy as an ordered rule table.
- **`escalate.py`**: Turns escalations into the coordinator's one kind of proposal.
- **`portfolio.py`**: The exploit/explore/critic slots and `SlotScheduler`, which pins a worker class to its frontier category.
- **`rounds.py`**: One round for one proof — `work` and `assess` — without LangGraph.
- **`mechanisms.py`**: The five-mechanism answer contract.
- **`graph.py`**: The LangGraph `StateGraph` and `Coordinator`.
- **`runtime.py`**: The registered synchronous calling facade used by the graph, with consistent reads, category-selected slots, and revision-bound escalation batches.

## Runtime calls

The trusted host registers its existing `ProofWorkspace` objects. Registration supplies the store factory, dispatcher and adapters; individual runtime calls cannot select an actor, worker implementation, database path or fencing token. Each store factory call must return a new caller-owned connection, which the coordinator closes after use.

```python
from coordinator.langgraph import CoordinatorRuntime, Escalation, StaleAssessment
from coordinator.langgraph.rounds import new_track

# workspace is a ProofWorkspace configured by the trusted service host.
runtime = CoordinatorRuntime([workspace])
proof_id = workspace.proof_id

state = runtime.read_coordination(proof_id)
print(state.revision, state.progress.progress)
print(state.to_dict())

slot = runtime.run_slot(
    proof_id, "critic-task", expected_revision=state.revision
)
print(slot["outcome"], slot["accepted"], slot["rejections"])

# Refresh before making a snapshot-dependent decision.
state = runtime.read_coordination(proof_id)
receipt = runtime.queue_escalations(
    proof_id,
    [Escalation("global-audit", proof_id, "Audit the current proof")],
    expected_revision=state.revision,
    round_number=1,
)
print(receipt["outcome"], receipt["lease_events"])
print(runtime.read_escalations(proof_id))

# Alternatively run the existing work/assess functions as one convenience call.
round_result = runtime.run_round(proof_id, 2, new_track())
```

`read_coordination` returns all scoped nodes, edges, frontier entries, progress and recorded escalations at one journal revision/hash. Its records are deeply immutable and `to_dict()` returns JSON-compatible values. A moving journal is retried at most three times. Reads do not advance the journal revision.

`run_slot` supports `formal-run`, `research-move` and `critic-task`. It serializes calls within a registered proof while allowing different proofs to execute concurrently. Each call is an intentional new cycle; it is not an idempotent remote submission. Its outcome distinguishes `accepted`, `rejected`, `idle`, `waiting-on-lease`, `stale-assessment`, `invalid-worker-result`, and `execution-unknown`. A successful function return does not imply formal closure.

`queue_escalations` submits at most 100 complete escalations as one gate proposal. It preserves target IDs, optional claim links, details, progress basis and round. Duplicate policy intents are removed using committed coordinator records; a restarted runtime with a fresh revision therefore does not recreate them. Stale assessment revisions raise `StaleAssessment` before mutation or produce a gate rejection if a race occurs during submission. Refresh and reassess rather than silently rebasing a previous decision.

The existing proof-wide escalation lease still supersedes active dispatches. The receipt includes observed lease audit events so supersession is visible, without exposing fencing tokens. This is not a non-preempting queue. The audit is separate from the proof hash chain.

## Optional SDK coordination reads

`SDKBackend` reads a complete proof graph and frontier through the standalone
`persistent-atp-client` SDK. It negotiates coordination reads under contract
`2.1`, checks every page against one journal revision and hash, and rejects
incomplete or inconsistent snapshots. Reads have configurable page, record,
byte, retry and time bounds. Returned records are immutable copies; receiver
component modes, including scripted modes, remain visible on the snapshot.

Install the SDK from its separate checkout first, then the optional binding:

```bash
uv pip install --python .venv/bin/python \
  -e '/path/to/sdk-checkout/packages/persistent-atp-client[http]'
uv pip install --python .venv/bin/python -e '.[langgraph,sdk]'
```

With a compatible receiver and an existing proof, read its committed state:

```python
import asyncio
import os

from persistent_atp_client import AsyncClient
from persistent_atp_client.adapters.http import HTTPReceiverAdapter
from coordinator.langgraph.sdk_backend import SDKBackend

async def main():
    port = HTTPReceiverAdapter(
        os.environ["PERSISTENT_ATP_URL"],
        os.environ["PERSISTENT_ATP_TOKEN"],
    )
    async with AsyncClient(port) as client:
        backend = SDKBackend(client, workspace_id="reference", proof_ids=("existing-proof",))
        state = await backend.aread_coordination("existing-proof")
        print(state.revision, state.progress.solved, state.component_mode)

asyncio.run(main())
```

Synchronous graph worker threads can use `backend.read_coordination(proof_id)`.
The caller owns the client's lifetime and credentials. The current SDK HTTP
adapter supports separate event loops; a custom local port must provide the
same property when shared across synchronous worker threads.

This binding currently supports **reads only**. `run_slot` and
`queue_escalations` raise `UnsupportedCapability` before submitting anything.
Running coordinator rounds remotely still requires durable request
reconciliation, execution-time revision checks, and a structured atomic
escalation contract. The SDK's generic guidance action does not preserve all
coordinator escalation metadata, so it is not used as a substitute.

The opt-in HTTP test reads an existing idle proof without submitting work.
Keep other workers paused on that proof while comparing the two reads:

```bash
export PERSISTENT_ATP_COORDINATION_TEST_URL=http://127.0.0.1:8000
# Set PERSISTENT_ATP_COORDINATION_TEST_TOKEN to your receiver token.
export PERSISTENT_ATP_COORDINATION_TEST_WORKSPACE=reference
export PERSISTENT_ATP_COORDINATION_TEST_PROOF=existing-proof
.venv/bin/python -m pytest -q -rs \
  src/coordinator/langgraph/tests/test_sdk_backend_http.py
```

The test explicitly skips if its optional SDK/HTTP dependency or receiver
configuration is missing. A reference receiver establishes read interoperability
with its recorded component mode; it does not establish production persistence
or independent Lean replay.

The facade and LangGraph graph reuse the original round, portfolio, scheduler, dispatcher and gate implementations. There is no `persistent-atp-client` dependency in this stage.

## Running

The core needs only the commit gate and the scheduler. The graph needs the optional extra:

```
uv sync --extra langgraph
python -m pytest src/coordinator/langgraph/tests
```

`test_graph.py` is skipped when LangGraph is not installed.

Verification on Python 3.13 with LangGraph 1.2.19:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q \
  src/coordinator/langgraph/tests
# 63 passed, including all five LangGraph graph tests

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q --tb=short
# 475 passed, 15 failed, 85 subtests passed; no skipped tests
```

Before changes, the same branch had 451 passed, 15 failed, five skipped graph tests and 85 subtests passed. The remaining 15 failures are PostgreSQL artifact tests: the configured `user` cannot authenticate at `127.0.0.1:5432`. Coordinator integration tests use a real gate and SQLite journal, scripted workers, and `FakeFormalATP`. Completion fixtures contain explicitly synthetic bound replay records; no real Lean execution or independent replay was tested.

## Not yet covered

- Two cycles on the same proof cannot overlap: the gate requires a proposal's base revision to equal the head, so a second commit on one proof would be rejected as stale. Parallelism is across proofs until the gate can rebase a disjoint proposal.
- Workers are scripted in the tests. A real LLM worker for the five-mechanism request, and routing `alignment-review` moves to the alignment reviewer instead of the research slot, are next steps.
- Replay is not driven by the coordinator; a proof whose certificate awaits replay is parked.
- Rules 3 and 4 read obstructions directly; the per-run routing of Section 8.7 (`routing.evaluate_run`, on `feat/scheduler-routing-concurrency`) is not on this branch yet and can feed them once it is.
- Calls are synchronous. A worker can block its caller; a lease TTL fences a late commit but does not terminate the worker. Unknown execution retains its lease and stops automatic rounds until the host reconciles the outcome.
- There is no remote admission ledger, durable cycle request ID, automatic mutation retry, cancellation or package integration in this facade. Those require a later receiver integration; the local runtime does not claim exactly-once cycle execution across process death.
- Reads currently use `MemoryView` enumeration. MORK parity and a non-preempting lease policy have not been implemented by this change.
