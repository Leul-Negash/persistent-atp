"""The five-mechanism operator's answer contract (Section 10.5).

A `five-mechanisms` escalation asks a research worker for five genuinely
different ways past the current gap. The coordinator does not write them --
it has no mathematical opinion -- but it fixes what an answer must state and
refuses one that is incomplete or not diverse, before any of it becomes a
queued move. Diversity is checked by technique tags: two mechanisms from the
same technique family are one idea twice.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from commit_gate.state import ReadView

__all__ = [
    "MECHANISM_COUNT",
    "MECHANISM_FIELDS",
    "STANCES",
    "check_mechanisms",
    "mechanism_moves",
]

MECHANISM_COUNT = 5

MECHANISM_FIELDS = (
    "operator",              # cognitive operator and technique family
    "technique_tags",
    "idea",                  # the central idea
    "bridge",                # the exact bridge lemma or reduction
    "why",                   # why it would close or clarify the gap
    "prerequisites",         # prerequisite claim IDs
    "formal_manifestation",  # what it is expected to look like in Lean
    "falsification_test",    # the earliest inexpensive test that could kill it
    "information_if_false",
    "promise",
    "cost",
    "novelty",
    "risk",
    "relation",              # relation to existing branches
    "stance",                # proves, bypasses, generalizes or reframes
    "first_step",            # first executable research or formal step
)

SCORES = ("promise", "cost", "novelty", "risk")
STANCES = frozenset({"proves", "bypasses", "generalizes", "reframes"})


def check_mechanisms(
    mechanisms: Sequence[Mapping[str, Any]], view: ReadView | None = None
) -> list[str]:
    """Every way `mechanisms` fails the contract; empty when it holds."""
    problems: list[str] = []
    if not isinstance(mechanisms, (list, tuple)):
        return ["mechanisms must be a list or tuple"]
    if len(mechanisms) != MECHANISM_COUNT:
        problems.append(f"expected {MECHANISM_COUNT} mechanisms, got {len(mechanisms)}")
    seen_tags: dict[str, int] = {}
    for position, mechanism in enumerate(mechanisms):
        name = f"mechanism {position + 1}"
        if not isinstance(mechanism, Mapping):
            problems.append(f"{name} must be a mapping")
            continue
        missing = [field for field in MECHANISM_FIELDS if field not in mechanism]
        if missing:
            problems.append(f"{name} is missing {', '.join(missing)}")
            continue
        if not isinstance(mechanism["stance"], str) or mechanism["stance"] not in STANCES:
            problems.append(f"{name} has stance {mechanism['stance']!r}")
        for score in SCORES:
            value = mechanism[score]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
                problems.append(f"{name} scores {score} outside [0, 1]")
        tags = mechanism["technique_tags"]
        if not isinstance(tags, (list, tuple)) or not tags or any(
            not isinstance(tag, str) or not tag.strip() for tag in tags
        ):
            problems.append(f"{name} names no technique tag")
            tags = []
        for tag in tags:
            if tag in seen_tags:
                problems.append(
                    f"{name} repeats technique {tag!r} from mechanism {seen_tags[tag] + 1}"
                )
            else:
                seen_tags[tag] = position
        prerequisites = mechanism["prerequisites"]
        if not isinstance(prerequisites, (list, tuple)) or any(
            not isinstance(claim_id, str) for claim_id in prerequisites
        ):
            problems.append(f"{name} prerequisites must be claim IDs")
            prerequisites = []
        for field in (name for name in MECHANISM_FIELDS
                      if name not in SCORES + ("technique_tags", "prerequisites", "stance")):
            if not isinstance(mechanism[field], str) or not mechanism[field].strip():
                problems.append(f"{name} has empty or invalid {field}")
        if view is not None:
            for claim_id in prerequisites:
                record = view.node(claim_id)
                if record is None or record.label != "Claim":
                    problems.append(f"{name} depends on {claim_id!r}, which is not a claim")
    return problems


class InvalidMechanisms(ValueError):
    """A completed worker answer failed validation before proposal creation."""


class MechanismDispatcher:
    """Enforce the answer contract on the existing dispatcher boundary."""

    def __init__(self, dispatcher, view):
        self.dispatcher, self.view = dispatcher, view

    def run(self, lease, context):
        result = dict(self.dispatcher.run(lease, context))
        selected = self.view.node(lease.selected_move_id)
        if (selected is not None and selected.label == "ResearchMove"
                and selected.fields.get("escalation") == "five-mechanisms"):
            answer = result.get("mechanisms", result.get("moves"))
            problems = check_mechanisms(answer, self.view)
            if not problems and any(not claim_id.startswith(lease.proof_id + "/")
                    for mechanism in answer for claim_id in mechanism["prerequisites"]):
                problems.append("mechanism prerequisites must belong to the leased proof")
            if problems:
                raise InvalidMechanisms("; ".join(problems))
            result["moves"] = mechanism_moves(answer)
        return result


def mechanism_moves(mechanisms: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The research result's ``moves`` for a checked answer: one per mechanism."""
    return [
        {
            "detail": mechanism["first_step"],
            "mechanism": mechanism["idea"],
            **{field: mechanism[field] for field in MECHANISM_FIELDS},
        }
        for mechanism in mechanisms
    ]
