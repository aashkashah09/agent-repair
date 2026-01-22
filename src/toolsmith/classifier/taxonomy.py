"""Failure taxonomy.

Six mutually exclusive classes. The first five name a way an interface can send
an agent wrong; the sixth is the escape hatch for failures that are the agent's
own, and it exists so the optimizer is not handed a tool to rewrite every time
the model simply reasons badly. A trace gets exactly one label -- the earliest
point at which the episode was already lost.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FailureClass:
    key: str
    label: str
    definition: str
    tool_attributable: bool


WRONG_TOOL = FailureClass(
    key="wrong_tool_selection",
    label="Wrong tool selection",
    definition=(
        "The agent called a tool that cannot accomplish what it was trying to do, or "
        "failed to call the tool that could, when the information needed to choose "
        "correctly was not available from the tool descriptions."
    ),
    tool_attributable=True,
)

MALFORMED_ARGUMENTS = FailureClass(
    key="malformed_arguments",
    label="Malformed arguments",
    definition=(
        "The agent chose the right tool but constructed arguments the tool rejected or "
        "silently mishandled: an unlisted enum value, a wrong type, a mis-formatted "
        "timestamp, a missing conditionally-required field."
    ),
    tool_attributable=True,
)

LOOP = FailureClass(
    key="loop",
    label="Retry loop",
    definition=(
        "The agent repeated a call, or cycled through a short sequence of calls, without "
        "the state or the arguments changing in a way that could alter the outcome, until "
        "the episode ran out of budget."
    ),
    tool_attributable=True,
)

CONTEXT_LOSS = FailureClass(
    key="context_loss",
    label="Context loss",
    definition=(
        "The agent acted on stale or discarded information: a value a tool returned "
        "earlier, a constraint the customer stated, or a result the agent had already "
        "obtained and then contradicted."
    ),
    tool_attributable=True,
)

PREMATURE_TERMINATION = FailureClass(
    key="premature_termination",
    label="Premature termination",
    definition=(
        "The agent stopped and reported success, or reported that nothing could be done, "
        "while the task was still achievable with the tools it had -- typically after a "
        "return it read as final when it was not."
    ),
    tool_attributable=True,
)

AGENT_ATTRIBUTABLE = FailureClass(
    key="agent_attributable",
    label="Agent-attributable",
    definition=(
        "The failure does not trace to any tool interface. The schemas gave the agent "
        "what it needed and it still reasoned or communicated its way to a wrong result."
    ),
    tool_attributable=False,
)

CLASSES = (
    WRONG_TOOL,
    MALFORMED_ARGUMENTS,
    LOOP,
    CONTEXT_LOSS,
    PREMATURE_TERMINATION,
    AGENT_ATTRIBUTABLE,
)

BY_KEY = {failure.key: failure for failure in CLASSES}
KEYS = tuple(failure.key for failure in CLASSES)


def is_valid(key: str) -> bool:
    return key in BY_KEY


def tool_attributable(key: str) -> bool:
    return BY_KEY[key].tool_attributable


def definitions_block() -> str:
    """The taxonomy as the judge sees it."""
    return "\n".join(f"- {f.key}: {f.definition}" for f in CLASSES)
