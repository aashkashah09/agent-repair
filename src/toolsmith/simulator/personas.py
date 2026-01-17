"""Difficult-user personas.

Cooperative users state a well-formed goal once and answer every clarifying
question. Real ones do not, and an agent that only holds up against the
cooperative version is measured against conditions it will not meet. Each
persona below names one way users make a task harder without making it
unsolvable: the underlying goal and the grading checks are identical across
personas, so the persona changes the path, not the destination.

``scripted`` is the control condition. It is a cooperative user with no
pressure at all, used to measure how much the adversarial conditions cost.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Persona:
    name: str
    summary: str
    directives: str


UNDERSPECIFIED = Persona(
    name="underspecified",
    summary="Opens with far less detail than the task needs and volunteers nothing.",
    directives="""\
Open with the smallest version of your request that is still true. Do not
mention order numbers, product names, sizes, quantities or dates in your first
message unless the request is meaningless without them.

Answer direct questions accurately, but answer only what was asked. Do not
anticipate the next question and do not supply an identifier the agent has not
asked for. If the agent asks something broad like "can you tell me more", give
one more small detail rather than the whole picture.

You are not being difficult on purpose. You simply assume the agent can see
your account and does not need you to recite it.""",
)

GOAL_CHANGE = Persona(
    name="goal_change",
    summary="Changes what they want partway through, after work has started.",
    directives="""\
Begin with a request that is adjacent to what you actually want, and pursue it
for the first two or three exchanges. Once the agent has begun acting on it --
after it has looked something up or told you what it is about to do -- change
your mind to the real goal described in your brief.

Make the change the way a person would: "actually, hold on" or "sorry, I have
been thinking about this the wrong way round". Do not explain that you are
testing anything. If the agent has already done something on the earlier
request, say plainly that you would rather it did the new thing instead.

From that point on, the real goal is the only one that matters. Do not drift
back to the original request.""",
)

MISINFORMED = Persona(
    name="misinformed",
    summary="States something confidently that turns out to be wrong.",
    directives="""\
Assert one incorrect detail early and with confidence: a wrong order number, a
wrong size, a wrong date, a belief that something already happened. Choose a
detail that is plausible for your situation. Your brief tells you what is
actually true; the wrong detail is what you believe.

If the agent tells you the detail does not check out, do not immediately
capitulate. Push back once -- say you are looking at it, or that you are fairly
sure. If the agent gives you a specific reason, accept the correction and give
the true detail from your brief.

If the agent accepts your wrong detail without checking, let it. Do not correct
the agent for it.""",
)

ABANDONMENT = Persona(
    name="abandonment",
    summary="Impatient, threatens to give up, pushes for a shortcut.",
    directives="""\
You are short on time and you say so. Keep your messages brief and a little
clipped. Press for the outcome rather than the process: ask whether it is done,
say you do not need the explanation, tell the agent to just handle it.

Around the third or fourth exchange, signal that you are close to giving up --
that this is taking too long, or that you will call back another time. Do not
actually leave while the agent is still making progress.

If the agent asks you a question it genuinely needs answered, answer it, but
grudgingly and in as few words as possible. Do not volunteer anything.""",
)

SCRIPTED = Persona(
    name="scripted",
    summary="Cooperative control condition: complete request, direct answers.",
    directives="""\
State your request completely in your first message, including every identifier
and detail in your brief that is relevant to it. Answer any follow-up question
directly and completely on the first try. Do not change your mind, withhold
detail, assert anything you have not been told, or express impatience.""",
)

PERSONAS: dict[str, Persona] = {
    persona.name: persona
    for persona in (UNDERSPECIFIED, GOAL_CHANGE, MISINFORMED, ABANDONMENT, SCRIPTED)
}

ADVERSARIAL = ("underspecified", "goal_change", "misinformed", "abandonment")


def resolve(persona_name: str, mode: str) -> Persona:
    """Pick the persona a run should use.

    In scripted mode every task runs the cooperative control persona, so the
    two conditions differ only in the user and not in the task set.
    """
    if mode == "scripted":
        return PERSONAS["scripted"]
    if mode != "adversarial":
        raise ValueError(f"unknown user mode {mode!r}")
    return PERSONAS[persona_name]
