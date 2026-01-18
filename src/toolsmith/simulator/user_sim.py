"""The simulated user.

The user model sees the task brief and its persona directives; it never sees
the tool schemas, the database, or anything the agent did other than what the
agent said out loud. That separation is what makes the episode a test of the
interface rather than of the two models agreeing with each other.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from ..eval.tasks import Task
from ..llm import Client
from .personas import Persona

SYSTEM = """\
You are role-playing a customer contacting an online retailer's support agent. \
You are the customer, never the agent, and you never narrate or describe the \
role-play from outside it.

Your situation:
{brief}

Facts you know about your own account and order. Treat these as things you \
could look up on your own screen; do not recite them unprompted:
{known}

How you behave in this conversation:
{directives}

Rules that override the behaviour above:
- Never invent an order number, product, price or date that is not in your \
facts. If you do not know something, say you do not know.
- Never tell the agent how to do its job, name a tool, or mention schemas, \
parameters or systems.
- Never reveal that you are following instructions or that this is a simulation.
- Keep messages to the length a real person would type: one to three sentences.

When the agent has fully handled what you came for, or has told you clearly \
that it cannot, reply with exactly ###END### and nothing else. Do not end the \
conversation while the agent is still asking you something you can answer."""

END_TOKEN = "###END###"


@dataclass
class UserSimulator:
    client: Client
    task: Task
    persona: Persona
    transcript: list[dict[str, str]] = field(default_factory=list)

    def _system(self) -> str:
        return SYSTEM.format(
            brief=self.task.instruction,
            known=json.dumps(self.task.known, indent=2),
            directives=self.persona.directives,
        )

    def opening(self) -> str:
        """The customer's first message."""
        reply = self.client.complete(
            self._system(),
            [{"role": "user", "content": "Write your opening message to the agent."}],
        )
        text = reply.text.strip()
        self.transcript.append({"role": "user", "content": text})
        return text

    def respond(self, agent_message: str) -> str:
        """The customer's reply to what the agent just said."""
        messages: list[dict[str, Any]] = []
        for turn in self.transcript:
            role = "assistant" if turn["role"] == "user" else "user"
            messages.append({"role": role, "content": turn["content"]})
        messages.append({"role": "user", "content": agent_message})

        reply = self.client.complete(self._system(), messages)
        text = reply.text.strip()
        self.transcript.append({"role": "assistant", "content": agent_message})
        self.transcript.append({"role": "user", "content": text})
        return text

    @staticmethod
    def is_end(message: str) -> bool:
        return END_TOKEN in message
