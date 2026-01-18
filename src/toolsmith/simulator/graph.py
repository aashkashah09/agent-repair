"""The episode graph.

One episode is a conversation between the simulated user and the agent, with
the agent's tool calls executed against a fresh copy of the database. The graph
alternates user and agent nodes and halts when the user signals it is done, the
turn budget is spent, or the agent stops producing anything to reply to.

The state carried between nodes is deliberately small: the last message in each
direction, the running transcript, and the counters the harness reports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict

from langgraph.graph import END, StateGraph

from ..agent.policy import Agent
from .user_sim import UserSimulator


class EpisodeState(TypedDict, total=False):
    user_message: str
    agent_message: str
    transcript: list[dict[str, str]]
    turns: int
    ended_by: str


@dataclass
class Episode:
    agent: Agent
    user: UserSimulator
    max_turns: int = 24
    transcript: list[dict[str, str]] = field(default_factory=list)

    def _open(self, state: EpisodeState) -> EpisodeState:
        message = self.user.opening()
        self.transcript.append({"role": "user", "content": message})
        return {"user_message": message, "transcript": self.transcript, "turns": 0}

    def _agent_turn(self, state: EpisodeState) -> EpisodeState:
        message = self.agent.step(state["user_message"])
        self.transcript.append({"role": "agent", "content": message})
        return {
            "agent_message": message,
            "transcript": self.transcript,
            "turns": state.get("turns", 0) + 1,
        }

    def _user_turn(self, state: EpisodeState) -> EpisodeState:
        message = self.user.respond(state["agent_message"])
        self.transcript.append({"role": "user", "content": message})
        return {"user_message": message, "transcript": self.transcript}

    def _route(self, state: EpisodeState) -> Literal["user", "__end__"]:
        if not state.get("agent_message", "").strip():
            return END
        if state.get("turns", 0) >= self.max_turns:
            return END
        return "user"

    def _route_after_user(self, state: EpisodeState) -> Literal["agent", "__end__"]:
        if UserSimulator.is_end(state["user_message"]):
            return END
        return "agent"

    def build(self) -> Any:
        graph: StateGraph = StateGraph(EpisodeState)
        graph.add_node("open", self._open)
        graph.add_node("agent", self._agent_turn)
        graph.add_node("user", self._user_turn)
        graph.set_entry_point("open")
        graph.add_edge("open", "agent")
        graph.add_conditional_edges("agent", self._route, {"user": "user", END: END})
        graph.add_conditional_edges("user", self._route_after_user, {"agent": "agent", END: END})
        return graph.compile()

    def run(self) -> EpisodeState:
        final: EpisodeState = self.build().invoke(
            {"transcript": [], "turns": 0},
            {"recursion_limit": self.max_turns * 3 + 10},
        )
        turns = final.get("turns", 0)
        if UserSimulator.is_end(final.get("user_message", "")):
            final["ended_by"] = "user"
        elif turns >= self.max_turns:
            final["ended_by"] = "turn_limit"
        else:
            final["ended_by"] = "agent_silent"
        return final

    def last_agent_message(self) -> str:
        for entry in reversed(self.transcript):
            if entry["role"] == "agent":
                return entry["content"]
        return ""
