from __future__ import annotations

import re
from typing import Any


class ContextBuilder:
    """Build bounded model context while retaining the complete event log in storage."""

    def __init__(self, store, recent_turns: int = 12, summary_limit: int = 10000):
        self.store = store
        self.recent_turns = recent_turns
        self.summary_limit = summary_limit

    def build(self, conversation_id: str) -> dict[str, Any]:
        conversation = self.store.get_conversation(conversation_id)
        project = self.store.project_context(conversation["project_id"])
        recent = self.store.messages(conversation_id, self.recent_turns * 2)
        return {
            "conversation_id": conversation_id,
            "project": project,
            "summary": conversation["summary"],
            "state": conversation["state"],
            "recent_messages": [{"role": m["role"], "content": m["content"]} for m in recent],
        }

    def prompt(self, context: dict) -> str:
        project = context["project"]
        active = project.get("active_model", {})
        lines = [
            "DURABLE PROJECT CONTEXT",
            f"Project: {project['name']} ({project['id']})",
            f"Active model: {active.get('path', 'none')}",
            f"Active version: {active.get('id', 'none')}",
            f"Active measured data: {project.get('active_measured_data', {}).get('path', 'none')}",
            "Conversation summary:", context.get("summary") or "No older summary yet.",
            "Recent conversation:",
        ]
        lines.extend(f"{m['role'].upper()}: {m['content']}" for m in context["recent_messages"])
        return "\n".join(lines)

    def compact(self, conversation_id: str) -> None:
        messages = self.store.messages(conversation_id)
        keep = self.recent_turns * 2
        if len(messages) <= keep:
            return
        conversation = self.store.get_conversation(conversation_id)
        older = messages[:-keep]
        facts = []
        for message in older[-40:]:
            text = re.sub(r"\s+", " ", message["content"]).strip()
            if text:
                facts.append(f"- {message['role']}: {text[:500]}")
        prior = conversation["summary"].strip()
        summary = (prior + "\n" if prior else "") + "\n".join(facts)
        summary = summary[-self.summary_limit:]
        state = dict(conversation["state"])
        state["compacted_message_count"] = len(messages) - keep
        self.store.update_memory(conversation_id, summary, state)
