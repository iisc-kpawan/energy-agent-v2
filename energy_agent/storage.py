"""SQLite-backed event log and project/artifact registry.

SQLite is intentional for the local v1: it is durable, transactional and requires
no service. All SQL is isolated here so PostgreSQL can replace this repository.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Store:
    def __init__(self, database: Path, artifacts_dir: Path):
        self.database = database
        self.artifacts_dir = artifacts_dir
        self._lock = threading.RLock()
        database.parent.mkdir(parents=True, exist_ok=True)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def _init_schema(self) -> None:
        schema = """
        CREATE TABLE IF NOT EXISTS projects(
          id TEXT PRIMARY KEY, name TEXT NOT NULL, created_at TEXT NOT NULL,
          active_version_id TEXT
        );
        CREATE TABLE IF NOT EXISTS conversations(
          id TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES projects(id),
          title TEXT NOT NULL, summary TEXT NOT NULL DEFAULT '', state_json TEXT NOT NULL DEFAULT '{}',
          created_at TEXT NOT NULL, updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS messages(
          id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL REFERENCES conversations(id),
          role TEXT NOT NULL, content TEXT NOT NULL, metadata_json TEXT NOT NULL DEFAULT '{}',
          created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS messages_conversation ON messages(conversation_id, created_at);
        CREATE TABLE IF NOT EXISTS runs(
          id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL REFERENCES conversations(id),
          status TEXT NOT NULL, route_json TEXT NOT NULL DEFAULT '[]', events_json TEXT NOT NULL DEFAULT '[]',
          error TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL, updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS artifacts(
          id TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES projects(id), type TEXT NOT NULL,
          name TEXT NOT NULL, path TEXT NOT NULL, sha256 TEXT NOT NULL, metadata_json TEXT NOT NULL DEFAULT '{}',
          created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS model_versions(
          id TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES projects(id), parent_id TEXT,
          artifact_id TEXT NOT NULL REFERENCES artifacts(id), label TEXT NOT NULL,
          changes_json TEXT NOT NULL DEFAULT '[]', validation_status TEXT NOT NULL DEFAULT 'pending',
          created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS usage_events(
          id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(id), conversation_id TEXT NOT NULL,
          agent TEXT NOT NULL, model TEXT NOT NULL, api_calls INTEGER NOT NULL DEFAULT 0,
          prompt_tokens INTEGER NOT NULL DEFAULT 0, completion_tokens INTEGER NOT NULL DEFAULT 0,
          total_tokens INTEGER NOT NULL DEFAULT 0, tool_calls INTEGER NOT NULL DEFAULT 0,
          estimated INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS usage_run ON usage_events(run_id, created_at);
        """
        with self._connect() as db:
            db.executescript(schema)

    def create_project(self, name: str) -> dict[str, Any]:
        project_id, now = _id("prj"), _now()
        with self._connect() as db:
            db.execute("INSERT INTO projects(id,name,created_at) VALUES(?,?,?)", (project_id, name, now))
        return self.get_project(project_id)

    def list_projects(self) -> list[dict[str, Any]]:
        with self._connect() as db:
            return [dict(r) for r in db.execute("SELECT * FROM projects ORDER BY created_at DESC")]

    def get_project(self, project_id: str) -> dict[str, Any]:
        with self._connect() as db:
            row = db.execute("SELECT * FROM projects WHERE id=?", (project_id,)).fetchone()
        if not row:
            raise KeyError(f"Project not found: {project_id}")
        return dict(row)

    def create_conversation(self, project_id: str, title: str = "New conversation") -> dict[str, Any]:
        self.get_project(project_id)
        conversation_id, now = _id("con"), _now()
        with self._connect() as db:
            db.execute(
                "INSERT INTO conversations(id,project_id,title,created_at,updated_at) VALUES(?,?,?,?,?)",
                (conversation_id, project_id, title, now, now),
            )
        return self.get_conversation(conversation_id)

    def get_conversation(self, conversation_id: str) -> dict[str, Any]:
        with self._connect() as db:
            row = db.execute("SELECT * FROM conversations WHERE id=?", (conversation_id,)).fetchone()
        if not row:
            raise KeyError(f"Conversation not found: {conversation_id}")
        result = dict(row)
        result["state"] = json.loads(result.pop("state_json"))
        return result

    def list_conversations(self, project_id: str) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute("SELECT * FROM conversations WHERE project_id=? ORDER BY updated_at DESC", (project_id,))
            return [dict(r) for r in rows]

    def delete_conversation(self, conversation_id: str) -> bool:
        with self._lock, self._connect() as db:
            exists = db.execute("SELECT 1 FROM conversations WHERE id=?", (conversation_id,)).fetchone()
            if not exists:
                return False
            run_ids = [r["id"] for r in db.execute("SELECT id FROM runs WHERE conversation_id=?", (conversation_id,))]
            if run_ids:
                placeholders = ",".join("?" for _ in run_ids)
                db.execute(f"DELETE FROM usage_events WHERE run_id IN ({placeholders})", run_ids)
            db.execute("DELETE FROM messages WHERE conversation_id=?", (conversation_id,))
            db.execute("DELETE FROM runs WHERE conversation_id=?", (conversation_id,))
            db.execute("DELETE FROM conversations WHERE id=?", (conversation_id,))
        return True

    def add_message(self, conversation_id: str, role: str, content: str, metadata: dict | None = None) -> dict:
        message_id, now = _id("msg"), _now()
        with self._connect() as db:
            db.execute(
                "INSERT INTO messages VALUES(?,?,?,?,?,?)",
                (message_id, conversation_id, role, content, json.dumps(metadata or {}), now),
            )
            db.execute("UPDATE conversations SET updated_at=? WHERE id=?", (now, conversation_id))
        return {"id": message_id, "role": role, "content": content, "metadata": metadata or {}, "created_at": now}

    def messages(self, conversation_id: str, limit: int | None = None) -> list[dict]:
        sql = "SELECT * FROM messages WHERE conversation_id=? ORDER BY created_at DESC"
        args: list[Any] = [conversation_id]
        if limit:
            sql += " LIMIT ?"; args.append(limit)
        with self._connect() as db:
            rows = list(db.execute(sql, args))
        result = []
        for row in reversed(rows):
            item = dict(row); item["metadata"] = json.loads(item.pop("metadata_json")); result.append(item)
        return result

    def update_memory(self, conversation_id: str, summary: str, state: dict) -> None:
        with self._connect() as db:
            db.execute(
                "UPDATE conversations SET summary=?,state_json=?,updated_at=? WHERE id=?",
                (summary, json.dumps(state), _now(), conversation_id),
            )

    def create_run(self, conversation_id: str) -> str:
        run_id, now = _id("run"), _now()
        with self._connect() as db:
            db.execute("INSERT INTO runs(id,conversation_id,status,created_at,updated_at) VALUES(?,?,?,?,?)",
                       (run_id, conversation_id, "running", now, now))
        return run_id

    def update_run(self, run_id: str, status: str | None = None, route: list | None = None,
                   event: dict | None = None, error: str = "") -> None:
        with self._lock, self._connect() as db:
            row = db.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
            if not row: return
            events = json.loads(row["events_json"])
            if event: events.append({**event, "at": _now()})
            db.execute("UPDATE runs SET status=?,route_json=?,events_json=?,error=?,updated_at=? WHERE id=?", (
                status or row["status"], json.dumps(route if route is not None else json.loads(row["route_json"])),
                json.dumps(events), error or row["error"], _now(), run_id))

    def get_run(self, run_id: str) -> dict:
        with self._connect() as db:
            row = db.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        if not row: raise KeyError(f"Run not found: {run_id}")
        item = dict(row)
        item["route"] = json.loads(item.pop("route_json")); item["events"] = json.loads(item.pop("events_json"))
        return item

    def add_usage(self, run_id: str, conversation_id: str, agent: str, model: str,
                  api_calls: int, prompt_tokens: int, completion_tokens: int,
                  total_tokens: int, tool_calls: int, estimated: bool = False) -> None:
        with self._connect() as db:
            db.execute("INSERT INTO usage_events VALUES(?,?,?,?,?,?,?,?,?,?,?,?)", (
                _id("use"), run_id, conversation_id, agent, model, api_calls,
                prompt_tokens, completion_tokens, total_tokens, tool_calls,
                int(estimated), _now()))

    def usage_summary(self, conversation_id: str | None = None) -> dict:
        where, args = (" WHERE conversation_id=?", [conversation_id]) if conversation_id else ("", [])
        with self._connect() as db:
            totals = dict(db.execute(f"""SELECT COALESCE(SUM(api_calls),0) api_calls,
                COALESCE(SUM(prompt_tokens),0) prompt_tokens,
                COALESCE(SUM(completion_tokens),0) completion_tokens,
                COALESCE(SUM(total_tokens),0) total_tokens,
                COALESCE(SUM(tool_calls),0) tool_calls,
                COALESCE(SUM(estimated),0) estimated_events FROM usage_events{where}""", args).fetchone())
            highest = db.execute(f"""SELECT run_id,agent,model,prompt_tokens,completion_tokens,total_tokens,
                api_calls,tool_calls,estimated,created_at FROM usage_events{where}
                ORDER BY prompt_tokens DESC LIMIT 1""", args).fetchone()
            by_agent = [dict(r) for r in db.execute(f"""SELECT agent, SUM(api_calls) api_calls,
                SUM(prompt_tokens) prompt_tokens, SUM(completion_tokens) completion_tokens,
                SUM(total_tokens) total_tokens, SUM(tool_calls) tool_calls
                FROM usage_events{where} GROUP BY agent ORDER BY total_tokens DESC""", args)]
        totals["highest_prompt"] = dict(highest) if highest else None
        totals["by_agent"] = by_agent
        return totals

    def import_artifact(self, project_id: str, source: Path, artifact_type: str = "idf",
                        metadata: dict | None = None) -> dict:
        source = source.resolve()
        if not source.is_file(): raise FileNotFoundError(source)
        artifact_id = _id("art")
        target_dir = self.artifacts_dir / project_id / artifact_id
        target_dir.mkdir(parents=True, exist_ok=False)
        target = target_dir / source.name
        shutil.copy2(source, target)
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        with self._connect() as db:
            db.execute("INSERT INTO artifacts VALUES(?,?,?,?,?,?,?,?)", (
                artifact_id, project_id, artifact_type, source.name, str(target), digest,
                json.dumps(metadata or {}), _now()))
        return self.get_artifact(artifact_id)

    def get_artifact(self, artifact_id: str) -> dict:
        with self._connect() as db:
            row = db.execute("SELECT * FROM artifacts WHERE id=?", (artifact_id,)).fetchone()
        if not row: raise KeyError(f"Artifact not found: {artifact_id}")
        item = dict(row); item["metadata"] = json.loads(item.pop("metadata_json")); return item

    def list_artifacts(self, project_id: str, artifact_type: str | None = None) -> list[dict]:
        self.get_project(project_id)
        query, args = "SELECT * FROM artifacts WHERE project_id=?", [project_id]
        if artifact_type:
            query += " AND type=?"; args.append(artifact_type)
        query += " ORDER BY created_at DESC"
        with self._connect() as db:
            rows = db.execute(query, args).fetchall()
        items = []
        for row in rows:
            item = dict(row); item["metadata"] = json.loads(item.pop("metadata_json")); items.append(item)
        return items

    def create_model_version(self, project_id: str, artifact_id: str, label: str,
                             parent_id: str | None = None, changes: list | None = None) -> dict:
        artifact = self.get_artifact(artifact_id)
        if artifact["project_id"] != project_id or artifact["type"] != "idf":
            raise ValueError("The artifact must be an IDF belonging to this project")
        version_id = _id("idf")
        with self._connect() as db:
            db.execute("INSERT INTO model_versions VALUES(?,?,?,?,?,?,?,?)", (
                version_id, project_id, parent_id, artifact_id, label, json.dumps(changes or []), "pending", _now()))
            db.execute("UPDATE projects SET active_version_id=? WHERE id=?", (version_id, project_id))
        return self.get_model_version(version_id)

    def get_model_version(self, version_id: str) -> dict:
        with self._connect() as db:
            row = db.execute("SELECT * FROM model_versions WHERE id=?", (version_id,)).fetchone()
        if not row: raise KeyError(f"Model version not found: {version_id}")
        item = dict(row); item["changes"] = json.loads(item.pop("changes_json")); return item

    def project_context(self, project_id: str) -> dict:
        project = self.get_project(project_id)
        if project.get("active_version_id"):
            version = self.get_model_version(project["active_version_id"])
            artifact = self.get_artifact(version["artifact_id"])
            project["active_model"] = {**version, "path": artifact["path"], "sha256": artifact["sha256"]}
        measured = self.list_artifacts(project_id, "measured_csv")
        if measured:
            project["active_measured_data"] = measured[0]
        return project
