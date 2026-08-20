from pathlib import Path
from energy_agent.storage import Store
from energy_agent.memory import ContextBuilder


def test_persistent_conversation_and_compaction(tmp_path: Path):
    store = Store(tmp_path / "test.db", tmp_path / "artifacts")
    project = store.create_project("Office")
    conversation = store.create_conversation(project["id"])
    for i in range(12):
        store.add_message(conversation["id"], "user" if i % 2 == 0 else "assistant", f"message {i}")
    memory = ContextBuilder(store, recent_turns=2, summary_limit=2000)
    memory.compact(conversation["id"])
    loaded = store.get_conversation(conversation["id"])
    assert "message 7" in loaded["summary"]
    assert loaded["state"]["compacted_message_count"] == 8


def test_immutable_artifact_and_model_version(tmp_path: Path):
    store = Store(tmp_path / "test.db", tmp_path / "artifacts")
    project = store.create_project("Office")
    source = tmp_path / "model.idf"; source.write_text("Version,26.1;", encoding="utf-8")
    artifact = store.import_artifact(project["id"], source)
    version = store.create_model_version(project["id"], artifact["id"], "Baseline")
    assert Path(artifact["path"]).read_text() == "Version,26.1;"
    assert store.project_context(project["id"])["active_model"]["id"] == version["id"]


def test_usage_summary_tracks_largest_prompt_and_calls(tmp_path: Path):
    store = Store(tmp_path / "test.db", tmp_path / "artifacts")
    project = store.create_project("Office")
    conversation = store.create_conversation(project["id"])
    run = store.create_run(conversation["id"])
    store.add_usage(run, conversation["id"], "model_analyst", "test-model", 2, 120, 30, 150, 3)
    store.add_usage(run, conversation["id"], "supervisor", "test-model", 1, 200, 40, 240, 0)
    usage = store.usage_summary(conversation["id"])
    assert usage["api_calls"] == 3
    assert usage["total_tokens"] == 390
    assert usage["highest_prompt"]["agent"] == "supervisor"


def test_delete_conversation_cascades_messages_runs_and_usage(tmp_path: Path):
    store = Store(tmp_path / "test.db", tmp_path / "artifacts")
    project = store.create_project("Office")
    conversation = store.create_conversation(project["id"])
    store.add_message(conversation["id"], "user", "hello")
    run = store.create_run(conversation["id"])
    store.add_usage(run, conversation["id"], "supervisor", "test-model", 1, 10, 2, 12, 0)
    assert store.delete_conversation(conversation["id"]) is True
    assert store.delete_conversation(conversation["id"]) is False
    assert store.list_conversations(project["id"]) == []
    assert store.usage_summary(conversation["id"])["api_calls"] == 0
