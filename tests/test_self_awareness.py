import json

from energy_agent.self_awareness import self_awareness_tools


def _tools(tmp_path):
    registry = {"tools": []}
    tools = self_awareness_tools(
        registry,
        [{"id": "analyst", "name": "Analyst", "purpose": "Inspect", "tools": {"list_registered_tools"}}],
        tmp_path, tmp_path / "data", tmp_path / "outputs",
    )
    registry["tools"] = tools
    return {item.name: item for item in tools}


def test_live_registry_reports_exact_unique_count(tmp_path):
    tools = _tools(tmp_path)
    result = json.loads(tools["list_registered_tools"].invoke({"include_descriptions": False}))
    assert result["status"] == "verified"
    assert result["unique_tool_count"] == 4
    assert {row["name"] for row in result["tools"]} == set(tools)


def test_registry_category_filter_and_agent_permissions(tmp_path):
    tools = _tools(tmp_path)
    result = json.loads(tools["list_registered_tools"].invoke(
        {"category": "platform_awareness", "include_descriptions": False}
    ))
    assert result["unique_tool_count"] == 4
    agents = json.loads(tools["describe_agent_team"].invoke({}))
    assert agents["agent_count"] == 1
    assert agents["agents"][0]["permitted_tools"] == ["list_registered_tools"]


def test_capabilities_are_honest_and_secret_safe(tmp_path):
    tools = _tools(tmp_path)
    result = json.loads(tools["get_platform_capabilities"].invoke({}))
    assert "lighting-only optimization" in result["limitations"]
    serialized = json.dumps(result).lower()
    assert "google_api_key" not in serialized
    assert "mcp_token" not in serialized
