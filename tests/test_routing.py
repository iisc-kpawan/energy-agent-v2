from energy_agent.orchestrator import MultiAgentOrchestrator, SPECIALISTS

ORIGINAL_MCP_TOOLS = {
    "copy_file", "load_idf_model", "get_model_summary", "check_simulation_settings",
    "inspect_schedules", "inspect_people", "modify_people", "inspect_lights", "modify_lights",
    "inspect_electric_equipment", "modify_electric_equipment", "modify_simulation_control",
    "modify_run_period", "change_infiltration_by_mult", "add_window_film_outside",
    "add_coating_outside", "list_zones", "get_surfaces", "get_materials", "validate_idf",
    "get_output_variables", "get_output_meters", "add_output_variables", "add_output_meters",
    "list_available_files", "get_server_configuration", "get_server_status", "discover_hvac_loops",
    "get_loop_topology", "visualize_loop_diagram", "run_energyplus_simulation",
    "create_interactive_plot", "get_server_logs", "get_error_logs", "clear_logs",
}


def test_simple_inspection_routes_to_model_analyst():
    assert MultiAgentOrchestrator.route("Summarize this IDF model") == ["model_analyst"]


def test_complex_change_has_planning_analysis_write_and_qa():
    route = MultiAgentOrchestrator.route("Find the best way to reduce annual energy and modify the model")
    assert route[:3] == ["planner", "model_analyst", "retrofit_engineer"]
    assert "qa_reviewer" in route


def test_simulation_is_reviewed():
    assert MultiAgentOrchestrator.route("Run an annual simulation") == ["simulation_engineer", "qa_reviewer"]


def test_end_to_end_wording_routes_simulation_and_results():
    route = MultiAgentOrchestrator.route("Handle all annual simulation and EPI calculations")
    assert route == ["simulation_engineer", "results_analyst", "qa_reviewer"]


def test_broad_simulation_request_cannot_drop_execution_tool():
    selected = MultiAgentOrchestrator._select_tools(
        SPECIALISTS["simulation_engineer"],
        "Handle file discovery, inspection, validation, annual simulation, EPI, results, outputs, and diagnostics",
    )
    assert "run_energyplus_simulation" in selected


def test_results_agent_cannot_run_or_calculate_from_model_tools():
    selected = MultiAgentOrchestrator._select_tools(
        SPECIALISTS["results_analyst"],
        "Calculate EPI and results after the annual simulation",
    )
    assert selected == {"calculate_energy_performance", "get_error_logs"}


def test_original_write_capabilities_remain_reachable():
    cases = {
        "modify people occupancy": "modify_people",
        "modify lighting power": "modify_lights",
        "modify equipment plug load": "modify_electric_equipment",
        "change the run period": "modify_run_period",
        "change infiltration": "change_infiltration_by_mult",
        "add window film": "add_window_film_outside",
        "add an outside coating": "add_coating_outside",
        "add output variables": "add_output_variables",
        "add output meters": "add_output_meters",
    }
    specialist = SPECIALISTS["retrofit_engineer"]
    for request, expected in cases.items():
        assert expected in MultiAgentOrchestrator._select_tools(specialist, request)


def test_original_log_admin_capability_is_assigned_and_reachable():
    selected = MultiAgentOrchestrator._select_tools(
        SPECIALISTS["simulation_engineer"], "clear server logs"
    )
    assert "clear_logs" in selected
    assert "simulation_engineer" in MultiAgentOrchestrator.route("clear server logs")


def test_every_original_mcp_tool_is_assigned_to_at_least_one_agent():
    assigned = set().union(*(specialist.tools for specialist in SPECIALISTS.values()))
    assert ORIGINAL_MCP_TOOLS <= assigned


def test_mentioning_a_future_simulation_does_not_trigger_execution():
    assert MultiAgentOrchestrator.route("List samples for my first simulation") == ["model_analyst"]


def test_listing_exposes_only_small_relevant_tool_set():
    specialist = __import__("energy_agent.orchestrator", fromlist=["SPECIALISTS"]).SPECIALISTS["model_analyst"]
    selected = MultiAgentOrchestrator._select_tools(specialist, "List available sample model files")
    assert selected == {"list_sample_files"}


def test_memory_does_not_expand_current_request_tools():
    specialist = __import__("energy_agent.orchestrator", fromlist=["SPECIALISTS"]).SPECIALISTS["model_analyst"]
    task = "old memory mentions HVAC schedules and materials\nCURRENT USER REQUEST:\nList available samples"
    assert MultiAgentOrchestrator._select_tools(specialist, task) == {"list_sample_files"}


def test_epi_routes_only_results_and_qa_with_one_calculation_tool():
    prompt = "Calculate EPI/EUI from completed simulation results in /outputs/run-1 and give an execution summary."
    assert MultiAgentOrchestrator.route(prompt) == ["results_analyst", "qa_reviewer"]
    results = __import__("energy_agent.orchestrator", fromlist=["SPECIALISTS"]).SPECIALISTS["results_analyst"]
    assert MultiAgentOrchestrator._select_tools(results, prompt) == {"calculate_energy_performance", "get_error_logs"}
    qa = __import__("energy_agent.orchestrator", fromlist=["SPECIALISTS"]).SPECIALISTS["qa_reviewer"]
    assert MultiAgentOrchestrator._select_tools(qa, prompt + "\nPRIOR SPECIALIST REPORTS:\nevidence") == set()
