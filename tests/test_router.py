"""Router tests (Step 6). Pure logic, no Ollama required."""
from agent import router

BIG = "qwen2.5-coder:7b"
SMALL = "qwen2.5-coder:1.5b"


def test_repo_question_gets_tools_even_without_verb():
    # "what does this function return?" has no tool verb, but it's a repo
    # question -- the code keyword "function" sends it down the tool path, which
    # is what the article says a real router should do (vs the bare small model).
    model, tools = router.route("what does this function return?", BIG, SMALL)
    assert model == BIG
    assert tools is True


def test_edit_request_gets_tools():
    # The manual-test failure: an edit request with no file name or identifier
    # must still attach tools, or the agent narrates instead of writing.
    model, tools = router.route("Add a multiply method to the Calculator class", BIG, SMALL)
    assert model == BIG
    assert tools is True


def test_tool_verb_goes_big_with_tools():
    model, tools = router.route("read config.py and find the timeout", BIG, SMALL)
    assert model == BIG
    assert tools is True


def test_long_input_goes_big_even_without_verb():
    long_q = "please explain in detail how the whole system fits together " * 3
    model, tools = router.route(long_q, BIG, SMALL)
    assert model == BIG


def test_word_boundary_avoids_false_positive():
    # "address" contains "add" but must not trip a tool verb.
    model, tools = router.route("what is an ip address?", BIG, SMALL)
    assert model == SMALL
    assert tools is False


def test_empty_input():
    model, tools = router.route("", BIG, SMALL)
    assert model == SMALL
    assert tools is False


def test_snake_case_identifier_triggers_tools():
    # No tool verb, but names a symbol -> still a repo question.
    model, tools = router.route("where is process_payment defined?", BIG, SMALL)
    assert model == BIG
    assert tools is True


def test_camelcase_identifier_triggers_tools():
    model, tools = router.route("list the methods of the CodingAgent class", BIG, SMALL)
    assert model == BIG
    assert tools is True


def test_backtick_span_triggers_tools():
    model, tools = router.route("what does `run_command` restrict?", BIG, SMALL)
    assert model == BIG
    assert tools is True


def test_generic_question_stays_small():
    model, tools = router.route("what is recursion?", BIG, SMALL)
    assert model == SMALL
    assert tools is False


# --- Session-level routing (the mid-tool-loop pin) ---------------------------
from agent.core import CodingAgent


def _msgs(*entries):
    return list(entries)


def test_generic_question_routes_small_even_after_earlier_tool_turns():
    # The manual-session bug: turn 1 used tools, and 'what is recursion?' on
    # turn 2 was pinned to the big model because the pin scanned all history.
    a = CodingAgent(model=BIG)
    messages = _msgs(
        {"role": "system", "content": "..."},
        {"role": "user", "content": "where is process_payment defined?"},
        {"role": "assistant", "content": None, "tool_calls": [{}]},
        {"role": "tool", "tool_call_id": "1", "content": "..."},
        {"role": "assistant", "content": "lines 23-27"},
        {"role": "user", "content": "what is recursion?"},
    )
    model, tools = a._route_for(messages)
    assert model == SMALL
    assert tools is False


def test_mid_tool_loop_pins_big_model():
    # Tool messages AFTER the latest user turn = the agent is mid-task.
    a = CodingAgent(model=BIG)
    messages = _msgs(
        {"role": "user", "content": "what is recursion?"},
        {"role": "assistant", "content": None, "tool_calls": [{}]},
        {"role": "tool", "tool_call_id": "1", "content": "..."},
    )
    model, tools = a._route_for(messages)
    assert model == BIG
    assert tools is True


def test_router_disabled_always_big():
    a = CodingAgent(model=BIG, use_router=False)
    model, tools = a._route_for(_msgs({"role": "user", "content": "hi"}))
    assert model == BIG
    assert tools is True


# --- Three-tier routing (optional edit model) ---------------------------------
EDIT = "qwen3:32b"


def test_write_request_routes_to_edit_model_when_set():
    model, tools = router.route("Add a multiply method to the Calculator class",
                                BIG, SMALL, edit_model=EDIT)
    assert model == EDIT and tools is True


def test_read_question_stays_on_big_model_even_with_edit_model():
    model, tools = router.route("where is process_payment defined?",
                                BIG, SMALL, edit_model=EDIT)
    assert model == BIG and tools is True


def test_generic_question_stays_small_even_with_edit_model():
    model, tools = router.route("what is recursion?", BIG, SMALL, edit_model=EDIT)
    assert model == SMALL and tools is False


def test_no_edit_model_falls_back_to_big():
    model, tools = router.route("Add a multiply method", BIG, SMALL)
    assert model == BIG and tools is True


def test_mid_loop_keeps_edit_model():
    # An edit that started on the edit model must finish on it.
    a = CodingAgent(model=BIG, edit_model=EDIT)
    messages = [
        {"role": "user", "content": "Add a multiply method to the Calculator class"},
        {"role": "assistant", "content": None, "tool_calls": [{}]},
        {"role": "tool", "tool_call_id": "1", "content": "..."},
    ]
    model, tools = a._route_for(messages)
    assert model == EDIT and tools is True
