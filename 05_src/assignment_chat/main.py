"""LangGraph wiring for Sophia's Kitchen chef chat.

Graph shape::

    START -> pre_guard -> call_model <-> tools
                              |
                              v
                            END

* ``pre_guard`` short-circuits restricted topics and prompt-injection so the
  LLM never sees a hostile prompt.
* ``call_model`` runs short-term memory management (trim + optional summary)
  on the messages before invoking the bound model.
* ``tools_condition`` routes to the standard ``ToolNode`` when the model emits
  tool calls.
"""

from __future__ import annotations

import os


def _silence_langsmith() -> None:
    """Disable LangSmith tracing and clear placeholder API keys.

    The course's ``.secrets`` template ships with literal placeholder strings
    like ``LANGSMITH_API_KEY=<Optional Langsmit Key>``. ``load_dotenv`` would
    otherwise inject that as an env var and ``langsmith`` would try to POST
    traces to the cloud, hit 403, and spam the terminal. We turn tracing off
    explicitly and drop the placeholder so it can't be accidentally trusted.
    """
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    for key in ("LANGSMITH_API_KEY", "LANGCHAIN_API_KEY"):
        val = os.environ.get(key, "")
        if not val or val.startswith("<") or val.lower().startswith("optional"):
            os.environ.pop(key, None)


_silence_langsmith()

from typing import Literal

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    trim_messages,
)
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt.tool_node import ToolNode, tools_condition

from assignment_chat.guardrails import check_user_input, scrub_assistant_output
from assignment_chat.prompts import (
    PROMPT_LEAK_REFUSAL,
    REFUSAL_LINE,
    return_instructions,
)
from assignment_chat.tools_kitchen_math import (
    convert_temperature,
    convert_volume,
    convert_weight,
    scale_recipe,
)
from assignment_chat.tools_recipes_api import (
    lookup_recipe_by_ingredient,
    lookup_recipe_details,
    random_recipe,
    search_recipe_by_name,
)
from assignment_chat.tools_recipes_search import recommend_recipes
from utils.logger import get_logger

_logs = get_logger(__name__)

load_dotenv(".env")
load_dotenv(".secrets")
_silence_langsmith()


TOOLS = [
    search_recipe_by_name,
    lookup_recipe_by_ingredient,
    lookup_recipe_details,
    random_recipe,
    recommend_recipes,
    convert_volume,
    convert_weight,
    convert_temperature,
    scale_recipe,
]

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "openai:gpt-4o-mini")
chat_agent = init_chat_model(OPENAI_MODEL, temperature=0.6)
chat_agent_with_tools = chat_agent.bind_tools(TOOLS)

INSTRUCTIONS = return_instructions()
SYSTEM_MESSAGE = SystemMessage(content=INSTRUCTIONS)

TRIM_THRESHOLD = 10
SUMMARY_THRESHOLD = 14
SUMMARY_KEEP_TAIL = 6


def _last_user_message(messages: list[AnyMessage]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            content = msg.content
            return content if isinstance(content, str) else str(content)
    return ""


def pre_guard(state: MessagesState) -> dict:
    """Short-circuit restricted topics / prompt-injection before the LLM runs."""
    user_text = _last_user_message(state["messages"])
    result = check_user_input(user_text)
    if result.allowed:
        return {}

    _logs.info(f"pre_guard blocked: {result.reason}")
    refusal = PROMPT_LEAK_REFUSAL if result.category == "prompt_injection" else REFUSAL_LINE
    return {"messages": [AIMessage(content=refusal)]}


def _route_after_guard(state: MessagesState) -> Literal["call_model", END]:
    """If pre_guard injected a refusal AIMessage, end the turn here."""
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and not getattr(last, "tool_calls", None):
        return END
    return "call_model"


def _summarize_old_messages(old_messages: list[AnyMessage]) -> SystemMessage:
    """Ask the base model (no tools) for a one-paragraph recap of old turns."""
    prompt = (
        "Summarize the following conversation between Sophia and a user in "
        "ONE short paragraph. Keep names, dishes, dietary preferences, and any "
        "in-flight requests. Drop pleasantries.\n\n"
    )
    convo = []
    for m in old_messages:
        role = (
            "User" if isinstance(m, HumanMessage)
            else "Sophia" if isinstance(m, AIMessage)
            else "Tool" if isinstance(m, ToolMessage)
            else "System"
        )
        text = m.content if isinstance(m.content, str) else str(m.content)
        convo.append(f"{role}: {text[:500]}")
    full_prompt = prompt + "\n".join(convo)

    try:
        resp = chat_agent.invoke([HumanMessage(content=full_prompt)])
        summary_text = resp.content if isinstance(resp.content, str) else str(resp.content)
    except Exception as exc:
        _logs.warning(f"summary failed, falling back to truncation: {exc}")
        summary_text = " ".join(c.content for c in old_messages if isinstance(c.content, str))[:1000]

    return SystemMessage(content=f"Conversation so far: {summary_text}")


def _manage_memory(messages: list[AnyMessage]) -> list[AnyMessage]:
    """Apply the memory policy described in the plan."""
    n = len(messages)

    if n <= TRIM_THRESHOLD:
        return messages

    if n <= SUMMARY_THRESHOLD:
        trimmed = trim_messages(
            messages,
            strategy="last",
            token_counter=len,
            max_tokens=TRIM_THRESHOLD,
            include_system=True,
            allow_partial=False,
            start_on="human",
        )
        _logs.info(f"trimmed messages: {n} -> {len(trimmed)}")
        return trimmed

    cut = max(0, n - SUMMARY_KEEP_TAIL)
    old, recent = messages[:cut], messages[cut:]
    summary_msg = _summarize_old_messages(old)
    new_messages = [summary_msg, *recent]
    _logs.info(f"summarized {cut} old messages into 1; kept {len(recent)} recent")
    return new_messages


def call_model(state: MessagesState) -> dict:
    """Invoke the LLM with system prompt + memory-managed history."""
    managed = _manage_memory(state["messages"])
    response = chat_agent_with_tools.invoke([SYSTEM_MESSAGE, *managed])

    if isinstance(response, AIMessage) and isinstance(response.content, str) and not response.tool_calls:
        scrubbed = scrub_assistant_output(response.content)
        if scrubbed != response.content:
            _logs.info("scrub_assistant_output replaced restricted output")
            response = AIMessage(content=scrubbed)

    return {"messages": [response]}


def get_graph():
    """Build and compile the chat graph."""
    builder = StateGraph(MessagesState)
    builder.add_node("pre_guard", pre_guard)
    builder.add_node("call_model", call_model)
    builder.add_node("tools", ToolNode(TOOLS))

    builder.add_edge(START, "pre_guard")
    builder.add_conditional_edges("pre_guard", _route_after_guard, ["call_model", END])
    builder.add_conditional_edges("call_model", tools_condition, {"tools": "tools", END: END})
    builder.add_edge("tools", "call_model")

    return builder.compile()
