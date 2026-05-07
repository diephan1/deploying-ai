"""Gradio chat front-end for Sophia's Kitchen.

Run from inside ``05_src/`` so the ``assignment_chat`` and ``utils`` packages
are importable::

    cd 05_src
    uv run python -m assignment_chat.app
"""

from __future__ import annotations

from assignment_chat.main import _silence_langsmith, get_graph

_silence_langsmith()

import gradio as gr
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

from utils.logger import get_logger

_logs = get_logger(__name__)

load_dotenv(".env")
load_dotenv(".secrets")
_silence_langsmith()

graph = get_graph()


def chef_chat(message: str, history: list[dict]) -> str:
    """Convert Gradio history -> LangChain messages, invoke graph, return text."""
    langchain_messages = []
    for msg in history:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "user":
            langchain_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            langchain_messages.append(AIMessage(content=content))
    langchain_messages.append(HumanMessage(content=message))

    result = graph.invoke({"messages": langchain_messages})
    last = result["messages"][-1]
    return last.content if isinstance(last.content, str) else str(last.content)


chat = gr.ChatInterface(
    fn=chef_chat,
    type="messages",
    title="Sophia's Kitchen",
    description=(
        "Ciao tesoro! I am Sophia. Ask me for a recipe, tell me what's in "
        "your pantry, or have me convert and scale ingredients. Mangia!"
    ),
    examples=[
        "What should I cook tonight with eggplant?",
        "Give me an Indian vegetarian dish",
        "Surprise me with a random recipe",
        "Convert 2 cups of flour to grams",
        "Scale this for 6 people instead of 4: 200g flour, 2 eggs, 100ml milk",
    ],
)


if __name__ == "__main__":
    _logs.info("Starting Sophia's Kitchen chat app...")
    chat.launch()
