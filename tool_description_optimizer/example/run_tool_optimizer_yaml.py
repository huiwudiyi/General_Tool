from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Annotated, Any, Callable, Dict, List, Mapping, Optional, Sequence, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from tool_optimizer_state import ToolOptimizerState

prompt_path = "config/prompts.json"
flow_config_path = "config/agent_config.yaml"
thread_id = "zachrychu"
write_default_flow = "store"


dag = ToolOptimizerGraph(
        llm_client  = None,
        prompt_path = prompt_path ,
        flow_config_path = flow_config_path,
        checkpointer = None,
)

graph = StateGraph(ToolOptimizerState)
dag._register_nodes(graph)
graph.add_edge(START, "guardrail")
graph.add_edge("guardrail", "intent")
graph.add_conditional_edges(
            "intent",
            dag.route_after_intent,
            dag.flow_config.get_route_entry_map(),
        )
dag._add_yaml_flows(graph)

graph =  graph.compile()

img_data = graph.get_graph().draw_mermaid_png()
with open("flow.png", "wb") as fp:
    fp.write(img_data)