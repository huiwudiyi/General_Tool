import os
from typing import  Any, Dict

import yaml
from langgraph.graph import END, START
from tool_description_optimizer.common.utils.utils import ensure_dict


class FlowConfig:
    """YAML-driven route and flow config."""

    def __init__(self, path: str = "config/agent_config.yaml") -> None:
        self.path = path
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f) or {}
        self.default_client: Dict[str, Any] = ensure_dict(self.config.get("default_client", {}))
        self.clients: Dict[str, Any] = ensure_dict(self.config.get("clients", {}))
        self.node_model_map: Dict[str, Any] = ensure_dict(self.config.get("node_model_map", {}))
        self.node_temperature_map: Dict[str, Any] = ensure_dict(self.config.get("node_temperature_map", {}))
        self.node_llm_client_map: Dict[str, Any] = ensure_dict(self.config.get("node_llm_client_map", {}))

    @staticmethod
    def resolve_node(name: str) -> Any:
        if name == "END":
            return END
        elif name == "START":
            return START
        else:
            return name



if __name__ == "__main__":
    pass