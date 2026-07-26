import json
import os
import sys
import time
from typing import  Any, Callable, Dict, List, Mapping, Optional, Sequence

class PromptRegistry:
    def __init__(self, path: str = "prompts.json") -> None:
        self.path = path
        self.prompts = {}
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                self.prompts = json.load(f)

    def get_promts(self):
        return self.prompts

    # def format(self, name: str, **kwargs: Any) -> str:
    #     if name not in self.prompts:
    #         raise KeyError(f"Prompt {name!r} not found in {self.path or 'DEFAULT_PROMPTS'}")

    #     prompt = self.prompts[name]
    #     str_kwargs = {
    #         k: safe_json_dumps(v) if isinstance(v, (dict, list)) else str(v)
    #         for k, v in kwargs.items()
    #     }
    #     return prompt.format(**str_kwargs)