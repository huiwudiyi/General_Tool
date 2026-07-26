from __future__ import annotations

import argparse
import json
import os
import sys
import time
import random
from typing import Annotated, Any, Callable, Dict, List, Mapping, Optional, Sequence, TypedDict

from state import ToolOptimizerState

class LLMDescriptionGenerator:
    @staticmethod
    def _gen_prompt(state: ToolOptimizerState, prompt: str, query_labelData_dict: Dict[str, str]):
        recall_content = []
        for query, content in query_labelData_dict.items():
            recall_content.append({
                "query": query,
                "recal_data": content
            })
        return prompt.replace("{{content}}", json.dumps(recall_content, ensure_ascii=False))
    @staticmethod
    def _vertify_result(response: dict):
        
        # check response result
        if len(set(response.keys()) - set(['summary', 'display', 'interaction'])) > 0:
            return {}, False , "key error"
        # check optimized_description 字符串
        summary = response["summary"]
        display = response['display']
        interaction = response['interaction']

        if len(summary) < 10 or len(display) < 10 or len(interaction) < 10:
            return {}, False, "optimizer_description error"
        return {
            "summary":summary,
            "display":display,
            "interaction":interaction,
            }, True, ""

            
