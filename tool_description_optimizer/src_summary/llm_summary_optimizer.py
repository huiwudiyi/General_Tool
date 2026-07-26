from __future__ import annotations

import argparse
import json
import os
import sys
import time
import random
from typing import Annotated, Any, Callable, Dict, List, Mapping, Optional, Sequence, TypedDict

from state import ToolOptimizerState

class LLMDescriptionSummary:
    @staticmethod
    def _gen_prompt(state: ToolOptimizerState, prompt: str, query_list: list):
        best_record = state.get("best_record", None)
        if not best_record:
            return ""
        description = best_record.description
        sampled_queries = query_list
        if len(query_list) > 22:
            sampled_queries = random.sample(query_list, 20)
        length = str(min(len(description), 350))
        return prompt.replace("{{tool_description}}", description).replace("{{number}}", length).replace("{{querys}}", "、".join(sampled_queries))

    @staticmethod
    def _vertify_result(response: dict):
        
        # check response result
        if len(set(response.keys()) - set(['think', 'optimizer_description'])) > 0:
            return {}, False , "key error"
        # check optimized_description 字符串
        optimized_description = response["optimizer_description"]
        if len(optimized_description) < 100:
            return {}, False, "optimizer_description error"
        return {
            'optimizer_description': optimized_description
        }, True, ""

            
