from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Annotated, Any, Callable, Dict, List, Mapping, Optional, Sequence, TypedDict

from state import ToolOptimizerState

class LLMDescriptionJudge:
    @staticmethod
    def _gen_prompt(state: ToolOptimizerState, prompt: str):
        optimizer_history = state.get("optimizer_history", [])
        if len(optimizer_history) == 0:
            return None
        infoRecord = optimizer_history[-1]
        original_description = state.get("original_description", "")
        best_description = infoRecord.info.get("optimizer_description", "")
        if len(original_description) < 10 or len(best_description) < 10:
            return None
        return prompt.replace("{{before_text}}", original_description).replace("{{after_text}}", best_description), best_description

    @staticmethod
    def _vertify_result(response: dict):
        
        # check response result
        if len(set(response.keys()) - set(['semantic_analysis', 'scene_comparison', 'function_comparison', 'content_quality', 'relevance_reason', 'relevance_score'])) > 0:
            return {}, False , "key error"
        # check semantic_analysis 字符串
        semantic_analysis = response["semantic_analysis"]
        if len(semantic_analysis) < 10:
            return {}, False, "semantic_analysis error"
        
        # check scene_comparison 字符串
        scene_comparison = response["scene_comparison"]  
        if len(scene_comparison) < 10:
            return {}, False, "scene_comparison error"
        
        # check function_comparison 字符串
        function_comparison = response["function_comparison"]  
        if len(function_comparison) < 10:
            return {}, False, "function_comparison error"

        # check content_quality 字符串
        content_quality = response["content_quality"]  
        if len(content_quality) < 10:
            return {}, False, "content_quality error"

        # check relevance_reason 字符串
        relevance_reason = response["relevance_reason"]  
        if len(relevance_reason) < 10:
            return {}, False, "relevance_reason error"

        # check relevance_score 类别
        relevance_score = response['relevance_score']
        if isinstance(relevance_score, (float, str)):
            relevance_score = int(relevance_score) 
        return {
            'semantic_analysis': semantic_analysis, 
            'scene_comparison': scene_comparison, 
            'function_comparison': function_comparison, 
            'content_quality':content_quality,
            'relevance_reason': relevance_reason,
            'relevance_score': relevance_score
        }, True, ""

