from __future__ import annotations

import argparse
import json
import os
import sys
import time
from utils import *
from typing import Annotated, Any, Callable, Dict, List, Mapping, Optional, Sequence, TypedDict


from state import ToolOptimizerState

class LLMDescriptionOptimizer:
    @staticmethod
    def _gen_prompt(state: ToolOptimizerState, prompt: str, tools_dict: dict):
        best_record = state.get("best_record", None)
        if not best_record:
            return ""
        title = state.get("title", "")
        description = best_record.description
        top_case  = best_record.top_case
        miss_case = ""
        for k, v in top_case.items():
            if tools_dict.get(k, "") == "" or k == NO_CALL:
                miss_case += "\n- 未召回query集合: " + "、".join(v)
            else:
                miss_case += "\n[[负例样本详情]] \n- 未召回query集合：" + "、".join(v) + "\n- 上面的query召回其他的混淆工具描述如下：\n" + tools_dict[k]["description"]

        return prompt.replace("{{original_description}}", description).replace("{{case}}", miss_case).replace("{{title}}", title)

    @staticmethod
    def _vertify_result(response: dict):
        
        # check response result
        if len(set(response.keys()) - set(['thought', 'optimizer_description', 'change_reason', 'optimizer_keys', 'risk'])) > 0:
            return {}, False , "key error"
        # check optimizer_description 字符串
        optimizer_description = response["optimizer_description"]
        if len(optimizer_description) < 100:
            return {}, False, "optimizer_description error"
        
        # check change_reason 字符串
        change_reason = response["change_reason"]  #isinstance(raw_data, (dict, list))
        if isinstance(change_reason, list):
            change_reason = "".join(change_reason)
        
        # check optimizer_keys 字符串
        optimizer_keys = response['optimizer_keys']
        if isinstance(optimizer_keys, list):
            optimizer_keys = " ".join(optimizer_keys) 

        # check risk 字符串
        risk = response['risk']
        if isinstance(risk, list):
            risk = " ".join(risk) 
        return {
            'optimizer_description': optimizer_description, 
            'change_reason': change_reason, 
            'optimizer_keys': optimizer_keys, 
            'risk': risk
        }, True, ""

