from __future__ import annotations

import argparse
import json
import os
import sys
import time
import random
from typing import Annotated, Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, TypedDict

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


class LLMPolicyGenerator:
    """描述生成器 π_θ 的输出校验。

    对应 prompt config/prompts.json 的 "policy_generator"：
    策略输出结构化描述 d = {功能区, 边界区}，功能区描述能力与覆盖意图，
    边界区列出不适用场景。两段都过短说明策略没按格式产出，判为无效样本。
    """

    ALLOWED_KEYS = {"功能区", "边界区"}
    MIN_FUNCTION_LENGTH = 20
    MIN_BOUNDARY_LENGTH = 10

    @staticmethod
    def _vertify_result(response: Any) -> Tuple[Dict[str, Any], bool, str]:
        """校验策略输出的结构化描述 d = {功能区, 边界区}。"""
        if not isinstance(response, Mapping):
            return {}, False, "response type error"
        if len(set(response.keys()) - LLMPolicyGenerator.ALLOWED_KEYS) > 0:
            return {}, False, "key error"
        function_area = str(response.get("功能区", "") or "").strip()
        boundary_area = str(response.get("边界区", "") or "").strip()
        if len(function_area) < LLMPolicyGenerator.MIN_FUNCTION_LENGTH:
            return {}, False, "功能区 too short"
        if len(boundary_area) < LLMPolicyGenerator.MIN_BOUNDARY_LENGTH:
            return {}, False, "边界区 too short"
        return {"功能区": function_area, "边界区": boundary_area}, True, ""


class LLMAladdinGenerator:
    """阿拉丁卡片筛选与填参助手。

    对应 prompt：给定「用户问题 / 已生成答案 / 候选阿拉丁列表」，
    让模型挑出强相关的卡片并按各卡 tool.parameters 生成调用参数，
    输出 {"reason": ..., "selected": [{"srcid": ..., "parameters": {...}}]}。

    prompt 占位符支持 ${query} / ${answer} / ${aladdin} 两种写法（同时兼容 {{query}} 风格）。
    """

    # 精准卡以卡名是否含"精准"判定；精准卡只按实体检索，question 不得带修饰/维度词
    PRECISE_KEYWORD = "精准"
    MODIFIER_TOKENS: Tuple[str, ...] = (
        "多少画", "几画", "笔画", "笔顺", "部首", "拼音", "读音", "怎么读",
        "近义词", "反义词", "同义词", "意思", "含义", "释义", "解释", "翻译",
        "出处", "作者", "全文", "上一句", "下一句", "组词", "造句", "英文",
        "是什么", "什么意思",
    )
    # 明显的占位/通配取值，属于"不留空、不用通配符"约束的兜底拦截
    PLACEHOLDER_VALUES = {"", "*", "**", "-", "--", "n/a", "na", "null", "none", "无", "待填", "xxx", "?", "？"}
    ALLOWED_KEYS = {"reason", "selected"}
    ALLOWED_ITEM_KEYS = {"srcid", "parameters"}

    # ---------------- prompt 拼装 ----------------
    @staticmethod
    def _dump(value: Any) -> str:
        """把候选列表/答案等结构体安全转成 prompt 里可读的文本。"""
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False, indent=2)

    @staticmethod
    def _resolve_query(state: Optional[ToolOptimizerState], query: Any) -> str:
        """优先用显式传入的用户问题，缺省时回退到 state 里的 query。"""
        if isinstance(query, str) and query.strip():
            return query.strip()
        if isinstance(query, (list, tuple)) and query:
            return str(query[0]).strip()
        state_query = state.get("query", None) if state else None
        if isinstance(state_query, str):
            return state_query.strip()
        if isinstance(state_query, (list, tuple)) and state_query:
            return str(state_query[0]).strip()
        return ""

    @staticmethod
    def _gen_prompt(
        state: Optional[ToolOptimizerState],
        prompt: str,
        query: Any = None,
        answer: Any = "",
        aladdin_candidates: Any = None,
    ) -> str:
        """填充 ${query} / ${answer} / ${aladdin} 三个占位符。

        Args:
            state: 全局状态，query 缺省时从这里回退取值。
            prompt: prompt 模板原文。
            query: 原始用户问题。
            answer: 已生成的文本答案。
            aladdin_candidates: 候选阿拉丁列表，list 或 {srcid: card} 形式均可。

        Returns:
            填充后的 prompt；候选列表为空时返回 ""，交由上层跳过该次调用。
        """
        candidates = LLMAladdinGenerator._index_candidates(aladdin_candidates)
        if not candidates:
            print("LLMAladdinGenerator: 候选阿拉丁列表为空，跳过筛选")
            return ""

        query_text = LLMAladdinGenerator._resolve_query(state, query)
        answer_text = LLMAladdinGenerator._dump(answer)
        aladdin_text = LLMAladdinGenerator._dump(list(candidates.values()))

        for name, value in (("query", query_text), ("answer", answer_text), ("aladdin", aladdin_text)):
            prompt = prompt.replace("${%s}" % name, value).replace("{{%s}}" % name, value)
        return prompt

    # ---------------- 候选卡片解析 ----------------
    @staticmethod
    def _index_candidates(aladdin_candidates: Any) -> Dict[str, Dict[str, Any]]:
        """把候选列表按 srcid 建索引，兼容 list / {srcid: card} 两种入参。"""
        indexed: Dict[str, Dict[str, Any]] = {}
        if not aladdin_candidates:
            return indexed

        if isinstance(aladdin_candidates, Mapping):
            items = list(aladdin_candidates.values())
        elif isinstance(aladdin_candidates, (list, tuple)):
            items = list(aladdin_candidates)
        else:
            return indexed

        for item in items:
            if not isinstance(item, Mapping):
                continue
            srcid = str(item.get("srcid", "") or item.get("src_id", "")).strip()
            if srcid:
                indexed[srcid] = dict(item)
        return indexed

    @staticmethod
    def _param_schema(candidate: Mapping[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
        """取出该卡 tool.parameters 的 properties 与 required。"""
        tool = candidate.get("tool", {}) or {}
        parameters = {}
        if isinstance(tool, Mapping):
            parameters = tool.get("parameters", {}) or {}
        if not parameters:
            parameters = candidate.get("parameters", {}) or {}
        if not isinstance(parameters, Mapping):
            return {}, []
        properties = parameters.get("properties", {}) or {}
        required = parameters.get("required", []) or []
        if not isinstance(properties, Mapping):
            properties = {}
        if not isinstance(required, (list, tuple)):
            required = []
        return dict(properties), [str(name) for name in required]

    @staticmethod
    def _is_blank(value: Any) -> bool:
        """判断参数取值是否为空/占位符。"""
        if value is None:
            return True
        if isinstance(value, str):
            return value.strip().lower() in LLMAladdinGenerator.PLACEHOLDER_VALUES
        if isinstance(value, (list, tuple, dict, set)):
            return len(value) == 0
        return False

    # ---------------- 结果校验 ----------------
    @staticmethod
    def _vertify_result(
        response: dict,
        aladdin_candidates: Any = None,
        check_parameters: bool = True,
        check_precise_question: bool = True,
    ) -> Tuple[Dict[str, Any], bool, str]:
        """校验模型输出的筛选与填参结果。

        Args:
            response: 模型输出的 JSON。
            aladdin_candidates: 候选列表，传入后才能校验 srcid 是否编造、参数是否合规。
            check_parameters: 是否按 tool.parameters 校验 required/多余参数。
            check_precise_question: 是否校验精准卡 question 不带修饰词。

        Returns:
            (result, flag, error_type)。selected 为空列表属于合法结果（无合适卡片）。
        """
        if not isinstance(response, Mapping):
            return {}, False, "response type error"

        if len(set(response.keys()) - LLMAladdinGenerator.ALLOWED_KEYS) > 0:
            return {}, False, "key error"
        if "selected" not in response:
            return {}, False, "selected missing"

        selected = response["selected"]
        if selected is None:
            selected = []
        if not isinstance(selected, (list, tuple)):
            return {}, False, "selected type error"

        candidates = LLMAladdinGenerator._index_candidates(aladdin_candidates)
        cleaned: List[Dict[str, Any]] = []
        seen_srcid: set = set()
        seen_question: Dict[str, str] = {}

        for item in selected:
            if not isinstance(item, Mapping):
                return {}, False, "selected item type error"
            if len(set(item.keys()) - LLMAladdinGenerator.ALLOWED_ITEM_KEYS) > 0:
                return {}, False, "selected item key error"

            srcid = str(item.get("srcid", "") or "").strip()
            if not srcid:
                return {}, False, "srcid empty"
            if candidates and srcid not in candidates:
                return {}, False, "srcid not in candidates: %s" % srcid
            if srcid in seen_srcid:
                return {}, False, "duplicate srcid: %s" % srcid
            seen_srcid.add(srcid)

            parameters = item.get("parameters", {})
            if parameters is None:
                parameters = {}
            if not isinstance(parameters, Mapping):
                return {}, False, "parameters type error: %s" % srcid

            result, flag, error_type = LLMAladdinGenerator._check_one_card(
                srcid=srcid,
                parameters=parameters,
                candidate=candidates.get(srcid),
                seen_question=seen_question,
                check_parameters=check_parameters,
                check_precise_question=check_precise_question,
            )
            if not flag:
                return {}, False, error_type
            cleaned.append(result)

        return {
            "reason": str(response.get("reason", "") or ""),
            "selected": cleaned,
        }, True, ""

    @staticmethod
    def _check_one_card(
        srcid: str,
        parameters: Mapping[str, Any],
        candidate: Optional[Mapping[str, Any]],
        seen_question: Dict[str, str],
        check_parameters: bool = True,
        check_precise_question: bool = True,
    ) -> Tuple[Dict[str, Any], bool, str]:
        """单张卡的参数与 question 校验。"""
        # 按 tool.parameters 校验：required 必填非空、不生成多余参数、不留通配符
        if check_parameters and candidate is not None:
            properties, required = LLMAladdinGenerator._param_schema(candidate)
            if properties:
                extra = sorted(set(parameters.keys()) - set(properties.keys()))
                if extra:
                    return {}, False, "extra parameters %s: %s" % (extra, srcid)
            for name in required:
                if name not in parameters:
                    return {}, False, "required missing '%s': %s" % (name, srcid)
                if LLMAladdinGenerator._is_blank(parameters[name]):
                    return {}, False, "required blank '%s': %s" % (name, srcid)
            for name, value in parameters.items():
                if isinstance(value, str) and value.strip().lower() in LLMAladdinGenerator.PLACEHOLDER_VALUES:
                    return {}, False, "placeholder value '%s': %s" % (name, srcid)

        # question 校验：多卡不得共用同一 question；精准卡 question 只填实体本身
        question = parameters.get("question", None)
        if isinstance(question, str) and question.strip():
            text = question.strip()
            if text in seen_question:
                return {}, False, "question reused by %s and %s" % (seen_question[text], srcid)
            seen_question[text] = srcid

            if check_precise_question and candidate is not None:
                name = str(candidate.get("name", "") or "")
                if LLMAladdinGenerator.PRECISE_KEYWORD in name:
                    hit = [tok for tok in LLMAladdinGenerator.MODIFIER_TOKENS if tok in text]
                    if hit:
                        return {}, False, "precise card question with modifier %s: %s" % (hit, srcid)

        return {"srcid": srcid, "parameters": dict(parameters)}, True, ""


            
