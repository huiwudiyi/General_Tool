#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Project : General_Tool
@File    : llm_description_refiner.py
@Desc    : 基于失败案例的工具描述定向精修

为什么需要这个模块——现有 judge（llm_description_judge.LLMDescriptionJudge）的局限：

1. 只裁决、不改进：judge 产出 semantic_analysis / scene_comparison /
   function_comparison / content_quality / relevance_reason 五段分析，
   但流程里只取了 relevance_score 做二值拦截，那些「哪里不好」的文字全被丢进
   history 无人消费。
2. 区分度被压成 1 bit：score<2 归零、>=2 不影响奖励，于是 2 分和 3 分完全等价。
3. 拦截即丢弃：被拦的候选直接 reward=0，没有"按意见修一版再评"的机会；
   整组被拦时一个 epoch 完全空转。
4. 判据不足：judge 只看「原描述 vs 新描述」的语义一致性，看不到 query、
   看不到正负例，因此能判"漂移"，判不出"不够"。
5. 与奖励脱耦：judge 跑在 retriever/selector 之前，做判断时拿不到该描述的
   真实检索/选择表现——而恰恰是「在哪些 query 上失败了」最有指导价值。

本模块补的正是第 3~5 点：在奖励算完之后，拿本轮最优描述的**具体失败案例**
（检索漏召 / 选择漏选 / 边界误选）加上 judge 的文字意见，让模型做一次定向重写，
再用同一套奖励复评，只有变好才接受。

失败案例的三分类与三路奖励一一对应：
- missed_positive        ：正例，目标卡没进 topk        → 检索侧问题，功能区覆盖不足
- unselected_positive    ：正例，进了 topk 但裁判没选    → 表述不够明确，裁判认不出
- false_selected_negative：负例，裁判误选了目标卡        → 边界区没写清
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from state import ToolOptimizerState

class LLMDescriptionRefiner:
    """按失败案例定向精修描述。"""

    ALLOWED_KEYS = {"诊断", "功能区", "边界区"}
    MIN_FUNCTION_LENGTH = 20
    MIN_BOUNDARY_LENGTH = 10
    MIN_DIAGNOSIS_LENGTH = 10
    # 单类失败案例最多带入 prompt 的条数，避免 prompt 过长冲淡重点
    MAX_CASES_PER_TYPE = 8

    CASE_TYPES = ("missed_positive", "unselected_positive", "false_selected_negative")

    @staticmethod
    def collect_failure_cases(
        positive_detail: Any,
        negative_detail: Any,
        max_k: int,
    ) -> Dict[str, List[str]]:
        """从 RerankScorer 的明细里归纳三类失败案例。

        Args:
            positive_detail: 正例打分明细（RerankScorer.score_batch 的 detail_df）。
            negative_detail: 负例打分明细。
            max_k: 取哪个 k 的明细行。

        Returns:
            {"missed_positive": [...], "unselected_positive": [...], "false_selected_negative": [...]}
        """
        cases: Dict[str, List[str]] = {name: [] for name in LLMDescriptionRefiner.CASE_TYPES}

        if positive_detail is not None and len(positive_detail) > 0:
            rows = positive_detail[positive_detail["k"] == max_k]
            for _, row in rows.iterrows():
                query = str(row["query"])
                # in_candidates 为 False 说明检索层就没把目标卡送进 topk
                if row["in_candidates"] is False:
                    cases["missed_positive"].append(query)
                elif not int(row["hit_at_k"]):
                    cases["unselected_positive"].append(query)

        if negative_detail is not None and len(negative_detail) > 0:
            rows = negative_detail[negative_detail["k"] == max_k]
            for _, row in rows.iterrows():
                if int(row["hit_at_k"]):
                    cases["false_selected_negative"].append(str(row["query"]))

        return cases

    @staticmethod
    def has_failure(cases: Optional[Mapping[str, Sequence[str]]]) -> bool:
        """是否存在任何失败案例；没有则无需精修。"""
        if not cases:
            return False
        return any(len(cases.get(name) or []) > 0 for name in LLMDescriptionRefiner.CASE_TYPES)

    @staticmethod
    def _format_cases(cases: Optional[Mapping[str, Sequence[str]]]) -> str:
        """把失败案例渲染成 prompt 片段，空类别显式写"无"，避免模型臆测。"""
        labels = {
            "missed_positive": "检索漏召（应命中本工具，但没被检索进候选）",
            "unselected_positive": "选择漏选（已进候选，但裁判没选中本工具）",
            "false_selected_negative": "边界误选（不该命中，裁判却选中了本工具）",
        }
        lines: List[str] = []
        for name in LLMDescriptionRefiner.CASE_TYPES:
            items = list((cases or {}).get(name) or [])[: LLMDescriptionRefiner.MAX_CASES_PER_TYPE]
            body = "、".join(items) if items else "无"
            lines.append(f"- {labels[name]}：{body}")
        return "\n".join(lines)

    @staticmethod
    def _format_judge(judge_info: Optional[Mapping[str, Any]]) -> str:
        """把 judge 的文字意见摘进 prompt——这正是原流程丢掉的信息。"""
        if not judge_info:
            return "无"
        keys = ("relevance_score", "relevance_reason", "content_quality",
                "function_comparison", "scene_comparison")
        parts = []
        for key in keys:
            value = judge_info.get(key)
            if value not in (None, ""):
                parts.append(f"{key}: {value}")
        return "\n".join(parts) if parts else "无"

    @staticmethod
    def _gen_prompt(
        state: Optional[ToolOptimizerState],
        prompt: str,
        description: str,
        failure_cases: Optional[Mapping[str, Sequence[str]]] = None,
        judge_info: Optional[Mapping[str, Any]] = None,
    ) -> str:
        """拼装精修 prompt。

        无失败案例时返回 ""，由上层跳过——没有失败就没有定向修改的依据，
        此时让模型自由重写只会引入无根据的漂移。
        """
        if not description or not str(description).strip():
            return ""
        if not LLMDescriptionRefiner.has_failure(failure_cases):
            return ""

        title = str((state or {}).get("title", "") or "")
        original = str((state or {}).get("original_description", "") or "")
        return (
            prompt.replace("{{title}}", title)
            .replace("{{original_description}}", original)
            .replace("{{current_description}}", str(description))
            .replace("{{failure_cases}}", LLMDescriptionRefiner._format_cases(failure_cases))
            .replace("{{judge_opinion}}", LLMDescriptionRefiner._format_judge(judge_info))
        )

    @staticmethod
    def _vertify_result(response: Any) -> Tuple[Dict[str, Any], bool, str]:
        """校验精修输出 {诊断, 功能区, 边界区}。"""
        if not isinstance(response, Mapping):
            return {}, False, "response type error"
        missing = LLMDescriptionRefiner.ALLOWED_KEYS - set(response.keys())
        if missing:
            return {}, False, "missing keys %s" % sorted(missing)
        if len(set(response.keys()) - LLMDescriptionRefiner.ALLOWED_KEYS) > 0:
            return {}, False, "key error"

        diagnosis = str(response.get("诊断", "") or "").strip()
        function_area = str(response.get("功能区", "") or "").strip()
        boundary_area = str(response.get("边界区", "") or "").strip()
        if len(diagnosis) < LLMDescriptionRefiner.MIN_DIAGNOSIS_LENGTH:
            return {}, False, "诊断 too short"
        if len(function_area) < LLMDescriptionRefiner.MIN_FUNCTION_LENGTH:
            return {}, False, "功能区 too short"
        if len(boundary_area) < LLMDescriptionRefiner.MIN_BOUNDARY_LENGTH:
            return {}, False, "边界区 too short"
        return {"诊断": diagnosis, "功能区": function_area, "边界区": boundary_area}, True, ""

    @staticmethod
    def build_description(structured: Mapping[str, Any]) -> str:
        """与 policy 节点保持一致的描述拼装格式。"""
        return "功能：%s\n边界：%s" % (
            str(structured.get("功能区", "") or ""),
            str(structured.get("边界区", "") or ""),
        )

