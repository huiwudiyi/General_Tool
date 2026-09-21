#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Project : General_Tool
@File    : llm_judge_agent.py
@Desc    : 2.3 judge agent —— 描述 × 召回文本 × 截图 三方研判

分别对「文本一致性」和「截图一致性」给结论，判断描述是否合理、功能是否完整，
并且必须给出证据（指向截图区域或文本片段）与可执行的优化建议。

为什么要求给证据：只给"不合理"的结论，下游 generator 无从下手；
带证据的建议才能定向修改，这一点和 src_summary 里 refiner 需要失败案例同理。
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from state import MultiModalState


class LLMMultiModalJudge:
    """研判 agent 的 prompt 拼装与结果校验。"""

    ALLOWED_KEYS = {"布局研判", "功能研判", "证据", "优化建议", "评分"}
    MIN_TEXT_LENGTH = 15
    # 评分 1-3：3=合理且完整，2=基本可用有小问题，1=不合理或严重缺失
    VALID_SCORES = (1, 2, 3)

    @staticmethod
    def _gen_prompt(
        state: Optional[MultiModalState],
        prompt: str,
        description: Optional[str] = None,
        label_data: Optional[str] = None,
    ) -> str:
        """拼装研判 prompt；描述为空时返回 "" 由上层跳过。"""
        text_description = description
        if text_description is None and state is not None:
            text_description = state.get("description", "")
        text_description = str(text_description or "").strip()
        if not text_description:
            return ""

        text_label = label_data
        if text_label is None and state is not None:
            text_label = state.get("label_data", "")
        text_label = str(text_label or "").strip() or "（无文本召回数据）"

        return (
            prompt.replace("{{description}}", text_description)
            .replace("{{label_data}}", text_label)
        )

    @classmethod
    def _vertify_result(cls, response: Any) -> Tuple[Dict[str, Any], bool, str]:
        """校验研判输出。

        评分不合理时不直接判失败，而是夹到合法区间——模型偶尔会返回 0 或 5，
        为这种小偏差重试三轮不值得，夹住并告警即可。
        """
        if not isinstance(response, Mapping):
            return {}, False, "response type error"
        missing = cls.ALLOWED_KEYS - set(response.keys())
        if missing:
            return {}, False, "missing keys %s" % sorted(missing)
        if len(set(response.keys()) - cls.ALLOWED_KEYS) > 0:
            return {}, False, "key error"

        layout_judge = str(response.get("布局研判", "") or "").strip()
        function_judge = str(response.get("功能研判", "") or "").strip()
        if len(layout_judge) < cls.MIN_TEXT_LENGTH:
            return {}, False, "布局研判 too short"
        if len(function_judge) < cls.MIN_TEXT_LENGTH:
            return {}, False, "功能研判 too short"

        evidence = cls._as_list(response.get("证据"))
        suggestions = cls._as_list(response.get("优化建议"))
        if not evidence:
            return {}, False, "证据 empty"

        try:
            score = int(float(response.get("评分")))
        except (TypeError, ValueError):
            return {}, False, "评分 not a number"
        if score not in cls.VALID_SCORES:
            clamped = min(max(score, min(cls.VALID_SCORES)), max(cls.VALID_SCORES))
            print(f"[judge] 评分 {score} 越界，夹到 {clamped}")
            score = clamped

        # 评分未满分却不给建议，说明研判没落到可执行动作上，打回重试
        if score < max(cls.VALID_SCORES) and not suggestions:
            return {}, False, "评分未满分但未给出优化建议"

        return {
            "布局研判": layout_judge,
            "功能研判": function_judge,
            "证据": evidence,
            "优化建议": suggestions,
            "评分": score,
        }, True, ""

    @staticmethod
    def _as_list(value: Any) -> List[str]:
        """证据/建议允许模型返回字符串或列表，统一成非空字符串列表。"""
        if value is None:
            return []
        if isinstance(value, str):
            items = [line.strip(" -•\t") for line in value.splitlines()]
            return [item for item in items if item]
        if isinstance(value, (list, tuple)):
            return [str(item).strip() for item in value if str(item).strip()]
        return [str(value).strip()]
