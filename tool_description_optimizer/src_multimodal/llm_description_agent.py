#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Project : General_Tool
@File    : llm_description_agent.py
@Desc    : 2.2 描述 agent —— 截图 + 召回文本 → 工具功能描述

输出两段：界面布局描述（客观还原视觉结构）、功能与服务总结（由布局推断能力）。
总字数目标 300-500。
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from state import MultiModalState, build_description


class LLMMultiModalDescription:
    """描述生成 agent 的 prompt 拼装与结果校验。"""

    ALLOWED_KEYS = {"界面布局描述", "功能与服务总结"}
    # 需求规定 300-500 字。字数硬卡死会让整条流水线因为差几个字而产不出描述，
    # 因此分两档：超出 HARD 区间判失败重试；只是偏离 TARGET 区间则放行并告警。
    TARGET_MIN, TARGET_MAX = 300, 500
    HARD_MIN, HARD_MAX = 200, 800
    MIN_SECTION_LENGTH = 60

    @staticmethod
    def _gen_prompt(
        state: Optional[MultiModalState],
        prompt: str,
        label_data: Optional[str] = None,
    ) -> str:
        """把召回文本注入 prompt。

        截图不走 prompt，而是作为 image_url 进 messages（见 image_utils），
        这里只负责文本部分。
        """
        text = label_data
        if text is None and state is not None:
            text = state.get("label_data", "")
        text = str(text or "").strip() or "（无文本召回数据，请仅依据截图分析）"

        if "{{label_data}}" in prompt:
            return prompt.replace("{{label_data}}", text)
        # prompt 里没占位符时追加一段，保证 label_data 一定被带上
        return "%s\n\n# 该组件的召回文本数据（与截图相互印证，综合考量）\n%s\n" % (prompt.rstrip(), text)

    @classmethod
    def _vertify_result(cls, response: Any) -> Tuple[Dict[str, Any], bool, str]:
        """校验描述输出。"""
        if not isinstance(response, Mapping):
            return {}, False, "response type error"
        missing = cls.ALLOWED_KEYS - set(response.keys())
        if missing:
            return {}, False, "missing keys %s" % sorted(missing)
        if len(set(response.keys()) - cls.ALLOWED_KEYS) > 0:
            return {}, False, "key error"

        layout = str(response.get("界面布局描述", "") or "").strip()
        function = str(response.get("功能与服务总结", "") or "").strip()
        if len(layout) < cls.MIN_SECTION_LENGTH:
            return {}, False, "界面布局描述 too short"
        if len(function) < cls.MIN_SECTION_LENGTH:
            return {}, False, "功能与服务总结 too short"

        total = len(layout) + len(function)
        if total < cls.HARD_MIN or total > cls.HARD_MAX:
            return {}, False, "总字数 %d 超出可接受区间 [%d, %d]" % (total, cls.HARD_MIN, cls.HARD_MAX)
        length_ok = cls.TARGET_MIN <= total <= cls.TARGET_MAX
        if not length_ok:
            print(
                "[description] 字数 %d 偏离目标区间 [%d, %d]，已放行"
                % (total, cls.TARGET_MIN, cls.TARGET_MAX)
            )

        return {
            "界面布局描述": layout,
            "功能与服务总结": function,
            "description": build_description(layout, function),
            "total_length": total,
            "length_ok": length_ok,
        }, True, ""
