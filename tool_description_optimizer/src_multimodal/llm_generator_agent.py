#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Project : General_Tool
@File    : llm_generator_agent.py
@Desc    : 2.4 generator agent —— 依据研判的优化建议定向优化描述

只按 judge 给出的证据与建议改，不做无关重写：无建议则不调用，
避免在没有依据的情况下引入新的偏差。
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from state import MultiModalState, build_description


class LLMMultiModalGenerator:
    """优化 agent 的 prompt 拼装与结果校验。"""

    ALLOWED_KEYS = {"优化说明", "界面布局描述", "功能与服务总结"}
    MIN_SECTION_LENGTH = 60
    MIN_NOTE_LENGTH = 10
    TARGET_MIN, TARGET_MAX = 300, 500
    HARD_MIN, HARD_MAX = 200, 800

    @staticmethod
    def _gen_prompt(
        state: Optional[MultiModalState],
        prompt: str,
        description: Optional[str] = None,
        suggestions: Optional[Sequence[str]] = None,
        evidence: Optional[Sequence[str]] = None,
        label_data: Optional[str] = None,
    ) -> str:
        """拼装优化 prompt。

        没有描述或没有优化建议时返回 ""，由上层跳过——
        无建议就没有定向依据，此时重写只会引入无根据的改动。
        """
        text_description = description
        if text_description is None and state is not None:
            text_description = state.get("description", "")
        text_description = str(text_description or "").strip()
        if not text_description:
            return ""

        items = list(suggestions if suggestions is not None else (state or {}).get("suggestions", []) or [])
        items = [str(item).strip() for item in items if str(item).strip()]
        if not items:
            return ""

        proofs = list(evidence if evidence is not None else (state or {}).get("evidence", []) or [])
        proofs = [str(item).strip() for item in proofs if str(item).strip()]

        text_label = label_data
        if text_label is None and state is not None:
            text_label = state.get("label_data", "")
        text_label = str(text_label or "").strip() or "（无文本召回数据）"

        return (
            prompt.replace("{{description}}", text_description)
            .replace("{{suggestions}}", "\n".join("- %s" % item for item in items))
            .replace("{{evidence}}", "\n".join("- %s" % item for item in proofs) or "（无）")
            .replace("{{label_data}}", text_label)
        )

    @classmethod
    def _vertify_result(cls, response: Any) -> Tuple[Dict[str, Any], bool, str]:
        """校验优化输出，字数口径与描述 agent 保持一致。"""
        if not isinstance(response, Mapping):
            return {}, False, "response type error"
        missing = cls.ALLOWED_KEYS - set(response.keys())
        if missing:
            return {}, False, "missing keys %s" % sorted(missing)
        if len(set(response.keys()) - cls.ALLOWED_KEYS) > 0:
            return {}, False, "key error"

        note = str(response.get("优化说明", "") or "").strip()
        layout = str(response.get("界面布局描述", "") or "").strip()
        function = str(response.get("功能与服务总结", "") or "").strip()
        if len(note) < cls.MIN_NOTE_LENGTH:
            return {}, False, "优化说明 too short"
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
                "[generator] 字数 %d 偏离目标区间 [%d, %d]，已放行"
                % (total, cls.TARGET_MIN, cls.TARGET_MAX)
            )

        return {
            "优化说明": note,
            "界面布局描述": layout,
            "功能与服务总结": function,
            "description": build_description(layout, function),
            "total_length": total,
            "length_ok": length_ok,
        }, True, ""
