#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Project : General_Tool
@File    : state.py
@Desc    : 多模态工具功能描述流程的全局状态定义

与 src_summary/state.py 的分工：那边服务于「向量召回描述优化 + GRPO」，
这边服务于「截图 + 召回文本 → 功能描述 → 研判 → 优化」的三段式流程，
状态字段完全不同，因此独立定义，不复用。
"""

import os
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, TypedDict

from utils import *


@dataclass(frozen=True)
class StageRecord:
    """单个阶段的不可变输出快照，用于追溯每一轮的中间结果。"""

    srcid: str
    round_id: int
    stage: str
    info: Dict[str, Any]

    def to_dict(self) -> dict:
        return {
            "srcid": self.srcid,
            "round_id": self.round_id,
            "stage": self.stage,
            "info": self.info,
        }


class MultiModalState(TypedDict):
    """多模态描述流程的状态。

    一个 srcid 一条流程：把该 srcid 下所有 query 的召回文本与截图聚合起来，
    先生成描述，再用「文本 + 截图」双路研判，最后按建议优化，不达标则再研判。
    """

    # ---- 输入基线（只读） ----
    srcid: str
    queries: list[str]
    label_data: str
    screenshots: list[str]

    # ---- 当前工作值 ----
    description: str
    layout_text: str
    function_text: str

    # ---- 研判结果 ----
    judge_score: Optional[int]
    judge_passed: bool
    evidence: list[str]
    suggestions: list[str]

    # ---- 循环控制 ----
    round_id: int
    max_rounds: int
    min_pass_score: int

    # ---- 产物 ----
    final_description: str
    accepted: bool

    # ---- 各阶段历史 ----
    description_history: list[StageRecord]
    judge_history: list[StageRecord]
    generator_history: list[StageRecord]


# ====================== 通用工具函数 ======================

def append_description_history(state: MultiModalState, record: StageRecord) -> list[StageRecord]:
    """追加描述生成记录，不修改原有记录。"""
    return [*state.get("description_history", []), record]


def append_judge_history(state: MultiModalState, record: StageRecord) -> list[StageRecord]:
    """追加研判记录，不修改原有记录。"""
    return [*state.get("judge_history", []), record]


def append_generator_history(state: MultiModalState, record: StageRecord) -> list[StageRecord]:
    """追加优化记录，不修改原有记录。"""
    return [*state.get("generator_history", []), record]


def build_description(layout_text: str, function_text: str) -> str:
    """统一的描述拼装格式，三个 agent 共用，避免各处拼法不一致。"""
    return "界面布局描述：%s\n功能与服务总结：%s" % (
        str(layout_text or "").strip(),
        str(function_text or "").strip(),
    )
