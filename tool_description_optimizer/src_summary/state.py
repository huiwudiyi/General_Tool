import argparse
import os
import sys
from typing import Annotated, Any, Callable, Dict, List, Mapping, Optional, Sequence, TypedDict
from dataclasses import dataclass
from utils import *


# =========================
# State definition
# =========================
@dataclass(frozen=True)
class InfoRecord:
    """不可变版本快照，每条迭代记录永久存入history"""
    version_id: int
    stage: str
    info: Dict[str,any]
    
    def to_dict(self) -> dict:
        """序列化导出，用于持久化JSON/入库"""
        return {
            "version_id": self.version_id,
            "stage": self.stage,
            "info": self.info,
        }


@dataclass(frozen=True)
class VersionRecord:
    """不可变版本快照，每条迭代记录永久存入history"""
    version_id: int
    stage: str
    description: str
    tool_path: str
    top_case: Dict[str, any]
    case_result: Dict[str, any]
    parent_version_id: Optional[int]
    recall1: float
    precision1: float
    recall3: float
    precision3: float
    accepted: bool = False
    reason: str = ""

    @classmethod
    def new_record(
        cls,
        version_id: int,
        stage: str,
        description: str,
        tool_path: str,
        top_case: Dict[str, any],
        case_result: Dict[str, any],
        parent_version_id: Optional[int],
        recall1: float,
        precision1: float,
        recall3: float,
        precision3: float,
        accepted: bool = False,
        reason: str = ""
    ) -> 'VersionRecord':
        """构造工厂，简化重复创建逻辑"""
        return cls(
            version_id=version_id,
            parent_version_id=parent_id,
            stage=stage,
            description=description,
            tool_path=tool_path,
            top_case=top_case,
            case_result=case_result,
            recall1=recall1,
            precision1=precision1,
            recall3=recall3,
            precision3=precision3,
            accepted=accepted,
            reason=reason
        )

    def to_dict(self) -> dict:
        """序列化导出，用于持久化JSON/入库"""
        return {
            "version_id": self.version_id,
            "parent_version_id": self.parent_version_id,
            "stage": self.stage,
            "description": self.description,
            "tool_path": self.tool_path,  # 修正：使用一致的字段名
            "top_case": self.top_case,
            "case_result": self.case_result,
            "recall1": self.recall1,
            "precision1": self.precision1,
            "recall3": self.recall3,
            "precision3": self.precision3,
            "accepted": self.accepted,
            "reason": self.reason,
        }

# ====================== 全局状态定义（内置版本指针与历史仓库） ======================
class ToolOptimizerState(TypedDict):
    # 原始基线（只读不变）
    query: Optional[list[str]]
    title: str
    original_description: str
    resource_id: str

    # 当前工作基线指针（可手动/自动回滚）
    current_description: str
    current_version_id: int
    
    # 全局单调自增版本分配器
    next_version_id: int

    # 全局最优
    best_description: Optional[str]
    best_version_id: Optional[int]
    best_record: Optional[VersionRecord]

    # 超参数
    max_iterations: int
    iteration: int

    # 版本历史仓库：全量快照存储（版本控制核心存储）
    version_history: list[VersionRecord]

    # 各阶段输出缓存，调试追溯
    optimizer_history: list[InfoRecord]


# ====================== 通用工具函数 ======================

def append_version_history(state: ToolOptimizerState, record: VersionRecord) -> list[VersionRecord]:
    """追加版本快照至历史仓库，不修改原有记录"""
    return [*state["version_history"], record]

def append_optimizer_history(state: ToolOptimizerState, record: InfoRecord) -> list[InfoRecord]:
    """追加版本快照至历史仓库，不修改原有记录"""
    return [*state["optimizer_history"], record]


def next_version_pair(state: ToolOptimizerState) -> tuple[int, int]:
    """分配全新全局唯一版本ID"""
    curr_vid = state["next_version_id"]
    return curr_vid, curr_vid + 1


# ====================== 版本控制工具集（核心扩展能力） ======================
def get_version_by_id(state: ToolOptimizerState, vid: int) -> Optional[VersionRecord]:
    """根据版本ID精准查询快照"""
    for record in state["history"]:
        if record.version_id == vid:
            return record
    return None

def trace_version_chain(state: ToolOptimizerState, start_vid: int) -> list[VersionRecord]:
    """递归追溯版本完整父链路，生成演化树"""
    chain = []
    curr_vid = start_vid
    while curr_vid is not None:
        rec = get_version_by_id(state, curr_vid)
        if rec is None:
            break
        chain.append(rec)
        curr_vid = rec.parent_version_id
    return chain

def list_accepted_versions(state: ToolOptimizerState) -> list[VersionRecord]:
    """筛选所有纳入主线的优质版本"""
    return [r for r in state["history"] if r.accepted]

def list_rejected_versions(state: ToolOptimizerState) -> list[VersionRecord]:
    """筛选所有被驳回的试验版本"""
    return [r for r in state["history"] if not r.accepted]

def rollback_to_version(state: ToolOptimizerState, target_vid: int) -> dict[str, Any]:
    """手动回滚基线至任意历史版本，仅修改指针不删除历史"""
    target_rec = get_version_by_id(state, target_vid)
    if target_rec is None:
        raise ValueError(f"目标版本 {target_vid} 不存在")
    return {
        "current_version_id": target_rec.version_id,
        "current_text": target_rec.text,
        "candidate_version_id": target_rec.version_id,
        "candidate_text": target_rec.text,
        "candidate_score": target_rec.score,
    }

def export_version_history(state: ToolOptimizerState) -> list[dict]:
    """导出全量版本序列化数据，用于本地存档/入库"""
    return [rec.to_dict() for rec in state["history"]]
