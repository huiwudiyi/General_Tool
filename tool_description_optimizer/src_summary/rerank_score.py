#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Project : General_Tool
@File    : rerank_score.py
@Desc    : 基于 llm_generator.LLMAladdinGenerator 的筛选结果，
           以 state["resource_id"] 为 gold 计算 rerank 命中指标。

场景说明：
- 上游召回给出一批候选阿拉丁卡片，LLM 从中筛选出 selected（有序）。
- 本模块的 gold 只有一个——当前正在优化的资源号 state["resource_id"]。
- 因此核心关注：目标卡有没有被选中、被选在第几位、以及上游候选是否包含它。

指标口径：
- hit@k        ：目标 resource_id 是否出现在 selected 前 k 个中。
- MRR@k        ：命中位次的倒数均值；gold 只有一个时 MAP@k 与 MRR@k 恒等，故不再重复输出。
- precision@k  ：hit@k / min(k, 选中数)。因其他被选卡未必是错的（只是没有标注），
                 该值系统性偏低，仅用于观察"是否过度多选"，不适合作为主指标。
- candidate_hit_rate  ：目标卡进入候选的比例，即 rerank 的天花板。
- conditional_hit_rate：目标卡进入候选的前提下被选中的比例，才是 rerank 自身的能力。
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from state import ToolOptimizerState


class RerankScorer:
    """rerank 命中指标计算器。"""

    # tool_generater_main 里用 "-1" 表示资源号缺失
    INVALID_RESOURCE_IDS = {"", "-1", "none", "null", "nan"}

    # ---------------- 输入解析 ----------------
    @staticmethod
    def extract_srcids(selected: Any) -> List[str]:
        """取出 LLM 选中的 srcid，保持模型输出顺序并去重。

        兼容三种入参：
        - `_vertify_result` 的完整结果 {"reason":..., "selected":[...]}
        - selected 列表 [{"srcid":..., "parameters":...}, ...]
        - 纯 srcid 列表 ["111", "222"]
        """
        if selected is None:
            return []
        if isinstance(selected, Mapping):
            selected = selected.get("selected", []) or []
        if isinstance(selected, str):
            selected = [selected]
        if not isinstance(selected, (list, tuple)):
            return []

        srcids: List[str] = []
        seen = set()
        for item in selected:
            if isinstance(item, Mapping):
                srcid = str(item.get("srcid", "") or item.get("src_id", "")).strip()
            else:
                srcid = str(item).strip()
            if srcid and srcid not in seen:
                srcids.append(srcid)
                seen.add(srcid)
        return srcids

    @classmethod
    def resolve_resource_id(
        cls,
        state: Optional[ToolOptimizerState] = None,
        resource_id: Any = None,
    ) -> str:
        """确定 gold 资源号：优先显式入参，其次 state；非法值统一返回 ""。"""
        value = resource_id
        if value is None and state is not None:
            value = state.get("resource_id", None)
        text = str(value).strip() if value is not None else ""
        if text.lower() in cls.INVALID_RESOURCE_IDS:
            return ""
        return text

    @staticmethod
    def _norm_candidates(candidates: Any) -> Optional[List[str]]:
        """把候选列表转成 srcid 列表；未提供则返回 None（表示不做候选层判断）。"""
        if candidates is None:
            return None
        if isinstance(candidates, Mapping):
            return [str(key).strip() for key in candidates.keys() if str(key).strip()]
        srcids = []
        for item in candidates:
            if isinstance(item, Mapping):
                srcid = str(item.get("srcid", "") or item.get("src_id", "")).strip()
            else:
                srcid = str(item).strip()
            if srcid:
                srcids.append(srcid)
        return srcids

    # ---------------- 单条打分 ----------------
    @classmethod
    def score_one(
        cls,
        selected: Any,
        resource_id: Any = None,
        state: Optional[ToolOptimizerState] = None,
        k_list: Optional[Sequence[int]] = None,
        candidates: Any = None,
    ) -> Dict[str, Any]:
        """计算单条 case 的命中指标。

        Args:
            selected: LLM 筛选结果，见 extract_srcids 支持的三种形态。
            resource_id: gold 资源号；缺省时从 state 取。
            state: 全局状态。
            k_list: 需要统计的 k，默认 [1, 3]。
            candidates: 候选阿拉丁列表，传入后才能区分"召回没给"与"rerank 漏选"。

        Returns:
            单条明细 dict；resource_id 非法时 valid=False，上层应排除出分母。
        """
        if k_list is None:
            k_list = [1, 3]
        k_list = sorted({int(k) for k in k_list if int(k) > 0})

        gold = cls.resolve_resource_id(state=state, resource_id=resource_id)
        srcids = cls.extract_srcids(selected)
        candidate_srcids = cls._norm_candidates(candidates)

        rank: Optional[int] = None
        if gold:
            for idx, srcid in enumerate(srcids, start=1):
                if srcid == gold:
                    rank = idx
                    break

        in_candidates: Optional[bool] = None
        if gold and candidate_srcids is not None:
            in_candidates = gold in set(candidate_srcids)

        hit_at_k: Dict[int, int] = {}
        precision_at_k: Dict[int, float] = {}
        for k in k_list:
            hit = int(rank is not None and rank <= k)
            hit_at_k[k] = hit
            denom = min(k, len(srcids))
            precision_at_k[k] = (hit / denom) if denom > 0 else 0.0

        return {
            "valid": bool(gold),
            "resource_id": gold,
            "selected_srcids": srcids,
            "selected_count": len(srcids),
            "candidate_count": len(candidate_srcids) if candidate_srcids is not None else None,
            "in_candidates": in_candidates,
            "hit": int(rank is not None),
            "rank": rank,
            "reciprocal_rank": (1.0 / rank) if rank is not None else 0.0,
            "hit_at_k": hit_at_k,
            "precision_at_k": precision_at_k,
        }

    # ---------------- 批量聚合 ----------------
    @classmethod
    def score_batch(
        cls,
        records: Sequence[Mapping[str, Any]],
        k_list: Optional[Sequence[int]] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """批量计算并汇总指标。

        Args:
            records: 每条 case 一个 dict，可用字段：
                {
                    "query": "原始用户问题",              # 可选，仅用于明细定位
                    "resource_id": "目标资源号",           # 必需（或用 state 提供）
                    "state": {...},                        # 可选，resource_id 缺省时的来源
                    "selected": <_vertify_result 的结果>,  # 必需
                    "candidates": [...],                   # 可选，用于算候选天花板
                }
            k_list: 需要统计的 k，默认 [1, 3]。

        Returns:
            (summary_df, detail_df)：summary 每个 k 一行，detail 每条 case × k 一行。
        """
        if k_list is None:
            k_list = [1, 3]
        k_list = sorted({int(k) for k in k_list if int(k) > 0})

        stat = defaultdict(
            lambda: {
                "case_count": 0,
                "valid_case_count": 0,
                "candidate_known_count": 0,
                "candidate_hit_count": 0,
                "hit_count": 0,
                "reciprocal_rank_sum": 0.0,
                "precision_sum": 0.0,
                "rank_sum": 0,
                "rank_count": 0,
                "selected_count_sum": 0,
                "empty_selected_count": 0,
            }
        )
        detail_rows: List[Dict[str, Any]] = []

        for index, record in enumerate(records):
            metric = cls.score_one(
                selected=record.get("selected", record.get("response")),
                resource_id=record.get("resource_id"),
                state=record.get("state"),
                k_list=k_list,
                candidates=record.get("candidates"),
            )

            for k in k_list:
                hit = metric["hit_at_k"][k]
                rr = (1.0 / metric["rank"]) if (metric["rank"] is not None and metric["rank"] <= k) else 0.0

                bucket = stat[k]
                bucket["case_count"] += 1
                if not metric["valid"]:
                    # resource_id 非法：只计入总数，不参与任何比率
                    detail_rows.append(cls._detail_row(index, record, metric, k, hit, rr))
                    continue

                bucket["valid_case_count"] += 1
                bucket["hit_count"] += hit
                bucket["reciprocal_rank_sum"] += rr
                bucket["precision_sum"] += metric["precision_at_k"][k]
                bucket["selected_count_sum"] += metric["selected_count"]
                if metric["selected_count"] == 0:
                    bucket["empty_selected_count"] += 1
                if hit and metric["rank"] is not None:
                    bucket["rank_sum"] += metric["rank"]
                    bucket["rank_count"] += 1
                if metric["in_candidates"] is not None:
                    bucket["candidate_known_count"] += 1
                    bucket["candidate_hit_count"] += int(metric["in_candidates"])

                detail_rows.append(cls._detail_row(index, record, metric, k, hit, rr))

        summary_df = pd.DataFrame(
            [cls._summary_row(k, stat[k]) for k in k_list]
        ).sort_values("k").reset_index(drop=True)
        detail_df = pd.DataFrame(detail_rows)
        if not detail_df.empty:
            detail_df = detail_df.sort_values(["k", "case_index"]).reset_index(drop=True)
        return summary_df, detail_df

    @staticmethod
    def _detail_row(
        index: int,
        record: Mapping[str, Any],
        metric: Mapping[str, Any],
        k: int,
        hit: int,
        rr: float,
    ) -> Dict[str, Any]:
        """组装单条明细行。"""
        return {
            "case_index": index,
            "query": record.get("query", ""),
            "resource_id": metric["resource_id"],
            "valid": metric["valid"],
            "k": k,
            "selected_srcids": metric["selected_srcids"],
            "selected_count": metric["selected_count"],
            "candidate_count": metric["candidate_count"],
            "in_candidates": metric["in_candidates"],
            "hit_at_k": hit,
            "rank": metric["rank"],
            "reciprocal_rank": rr,
            "precision_at_k": metric["precision_at_k"][k],
        }

    @staticmethod
    def _summary_row(k: int, value: Mapping[str, Any]) -> Dict[str, Any]:
        """把累加器折算成汇总指标。"""
        valid = value["valid_case_count"]
        candidate_known = value["candidate_known_count"]
        candidate_hit = value["candidate_hit_count"]

        hit_rate = value["hit_count"] / valid if valid else 0.0
        candidate_hit_rate = candidate_hit / candidate_known if candidate_known else None
        # 目标卡进入候选的前提下才被选中的比例，剔除上游召回缺失的影响
        conditional_hit_rate = value["hit_count"] / candidate_hit if candidate_hit else None

        return {
            "k": k,
            "case_count": value["case_count"],
            "valid_case_count": valid,
            "hit_count": value["hit_count"],
            "hit_at_k": hit_rate,
            "mrr_at_k": value["reciprocal_rank_sum"] / valid if valid else 0.0,
            "precision_at_k": value["precision_sum"] / valid if valid else 0.0,
            "candidate_hit_rate": candidate_hit_rate,
            "conditional_hit_at_k": conditional_hit_rate,
            "avg_hit_rank": value["rank_sum"] / value["rank_count"] if value["rank_count"] else None,
            "avg_selected_count": value["selected_count_sum"] / valid if valid else 0.0,
            "empty_selected_count": value["empty_selected_count"],
        }

    # ---------------- 结果保存 ----------------
    @staticmethod
    def save_rerank_results(
        summary_df: pd.DataFrame,
        detail_df: pd.DataFrame,
        output_dir: str = "../rerank_outputs",
        prefix: str = "aladdin_rerank_score",
        save_csv: bool = True,
        save_jsonl: bool = True,
    ) -> Dict[str, str]:
        """保存 summary/detail 结果，落盘风格与 nn_recall_passk 保持一致。"""
        os.makedirs(output_dir, exist_ok=True)
        saved_files: Dict[str, str] = {}

        if save_csv:
            summary_path = os.path.join(output_dir, f"{prefix}_summary.csv")
            detail_path = os.path.join(output_dir, f"{prefix}_detail.csv")
            summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
            detail_df.to_csv(detail_path, index=False, encoding="utf-8-sig")
            saved_files["summary_csv"] = summary_path
            saved_files["detail_csv"] = detail_path

        if save_jsonl:
            summary_path = os.path.join(output_dir, f"{prefix}_summary.jsonl")
            detail_path = os.path.join(output_dir, f"{prefix}_detail.jsonl")
            summary_df.to_json(summary_path, orient="records", lines=True, force_ascii=False)
            detail_df.to_json(detail_path, orient="records", lines=True, force_ascii=False)
            saved_files["summary_jsonl"] = summary_path
            saved_files["detail_jsonl"] = detail_path

        print("rerank 指标保存完成：")
        for name, path in saved_files.items():
            print(f"{name}: {path}")
        return saved_files


def rerank_score_function(
    records: Sequence[Mapping[str, Any]],
    k_list: Optional[Sequence[int]] = None,
    output_dir: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """便捷入口：批量打分，可选落盘。与 recall_passk_function 的用法对齐。"""
    summary_df, detail_df = RerankScorer.score_batch(records=records, k_list=k_list)
    if output_dir:
        try:
            RerankScorer.save_rerank_results(
                summary_df=summary_df,
                detail_df=detail_df,
                output_dir=output_dir,
            )
        except Exception as exc:
            print(f"rerank 指标保存失败：{exc}")
    return summary_df, detail_df

