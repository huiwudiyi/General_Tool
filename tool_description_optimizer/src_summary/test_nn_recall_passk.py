#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nn_recall_passk.py 指标计算测试用例

覆盖：
1. calc_pass_at_k 单条 query 级别的 pass@k / RR@k / AP@k
2. evaluate 端到端的 pass@k / mrr_at_k / map_at_k 汇总与明细列

运行：python3 test_nn_recall_passk.py
"""

import os
import sys
import types
import unittest

import numpy as np


def _install_stubs() -> None:
    """桩掉环境中未安装的重依赖，使被测模块可以在无 GPU / 无模型时导入。"""
    if "sentence_transformers" not in sys.modules:
        stub = types.ModuleType("sentence_transformers")

        class SentenceTransformer:  # noqa: D401 - 测试桩
            def __init__(self, *args, **kwargs) -> None:
                pass

            def encode(self, texts, **kwargs):
                return np.zeros((len(texts), 2), dtype=np.float64)

        stub.SentenceTransformer = SentenceTransformer
        sys.modules["sentence_transformers"] = stub

    if "json_repair" not in sys.modules:
        stub = types.ModuleType("json_repair")
        stub.repair_json = lambda text, **kwargs: text
        sys.modules["json_repair"] = stub


_install_stubs()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nn_recall_passk import ToolPassAtKRecallEvaluator  # noqa: E402


def make_recall_results(ids):
    """把 id 列表包装成 recall_results 结构。"""
    return [
        {"rank": i, "id": item_id, "score": 1.0 - i * 0.01, "view": "description"}
        for i, item_id in enumerate(ids, start=1)
    ]


class TestCalcPassAtK(unittest.TestCase):
    """单条 query 级别指标。"""

    def _calc(self, ranked_ids, gold_ids, k):
        return ToolPassAtKRecallEvaluator.calc_pass_at_k(
            recall_results=make_recall_results(ranked_ids),
            gold_ids=gold_ids,
            k=k,
        )

    def test_hit_at_rank2_partial(self):
        """gold={A,C}，排序 [B,A,C]：RR=1/2，AP@3=(1/2+2/3)/2。"""
        m = self._calc(["B", "A", "C"], ["A", "C"], 3)
        self.assertEqual(m["pass"], 1)
        self.assertEqual(m["first_hit_rank"], 2)
        self.assertAlmostEqual(m["reciprocal_rank"], 0.5, places=6)
        self.assertAlmostEqual(m["average_precision"], (0.5 + 2 / 3) / 2, places=6)

    def test_perfect_ranking(self):
        """gold 全部排在最前：RR=1，AP=1。"""
        m = self._calc(["A", "C", "B"], ["A", "C"], 3)
        self.assertEqual(m["pass"], 1)
        self.assertAlmostEqual(m["reciprocal_rank"], 1.0, places=6)
        self.assertAlmostEqual(m["average_precision"], 1.0, places=6)

    def test_no_hit(self):
        """top-k 内无命中：三项指标均为 0。"""
        m = self._calc(["X", "Y"], ["A"], 2)
        self.assertEqual(m["pass"], 0)
        self.assertIsNone(m["first_hit_rank"])
        self.assertAlmostEqual(m["reciprocal_rank"], 0.0, places=6)
        self.assertAlmostEqual(m["average_precision"], 0.0, places=6)

    def test_empty_gold(self):
        """gold 为空：num_relevant=0，AP 记 0（上层会排除出均值分母）。"""
        m = self._calc(["A", "B"], [], 2)
        self.assertEqual(m["num_relevant"], 0)
        self.assertAlmostEqual(m["average_precision"], 0.0, places=6)
        self.assertAlmostEqual(m["reciprocal_rank"], 0.0, places=6)

    def test_recall_below_relevant_count(self):
        """R=3 但 top-3 只命中 2 个：AP 分母取 min(R,k)=3。"""
        m = self._calc(["A", "X", "B"], ["A", "B", "C"], 3)
        self.assertAlmostEqual(m["reciprocal_rank"], 1.0, places=6)
        self.assertAlmostEqual(m["average_precision"], (1.0 + 2 / 3) / 3, places=6)

    def test_gold_dedup(self):
        """重复 gold id 需去重，R 按去重后计算。"""
        m = self._calc(["A", "B"], ["A", "A"], 2)
        self.assertEqual(m["num_relevant"], 1)
        self.assertAlmostEqual(m["average_precision"], 1.0, places=6)

    def test_k_truncation(self):
        """k 截断生效：命中在 rank 3，k=2 时视为未命中。"""
        m = self._calc(["X", "Y", "A", "B"], ["A"], 2)
        self.assertEqual(m["pass"], 0)
        self.assertAlmostEqual(m["reciprocal_rank"], 0.0, places=6)
        m3 = self._calc(["X", "Y", "A", "B"], ["A"], 3)
        self.assertEqual(m3["pass"], 1)
        self.assertAlmostEqual(m3["reciprocal_rank"], 1 / 3, places=6)


class TestEvaluateEndToEnd(unittest.TestCase):
    """端到端汇总指标。

    用 one-hot 工具向量构造可控排序：工具 d1~d4 分别占据 4 个维度，
    query 向量各维取值即为对该工具的相似度，从而精确控制召回顺序。
    """

    TOOLS = {
        "idx1": {"id": "idx1", "title": "t1", "description": "d1"},
        "idx2": {"id": "idx2", "title": "t2", "description": "d2"},
        "idx3": {"id": "idx3", "title": "t3", "description": "d3"},
        "idx4": {"id": "idx4", "title": "t4", "description": "d4"},
    }

    # q1 排序：idx2(1.0) > idx1(0.9) > idx3(0.8) > idx4(0.5)
    # q2 排序：idx4(0.9) > idx2(0.3) > idx1(0.2) > idx3(0.1)
    VECTORS = {
        "d1": [1.0, 0.0, 0.0, 0.0],
        "d2": [0.0, 1.0, 0.0, 0.0],
        "d3": [0.0, 0.0, 1.0, 0.0],
        "d4": [0.0, 0.0, 0.0, 1.0],
        "q1": [0.9, 1.0, 0.8, 0.5],
        "q2": [0.2, 0.3, 0.1, 0.9],
    }

    QUERY_GOLD_IDS = {
        "q1": ["idx1", "idx3"],
        "q2": ["idx4"],
    }

    def _encode_fn(self, texts, **kwargs):
        return np.asarray([self.VECTORS[t] for t in texts], dtype=np.float64)

    def setUp(self):
        self.evaluator = ToolPassAtKRecallEvaluator(
            tools=self.TOOLS,
            query_gold_ids=self.QUERY_GOLD_IDS,
            embeding_term=["description"],
            encode_fn=self._encode_fn,
            batch_size=8,
        )
        self.evaluator.build_vector_index()
        self.summary_df, self.detail_df = self.evaluator.evaluate(
            k_list=[1, 3], eval_views=["description"]
        )

    def _summary_row(self, k):
        rows = self.summary_df[self.summary_df["k"] == k]
        self.assertEqual(len(rows), 1)
        return rows.iloc[0]

    def test_ranking_is_as_designed(self):
        """先确认召回顺序符合设计，后续指标断言才有意义。"""
        q1 = self.detail_df[(self.detail_df["query"] == "q1") & (self.detail_df["k"] == 3)].iloc[0]
        self.assertEqual(list(q1["recall_ids"]), ["idx2", "idx1", "idx3"])
        q2 = self.detail_df[(self.detail_df["query"] == "q2") & (self.detail_df["k"] == 3)].iloc[0]
        self.assertEqual(list(q2["recall_ids"]), ["idx4", "idx2", "idx1"])

    def test_new_columns_exist(self):
        for col in ("mrr_at_k", "map_at_k"):
            self.assertIn(col, self.summary_df.columns)
        for col in ("reciprocal_rank", "average_precision"):
            self.assertIn(col, self.detail_df.columns)

    def test_metrics_at_k1(self):
        """k=1：q1 首位 idx2 未命中；q2 首位 idx4 命中。"""
        row = self._summary_row(1)
        self.assertAlmostEqual(row["pass_at_k"], 0.5, places=6)
        self.assertAlmostEqual(row["mrr_at_k"], (0.0 + 1.0) / 2, places=6)
        self.assertAlmostEqual(row["map_at_k"], (0.0 + 1.0) / 2, places=6)

    def test_metrics_at_k3(self):
        """k=3：q1 RR=1/2、AP=(1/2+2/3)/2；q2 RR=1、AP=1。"""
        row = self._summary_row(3)
        ap_q1 = (0.5 + 2 / 3) / 2
        self.assertAlmostEqual(row["pass_at_k"], 1.0, places=6)
        self.assertAlmostEqual(row["mrr_at_k"], (0.5 + 1.0) / 2, places=6)
        self.assertAlmostEqual(row["map_at_k"], (ap_q1 + 1.0) / 2, places=6)

    def test_empty_gold_excluded_from_denominator(self):
        """空 gold 的 query 不计入 MRR/MAP 分母，但仍计入 pass@k 分母。"""
        query_gold_ids = dict(self.QUERY_GOLD_IDS)
        query_gold_ids["q1"] = []  # q1 无 gold
        evaluator = ToolPassAtKRecallEvaluator(
            tools=self.TOOLS,
            query_gold_ids=query_gold_ids,
            embeding_term=["description"],
            encode_fn=self._encode_fn,
            batch_size=8,
        )
        evaluator.build_vector_index()
        summary_df, _ = evaluator.evaluate(k_list=[3], eval_views=["description"])
        row = summary_df.iloc[0]
        # 只剩 q2 参与 MRR/MAP，二者均为 1.0
        self.assertAlmostEqual(row["mrr_at_k"], 1.0, places=6)
        self.assertAlmostEqual(row["map_at_k"], 1.0, places=6)
        # pass@k 分母仍是 2 条 query
        self.assertEqual(row["query_count"], 2)
        self.assertAlmostEqual(row["pass_at_k"], 0.5, places=6)

    def test_metric_ranges(self):
        """所有指标应落在 [0, 1]。"""
        for _, row in self.summary_df.iterrows():
            for col in ("pass_at_k", "mrr_at_k", "map_at_k"):
                self.assertGreaterEqual(row[col], 0.0)
                self.assertLessEqual(row[col], 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
