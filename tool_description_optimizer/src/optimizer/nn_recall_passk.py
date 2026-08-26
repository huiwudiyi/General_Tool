#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Project : General_Tool
@File    : nn_recall_passk.py.py
@Author  : zhuzerun
@Date    : 2026-06-29 19:06
@Version : 1.0
@Desc    : 
@Contact : zachary6chu@gmail.com
"""
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
工具向量召回 pass@k 评估脚本

输入格式：
1. tool 集合：
[
    {"id": "idx1", "title": "xxx", "description": "xxx"},
    {"id": "idx2", "title": "yyy", "description": "yyy"}
]

2. query 测试集合：
{
    "query2": ["idx1", "idx4"],
    "query3": ["idx2"]
}

评估逻辑：
- 使用 title / description / title + description 三路文本构建工具底库向量
- 使用 query 文本生成 query 向量
- query 向量召回 tool 集合
- 分别计算 title / description / title_description / merged 的 pass@k
"""

import os
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tool_description_optimizer.common.utils.utils import l2_normalize, cosine_similarity, load_tools_from_json

MODEL_PATH = "../../../afs_data/model/Qwen3-Embedding-8B"
EMB_MODEL = SentenceTransformer(MODEL_PATH, device='cuda:1')


def encode_fn(texts: List[str], batch_size=64, normalize_embeddings=False, show_progress_bar=False) -> np.ndarray:
    """向量模型编码函数。"""
    embeddings = EMB_MODEL.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=normalize_embeddings,
        show_progress_bar=show_progress_bar,
    )
    return np.asarray(embeddings)


class DescriptionComparator:
    """Compares original and optimized descriptions with exact, normalized and semantic checks."""

    def __init__(self, encoder: Any | None = None, semantic_threshold: float = 0.985) -> None:
        self.encode_fn = encode_fn or encoder
        self.semantic_threshold = semantic_threshold

    def compare(self, original_description: str, optimized_description: str) -> dict[str, Any]:
        same_exact = original_description == optimized_description
        same_normalized = normalize_text(original_description) == normalize_text(optimized_description)
        embeddings = self.encode_fn([original_description, optimized_description], batch_size=2)
        semantic_similarity = 1.0 if same_normalized else cosine_similarity(embeddings[0], embeddings[1])

        changed = not (same_exact or same_normalized or semantic_similarity > self.semantic_threshold)

        return {
            "same_exact": same_exact,
            "same_normalized": same_normalized,
            "semantic_similarity": round(semantic_similarity, 6),
            "changed": changed,
        }


class ToolPassAtKRecallEvaluator:
    """工具集合向量召回 pass@k 评估器。"""

    def __init__(
            self,
            tools: Dict[str, Dict],
            query_gold_ids: Dict[str, Sequence[str]],
            embeding_term: List[str],
            encode_fn: Callable[[List[str]], np.ndarray],
            batch_size: int = 64,
    ):
        """
        Args:
            tools:
                工具集合，格式：
                {
                    "idx1":{"id": "idx1", "title": "xxx", "description": "xxx"},
                    ...
                }

            query_gold_ids:
                测试 query 集合，格式：
                {
                    "query2": ["idx1", "idx4"],
                    ...
                }
                其中 value 是该 query 对应的正确工具 id 列表。

            encode_fn:
                向量模型编码函数，输入 List[str]，返回 np.ndarray，shape=[N, dim]。

            batch_size:
                embedding 批大小。
        """
        self.tools = tools
        self.query_gold_ids = query_gold_ids
        self.encode_fn = encode_fn
        self.batch_size = batch_size
        self.embeding_term = embeding_term

        self.ids: List[str] = []
        self.id_to_tool: Dict[str, Dict[str, Any]] = {}
        self.view_texts: Dict[str, List[str]] = {
            "title": [],
            "description": [],
            "title_description": [],
        }
        self.view_embeddings: Dict[str, np.ndarray] = {}

    def _batch_encode(self, texts: List[str]) -> np.ndarray:
        """分批编码文本，并做 L2 normalize。"""
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i: i + self.batch_size]
            emb = self.encode_fn(batch_texts)
            if not isinstance(emb, np.ndarray):
                emb = np.asarray(emb)
            all_embeddings.append(emb)

        embeddings = np.vstack(all_embeddings)
        embeddings = l2_normalize(embeddings)
        return embeddings

    @staticmethod
    def _clean_text(value: Any) -> str:
        """将 None / nan / 其他对象安全转为字符串。"""
        if value is None:
            return ""
        try:
            if pd.isna(value):
                return ""
        except Exception:
            pass
        return str(value).strip()

    @staticmethod
    def _normalize_gold_ids(gold_ids: Sequence[str]) -> List[str]:
        """标准化 gold ids，去空、去重、转 string。"""
        if gold_ids is None:
            return []
        if isinstance(gold_ids, str):
            gold_ids = [gold_ids]

        seen = set()
        normalized = []
        for item in gold_ids:
            item_id = str(item).strip()
            if item_id and item_id not in seen:
                normalized.append(item_id)
                seen.add(item_id)
        return normalized

    def build_vector_index(self) -> None:
        """构建工具底库向量：title / description / title + description 三路。"""
        self.ids = []
        self.id_to_tool = {}
        self.view_texts = {
            "title": [],
            "description": [],
            "title_description": [],
        }
        self.view_embeddings = {}

        duplicate_ids = []
        for key, item in self.tools.items():
            item_id = self._clean_text(item.get("id"))
            if not item_id:
                continue

            if item_id in self.id_to_tool:
                duplicate_ids.append(item_id)
                continue

            title = self._clean_text(item.get("title"))
            description = self._clean_text(item.get("description"))
            title_description = f"{title}\n{description}".strip()

            self.ids.append(item_id)
            self.id_to_tool[item_id] = item
            self.view_texts["title"].append(title)
            self.view_texts["description"].append(description)
            self.view_texts["title_description"].append(title_description)

        if duplicate_ids:
            print(f"Warning: skipped duplicate tool ids: {duplicate_ids[:10]}, total={len(duplicate_ids)}")

        for view_name, texts in self.view_texts.items():
            if view_name not in self.embeding_term:
                continue
            print("开始建库:", view_name)
            print(f"Building embeddings for view={view_name}, size={len(texts)}")
            self.view_embeddings[view_name] = self._batch_encode(texts)

        print(f"Total tools indexed: {len(self.ids)}")
        print(f"Total eval queries: {len(self.query_gold_ids)}")

    def recall_one_view(
            self,
            query_embedding: np.ndarray,
            view_name: str,
            top_k: int,
    ) -> List[Dict[str, Any]]:
        """单路召回。"""
        if view_name not in self.view_embeddings:
            raise ValueError(f"Unknown view_name={view_name}, available={list(self.view_embeddings)}")

        doc_embeddings = self.view_embeddings[view_name]
        scores = np.dot(doc_embeddings, query_embedding)
        real_top_k = min(top_k, len(scores))
        top_indices = np.argsort(-scores)[:real_top_k]

        results = []
        for rank, idx in enumerate(top_indices, start=1):
            item_id = self.ids[idx]
            tool = self.id_to_tool.get(item_id, {})
            results.append(
                {
                    "rank": rank,
                    "id": item_id,
                    "score": float(scores[idx]),
                    "view": view_name,
                    "title": self._clean_text(tool.get("title")),
                    "description": self._clean_text(tool.get("description")),
                }
            )
        return results

    def recall_merged(
            self,
            query_embedding: np.ndarray,
            top_k: int,
    ) -> List[Dict[str, Any]]:
        """
        多路融合召回。

        对每个工具 id，计算其在 title / description / title_description 三路中的最高相似度，
        最终按最高相似度排序。
        """
        id_best_result: Dict[str, Dict[str, Any]] = {}

        for view_name, doc_embeddings in self.view_embeddings.items():
            scores = np.dot(doc_embeddings, query_embedding)

            for idx, score in enumerate(scores):
                item_id = self.ids[idx]
                score = float(score)
                tool = self.id_to_tool.get(item_id, {})

                if item_id not in id_best_result or score > id_best_result[item_id]["score"]:
                    id_best_result[item_id] = {
                        "id": item_id,
                        "score": score,
                        "view": view_name,
                        "title": self._clean_text(tool.get("title")),
                        "description": self._clean_text(tool.get("description")),
                    }

        merged_results = sorted(
            id_best_result.values(),
            key=lambda x: x["score"],
            reverse=True,
        )[: min(top_k, len(id_best_result))]

        for rank, item in enumerate(merged_results, start=1):
            item["rank"] = rank

        return merged_results

    @staticmethod
    def calc_pass_at_k(
            recall_results: List[Dict[str, Any]],
            gold_ids: Sequence[str],
            k: int,
    ) -> Dict[str, Any]:
        """计算单条 query 的 pass@k。"""
        normalized_gold_ids = set(ToolPassAtKRecallEvaluator._normalize_gold_ids(gold_ids))
        top_results = recall_results[:k]
        top_ids = [x["id"] for x in top_results]
        hit_ids = [item_id for item_id in top_ids if item_id in normalized_gold_ids]

        first_hit_rank: Optional[int] = None
        for idx, item_id in enumerate(top_ids, start=1):
            if item_id in normalized_gold_ids:
                first_hit_rank = idx
                break

        return {
            "pass": int(len(hit_ids) > 0),
            "hit_ids": hit_ids,
            "first_hit_rank": first_hit_rank,
            "top_ids": top_ids,
        }

    def evaluate(
            self,
            k_list: List[int] = None,
            eval_views: List[str] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        计算 pass@k。

        Args:
            k_list:
                需要计算的 k，例如 [1, 3, 5, 10]。

            eval_views:
                需要评估的召回视角，默认：
                ["title", "description", "title_description", "merged"]。

        Returns:
            summary_df:
                汇总指标，每个 view + k 一行。

            detail_df:
                明细结果，每个 query + view + k 一行。
        """
        if not self.view_embeddings:
            raise RuntimeError("Please call build_vector_index() first.")

        if k_list is None:
            k_list = [1, 3, 5, 10]
        k_list = sorted(set(int(k) for k in k_list if int(k) > 0))
        max_k = max(k_list)

        if eval_views is None:
            eval_views = ["title", "description", "title_description", "merged"]

        queries = [str(query).strip() for query in self.query_gold_ids.keys()]
        query_embeddings = self._batch_encode(queries)

        stat = defaultdict(
            lambda: {
                "query_count": 0,
                "pass_count": 0,
                "first_hit_rank_sum": 0.0,
                "first_hit_rank_count": 0,
                "missing_gold_query_count": 0,
            }
        )
        detail_rows: List[Dict[str, Any]] = []

        tool_id_set = set(self.ids)

        for i, query in enumerate(queries):
            query_embedding = query_embeddings[i]
            gold_ids = self._normalize_gold_ids(self.query_gold_ids.get(query, []))
            missing_gold_ids = [item_id for item_id in gold_ids if item_id not in tool_id_set]

            for view_name in eval_views:
                if view_name == "merged":
                    recall_results = self.recall_merged(query_embedding=query_embedding, top_k=max_k)
                else:
                    recall_results = self.recall_one_view(
                        query_embedding=query_embedding,
                        view_name=view_name,
                        top_k=max_k,
                    )

                for k in k_list:
                    metric = self.calc_pass_at_k(
                        recall_results=recall_results,
                        gold_ids=gold_ids,
                        k=k,
                    )
                    key = (view_name, k)
                    stat[key]["query_count"] += 1
                    stat[key]["pass_count"] += metric["pass"]
                    if missing_gold_ids:
                        stat[key]["missing_gold_query_count"] += 1

                    if metric["first_hit_rank"] is not None:
                        stat[key]["first_hit_rank_sum"] += metric["first_hit_rank"]
                        stat[key]["first_hit_rank_count"] += 1

                    detail_rows.append(
                        {
                            "query": query,
                            "gold_ids": gold_ids,
                            "missing_gold_ids": missing_gold_ids,
                            "view": view_name,
                            "k": k,
                            "pass_at_k": metric["pass"],
                            "hit_ids": metric["hit_ids"],
                            "first_hit_rank": metric["first_hit_rank"],
                            "recall_ids": metric["top_ids"],
                            "recall_scores": [x["score"] for x in recall_results[:k]],
                            "recall_views": [x["view"] for x in recall_results[:k]],
                        }
                    )

        summary_rows = []
        for (view_name, k), value in stat.items():
            query_count = value["query_count"]
            pass_count = value["pass_count"]
            first_hit_rank_count = value["first_hit_rank_count"]

            pass_at_k = pass_count / query_count if query_count else 0.0
            avg_first_hit_rank = (
                value["first_hit_rank_sum"] / first_hit_rank_count
                if first_hit_rank_count > 0
                else None
            )

            summary_rows.append(
                {
                    "view": view_name,
                    "k": k,
                    "query_count": query_count,
                    "pass_count": pass_count,
                    "pass_at_k": pass_at_k,
                    "avg_first_hit_rank": avg_first_hit_rank,
                    "missing_gold_query_count": value["missing_gold_query_count"],
                }
            )

        summary_df = pd.DataFrame(summary_rows).sort_values(["view", "k"]).reset_index(drop=True)
        detail_df = pd.DataFrame(detail_rows).sort_values(["view", "query", "k"]).reset_index(drop=True)
        return summary_df, detail_df

    @staticmethod
    def save_recall_results(
            summary_df: pd.DataFrame,
            detail_df: pd.DataFrame,
            output_dir: str = "../recall_outputs",
            prefix: str = "tool_pass_at_k_recall",
            save_csv: bool = True,
            save_jsonl: bool = True,
            save_excel: bool = True,
    ) -> Dict[str, str]:
        """保存 summary/detail 结果。"""
        os.makedirs(output_dir, exist_ok=True)
        saved_files = {}

        if save_csv:
            summary_csv_path = os.path.join(output_dir, f"{prefix}_summary.csv")
            detail_csv_path = os.path.join(output_dir, f"{prefix}_detail.csv")
            summary_df.to_csv(summary_csv_path, index=False, encoding="utf-8-sig")
            detail_df.to_csv(detail_csv_path, index=False, encoding="utf-8-sig")
            saved_files["summary_csv"] = summary_csv_path
            saved_files["detail_csv"] = detail_csv_path

        if save_jsonl:
            summary_jsonl_path = os.path.join(output_dir, f"{prefix}_summary.jsonl")
            detail_jsonl_path = os.path.join(output_dir, f"{prefix}_detail.jsonl")
            summary_df.to_json(summary_jsonl_path, orient="records", lines=True, force_ascii=False)
            detail_df.to_json(detail_jsonl_path, orient="records", lines=True, force_ascii=False)
            saved_files["summary_jsonl"] = summary_jsonl_path
            saved_files["detail_jsonl"] = detail_jsonl_path

        if save_excel:
            excel_path = os.path.join(output_dir, f"{prefix}.xlsx")
            with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
                summary_df.to_excel(writer, sheet_name="summary", index=False)
                detail_df.to_excel(writer, sheet_name="detail", index=False)
            saved_files["excel"] = excel_path

        print("召回结果保存完成：")
        for name, path in saved_files.items():
            print(f"{name}: {path}")
        return saved_files


def recall_passk_function(tools, query_gold_ids, srcid, output_dir, eval_view="title_description"):
    """
    eval_view: 用于上层指标计算的召回视角开关。
        - 'merged': 三路(title/description/title_description)取最高分融合
        - 'title_description': 仅 title+description 拼接单路召回（默认）
        - 'description': 仅 description 单路召回
    注意：无论 eval_view 取什么值，评估仍会计算并保存所有路的结果，
          eval_view 只影响返回给上层 statistic_check_tool 的 detaildf 过滤。
    """
    embeding_view_term = ["description", "title_description", "merged"]

    evaluator = ToolPassAtKRecallEvaluator(
        tools=tools,
        query_gold_ids=query_gold_ids,
        embeding_term=embeding_view_term,
        encode_fn=encode_fn,
        batch_size=64
    )
    evaluator.build_vector_index()

    summary_df, detail_df = evaluator.evaluate(
        k_list=[1, 3],
        eval_views=embeding_view_term
    )

    try:
        evaluator.save_recall_results(
            summary_df=summary_df,
            detail_df=detail_df,
            output_dir=output_dir,
            prefix="tool_pass_at_k_recall",
        )
    except:
        pass
    return detail_df


def main():
    """示例主函数：你可以替换为自己的 JSON 文件路径。"""
    tools = {
        "idx1": {"id": "idx1", "title": "工具1", "description": "这是工具1的描述"},
        "idx2": {"id": "idx2", "title": "工具2", "description": "这是工具2的描述"},
        "idx3": {"id": "idx3", "title": "工具3", "description": "这是工具3的描述"},
        "idx4": {"id": "idx4", "title": "工具4", "description": "这是工具4的描述"},
    }
    query_gold_ids = {"query2": ["idx1", "idx4"]}

    # 方式 1：直接写入 Python 变量
    tools = load_tools_from_json("../data/tool_descriptions.json")

    # 方式 2：从 JSON 文件读取
    query_gold_ids = load_tools_from_json("../data/query.json")
    all_query_ids = {}
    for srcid, qgs in query_gold_ids.items():
        for k, v in qgs.items():
            all_query_ids[k] = v

    embeding_view_term = ["description", "title_description", "merged"]

    evaluator = ToolPassAtKRecallEvaluator(
        tools=tools,
        query_gold_ids=all_query_ids,
        embeding_term=embeding_view_term,
        encode_fn=encode_fn,
        batch_size=64
    )
    evaluator.build_vector_index()

    summary_df, detail_df = evaluator.evaluate(
        k_list=[1, 3],
        eval_views=embeding_view_term
    )

    evaluator.save_recall_results(
        summary_df=summary_df,
        detail_df=detail_df,
        output_dir="./recall_outputs/",
        prefix="tool_pass_at_k_recall",
    )

# if __name__ == "__main__":
#     main()