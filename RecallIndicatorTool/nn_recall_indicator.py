#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Project : General_Tool
@File    : nn_recall_indicator.py
@Author  : zhuzerun
@Date    : 2026-06-11 17:31
@Version : 1.0
@Desc    : 
@Contact : zachary6chu@gmail.com
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Callable, Any, Tuple
from collections import defaultdict
from sentence_transformers import SentenceTransformer
import numpy as np


model = SentenceTransformer("/root/paddlejob/workspace/env_run/output/zacharychu/afs_data/model/Qwen3-Embedding-8B")


def l2_normalize(vectors: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    向量归一化，用于余弦相似度计算。
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, eps)


class MultiViewVectorRecallEvaluator:
    """
    多视角向量召回评估器。

    支持：
    1. title 向量
    2. description 向量
    3. title + description 向量
    4. merged：合并三路向量召回结果，同一个 id 取最高分
    """

    def __init__(
        self,
        data: List[Dict[str, Any]],
        encode_fn: Callable[[List[str]], np.ndarray],
        batch_size: int = 64,
    ):
        """
        Args:
            data:
                数据集，每条数据格式示例：
                {
                    "id": "1",
                    "title": "xxx",
                    "description": "xxx",
                    "call_querys_list": ["query1", "query2"]
                }

            encode_fn:
                向量模型编码函数。
                输入 List[str]，返回 np.ndarray，shape = [N, dim]

            batch_size:
                批量请求模型的大小。
        """
        self.data = data
        self.encode_fn = encode_fn
        self.batch_size = batch_size

        self.ids = []
        self.view_texts = {
            "title": [],
            "description": [],
            "title_description": [],
        }

        self.view_embeddings = {}
        self.query_records = []

    def _batch_encode(self, texts: List[str]) -> np.ndarray:
        """
        分批调用 embedding 模型。
        """
        all_embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            emb = self.encode_fn(batch_texts)

            if not isinstance(emb, np.ndarray):
                emb = np.array(emb)

            all_embeddings.append(emb)

        embeddings = np.vstack(all_embeddings)
        embeddings = l2_normalize(embeddings)
        return embeddings
    def save_recall_results(
        self,
        summary_df: pd.DataFrame,
        detail_df: pd.DataFrame,
        output_dir: str = "./recall_outputs",
        prefix: str = "recall_eval",
        save_csv: bool = True,
        save_jsonl: bool = True,
        save_excel: bool = True,
    ):
        """
        保存召回评估结果。

        会保存两类结果：
        1. summary_df：整体评估指标
        2. detail_df：每条 query 的召回详情

        Args:
            summary_df:
                evaluate() 返回的汇总结果。

            detail_df:
                evaluate() 返回的明细结果。

            output_dir:
                输出目录。

            prefix:
                输出文件名前缀。

            save_csv:
                是否保存 csv。

            save_jsonl:
                是否保存 jsonl。

            save_excel:
                是否保存 excel。

        Returns:
            保存后的文件路径字典。
        """
        import os
        import json

        os.makedirs(output_dir, exist_ok=True)

        saved_files = {}

        if save_csv:
            summary_csv_path = os.path.join(output_dir, f"{prefix}_summary.csv")
            detail_csv_path = os.path.join(output_dir, f"{prefix}_detail.csv")

            summary_df.to_csv(
                summary_csv_path,
                index=False,
                encoding="utf-8-sig",
            )
            detail_df.to_csv(
                detail_csv_path,
                index=False,
                encoding="utf-8-sig",
            )

            saved_files["summary_csv"] = summary_csv_path
            saved_files["detail_csv"] = detail_csv_path

        if save_jsonl:
            summary_jsonl_path = os.path.join(output_dir, f"{prefix}_summary.jsonl")
            detail_jsonl_path = os.path.join(output_dir, f"{prefix}_detail.jsonl")

            with open(summary_jsonl_path, "w", encoding="utf-8") as f:
                for row in summary_df.to_dict(orient="records"):
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

            with open(detail_jsonl_path, "w", encoding="utf-8") as f:
                for row in detail_df.to_dict(orient="records"):
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

            saved_files["summary_jsonl"] = summary_jsonl_path
            saved_files["detail_jsonl"] = detail_jsonl_path

        if save_excel:
            excel_path = os.path.join(output_dir, f"{prefix}.xlsx")

            with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
                summary_df.to_excel(
                    writer,
                    sheet_name="summary",
                    index=False,
                )
                detail_df.to_excel(
                    writer,
                    sheet_name="detail",
                    index=False,
                )

            saved_files["excel"] = excel_path

        print("召回结果保存完成：")
        for name, path in saved_files.items():
            print(f"{name}: {path}")

        return saved_files
    def build_vector_index(self):
        """
        构建 title / description / title + description 三路向量集合。
        """
        self.ids = []
        self.view_texts = {
            "title": [],
            "description": [],
            "title_description": [],
        }
        self.query_records = []

        for item in self.data:
            item_id = str(item["id"])
            title = item.get("title", "") or ""
            description = item.get("description", "") or ""

            title = title.strip()
            description = description.strip()
            title_description = f"{title}\n{description}".strip()

            self.ids.append(item_id)
            self.view_texts["title"].append(title)
            self.view_texts["description"].append(description)
            self.view_texts["title_description"].append(title_description)

            for query in item.get("call_querys_list", []):
                query = str(query).strip()
                if query:
                    self.query_records.append({
                        "query": query,
                        "target_id": item_id,
                    })

        for view_name, texts in self.view_texts.items():
            print(f"Building embeddings for view: {view_name}, size={len(texts)}")
            self.view_embeddings[view_name] = self._batch_encode(texts)

        print(f"Total ids: {len(self.ids)}")
        print(f"Total eval queries: {len(self.query_records)}")

    def recall_one_view(
        self,
        query_embedding: np.ndarray,
        view_name: str,
        top_k: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        单路向量召回。
        """
        doc_embeddings = self.view_embeddings[view_name]

        scores = np.dot(doc_embeddings, query_embedding)

        top_indices = np.argsort(-scores)[:top_k]

        results = []
        for idx in top_indices:
            results.append({
                "id": self.ids[idx],
                "score": float(scores[idx]),
                "view": view_name,
            })

        return results

    def recall_merged(
        self,
        query_embedding: np.ndarray,
        top_k: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        合并 title / description / title_description 三路召回结果。

        同一个 id 多次出现时，保留最高相似度。
        """
        id_best_result = {}

        for view_name in self.view_embeddings.keys():
            view_results = self.recall_one_view(
                query_embedding=query_embedding,
                view_name=view_name,
                top_k=top_k,
            )

            for result in view_results:
                item_id = result["id"]
                score = result["score"]

                if item_id not in id_best_result:
                    id_best_result[item_id] = result
                else:
                    if score > id_best_result[item_id]["score"]:
                        id_best_result[item_id] = result

        merged_results = list(id_best_result.values())
        merged_results.sort(key=lambda x: x["score"], reverse=True)

        return merged_results[:top_k]

    @staticmethod
    def filter_by_threshold(
        results: List[Dict[str, Any]],
        threshold: float,
    ) -> List[Dict[str, Any]]:
        """
        按相似度阈值过滤召回结果。
        """
        return [x for x in results if x["score"] >= threshold]

    @staticmethod
    def calc_one_query_metrics(
        filtered_results: List[Dict[str, Any]],
        target_id: str,
    ) -> Dict[str, Any]:
        """
        计算单个 query 的召回指标。
        """
        candidate_ids = [x["id"] for x in filtered_results]

        hit = target_id in candidate_ids

        rank = None
        reciprocal_rank = 0.0

        if hit:
            rank = candidate_ids.index(target_id) + 1
            reciprocal_rank = 1.0 / rank

        top1_hit = len(candidate_ids) > 0 and candidate_ids[0] == target_id

        # 单正样本场景下的 precision，更多是观察过滤后的候选纯度
        precision = 0.0
        if len(candidate_ids) > 0:
            precision = 1.0 / len(candidate_ids) if hit else 0.0

        return {
            "hit": int(hit),
            "top1_hit": int(top1_hit),
            "rank": rank,
            "reciprocal_rank": reciprocal_rank,
            "candidate_count": len(candidate_ids),
            "precision": precision,
        }

    def evaluate(
        self,
        thresholds: List[float],
        top_k: int = 50,
        eval_views: List[str] = None,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        评估多种召回效果。

        Args:
            thresholds:
                多个相似度阈值，例如 [0.3, 0.4, 0.5, 0.6, 0.7]

            top_k:
                每路召回 top_k

            eval_views:
                需要评估的召回方式。
                默认：
                ["title", "description", "title_description", "merged"]

        Returns:
            summary_results:
                汇总统计结果

            detail_results:
                每个 query 的详细召回结果
        """
        if eval_views is None:
            eval_views = [
                "title",
                "description",
                "title_description",
                "merged",
            ]

        if not self.view_embeddings:
            raise RuntimeError("Please call build_vector_index() first.")

        queries = [x["query"] for x in self.query_records]
        query_embeddings = self._batch_encode(queries)

        stat = defaultdict(lambda: {
            "query_count": 0,
            "hit_count": 0,
            "top1_hit_count": 0,
            "mrr_sum": 0.0,
            "candidate_count_sum": 0,
            "precision_sum": 0.0,
            "rank_sum": 0.0,
            "rank_hit_count": 0,
        })

        detail_results = []

        for i, record in enumerate(self.query_records):
            query = record["query"]
            target_id = record["target_id"]
            query_embedding = query_embeddings[i]

            for view_name in eval_views:
                if view_name == "merged":
                    raw_results = self.recall_merged(
                        query_embedding=query_embedding,
                        top_k=top_k,
                    )
                else:
                    raw_results = self.recall_one_view(
                        query_embedding=query_embedding,
                        view_name=view_name,
                        top_k=top_k,
                    )

                for threshold in thresholds:
                    filtered_results = self.filter_by_threshold(
                        raw_results,
                        threshold,
                    )

                    one_metrics = self.calc_one_query_metrics(
                        filtered_results=filtered_results,
                        target_id=target_id,
                    )

                    key = (view_name, threshold)

                    stat[key]["query_count"] += 1
                    stat[key]["hit_count"] += one_metrics["hit"]
                    stat[key]["top1_hit_count"] += one_metrics["top1_hit"]
                    stat[key]["mrr_sum"] += one_metrics["reciprocal_rank"]
                    stat[key]["candidate_count_sum"] += one_metrics["candidate_count"]
                    stat[key]["precision_sum"] += one_metrics["precision"]

                    if one_metrics["rank"] is not None:
                        stat[key]["rank_sum"] += one_metrics["rank"]
                        stat[key]["rank_hit_count"] += 1

                    detail_results.append({
                        "query": query,
                        "target_id": target_id,
                        "view": view_name,
                        "threshold": threshold,
                        "hit": one_metrics["hit"],
                        "top1_hit": one_metrics["top1_hit"],
                        "rank": one_metrics["rank"],
                        "candidate_count": one_metrics["candidate_count"],
                        "precision": one_metrics["precision"],
                        "recall_ids": [x["id"] for x in filtered_results],
                        "recall_scores": [x["score"] for x in filtered_results],
                        "recall_views": [x["view"] for x in filtered_results],
                    })

        summary_results = []

        for key, value in stat.items():
            view_name, threshold = key
            query_count = value["query_count"]

            hit_count = value["hit_count"]
            top1_hit_count = value["top1_hit_count"]
            rank_hit_count = value["rank_hit_count"]

            recall_rate = hit_count / query_count if query_count else 0.0
            top1_acc = top1_hit_count / query_count if query_count else 0.0
            mrr = value["mrr_sum"] / query_count if query_count else 0.0
            avg_candidate_count = value["candidate_count_sum"] / query_count if query_count else 0.0
            avg_precision = value["precision_sum"] / query_count if query_count else 0.0

            avg_hit_rank = None
            if rank_hit_count > 0:
                avg_hit_rank = value["rank_sum"] / rank_hit_count

            summary_results.append({
                "view": view_name,
                "threshold": threshold,
                "query_count": query_count,
                "hit_count": hit_count,
                "recall_rate": recall_rate,
                "top1_acc": top1_acc,
                "mrr": mrr,
                "avg_candidate_count": avg_candidate_count,
                "avg_precision": avg_precision,
                "avg_hit_rank": avg_hit_rank,
            })

        summary_results.sort(key=lambda x: (x["view"], x["threshold"]))

        return summary_results, detail_results

def encode_fn(texts):
    embeddings = model.encode(
        texts,
        batch_size=64,
        normalize_embeddings=False,
        show_progress_bar=False,
    )
    return np.asarray(embeddings)


def main():
    """主函数"""
    data = []
    dataframe = pd.read_excel("阿拉丁卡片梳理-20250819.xlsx")
    dataframe.head()
    """
    data:
    [    {
            "id": "1",
            "title": "xxx",
            "description": "xxx",
            "call_querys_list": ["query1", "query2"]
        }
    ]
    """
    for indx, row in dataframe.iterrows():
        data.append({
            "id": row["name"],
            "title": row["name_for_human"],
            "description": row["description"],
            "call_querys_list": eval(
                row["call_querys"].replace(', \"《“骗骗”喜欢你》\"', '').replace("”", '"').replace("“", '"').replace('""',
                                                                                                                   '"').replace(
                    '\\\\', '').replace('\\"', '"'))
        })

    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]

    evaluator = MultiViewVectorRecallEvaluator(
        data=data,
        encode_fn=encode_fn,
        batch_size=64,
    )

    evaluator.build_vector_index()

    summary_results, detail_results = evaluator.evaluate(
        thresholds=thresholds,
        top_k=6,
    )

    summary_df = pd.DataFrame(summary_results)
    detail_df = pd.DataFrame(detail_results)

    evaluator.save_recall_results(
        summary_df=summary_df,
        detail_df=detail_df,
        output_dir="./recall_outputs",
        prefix="multi_view_faiss_recall",
    )

if __name__ == "__main__":
    main()
