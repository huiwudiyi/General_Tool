#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Project : General_Tool
@File    : KMedoids_build_embedding.py
@Author  : zhuzerun
@Date    : 2026-07-16 14:27
@Version : 1.0
@Desc    : 
@Contact : zachary6chu@gmail.com
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys
import pandas as pd
import json
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import torch
import numpy as np
import argparse
import pandas as pd
from sentence_transformers import SentenceTransformer


MODEL_PATH = "../../afs_data/model/Qwen3-Embedding-8B"
MODEL = SentenceTransformer(MODEL_PATH)

def l2_normalize(vectors: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2 归一化，归一化后点积等价于 cosine similarity。"""
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, eps)


def encode_fn(texts: List[str]) -> np.ndarray:
    """向量模型编码函数。"""
    embeddings = MODEL.encode(
        texts,
        batch_size=64,
        normalize_embeddings=False,
        show_progress_bar=False,
    )
    torch.cuda.empty_cache()
    return np.asarray(embeddings)

class QueryWeightKmeans:
    """工具集合向量召回 pass@k 评估器。"""

    def __init__(
        self,
        querys: List[Tuple[str, float]],
        encode_fn:  Callable[[List[str]], np.ndarray],
        batch_size: int = 64,
    ):
        """
        Args:
            querys:
                [
                    [query_text, weight_value],
                    [query_text, weight_value],
                    ...
                ]

            encode_fn:
                向量模型编码函数，输入 List[str]，返回 np.ndarray，shape=[N, dim]。

            batch_size:
                embedding 批大小。
        """
        self.raw_querys = querys
        self.encode_fn = encode_fn
        self.batch_size = batch_size

        # 解析后存储
        self.texts: List[str] = []
        self.weights: np.ndarray = np.array([])

        # 视图：只保留query视图
        self.view_embeddings = {
            "query": [],
            "weight": [],
            "embedding": []
        }

    def _batch_encode(self, texts: List[str]) -> np.ndarray:
        """分批编码文本，并做 L2 normalize。"""
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i : i + self.batch_size]
            emb = self.encode_fn(batch_texts)
            if not isinstance(emb, np.ndarray):
                emb = np.asarray(emb)
            all_embeddings.append(emb)

        embeddings = np.vstack(all_embeddings)
        embeddings = l2_normalize(embeddings)
        return embeddings.tolist()

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

    def build_vector_index(self) -> None:
        """构建工具底库向量：querys"""
        self.view_texts = {"query": []}
        query_list = []
        weight_list = []

        for item in self.raw_querys:
            if len(item) < 2:
                continue
            q = self._clean_text(item[0])
            w = float(item[1])
            if not q:
                continue
            query_list.append(q)
            weight_list.append(w)

        self.view_embeddings["query"] = query_list
        self.view_embeddings["weight"] = weight_list

        # 编码query向量

        print(f"Building embeddings for view= query , size={len(query_list)}")
        self.view_embeddings["embedding"] = self._batch_encode(query_list)
        print(f"Total indexed queries: {len(query_list)}")

    def dumps_vector(self, path, srcid) -> None:
        out_path = path.replace(".txt", "_" + str(srcid) + ".json")
        ## 这里注意，超大文件问题
        with open(out_path, "w") as wf:
            for q, w, emb in zip(self.view_embeddings["query"], self.view_embeddings["weight"], self.view_embeddings["embedding"]):
                wf.write(json.dumps({"q":q, "w":w, "emb": emb}, ensure_ascii=False) + "\n")

def list_file(folder_path):
    all_items = os.listdir(folder_path)
    # 只列出文件（不包括文件夹）
    files_only = [os.path.join(folder_path, item) for item in all_items if os.path.isfile(os.path.join(folder_path, item)) and item.endswith(".txt")]
    print("\n只列出文件:")
    return files_only
paths = list_file("./data")

def split_list_evenly(lst, num_srcids, skipList = None):
	# lst = [1,2,3,4,5,6,7,8,9]
	# num_workers = ["2","3","5","6"]
    """将文件列表均衡切分成 len(num_srcids) 份"""
    chunks = {}
    for srcid in num_srcids:
        chunks[srcid] = []
    for idx, item in enumerate(lst):
        # print(item)
        if skipList and item in skipList:
            print(item, "skip!!!")
            continue
        chunks[num_srcids[idx % len(num_srcids)]].append(item)
    return chunks

# path = "data/19192816.txt"
# MODEL = MODEL.half().to(f"cuda:1")
# dataframe = pd.read_csv(path, sep="\t", quoting=3, on_bad_lines="skip")
# query_items = []
# srcid = -1
# print("path:", path)
# for indx, row in dataframe.iterrows():
#     query = row["Query"]
#     srcid = row["阿拉丁资源id"]
#     try:
#         pv = int(row["展现量"] + row["跳转点击量"] + row["点击量"])
#         if pv < 2:
#             continue
#     except:
#         continue
#     query_items.append([query, pv])
# wkmeas = QueryWeightKmeans(querys = query_items,
#                             encode_fn = encode_fn,
#                             batch_size = 8)
# wkmeas.build_vector_index()
# wkmeas.dumps_vector(path, srcid)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=str, required=True,
                        help="指定需要使用的gpus，逗号分隔，例如 1,2,3,4")
    parser.add_argument("--gpuid", type=str, required=True,
                        help="指定需要使用的gpuid，逗号分隔，例如 4")
    args = parser.parse_args()

    # 解析gpu编号列表
    gpu_ids = [x.strip() for x in args.gpus.split(",")]
    gpuid = args.gpuid.strip()
    MODEL = MODEL.half().to(f"cuda:{gpuid}")

    lst = list_file("./data")

    num_srcids = gpu_ids

    chunks = split_list_evenly(lst, num_srcids)
    # print(chunks)
    print("执行文件", chunks[gpuid])

    for path in chunks[gpuid]:
        dataframe = pd.read_csv(path, sep="\t", quoting=3, on_bad_lines="skip")
        query_items = []
        srcid = -1
        print("path:", path)
        for indx, row in dataframe.iterrows():
            query = row["Query"]
            srcid = row["id"]
            try:
                pv = int(row["show"] + row["judge"] + row["click"])
                if pv < 2:
                    continue
            except:
                continue
            query_items.append([query, pv])
        wkmeas = QueryWeightKmeans(querys = query_items,
                                    encode_fn = encode_fn,
                                    batch_size = 8)
        wkmeas.build_vector_index()
        wkmeas.dumps_vector(path, srcid)
