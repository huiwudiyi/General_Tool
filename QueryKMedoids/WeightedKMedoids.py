#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Project : General_Tool
@File    : WeightedKMedoids.py
@Author  : zhuzerun
@Date    : 2026-07-16 14:31
@Version : 1.0
@Desc    : 
@Contact : zachary6chu@gmail.com
"""
import os
import json
import datetime
import numpy as np
import pandas as pd
from json_repair import repair_json
from sklearn.metrics.pairwise import pairwise_distances

class WeightedKMedoids:
    def __init__(self, n_clusters, max_iter=300, random_state=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.medoid_indices_ = None
        self.labels_ = None
        self.medoids_ = None

    def fit(self, X, weights, queries=None):
        n = len(X)
        w = weights
        dist_matrix = pairwise_distances(X, metric="cosine") # embedding优先余弦距离

        # 加权概率初始化medoid
        prob = w / w.sum()
        med_ids = np.random.choice(n, size=self.n_clusters, replace=False, p=prob)

        for _ in range(self.max_iter):
            dist_to_med = dist_matrix[:, med_ids]
            weighted_dist = dist_to_med * w[:, None]
            labels = np.argmin(weighted_dist, axis=1)

            improved = False
            for k in range(self.n_clusters):
                m_idx = med_ids[k]
                members = np.where(labels == k)[0]
                if len(members) == 0:
                    continue
                loss_old = np.sum(w[members] * dist_matrix[members, m_idx])
                for cand in members:
                    if cand == m_idx:
                        continue
                    loss_new = np.sum(w[members] * dist_matrix[members, cand])
                    if loss_new < loss_old:
                        med_ids[k] = cand
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break

        self.medoid_indices_ = med_ids
        self.medoids_ = X[med_ids]
        dist_to_med = dist_matrix[:, med_ids]
        weighted_dist = dist_to_med * w[:, None]
        self.labels_ = np.argmin(weighted_dist, axis=1)
        # 如果传入query列表，自动保存中心点文本
        if queries is not None:
            self.cluster_centers_query_ = [queries[idx] for idx in med_ids]
        return self
    def get_cluster_centers_query(self):
        """返回每个簇中心query文本列表，顺序：簇0,簇1,...簇k-1"""
        if self.cluster_centers_query_ is None:
            raise ValueError("fit时需要传入参数 queries=原始query列表")
        return self.cluster_centers_query_


def list_file(folder_path):
    all_items = os.listdir(folder_path)
    # 只列出文件（不包括文件夹）
    files_only = [os.path.join(folder_path, item) for item in all_items if os.path.isfile(os.path.join(folder_path, item)) and item.endswith("json")]
    print("\n只列出文件:")
    return files_only


def main():
    """主函数"""
    paths = list_file("./data")
    print(paths)
    for path in paths:
        view_embeddings = {}
        with open(path) as f:
            view_embeddings = json.load(f)
            querys = view_embeddings["query"]
            weight = np.asarray(view_embeddings["weight"], dtype=np.float32)
            embeddings = np.asarray(view_embeddings["embedding"], dtype=np.float32)

            # ==========核心：权重log变换==========
            w_log = np.log(weight)
            n_clusters = min(int(len(querys)/100), 100)
            # ========== 模型 fit ==================
            model = WeightedKMedoids(n_clusters=n_clusters)
            model.fit(embeddings, weights=w_log, queries=querys)
            # 簇统计（使用原始PV看业务指标）
            labels = model.labels_
            center_queries = model.get_cluster_centers_query()  # 获取所有簇中心query
            out_path = path.replace(".json", "_call_query")
            center_qs = []
            w = open(out_path, "w")
            for c in range(n_clusters):
                mask = labels == c
                center_q = center_queries[c]
                w.write(center_q + "\n")
                print(f"簇{c} | 中心query：{center_q} | item数量:{mask.sum()}, 总原始PV:{weight[mask].sum():.1f}")
            w.flush()
            w.close()


if __name__ == "__main__":
    main()
