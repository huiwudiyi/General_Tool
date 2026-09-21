"""全量批次编排脚本：KMeans 聚类 + 方案5分配 + 逐轮调用 run_parallel.py。

用法:
    # 默认 k=6 聚类，workers=6
    python run_batch.py

    # 自定义参数
    python run_batch.py --k 8 --workers 6 --max-iterations 3

流程:
    1. 读取 tool_descriptions.json，拼接 title+description 获取 embedding
    2. KMeans 聚类为 k 簇
    3. 方案5 分配：每轮 batch ≤ workers，同簇多点尽量距离最远
    4. 逐轮调用 run_parallel.py，每轮完成后 merge 回主文件
    5. 下一轮基于更新后的描述计算 baseline
"""
import os
import sys
import json
import math
import time
import subprocess
import argparse
import logging
import numpy as np
from datetime import datetime
from collections import defaultdict
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans

# =========== 路径配置 ===========
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TOOLS_DESC_PATH = os.path.join(SCRIPT_DIR, "../data/optimizer/tool_descriptions.json")
PARALLEL_SCRIPT = os.path.join(SCRIPT_DIR, "run_parallel.py")
LOG_DIR = os.path.join(SCRIPT_DIR, "../recall_outputs/optimizer/logs")
PYTHON_BIN = sys.executable

# no_proxy fix for local embedding service
for _key in ("no_proxy", "NO_PROXY"):
    _val = os.environ.get(_key, "")
    if "127.0.0.1" not in _val:
        os.environ[_key] = f"127.0.0.1,localhost,{_val}".rstrip(",")


def get_embeddings(texts, batch_size=64):
    """调用本地 SGLang embedding 服务获取向量。"""
    from openai import OpenAI
    client = OpenAI(base_url="http://127.0.0.1:8882/v1", api_key="zacharychu")
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        resp = client.embeddings.create(model="Qwen3-Embedding-8B", input=batch)
        all_embs.extend([item.embedding for item in resp.data])
    return np.array(all_embs)


def allocate_scheme5(ids, emb, k, batch_size, seed=42):
    """方案5：KMeans 聚类后分配到各轮次。

    每轮 batch <= batch_size；优先每簇每轮最多一个成员；
    若某簇成员数超过总轮数不得不同轮出现多个，则选择距该轮已有同簇成员最远的 batch。

    Returns:
        batches: List[List[str]]，每个元素是该轮的资源号列表。
    """
    n = len(ids)
    D = squareform(pdist(emb, metric="cosine"))
    R = math.ceil(n / batch_size)  # 总轮数

    km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(emb)
    labels = km.labels_

    cl = defaultdict(list)
    for i, lab in enumerate(labels):
        cl[lab].append(i)

    batches_idx = [[] for _ in range(R)]
    batch_labels = [set() for _ in range(R)]  # 每轮已包含的簇

    for lab, members in cl.items():
        remain = list(members)
        np.random.default_rng(seed + int(lab)).shuffle(remain)
        for it in remain:
            # 优先：不含该簇 且 未满 的 batch，取 size 最小
            cand = [r for r in range(R)
                    if lab not in batch_labels[r] and len(batches_idx[r]) < batch_size]
            if cand:
                r = min(cand, key=lambda r: (len(batches_idx[r]), r))
            else:
                # 兜底：已所有未满 batch 都含该簇，选(未满, size最小, 距同簇成员最远)
                def score(r):
                    same = [x for x in batches_idx[r] if labels[x] == lab]
                    min_d = min(D[it][x] for x in same) if same else 999.0
                    full_penalty = 0 if len(batches_idx[r]) < batch_size else 1
                    return (full_penalty, len(batches_idx[r]), -min_d)
                r = min(range(R), key=score)
            batches_idx[r].append(it)
            batch_labels[r].add(lab)

    batches = [[ids[i] for i in b] for b in batches_idx]
    return batches, labels, D


def run_round(round_idx, total_rounds, batch_ids, workers, max_iterations, eval_view, log):
    """调用 run_parallel.py 执行一轮优化。"""
    ids_str = ",".join(batch_ids)
    cmd = [
        PYTHON_BIN, PARALLEL_SCRIPT,
        "--ids", ids_str,
        "--workers", str(min(workers, len(batch_ids))),
        "--max-iterations", str(max_iterations),
        "--eval-view", eval_view,
    ]
    log.info(f"[round {round_idx+1}/{total_rounds}] batch={batch_ids} workers={min(workers, len(batch_ids))}")
    t0 = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=SCRIPT_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=1200,
        )
        elapsed = time.time() - t0
        success = result.returncode == 0
        if not success:
            log.warning(f"[round {round_idx+1}] returncode={result.returncode}")
            log.warning(result.stdout.decode("utf-8", errors="replace")[-500:])
        return {"round": round_idx, "success": success, "elapsed": round(elapsed, 1), "ids": batch_ids}
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        log.error(f"[round {round_idx+1}] TIMEOUT after {elapsed:.0f}s")
        return {"round": round_idx, "success": False, "elapsed": round(elapsed, 1), "ids": batch_ids, "error": "timeout"}


def main():
    parser = argparse.ArgumentParser(description="全量批次编排：聚类 + 逐轮并行优化")
    parser.add_argument("--k", type=int, default=6, help="KMeans 聚类中心数（默认6）")
    parser.add_argument("--workers", type=int, default=6, help="每轮并行 worker 数（默认6）")
    parser.add_argument("--max-iterations", type=int, default=3, help="每个资源号优化轮次（默认3）")
    parser.add_argument("--eval-view", type=str, default="title_description",
                        choices=["merged", "title_description", "description"],
                        help="召回评估视角（默认 title_description）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--dry-run", action="store_true", help="只打印分配方案，不执行")
    args = parser.parse_args()

    os.makedirs(LOG_DIR, exist_ok=True)

    # 日志
    log_file = os.path.join(LOG_DIR, "batch_orchestrator.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, mode="w", encoding="utf-8"),
        ],
    )
    log = logging.getLogger("batch")

    # 1. 读取工具集
    with open(TOOLS_DESC_PATH, encoding="utf-8") as f:
        tools = json.load(f)
    ids = list(tools.keys())
    n = len(ids)
    texts = [f"{tools[i].get('title', '')}\n{tools[i].get('description', '')}".strip() for i in ids]

    log.info("=" * 70)
    log.info(f"[batch] #### BATCH ORCHESTRATOR START ####")
    log.info(f"[batch] total_resources : {n}")
    log.info(f"[batch] k_clusters      : {args.k}")
    log.info(f"[batch] workers         : {args.workers}")
    log.info(f"[batch] max_iterations  : {args.max_iterations}")
    log.info(f"[batch] eval_view       : {args.eval_view}")
    log.info(f"[batch] seed            : {args.seed}")
    log.info("=" * 70)

    # 2. 获取 embedding
    log.info("[batch] 获取 embedding ...")
    emb = get_embeddings(texts)
    log.info(f"[batch] embedding shape: {emb.shape}")

    # 3. 聚类 + 方案5分配
    log.info("[batch] KMeans 聚类 + 方案5 分配 ...")
    batches, labels, D = allocate_scheme5(ids, emb, k=args.k, batch_size=args.workers, seed=args.seed)
    total_rounds = len(batches)

    # 打印分配概况
    cluster_sizes = np.bincount(labels)
    log.info(f"[batch] 簇大小分布: {sorted(cluster_sizes.tolist(), reverse=True)}")
    log.info(f"[batch] 总轮数: {total_rounds}")
    for i, b in enumerate(batches):
        log.info(f"[batch]   round {i+1}: {b}")

    if args.dry_run:
        log.info("[batch] --dry-run 模式，不执行优化。")
        return

    # 4. 逐轮执行
    all_results = []
    t_start = time.time()
    for i, batch_ids in enumerate(batches):
        if not batch_ids:
            continue
        result = run_round(i, total_rounds, batch_ids, args.workers, args.max_iterations, args.eval_view, log)
        all_results.append(result)
        elapsed_total = time.time() - t_start
        log.info(f"[batch] round {i+1} done in {result['elapsed']:.1f}s "
                 f"({'OK' if result['success'] else 'FAIL'}) "
                 f"| 累计 {elapsed_total:.0f}s")

    # 5. 汇总
    total_elapsed = time.time() - t_start
    success_rounds = sum(1 for r in all_results if r["success"])
    log.info("=" * 70)
    log.info(f"[batch] #### BATCH ORCHESTRATOR COMPLETE ####")
    log.info(f"[batch] 总轮数: {total_rounds}, 成功: {success_rounds}, 失败: {total_rounds - success_rounds}")
    log.info(f"[batch] 总耗时: {total_elapsed:.0f}s ({total_elapsed/60:.1f}min)")
    log.info("=" * 70)

    # 保存运行日志
    summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "params": {"k": args.k, "workers": args.workers, "max_iterations": args.max_iterations,
                   "eval_view": args.eval_view, "seed": args.seed},
        "total_resources": n,
        "total_rounds": total_rounds,
        "success_rounds": success_rounds,
        "total_elapsed_s": round(total_elapsed, 1),
        "cluster_sizes": sorted(cluster_sizes.tolist(), reverse=True),
        "rounds": all_results,
    }
    summary_path = os.path.join(LOG_DIR, "batch_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log.info(f"[batch] 汇总保存: {summary_path}")


if __name__ == "__main__":
    main()
