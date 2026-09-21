"""并行调度脚本：对一批资源号并行执行优化，完成后统一合并结果到主文件。

用法:
    # 方式1：命令行直接传入资源号
    python run_parallel.py --ids 23,47190,62206

    # 方式2：从文件读取（每行一个资源号）
    python run_parallel.py --ids-file todo_ids.txt

    # 指定并行度（默认6）
    python run_parallel.py --ids 23,47190,62206 --workers 4

外层批次脚本可多次调用本脚本，每次传入一批资源号。
"""
import os
import sys
import json
import time
import subprocess
import argparse
import logging
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

# =========== 配置 ===========
PYTHON_BIN = sys.executable
MAIN_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tool_optimizer_main.py")
TOOLS_DESC_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/optimizer/tool_descriptions.json")
BEST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../recall_outputs/optimizer/version_best")
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../recall_outputs/optimizer/logs")


def run_one_resource(resource_id: str, max_iterations: int, eval_view: str = "title_description") -> dict:
    """在子进程中执行单个资源号的优化。返回执行结果摘要。"""
    cmd = [
        PYTHON_BIN, MAIN_SCRIPT,
        "--resource-id", resource_id,
        "--no-writeback",
        "--max-iterations", str(max_iterations),
        "--eval-view", eval_view,
    ]
    log_file = os.path.join(LOG_DIR, f"{resource_id}.log")

    t0 = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=os.path.dirname(MAIN_SCRIPT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=600,  # 单资源号超时10分钟
        )
        elapsed = time.time() - t0
        # 追加 subprocess 的 stdout 到日志文件（主脚本内部也写了同名文件，这里确保有完整输出）
        with open(log_file, "ab") as f:
            f.write(result.stdout)
        return {
            "resource_id": resource_id,
            "returncode": result.returncode,
            "elapsed": round(elapsed, 1),
            "success": result.returncode == 0,
        }
    except subprocess.TimeoutExpired:
        return {
            "resource_id": resource_id,
            "returncode": -1,
            "elapsed": round(time.time() - t0, 1),
            "success": False,
            "error": "timeout",
        }
    except Exception as e:
        return {
            "resource_id": resource_id,
            "returncode": -1,
            "elapsed": round(time.time() - t0, 1),
            "success": False,
            "error": str(e),
        }


def _find_best_summary(resource_id: str) -> dict:
    """在 version_best 目录下找到该资源号的 best_summary.json。"""
    if not os.path.isdir(BEST_DIR):
        return {}
    for folder in os.listdir(BEST_DIR):
        if folder.startswith(f"{resource_id}_"):
            p = os.path.join(BEST_DIR, folder, "best_summary.json")
            if os.path.isfile(p):
                with open(p, encoding="utf-8") as f:
                    return json.load(f)
    return {}


def merge_best_to_main(resource_ids: list, log) -> dict:
    """所有子进程完成后，单进程一次性把各资源号的最佳 description 合并回主文件。

    没有锁，因为此时并行阶段已经全部结束，只有主进程在写。
    """
    with open(TOOLS_DESC_PATH, encoding="utf-8") as f:
        tools_all = json.load(f)

    updated, skipped = [], []
    for rid in resource_ids:
        summary = _find_best_summary(rid)
        if not summary:
            skipped.append({"resource_id": rid, "reason": "no_best_summary"})
            continue
        new_desc = summary.get("description", "")
        if not new_desc:
            skipped.append({"resource_id": rid, "reason": "empty_description"})
            continue
        if rid not in tools_all:
            skipped.append({"resource_id": rid, "reason": "not_in_main_file"})
            continue
        old_desc = tools_all[rid].get("description", "")
        if old_desc == new_desc:
            skipped.append({"resource_id": rid, "reason": "unchanged"})
            continue
        tools_all[rid]["description"] = new_desc
        updated.append(rid)
        log.info(f"[merge] resource_id={rid} 描述已更新 "
                 f"(best_version={summary.get('best_version_id')}, "
                 f"R@1={summary.get('recall1')}, R@3={summary.get('recall3')})")

    if updated:
        with open(TOOLS_DESC_PATH, "w", encoding="utf-8") as f:
            json.dump(tools_all, f, ensure_ascii=False, indent=2)
        log.info(f"[merge] 已写入主文件，共更新 {len(updated)} 个资源号")
    else:
        log.info("[merge] 无资源号需要更新，主文件未改动")

    for item in skipped:
        log.info(f"[merge] 跳过 resource_id={item['resource_id']} ({item['reason']})")

    return {"updated": updated, "skipped": skipped}


def main():
    parser = argparse.ArgumentParser(description="并行调度优化器")
    parser.add_argument("--ids", type=str, default=None,
                        help="逗号分隔的资源号列表")
    parser.add_argument("--ids-file", type=str, default=None,
                        help="资源号文件路径（每行一个）")
    parser.add_argument("--workers", type=int, default=6,
                        help="并行 worker 数（默认6）")
    parser.add_argument("--max-iterations", type=int, default=3,
                        help="每个资源号的最大优化轮次")
    parser.add_argument("--eval-view", type=str, default="title_description",
                        choices=["merged", "title_description", "description"],
                        help="召回视角开关(默认title_description)")
    args = parser.parse_args()

    # 解析资源号列表
    if args.ids:
        resource_ids = [x.strip() for x in args.ids.split(",") if x.strip()]
    elif args.ids_file:
        with open(args.ids_file, "r") as f:
            resource_ids = [line.strip() for line in f if line.strip()]
    else:
        print("ERROR: 必须指定 --ids 或 --ids-file")
        sys.exit(1)

    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(BEST_DIR, exist_ok=True)

    # 设置本脚本自身的日志
    log_file = os.path.join(LOG_DIR, "parallel_dispatch.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, mode="w", encoding="utf-8"),
        ],
    )
    log = logging.getLogger("parallel")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    banner = "=" * 70
    log.info(banner)
    log.info(banner)
    log.info(f"[dispatch] ####  NEW RUN  ####")
    log.info(f"[dispatch] run_id       : {run_id}")
    log.info(f"[dispatch] start_time   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"[dispatch] total_ids    : {len(resource_ids)}")
    log.info(f"[dispatch] workers      : {args.workers}")
    log.info(f"[dispatch] max_iter     : {args.max_iterations}")
    log.info(f"[dispatch] eval_view    : {args.eval_view}")
    log.info(f"[dispatch] resource_ids : {resource_ids}")
    log.info(f"[dispatch] log_file     : {log_file}")
    log.info(banner)
    log.info(banner)

    # ============ 并行执行 ============
    results = []
    total = len(resource_ids)
    done_count = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(run_one_resource, rid, args.max_iterations, args.eval_view): rid
            for rid in resource_ids
        }
        for future in as_completed(futures):
            rid = futures[future]
            try:
                res = future.result()
            except Exception as e:
                res = {"resource_id": rid, "success": False, "error": str(e), "elapsed": 0}
            results.append(res)
            done_count += 1
            status = "OK" if res["success"] else "FAIL"
            log.info(f"[dispatch] ({done_count}/{total}) {status} resource_id={rid} elapsed={res.get('elapsed', '?')}s")

    # ============ 统一合并结果到 tool_descriptions.json ============
    log.info(f"[merge] 开始合并优化结果到 {TOOLS_DESC_PATH}")
    merge_results = merge_best_to_main(resource_ids, log)

    # ============ 输出汇总 ============
    success_ids = [r["resource_id"] for r in results if r["success"]]
    failed_ids = [r["resource_id"] for r in results if not r["success"]]

    summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total": len(resource_ids),
        "success": len(success_ids),
        "failed": len(failed_ids),
        "updated": merge_results["updated"],
        "failed_ids": failed_ids,
        "details": results,
    }

    summary_path = os.path.join(LOG_DIR, "parallel_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    log.info(f"[dispatch] 完成。成功 {len(success_ids)}/{len(resource_ids)}，"
             f"更新 {len(merge_results['updated'])} 个描述")
    if failed_ids:
        log.info(f"[dispatch] 失败资源号: {failed_ids}")
    log.info(f"[dispatch] 汇总: {summary_path}")


if __name__ == "__main__":
    main()
