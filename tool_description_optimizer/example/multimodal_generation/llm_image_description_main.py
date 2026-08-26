#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Project : General_Tool
@File    : llm_image_description_main.py
@Author  : zhuzerun
@Date    : 2026-08-12 21:10
@Version : 1.0
@Desc    : 
@Contact : zachary6chu@gmail.com
"""
import random
import os
import sys
import shutil

import yaml
import pandas as pd
from json_repair import repair_json
from openai import OpenAI

try:
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import END, START, StateGraph
    from langgraph.graph.message import add_messages
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "Missing langgraph dependencies. Please run: "
        "pip install langgraph langchain-core"
    ) from exc

sys.path.append('../../common')
from aiapi.aiapapi import request_aiapi
from client.llm_client import LLMClient
from flow.flow_config import FlowConfig
from flow.prompt_registry import PromptRegistry
from passk.nn_server_passk import *

sys.path.append("../../src/multimodal_generation")
from llm_image_description import LLMImageDescription

class LLMImageOptimizerGraph:
    def __init__(
            self,
            resource_id: str,
            llm_client: Optional[LLMClient] = None,
            prompt_path: str = "../tool_description_optimizer/config/prompts.json",
            flow_config_path: str = "../tool_description_optimizer/config/agent_config.yaml",
            image_path: str = "../data/optimizer/tool_descriptions.json",
            test_data_path: str = "../../data/multimodal_generation/query_image.xlsx",
            checkpointer: Optional[Any] = None,
            eval_view: str = "title_description",
    ) -> None:
        # 设置通用的 参数
        self.prompts = PromptRegistry(prompt_path).get_promts()
        self.flow_config = FlowConfig(flow_config_path)
        self.checkpointer = checkpointer or InMemorySaver()
        self.image_path = image_path
        # 设置 tool resource_id
        self.resource_id = resource_id
        # 用于上层指标计算的召回视角: 'merged' | 'title_description' | 'description'
        self.eval_view = eval_view

        # 加载tools 这个列表
        self.tools_dict = load_tools_from_json(self.last_tools_description_path)

        # 加载测试集
        # test_data_path: str = "../data/optimizer/query.json"
        # load_tools_from_json(test_data_path)#
        self.query_good_all_dict = load_tools_from_json(test_data_path)
        self.query_good_dict = self.query_good_all_dict[resource_id]

        # node -> model
        self.node_model_map: Dict[str, str] = {}
        if self.flow_config.node_model_map:
            self.node_model_map.update({str(k): str(v) for k, v in self.flow_config.node_model_map.items()})

        # node -> temperature
        self.node_temperature_map: Dict[str, float] = {}
        if self.flow_config.node_temperature_map:
            self.node_temperature_map.update(
                {str(k): float(v) for k, v in self.flow_config.node_temperature_map.items()})

        llm_dict = {}
        if len(self.flow_config.clients) != 0:
            for k, v in self.flow_config.clients.items():
                if "tianchi" in v["base_url"]:
                    v['host'] = "tianchi-proxy.baidu-int.com"
                    v['appid'] = 'app-CdjpA4YQ'
                    llm_dict[k] = LLMClient(**v)

        # llm_client
        if llm_client:
            self.llm_client = llm_client
        else:
            self.llm_client = LLMClient(**self.flow_config.config["default_client"])

        # node_llm_client_map
        self.node_llm_clients: Dict[str, LLMClient] = {}
        if self.flow_config.node_llm_client_map:
            self.node_llm_clients.update({
                str(k): llm_dict[v] for k, v in self.flow_config.node_llm_client_map.items()
            })

    # ---------- generate text ----------

    # node -> client
    def _client_for(self, node_name: str) -> LLMClient:
        return self.node_llm_clients.get(node_name, self.llm_client)

    # node -> model name
    def _model_for(self, node_name: str) -> str:
        return self.node_model_map.get(node_name, "deepseek-v3.2")

    # node -> temperature
    def _temperature_for(self, node_name: str) -> float:
        return self.node_temperature_map.get(node_name, 0.8)

    def _generate_text(
            self,
            node_name: str,
            prompt: str,
            max_tokens: int = 2048,
            extra_body: Dict[str, Any] = {}
    ) -> Any:
        client = self._client_for(node_name)
        response = client.generate_text(
            prompt=prompt,
            model=self._model_for(node_name),
            temperature=self._temperature_for(node_name),
            extra_body=extra_body
        )
        return client.parse_chat_content(response)

    # --------- 统计和check工具 ----------------------
    def statistic_check_tool(self,
                             state: ToolOptimizerState):
        """示例主函数：你可以替换为自己的 JSON 文件路径。"""
        best_record = state.get("best_record", None)
        if best_record is not None:
            self.last_tools_description_path = best_record.tool_path

        self.tools_dict = load_tools_from_json(self.last_tools_description_path)
        # 如果 judge 已通过并更新了 description，用 state 中的最新版本覆盖
        current_desc = state.get("current_description", "")
        if current_desc and current_desc != self.tools_dict.get(self.resource_id, {}).get("description", ""):
            self.tools_dict[self.resource_id]["description"] = current_desc
        statistic_check_version = "version_" + str(state.get("current_version_id", 0))
        print(f"当前执行的版本： {statistic_check_version}, eval_view={self.eval_view}")
        statistic_output = "../recall_outputs/optimizer/" + statistic_check_version + "/" + self.resource_id

        detaildf = recall_passk_function(self.tools_dict, self.query_good_dict, self.resource_id, statistic_output,
                                         eval_view=self.eval_view)

        # 计算 top 1 的 precision
        detaildf_top1 = detaildf[(detaildf['k'] == 1) & (detaildf['view'] == self.eval_view)]
        relevants = detaildf_top1['gold_ids'].to_list()
        retrieveds = detaildf_top1['recall_ids'].to_list()
        precision1 = precision_at_k_batch(retrieveds, relevants, 1)
        recall1 = recall_at_k_batch(retrieveds, relevants, 1)

        # 计算 top 3 的 precision
        detaildf_top3 = detaildf[(detaildf['k'] == 3) & (detaildf['view'] == self.eval_view)]
        relevants = detaildf_top3['gold_ids'].to_list()
        retrieveds = detaildf_top3['recall_ids'].to_list()
        precision3 = precision_at_k_batch(retrieveds, relevants, 3)
        recall3 = recall_at_k_batch(retrieveds, relevants, 3)

        # 统计 top 3 的结果
        tools_query = {}
        tools_case_ids = []
        for indx, row in detaildf_top3.iterrows():
            recall_ids = row["recall_ids"]
            if len(recall_ids) == 0:
                recall_ids = [NO_CALL]
            if len(tools_query.get(recall_ids[0], [])) == 0:
                tools_query[recall_ids[0]] = []
            tools_query[recall_ids[0]].append(row["query"])
            tools_case_ids.append(recall_ids[0])
        top_tools_case = {k: tools_query[k] for k in get_k_tool(tools_case_ids, 3)}
        tools_case_all = {k: tools_query[k] for k in set(tools_case_ids)}

        version_id, next_version_id = next_version_pair(state)

        versionInfo = VersionRecord(
            version_id=statistic_check_version,
            parent_version_id=version_id,
            stage="statistic",
            description=self.tools_dict[self.resource_id]["description"],
            case_result=tools_case_all,
            top_case=top_tools_case,
            tool_path=statistic_output + "/tool_prompt.json",
            recall1=recall1,
            precision1=precision1,
            recall3=recall3,
            precision3=precision3
        )
        best_record = state.get("best_record", None)

        # 判断是否需要更新最佳记录
        # 条件1: best_record 不存在 (即为 None)
        # 条件2: 至少一个指标提升，另一个不下降
        should_update = (best_record is None) or (
                (recall1 >= best_record.recall1 and recall3 >= best_record.recall3) and
                (recall1 > best_record.recall1 or recall3 > best_record.recall3)
        )

        with open(statistic_output + "/tool_prompt.json", "w") as w:
            json.dump(self.tools_dict, w, ensure_ascii=False, indent=2)

        if should_update:
            return {
                "current_version_id": version_id,
                "next_version_id": next_version_id,
                "version_history": append_version_history(state, versionInfo),
                "current_description": versionInfo.description,
                "best_description": versionInfo.description,
                "best_version_id": state.get("current_version_id", 0),
                "best_record": versionInfo,

            }
        else:
            # 本轮候选未跑赢基线：回滚 current_description，避免劣化描述带入下一轮
            return {
                "current_version_id": version_id,
                "next_version_id": next_version_id,
                "version_history": append_version_history(state, versionInfo),
                "current_description": best_record.description
            }

    # ---------  LLMNodeOptimizer node--------------
    def llm_description_optimizer(self,
                                  state: ToolOptimizerState):
        prompt = LLMDescriptionOptimizer._gen_prompt(state, self.prompts['optimizer'], self.tools_dict)
        if not prompt:
            print(f"[optimizer] 无有效 best_record，跳过本轮优化")
            version_id = state.get("next_version_id", 1)
            return {
                "current_version_id": version_id,
                "next_version_id": version_id + 1,
            }

        for _ in range(3):
            response, stype, flag = self._generate_text(node_name="optimizer", prompt=prompt)
            response, flag, error_type = LLMDescriptionOptimizer._vertify_result(response)
            if flag:
                break
        if flag:
            optimizer_record = InfoRecord(
                version_id=state.get("current_version_id", "-1"),
                stage="optimizer",
                info=response
            )
            return {
                "optimizer_history": append_optimizer_history(state, optimizer_record)
            }
        else:
            print(f"[optimizer] 3 次重试后仍失败，跳过本轮")
            version_id = state.get("next_version_id", 1)
            return {
                "current_version_id": version_id,
                "next_version_id": version_id + 1,
            }

    # ---------  LLMNodeCritic node--------------
    def llm_description_judge(self, state: ToolOptimizerState):
        prompt_and_desc = LLMDescriptionJudge._gen_prompt(state, self.prompts['judge'])
        if not prompt_and_desc:
            print(f"[judge] optimizer_history 为空，跳过本轮评审")
            version_id = state.get("next_version_id", 1)
            return {
                "current_version_id": version_id,
                "next_version_id": version_id + 1,
            }
        prompt, optimizer_description = prompt_and_desc

        for _ in range(3):
            response, stype, flag = self._generate_text(node_name="judge", prompt=prompt)
            response, flag, error_type = LLMDescriptionJudge._vertify_result(response)
            if flag:
                break
        if flag:
            relevance_score = response["relevance_score"]
            judge_record = InfoRecord(
                version_id=state.get("current_version_id", "-1"),
                stage="judge",
                info=response
            )
            if relevance_score >= 2:
                self.tools_dict[self.resource_id]['description'] = optimizer_description
                return {
                    "current_description": optimizer_description,
                    "judge_history": append_judge_history(state, judge_record)
                }
            return {
                "judge_history": append_judge_history(state, judge_record)
            }
        else:
            print(f"[judge] 3 次重试后仍失败，跳过本轮评审")
            version_id = state.get("next_version_id", 1)
            return {
                "current_version_id": version_id,
                "next_version_id": version_id + 1,
            }

    # ---------- graph build ----------


def should_continue(state: ToolOptimizerState) -> Literal["description_optimizer", END]:
    # 达到最大轮次
    max_iteration = state.get("max_iterations", 3)
    if state.get("current_version_id", 10) > max_iteration:
        return END
    return "description_optimizer"


def export_best_version(state, resource_id, tools_description_path, write_back=True):
    """归档最佳版本，并（可选）把最佳 description 回写到原始工具描述文件。

    产出 ../recall_outputs/optimizer/version_best/{resource_id}_{best_round}/

    write_back: 是否立即把最佳 description 回写到 tools_description_path。
                并行模式下应设为 False，由外层调度脚本在所有子进程完成后统一合并，
                避免多个进程同时读-改-写同一个文件互相覆盖。
    """
    best_record = state.get("best_record", None)
    if best_record is None:
        print(f"[best] resource_id={resource_id} 无 best_record，跳过归档")
        return None, False

    # version_id 形如 "version_2"，取末尾轮次号
    best_round = str(best_record.version_id).split("_")[-1]
    src_dir = os.path.dirname(best_record.tool_path)
    best_root = "../recall_outputs/optimizer/version_best"
    dst_dir = os.path.join(best_root, f"{resource_id}_{best_round}")

    # 先清理该资源号所有历史归档（{rid}_* ），避免上一次运行选中的轮次残留，
    # 导致下游按前缀查找 best_summary.json 时读到过期结果
    if os.path.isdir(best_root):
        for folder in os.listdir(best_root):
            if folder.startswith(f"{resource_id}_") and folder.split("_")[0] == resource_id:
                stale = os.path.join(best_root, folder)
                if os.path.isdir(stale):
                    shutil.rmtree(stale)
                    print(f"[best] 清理历史归档: {stale}")

    if os.path.isdir(src_dir):
        shutil.copytree(src_dir, dst_dir)
        print(f"[best] 最佳版本 {best_record.version_id} 已归档: {dst_dir}")
    else:
        print(f"[best] 源目录不存在，跳过复制: {src_dir}")
        os.makedirs(dst_dir, exist_ok=True)

    # 归档一份指标摘要，便于直接查看该资源号选中了哪一轮
    with open(os.path.join(dst_dir, "best_summary.json"), "w", encoding="utf-8") as w:
        json.dump({
            "resource_id": resource_id,
            "best_version_id": best_record.version_id,
            "best_round": best_round,
            "description": best_record.description,
            "recall1": best_record.recall1,
            "precision1": best_record.precision1,
            "recall3": best_record.recall3,
            "precision3": best_record.precision3,
        }, w, ensure_ascii=False, indent=2)

    # 把最佳 description 回写到原始输入文件
    changed = False
    if not write_back:
        print(f"[best] write_back=False，跳过回写；结果已归档到 {dst_dir}")
        return dst_dir, changed
    tools_all = load_tools_from_json(tools_description_path)
    if resource_id in tools_all:
        old_desc = tools_all[resource_id].get("description", "")
        tools_all[resource_id]["description"] = best_record.description
        with open(tools_description_path, "w", encoding="utf-8") as w:
            json.dump(tools_all, w, ensure_ascii=False, indent=2)
        changed = old_desc != best_record.description
        print(f"[best] description 已回写 {tools_description_path} (changed={changed})")
    else:
        print(f"[best] {tools_description_path} 中无 resource_id={resource_id}，未回写")

    return dst_dir, changed


def _query_term_coverage(description: str, queries: list) -> float:
    """description 覆盖了多少比例的测试 query 字符 2-gram，用于观察语义靠拢程度。"""
    if not queries:
        return 0.0
    desc = description.replace(" ", "")
    hit = 0
    for q in queries:
        q = str(q).replace(" ", "")
        grams = {q[i:i + 2] for i in range(len(q) - 1)} or {q}
        if grams and len(grams & {desc[i:i + 2] for i in range(len(desc) - 1)}) / len(grams) >= 0.5:
            hit += 1
    return round(hit / len(queries), 4)


def export_diff_report(org_path, cur_path, test_data_path, updated_ids, out_path):
    """把所有被更新资源号的前后描述、指标、客观信号写入 Markdown 报告。

    主观优劣评述留空占位，由人工/模型事后补写。
    """
    org_tools = load_tools_from_json(org_path)
    cur_tools = load_tools_from_json(cur_path)
    query_all = load_tools_from_json(test_data_path)

    lines = [
        "# 描述优化前后对比报告",
        "",
        f"- 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 原始基线: `{org_path}`",
        f"- 当前结果: `{cur_path}`",
        f"- 更新资源号数: {len(updated_ids)}",
        "",
    ]

    for rid in updated_ids:
        org_desc = org_tools.get(rid, {}).get("description", "")
        new_desc = cur_tools.get(rid, {}).get("description", "")
        queries = list(query_all.get(rid, {}).keys())

        best_dir = os.path.join("../recall_outputs/optimizer/version_best")
        summary = {}
        for folder in os.listdir(best_dir) if os.path.isdir(best_dir) else []:
            if folder.startswith(f"{rid}_"):
                p = os.path.join(best_dir, folder, "best_summary.json")
                if os.path.isfile(p):
                    with open(p, encoding="utf-8") as f:
                        summary = json.load(f)
                break

        v0_csv = f"../recall_outputs/optimizer/version_0/{rid}/tool_pass_at_k_recall_summary.csv"
        v0_r1 = v0_r3 = None
        if os.path.isfile(v0_csv):
            import pandas as pd
            df = pd.read_csv(v0_csv)
            merged = df[df["view"] == "merged"]
            r1 = merged[merged["k"] == 1]["pass_at_k"].values
            r3 = merged[merged["k"] == 3]["pass_at_k"].values
            v0_r1 = float(r1[0]) if len(r1) else None
            v0_r3 = float(r3[0]) if len(r3) else None

        def _row(name, before, after):
            if before is None or after is None:
                return f"| {name} | - | - | - |"
            return f"| {name} | {before:.4f} | {after:.4f} | {after - before:+.4f} |"

        lines += [
            f"## 资源号 {rid}",
            "",
            f"- 标题: {cur_tools.get(rid, {}).get('title', '')}",
            f"- 最佳版本: {summary.get('best_version_id', 'N/A')}",
            f"- 测试 query 数: {len(queries)}",
            "",
            "| 指标 | version_0 | best | delta |",
            "| --- | --- | --- | --- |",
            _row("Recall@1", v0_r1, summary.get("recall1")),
            _row("Recall@3", v0_r3, summary.get("recall3")),
            "",
            f"- 描述长度: {len(org_desc)} -> {len(new_desc)} ({len(new_desc) - len(org_desc):+d})",
            f"- query 词覆盖率: {_query_term_coverage(org_desc, queries)} -> {_query_term_coverage(new_desc, queries)}",
            "",
            "### 原始描述",
            "",
            "```",
            org_desc,
            "```",
            "",
            "### 优化描述",
            "",
            "```",
            new_desc,
            "```",
            "",
            "### 优劣评述",
            "",
            "> TODO: 待人工/模型补充对本次改写的定性判断",
            "",
            "---",
            "",
        ]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return out_path


def run_single_resource(
        resource_id: str,
        tools_description_path: str = "../data/optimizer/tool_descriptions.json",
        test_data_path: str = "../data/optimizer/query.json",
        prompt_path: str = "../config/prompts.json",
        flow_config_path: str = "../config/agent_config.yaml",
        max_iterations: int = 3,
        write_back: bool = True,
        eval_view: str = "title_description",
):
    """对单个资源号执行完整的优化流程，返回 (state, changed)。

    write_back: 是否把最佳 description 回写到 tools_description_path。
                并行模式下应设为 False。
    eval_view: 召回视角开关 ('merged' / 'title_description' / 'description')。
    """
    dag = ToolOptimizerGraph(
        resource_id=resource_id,
        prompt_path=prompt_path,
        flow_config_path=flow_config_path,
        tools_description_path=tools_description_path,
        test_data_path=test_data_path,
        checkpointer=None,
        eval_view=eval_view,
    )

    graph = StateGraph(ToolOptimizerState)
    graph.add_node("statistic_check_tool", dag.statistic_check_tool)
    graph.add_node("description_optimizer", dag.llm_description_optimizer)
    graph.add_node("description_judge", dag.llm_description_judge)

    graph.set_entry_point("statistic_check_tool")
    graph.add_conditional_edges(
        "statistic_check_tool",
        should_continue,
        {
            "description_optimizer": "description_optimizer",
            END: END,
        },
    )
    graph.add_edge("description_optimizer", "description_judge")
    graph.add_edge("description_judge", "statistic_check_tool")

    compiled_graph = graph.compile()

    initial_state = {
        "resource_id": resource_id,
        "title": dag.tools_dict[resource_id]["title"],
        "original_description": dag.tools_dict[resource_id]["description"],
        "current_version_id": 0,
        "current_description": dag.tools_dict[resource_id]["description"],
        "next_version_id": 1,
        "max_iterations": max_iterations,
        "iteration": 0,
        "version_history": [],
        "optimizer_history": [],
        "critic_history": [],
        "generate_history": [],
        "judge_history": [],
        "verify_judge_history": [],
    }

    state = compiled_graph.invoke(initial_state)
    save_pickle(state, "../recall_outputs/optimizer/" + resource_id + ".pkl")
    _, changed = export_best_version(state, resource_id, tools_description_path, write_back=write_back)
    return state, changed


# =========================
# Main: batch loop (串行) 或 单资源号模式
# =========================
if __name__ == "__main__":
    import sys
    import argparse
    import logging
    from datetime import datetime

    parser = argparse.ArgumentParser(description="工具描述优化器")
    parser.add_argument("--resource-id", type=str, default=None,
                        help="指定单个资源号执行（供并行调度脚本调用）")
    parser.add_argument("--no-writeback", action="store_true",
                        help="不回写主文件，仅归档 best_summary.json（并行模式使用）")
    parser.add_argument("--max-iterations", type=int, default=3,
                        help="每个资源号的最大优化轮次")
    parser.add_argument("--eval-view", type=str, default="title_description",
                        choices=["merged", "title_description", "description"],
                        help="召回视角开关：title_description=标题+描述单路(默认); "
                             "merged=三路取最高分融合; description=仅描述单路")
    args = parser.parse_args()

    TOOLS_DESC_PATH = "../data/optimizer/tool_descriptions.json"
    TEST_DATA_PATH = "../data/optimizer/query.json"
    LOG_DIR = "../recall_outputs/optimizer"
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(os.path.join(LOG_DIR, "logs"), exist_ok=True)

    # ============ 单资源号模式（被并行调度脚本调用） ============
    if args.resource_id:
        rid = args.resource_id
        log_file = os.path.join(LOG_DIR, "logs", f"{rid}.log")
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(log_file, mode="w", encoding="utf-8"),
            ],
        )
        log = logging.getLogger("optimizer")
        log.info(f"[single] 开始执行 resource_id={rid}, write_back={not args.no_writeback}, eval_view={args.eval_view}")

        try:
            state, changed = run_single_resource(
                resource_id=rid,
                tools_description_path=TOOLS_DESC_PATH,
                test_data_path=TEST_DATA_PATH,
                max_iterations=args.max_iterations,
                write_back=(not args.no_writeback),
                eval_view=args.eval_view,
            )
            log.info(f"[single] resource_id={rid} 完成, changed={changed}")
            sys.exit(0)
        except Exception as e:
            log.error(f"[single] resource_id={rid} 执行失败: {e}")
            sys.exit(1)

    # ============ 批量串行模式（兼容原来的用法） ============
    LOG_PATH = os.path.join(LOG_DIR, "run_log.json")
    RUN_LOG_TXT = os.path.join(LOG_DIR, "run_log.txt")
    LIMIT = None  # 只跑前 N 个；None 表示全量

    os.makedirs(LOG_DIR, exist_ok=True)

    # 日志同时输出到控制台和文件
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(RUN_LOG_TXT, mode="w", encoding="utf-8"),
        ],
    )
    log = logging.getLogger("optimizer")

    # 加载所有资源号，每次运行都独立执行全部资源号
    query_gold_dict = load_tools_from_json(TEST_DATA_PATH)

    # 支持通过环境变量指定资源号列表，便于回归重跑
    only_ids_env = os.environ.get("OPTIMIZER_ONLY_IDS", "").strip()
    if only_ids_env:
        todo_ids = [x.strip() for x in only_ids_env.split(",") if x.strip()]
    else:
        todo_ids = list(query_gold_dict.keys())
        random.shuffle(todo_ids)
        if LIMIT is not None:
            todo_ids = todo_ids[:LIMIT]

    # 每次运行重建 log
    run_log = {"order": list(todo_ids), "updated": [], "failed": []}

    log.info(f"[main] 本轮待执行 {len(todo_ids)} 个资源号")
    log.info(f"[main] 执行顺序: {todo_ids[:10]}{'...' if len(todo_ids) > 10 else ''}")

    for i, resource_id in enumerate(todo_ids):
        log.info(f"{'=' * 60}")
        log.info(f"[main] ({i + 1}/{len(todo_ids)}) 开始执行 resource_id={resource_id}")
        log.info(f"{'=' * 60}")
        try:
            state, changed = run_single_resource(
                resource_id=resource_id,
                tools_description_path=TOOLS_DESC_PATH,
                test_data_path=TEST_DATA_PATH,
            )
            if changed:
                run_log["updated"].append(resource_id)
                log.info(f"[main] resource_id={resource_id} 描述已更新")
            else:
                log.info(f"[main] resource_id={resource_id} 描述未变化")
        except Exception as e:
            log.error(f"[main] resource_id={resource_id} 执行失败: {e}")
            run_log["failed"].append({"resource_id": resource_id, "error": str(e)})

        # 每个资源号结束后保存 log（防止中途中断丢失记录）
        with open(LOG_PATH, "w", encoding="utf-8") as f:
            json.dump(run_log, f, ensure_ascii=False, indent=2)

    log.info(f"[main] 全部完成。更新了 {len(run_log['updated'])} 个描述，失败 {len(run_log['failed'])} 个")
    log.info(f"[main] 日志: {LOG_PATH}")
    log.info(f"[main] 执行日志: {RUN_LOG_TXT}")

    if run_log["updated"]:
        report_path = export_diff_report(
            org_path="../data/optimizer/tool_descriptions_org.json",
            cur_path=TOOLS_DESC_PATH,
            test_data_path=TEST_DATA_PATH,
            updated_ids=run_log["updated"],
            out_path=os.path.join(LOG_DIR, "diff_report.md"),
        )
        log.info(f"[main] 前后对比报告: {report_path}")

