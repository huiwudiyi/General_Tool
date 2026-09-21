#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Project : General_Tool
@File    : tool_multimodal_main.py
@Desc    : 多模态工具功能描述主流程（LangGraph 编排）

        ┌──────────────── Excel 输入 ────────────────┐
        │  query / srcid / label_data / 截图          │
        └──────────────┬─────────────────────────────┘
                       │ 按 srcid 聚合（多 query 合并）
                       ▼
        ┌── ① 描述 agent（截图 + 召回文本 → 描述）───┐   describe 节点
        │  输出 {界面布局描述, 功能与服务总结}        │
        └──────────────┬─────────────────────────────┘
                       ▼
        ┌── ② judge agent（描述 × 文本 × 截图）──────┐   judge 节点
        │  输出 {布局研判, 功能研判, 证据, 建议, 评分} │
        └──────────────┬─────────────────────────────┘
              评分达标 │          │ 未达标且未超轮次
                       ▼          ▼
                     收尾    ┌── ③ generator agent ──┐  refiner 节点
                             │  按证据+建议定向优化   │
                             └────────┬───────────────┘
                                      └──→ 回到 ② 再研判

框架要件：
- langgraph  ：三节点 + 条件回边的状态机编排
- harness    ：run_harness 的调用守卫（异常分类/退避/熔断/降级）与检查点
- sqlite     ：langgraph 检查点持久化（节点级续跑）+ ResultStore 业务产物落库

代码风格对齐 src_summary：每个 agent 一个模块，静态方法 _gen_prompt / _vertify_result
返回 (result, flag, error_type)；节点方法挂在一个图类上，__main__ 遍历批量执行。
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, List, Literal, Mapping, Optional, Sequence, Tuple

import yaml
from json_repair import repair_json

from utils import *
from state import *
from llm_client import LLMClient

from flow_config import FlowConfig
from prompt_registry import PromptRegistry

from image_utils import build_multimodal_messages, count_usable
from excel_loader import load_samples, MultiModalSample
from llm_description_agent import LLMMultiModalDescription
from llm_judge_agent import LLMMultiModalJudge
from llm_generator_agent import LLMMultiModalGenerator
from result_store import ResultStore
from run_harness import CheckpointStore, LLMCallGuard

try:
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import END, START, StateGraph
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "Missing langgraph dependencies. Please run: pip install langgraph langchain-core"
    ) from exc

from enum import Enum


class Const(Enum):
    PATH = '../data/data_multimodal/'
    HOST = "tianchi-proxy.baidu-int.com"
    APPID = 'app-RNgOjXzL'
    DEFAULT_MODEL = "deepseek-v4-flash"
    OUTPUT_SUBDIR = "multimodal_output"
    EXCEL_NAME = "multimodal_input.xlsx"

def build_serde() -> Any:
    """登记本模块的自定义 dataclass，避免未来 langgraph 收紧反序列化后旧检查点失效。"""
    try:
        from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
    except ImportError:
        return None
    try:
        return JsonPlusSerializer(allowed_msgpack_modules=[("state", "StageRecord")])
    except TypeError:
        return None


def build_checkpointer(db_path: str) -> Any:
    """构造可持久化 checkpointer，支持跨进程的节点级续跑。

    装不上 SqliteSaver 时退回 InMemorySaver：流程照样能跑，只是中断后无法续跑。
    需额外安装：pip install langgraph-checkpoint-sqlite
    """
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
    except ImportError:
        print("[ckpt] 未安装 langgraph-checkpoint-sqlite，回退 InMemorySaver（中断后无法续跑）")
        return InMemorySaver()

    directory = os.path.dirname(db_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    serde = build_serde()
    try:
        import sqlite3

        conn = sqlite3.connect(db_path, check_same_thread=False)
        saver = SqliteSaver(conn, serde=serde) if serde is not None else SqliteSaver(conn)
        print(f"[ckpt] SqliteSaver 已就绪（节点级续跑可用）：{db_path}")
        return saver
    except Exception as exc:
        print(f"[ckpt] SqliteSaver(conn) 构造失败（{exc}），改试 from_conn_string")

    try:
        saver = SqliteSaver.from_conn_string(db_path)
        if hasattr(saver, "__enter__"):
            saver = saver.__enter__()
        print(f"[ckpt] SqliteSaver.from_conn_string 已就绪：{db_path}")
        return saver
    except Exception as exc:
        print(f"[ckpt] SqliteSaver 不可用（{exc}），回退 InMemorySaver")
        return InMemorySaver()


def pending_nodes(compiled_graph: Any, config: Mapping[str, Any]) -> Tuple[Any, ...]:
    """该 thread 是否还有未执行节点；非空即可用 invoke(None, config) 续跑。"""
    try:
        snapshot = compiled_graph.get_state(config)
    except Exception as exc:
        print(f"[ckpt] 读取图状态失败，按全新开始处理: {exc}")
        return ()
    return tuple(getattr(snapshot, "next", ()) or ())


class MultiModalDescriptionGraph:
    """描述 → 研判 → 优化 三段式流程。"""

    def __init__(
        self,
        srcid: str,
        sample: MultiModalSample,
        llm_client: Optional[LLMClient] = None,
        prompt_path: str = "../config/prompts_multimodal.json",
        flow_config_path: str = "../config/agent_config.yaml",
        max_rounds: int = 2,
        min_pass_score: int = 3,
        max_images: int = 4,
        result_store: Optional[ResultStore] = None,
        call_guard: Optional[LLMCallGuard] = None,
        checkpointer: Optional[Any] = None,
        checkpoint_root: Optional[str] = None,
    ) -> None:
        self.prompts = PromptRegistry(prompt_path).get_promts()
        self.flow_config = FlowConfig(flow_config_path)
        self.srcid = str(srcid)
        self.sample = sample
        self.max_rounds = max(1, int(max_rounds))
        self.min_pass_score = int(min_pass_score)
        self.max_images = max(1, int(max_images))
        self.result_store = result_store

        # harness：模型调用守卫。多模态请求体大、耗时长，超时比纯文本更常见，
        # 因此把基础退避拉长一些；400 这类不可重试错误仍然立刻放弃。
        self.call_guard = call_guard or LLMCallGuard(
            max_attempts=3, base_delay=4.0, max_delay=60.0, jitter=0.3, circuit_threshold=5
        )
        self.checkpointer = checkpointer or InMemorySaver()
        self.checkpoint = CheckpointStore(
            root=checkpoint_root or os.path.join(Const.PATH.value, Const.OUTPUT_SUBDIR.value),
            resource_id=self.srcid,
        )

        self.node_model_map: Dict[str, str] = {}
        if self.flow_config.node_model_map:
            self.node_model_map.update({str(k): str(v) for k, v in self.flow_config.node_model_map.items()})
        self.node_temperature_map: Dict[str, float] = {}
        if self.flow_config.node_temperature_map:
            self.node_temperature_map.update(
                {str(k): float(v) for k, v in self.flow_config.node_temperature_map.items()}
            )

        llm_dict: Dict[str, LLMClient] = {}
        if len(self.flow_config.clients) != 0:
            for name, cfg in self.flow_config.clients.items():
                if "tianchi" in cfg.get("base_url", ""):
                    cfg['host'] = Const.HOST.value
                    cfg['appid'] = Const.APPID.value
                    print(name, "cfg:", cfg)
                    llm_dict[name] = LLMClient(**cfg)
        print("llm_dict", llm_dict)
        self.llm_client = llm_client or LLMClient(**self.flow_config.config["default_client"])
        self.node_llm_clients: Dict[str, LLMClient] = {}
        if self.flow_config.node_llm_client_map:
            self.node_llm_clients.update(
                {str(k): llm_dict[v] for k, v in self.flow_config.node_llm_client_map.items() if v in llm_dict}
            )

    # ---------------- LLM 调用 ----------------
    def _client_for(self, node_name: str) -> LLMClient:
        return self.node_llm_clients.get(node_name, self.llm_client)

    def _model_for(self, node_name: str) -> str:
        return self.node_model_map.get(node_name, Const.DEFAULT_MODEL.value)

    def _temperature_for(self, node_name: str) -> float:
        return self.node_temperature_map.get(node_name, 0.3)

    def _generate(
        self,
        node_name: str,
        prompt: str,
        screenshots: Optional[Sequence[Any]] = None,
        extra_body: Dict[str, Any] = {"top_p": 0.9},
    ) -> Any:
        """带截图的模型调用，走 harness 守卫。

        失败统一返回 (None, "", False)：下游 _vertify_result 对非 Mapping 判
        "response type error"，节点走"重试→放弃"的既有路径，不让异常击穿流程。
        """
        client = self._client_for(node_name)
        messages = build_multimodal_messages(prompt, screenshots, max_images=self.max_images)
        response, ok, error = self.call_guard.call(
            client.generate_text,
            messages=messages,
            model=self._model_for(node_name),
            temperature=self._temperature_for(node_name),
            extra_body=extra_body,
        )
        if not ok:
            print(f"[{node_name}] 模型调用失败 kind={error.get('kind')}: {error.get('message')}")
            return None, "", False
        parsed, ok_parse, parse_error = self.call_guard.call(client.parse_chat_content, response)
        if not ok_parse:
            print(f"[{node_name}] 响应解析异常: {parse_error.get('message')}")
            return None, "", False
        return parsed

    def _with_checkpoint(self, node_name: str, node_fn: Callable[[Any], Any]) -> Callable[[Any], Any]:
        """节点包装：跑完落盘，异常先落盘再抛，便于定位死在哪个节点。"""

        def wrapped(state: MultiModalState):
            try:
                updates = node_fn(state) or {}
            except BaseException:
                self.checkpoint.save("%s:failed" % node_name, state)
                print(f"[ckpt] 节点 {node_name} 异常，已落盘中断点: {self.checkpoint.path}")
                raise
            self.checkpoint.save(node_name, {**dict(state), **dict(updates)})
            return updates

        return wrapped

    # ---------------- ① 描述 agent ----------------
    def describe(self, state: MultiModalState):
        """截图 + 召回文本 → 初版描述。"""
        screenshots = list(state.get("screenshots", []) or [])
        usable = count_usable(screenshots)
        if usable == 0:
            # 没有可用截图仍继续：纯文本也能产出功能总结，只是缺视觉证据，
            # 这里显式告警，方便事后区分"描述质量差"是模型问题还是输入缺图。
            print(f"[describe] srcid={self.srcid} 无可用截图，退化为纯文本分析")
        else:
            print(f"[describe] srcid={self.srcid} 可用截图 {usable}/{len(screenshots)} 张")

        prompt = LLMMultiModalDescription._gen_prompt(
            state=state, prompt=self.prompts["multimodal_description"]
        )
        result: Dict[str, Any] = {}
        flag = False
        for _ in range(3):
            response, stype, ok = self._generate("describe", prompt, screenshots)
            result, flag, error_type = LLMMultiModalDescription._vertify_result(response)
            if flag:
                break
            print(f"[describe] 校验失败: {error_type}")
        if not flag:
            print(f"[describe] srcid={self.srcid} 始终未产出合规描述，跳过该 srcid")
            return {"accepted": False}

        record = StageRecord(srcid=self.srcid, round_id=0, stage="describe", info=result)
        if self.result_store:
            self.result_store.log_stage(self.srcid, 0, "describe", result)
        print(f"[describe] 初版描述完成，{result['total_length']} 字")
        return {
            "description": result["description"],
            "layout_text": result["界面布局描述"],
            "function_text": result["功能与服务总结"],
            "round_id": 0,
            "description_history": append_description_history(state, record),
        }

    # ---------------- ② judge agent ----------------
    def judge(self, state: MultiModalState):
        """描述 × 召回文本 × 截图 三方研判。"""
        description = str(state.get("description", "") or "")
        if not description:
            print("[judge] 无描述可研判，跳过")
            return {"judge_passed": False}

        prompt = LLMMultiModalJudge._gen_prompt(
            state=state, prompt=self.prompts["multimodal_judge"], description=description
        )
        if not prompt:
            return {"judge_passed": False}

        result: Dict[str, Any] = {}
        flag = False
        for _ in range(3):
            response, stype, ok = self._generate("judge", prompt, state.get("screenshots", []))
            result, flag, error_type = LLMMultiModalJudge._vertify_result(response)
            if flag:
                break
            print(f"[judge] 校验失败: {error_type}")
        if not flag:
            # 研判失败不能当作"通过"，否则会把未核验的描述直接收下；
            # 这里判为不通过但也不给建议，下游 refine 会因无建议而跳过，流程收尾。
            print("[judge] 始终未产出合规研判，本轮按不通过处理且不再优化")
            return {"judge_passed": False, "suggestions": []}

        score = int(result["评分"])
        passed = score >= self.min_pass_score
        round_id = int(state.get("round_id", 0) or 0)
        record = StageRecord(srcid=self.srcid, round_id=round_id, stage="judge", info=result)
        if self.result_store:
            self.result_store.log_stage(self.srcid, round_id, "judge", result)
        print(
            "[judge] 评分=%d %s，证据 %d 条，建议 %d 条"
            % (score, "通过" if passed else "未通过", len(result["证据"]), len(result["优化建议"]))
        )
        return {
            "judge_score": score,
            "judge_passed": passed,
            "evidence": result["证据"],
            "suggestions": result["优化建议"],
            "judge_history": append_judge_history(state, record),
        }

    # ---------------- ③ generator agent ----------------
    def refiner(self, state: MultiModalState):
        """按研判的证据与建议定向优化描述。"""
        prompt = LLMMultiModalGenerator._gen_prompt(
            state=state, prompt=self.prompts["multimodal_generator"]
        )
        if not prompt:
            print("[refiner] 无描述或无优化建议，跳过优化")
            return {}

        result: Dict[str, Any] = {}
        flag = False
        for _ in range(3):
            response, stype, ok = self._generate("refiner", prompt, state.get("screenshots", []))
            result, flag, error_type = LLMMultiModalGenerator._vertify_result(response)
            if flag:
                break
            print(f"[refiner] 校验失败: {error_type}")
        if not flag:
            # 优化失败就保留上一版描述，不要把已有成果丢掉
            print("[refiner] 优化输出始终不合规，保留上一版描述")
            return {"round_id": int(state.get("round_id", 0) or 0) + 1}

        round_id = int(state.get("round_id", 0) or 0) + 1
        record = StageRecord(srcid=self.srcid, round_id=round_id, stage="refiner", info=result)
        if self.result_store:
            self.result_store.log_stage(self.srcid, round_id, "refiner", result)
        print(f"[refiner] 第 {round_id} 轮优化完成，{result['total_length']} 字")
        return {
            "description": result["description"],
            "layout_text": result["界面布局描述"],
            "function_text": result["功能与服务总结"],
            "round_id": round_id,
            "generator_history": append_generator_history(state, record),
        }

    # ---------------- 收尾 ----------------
    def finalize(self, state: MultiModalState):
        """定稿：把当前描述固化为产物并落库。"""
        description = str(state.get("description", "") or "")
        accepted = bool(state.get("judge_passed")) and bool(description)
        updates = {
            "final_description": description,
            "accepted": accepted,
        }
        merged = {**dict(state), **updates}
        if self.result_store:
            self.result_store.save_result(merged)
        print(
            "[finalize] srcid=%s 定稿%s（评分=%s，轮次=%s）"
            % (self.srcid, "（已通过研判）" if accepted else "（未通过研判，仍保留描述）",
               state.get("judge_score"), state.get("round_id"))
        )
        return updates

# ---------------- 路由 ----------------
def judge_router(state: MultiModalState) -> Literal["refiner", "finalize"]:
    """研判后分流：通过则定稿；未通过且还有轮次、且有可执行建议才去优化。

    三个条件都要满足才 refiner——只要有一个不满足，再转一轮也不会变好，
    只是白烧多模态调用。
    """
    if state.get("judge_passed"):
        return "finalize"
    if int(state.get("round_id", 0) or 0) >= int(state.get("max_rounds", 1) or 1):
        print("[router] 已达最大优化轮次，直接定稿")
        return "finalize"
    if not list(state.get("suggestions", []) or []):
        print("[router] 研判未给出可执行建议，直接定稿")
        return "finalize"
    return "refiner"


def describe_router(state: MultiModalState) -> Literal["judge", "finalize"]:
    """描述阶段失败（无描述）时直接收尾，不做无意义的研判。"""
    if str(state.get("description", "") or "").strip():
        return "judge"
    print("[router] 描述缺失，跳过研判直接收尾")
    return "finalize"


def build_multimodal_graph(dag: MultiModalDescriptionGraph):
    """describe → judge →（refiner → judge 回环）→ finalize。"""
    graph = StateGraph(MultiModalState)
    graph.add_node("describe", dag._with_checkpoint("describe", dag.describe))
    graph.add_node("judge", dag._with_checkpoint("judge", dag.judge))
    graph.add_node("refiner", dag._with_checkpoint("refiner", dag.refiner))
    graph.add_node("finalize", dag._with_checkpoint("finalize", dag.finalize))

    graph.set_entry_point("describe")
    graph.add_conditional_edges(
        source="describe",
        path=describe_router,
        path_map={"judge": "judge", "finalize": "finalize"},
    )
    graph.add_conditional_edges(
        source="judge",
        path=judge_router,
        path_map={"refiner": "refiner", "finalize": "finalize"},
    )
    # 优化后回到研判，形成「改一版 → 再判一次」的闭环
    graph.add_edge("refiner", "judge")
    graph.add_edge("finalize", END)
    return graph.compile(checkpointer=dag.checkpointer)


def build_initial_state(sample: MultiModalSample, max_rounds: int, min_pass_score: int) -> Dict[str, Any]:
    """从 Excel 聚合结果构造初始状态。"""
    return {
        "srcid": sample.srcid,
        "queries": list(sample.queries),
        "label_data": sample.label_data,
        "screenshots": list(sample.screenshots),
        "description": "",
        "layout_text": "",
        "function_text": "",
        "judge_score": None,
        "judge_passed": False,
        "evidence": [],
        "suggestions": [],
        "round_id": 0,
        "max_rounds": int(max_rounds),
        "min_pass_score": int(min_pass_score),
        "final_description": "",
        "accepted": False,
        "description_history": [],
        "judge_history": [],
        "generator_history": [],
    }


if __name__ == "__main__":
    output_root = Const.PATH.value + Const.OUTPUT_SUBDIR.value
    os.makedirs(output_root, exist_ok=True)

    EXCEL_PATH = Const.PATH.value + Const.EXCEL_NAME.value
    MAX_ROUNDS = 2
    MIN_PASS_SCORE = 3
    CKPT_DB = os.path.join(output_root, "checkpoints", "langgraph.sqlite")
    RESULT_DB = os.path.join(output_root, "multimodal_result.sqlite")

    store = ResultStore(RESULT_DB)
    finished = set(store.finished_srcids())
    samples = load_samples(EXCEL_PATH, image_dir=os.path.join(output_root, "embedded_images"))
    print(f"待处理 srcid={len(samples)}，已完成={len(finished)}")

    for srcid, sample in samples.items():
        if srcid in finished:
            print("skip !!!!", srcid)
            continue
        print("=" * 20, "开始执行：", srcid, "=" * 20)
        try:
            dag = MultiModalDescriptionGraph(
                srcid=srcid,
                sample=sample,
                prompt_path="../config/prompts.json",
                flow_config_path="../config/agent_config.yaml",
                max_rounds=MAX_ROUNDS,
                min_pass_score=MIN_PASS_SCORE,
                result_store=store,
                checkpointer=build_checkpointer(CKPT_DB),
                checkpoint_root=output_root,
            )
            compiled_graph = build_multimodal_graph(dag)
            run_config = {"configurable": {"thread_id": str(srcid)}}

            waiting = pending_nodes(compiled_graph, run_config)
            if waiting:
                print(f"[ckpt] 节点级续跑，待执行节点={waiting}")
                graph_input: Optional[Dict[str, Any]] = None
            else:
                graph_input = build_initial_state(sample, MAX_ROUNDS, MIN_PASS_SCORE)

            state = compiled_graph.invoke(graph_input, config=run_config)
            dag.checkpoint.clear()
            print("[guard] 调用统计:", dag.call_guard.stats.to_dict())
            if dag.call_guard.circuit_open:
                print("[guard] 熔断已打开，endpoint 可能不可用，终止后续 srcid")
                break
        except Exception as exc:
            print("srcid error:", srcid, exc)
            continue

    rows = store.export_rows()
    accepted = sum(1 for row in rows if row.get("accepted"))
    print(f"\n全部完成：产出 {len(rows)} 条，其中研判通过 {accepted} 条")
    print(f"产物库：{RESULT_DB}")
    store.close()

