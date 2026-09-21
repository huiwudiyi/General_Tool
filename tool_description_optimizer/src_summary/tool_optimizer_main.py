#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Project : General_Tool
@File    : tool_optimizer_main.py
@Desc    : 工具描述「库级共同进化」主流程（LangGraph 编排）

流程对应关系：

        ┌──────────────────────────────────┐
        │  描述生成器 π_θ (待训练 LLM)      │  ← policy_generate 节点
        │  输出结构化描述 d={功能区,边界区} │
        └──────────────┬───────────────────┘
                       │ 写入 / 替换工具描述
                       ▼
        ┌──────── 工具描述库 (向量索引) ────────┐  ← rebuild_index
        └──────────────┬────────────────────────┘
          ┌────────────┴────────────┐
          ▼                          ▼
   ① Retriever (冻结 embedding)   ② LLM Selector (冻结裁判)
   nn_recall_passk.py             llm_generator.LLMAladdinGenerator
          │                          │
          ▼                          ▼
   MRR 检索奖励                选择正确率 + 边界判别奖励
   retriever_reward 节点        selector_reward 节点(rerank_score.py 打分)
                     │
                     ▼
                 奖励聚合   ← aggregate_reward 节点
                     │
                     ▼   (每 epoch 全库重生成 → 重建索引 → 再评估)
              库级共同进化 (self-evolution)

复用的已有模块：
- nn_recall_passk.ToolPassAtKRecallEvaluator ：冻结 Retriever，提供 mrr_at_k（检索奖励）
- llm_generator.LLMPolicyGenerator           ：策略 π_θ 输出 {功能区,边界区} 的结构校验
- llm_generator.LLMAladdinGenerator          ：冻结 LLM Selector，query+topk → 选卡
- llm_description_judge.LLMDescriptionJudge  ：语义保真裁判，拦截语义漂移的候选描述
- rerank_score.RerankScorer                  ：把 Selector 的 srcid 与 resource_id 比对成指标

重要边界说明（GRPO）：
本文件实现的是「采样 → 奖励 → 组内相对优势 → 导出训练样本」这一段。
真正的策略梯度更新需要 trl/verl 之类的训练器与可训练权重，
因此 grpo_update 只是一个可注入的 hook：默认把带 advantage 的样本落盘成 JSONL，
不做任何权重更新。接入训练框架时把 grpo_trainer 传进构造函数即可。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Mapping, Optional, Sequence, Tuple

import yaml
from json_repair import repair_json

from utils import *
from state import *
from llm_client import LLMClient

from flow_config import FlowConfig
from prompt_registry import PromptRegistry

# 复用的评估 / 筛选 / 打分模块
from nn_server_passk import ToolPassAtKRecallEvaluator, encode_fn
from llm_generator import LLMAladdinGenerator, LLMPolicyGenerator
from llm_description_judge import LLMDescriptionJudge
from llm_description_refiner import LLMDescriptionRefiner
from rerank_score import RerankScorer
# 运行时保障：模型调用异常/超时守卫 + 中断恢复检查点
from run_harness import CheckpointStore, LLMCallGuard

try:
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import END, START, StateGraph
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "Missing langgraph dependencies. Please run: pip install langgraph langchain-core"
    ) from exc

from enum import Enum


def build_checkpointer(db_path: str) -> Any:
    """构造可持久化 checkpointer，用于跨进程的节点级续跑。

    InMemorySaver 随进程消亡，只能支持进程内恢复；要在 Ctrl+C / OOM / 机器重启后
    从中断的那个节点继续，检查点必须落盘。SqliteSaver 在 langgraph 各版本的构造
    方式不一致（有的收 Connection、有的用 from_conn_string 且返回上下文管理器），
    这里逐个兜住；都不行才退回 InMemorySaver，此时降级为 epoch 级恢复。

    需额外安装：pip install langgraph-checkpoint-sqlite
    """
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
    except ImportError:
        print("[ckpt] 未安装 langgraph-checkpoint-sqlite，回退 InMemorySaver（降级为 epoch 级恢复）")
        return InMemorySaver()

    directory = os.path.dirname(db_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    try:
        import sqlite3

        conn = sqlite3.connect(db_path, check_same_thread=False)
        saver = SqliteSaver(conn)
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
        print(f"[ckpt] SqliteSaver 不可用（{exc}），回退 InMemorySaver（降级为 epoch 级恢复）")
        return InMemorySaver()


def pending_nodes(compiled_graph: Any, config: Mapping[str, Any]) -> Tuple[Any, ...]:
    """查询该 thread 是否还有未执行的节点。

    LangGraph 的 StateSnapshot.next 是"下一步待执行的节点"元组：
    非空说明上次中断在半路，可以用 invoke(None, config) 从那里继续。
    """
    try:
        snapshot = compiled_graph.get_state(config)
    except Exception as exc:
        print(f"[ckpt] 读取图状态失败，按全新开始处理: {exc}")
        return ()
    return tuple(getattr(snapshot, "next", ()) or ())


class Const(Enum):
    PATH = '../data/data_not_import/'
    HOST = "tianchi-proxy.baidu-int.com"
    APPID = 'app-RNgOjXzL'
    DEFAULT_MODEL = "deepseek-v4-flash"
    OUTPUT_SUBDIR = "optimizer_output"


@dataclass
class RewardWeights:
    """三路奖励的聚合权重。"""

    retrieval: float = 0.4     # MRR 检索奖励
    selection: float = 0.4     # 正例上的选择正确率
    boundary: float = 0.2      # 负例上的边界判别（不该选中时没选中）

    def aggregate(self, retrieval: float, selection: float, boundary: float) -> float:
        total = self.retrieval + self.selection + self.boundary
        if total <= 0:
            return 0.0
        return (
            self.retrieval * retrieval
            + self.selection * selection
            + self.boundary * boundary
        ) / total

@dataclass
class PolicySample:
    """GRPO 一个 group 内的一条候选描述及其奖励明细。"""

    sample_index: int
    description: str
    structured: Dict[str, str] = field(default_factory=dict)
    retrieval_reward: float = 0.0
    selection_reward: float = 0.0
    boundary_reward: float = 0.0
    reward: float = 0.0
    advantage: float = 0.0
    # 语义保真裁判结果：分数越高说明与原描述的语义一致性越好
    judge_score: Optional[int] = None
    rejected: bool = False
    detail: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_index": self.sample_index,
            "description": self.description,
            "structured": self.structured,
            "retrieval_reward": self.retrieval_reward,
            "selection_reward": self.selection_reward,
            "boundary_reward": self.boundary_reward,
            "reward": self.reward,
            "advantage": self.advantage,
            "judge_score": self.judge_score,
            "rejected": self.rejected,
            "detail": self.detail,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PolicySample":
        """从 state 里的快照还原，用于节点级续跑。字段缺失时取默认值。"""
        return cls(
            sample_index=int(payload.get("sample_index", 0)),
            description=str(payload.get("description", "") or ""),
            structured=dict(payload.get("structured", {}) or {}),
            retrieval_reward=float(payload.get("retrieval_reward", 0.0) or 0.0),
            selection_reward=float(payload.get("selection_reward", 0.0) or 0.0),
            boundary_reward=float(payload.get("boundary_reward", 0.0) or 0.0),
            reward=float(payload.get("reward", 0.0) or 0.0),
            advantage=float(payload.get("advantage", 0.0) or 0.0),
            judge_score=payload.get("judge_score", None),
            rejected=bool(payload.get("rejected", False)),
            detail=dict(payload.get("detail", {}) or {}),
        )


# 缺省 prompt：config/prompts.json 里没有对应 key 时兜底使用
DEFAULT_POLICY_PROMPT = """<目标>
你是工具描述生成器。请为下面的工具重写描述，输出「功能区」与「边界区」两段结构化内容。
功能区：这个工具能解决什么问题、覆盖哪些查询意图、返回什么内容。
边界区：明确不适用的场景，即哪些相似但不该命中本工具的查询要排除。
</目标>

<工具标题>
{{title}}
</工具标题>

<当前描述>
{{description}}
</当前描述>

<正例查询（应命中本工具）>
{{positive_queries}}
</正例查询>

<负例查询（不应命中本工具）>
{{negative_queries}}
</负例查询>

<输出格式>
只输出 JSON：
```json
{"功能区": "...", "边界区": "..."}
```
</输出格式>
"""

DEFAULT_SELECTOR_PROMPT = """<目标>
你是冻结的工具选择裁判。给定用户 query 与检索到的候选工具描述，
判断哪些候选与 query 强相关、应当被调用。描述里的「边界」部分列出了不适用场景，
若 query 落在边界内则不要选中该工具。没有合适工具时 selected 输出空列表。
</目标>

<用户问题>
${query}
</用户问题>

<候选工具列表>
${aladdin}
</候选工具列表>

<输出格式>
只输出 JSON：
```json
{"reason": "选择/不选择的理由", "selected": [{"srcid": "工具srcid", "parameters": {}}]}
```
</输出格式>
"""

class ToolOptimizerGraph:
    """描述生成器 π_θ 的采样—评估—奖励聚合主流程。"""

    def __init__(
        self,
        resource_id: str,
        llm_client: Optional[LLMClient] = None,
        prompt_path: str = "../config/prompts.json",
        flow_config_path: str = "../config/agent_config.yaml",
        tools_description_path: str = "../data/summary/tool_descriptions.json",
        test_data_path: str = "../data/summary/query.json",
        group_size: int = 4,
        k_list: Sequence[int] = (1, 3),
        eval_view: str = "merged",
        max_negative_queries: int = 20,
        min_judge_score: int = 2,
        reward_weights: Optional[RewardWeights] = None,
        grpo_trainer: Optional[Callable[[str, List[PolicySample]], Any]] = None,
        checkpointer: Optional[Any] = None,
        call_guard: Optional[LLMCallGuard] = None,
        checkpoint_root: Optional[str] = None,
        checkpoint_enabled: bool = True,
    ) -> None:
        self.prompts = PromptRegistry(prompt_path).get_promts()
        self.flow_config = FlowConfig(flow_config_path)
        self.checkpointer = checkpointer or InMemorySaver()
        self.resource_id = resource_id

        # 模型调用守卫：异常分类 + 指数退避 + 熔断，失败返回降级值而不抛异常。
        # circuit_threshold=5：连续 5 次调用彻底失败就认为 endpoint 挂了，
        # 快速失败掉剩余流程，而不是把每个 resource_id 都重试一遍。
        self.call_guard = call_guard or LLMCallGuard(
            max_attempts=3, base_delay=2.0, max_delay=30.0, jitter=0.3, circuit_threshold=5
        )
        # 中断恢复检查点
        self.checkpoint = CheckpointStore(
            root=checkpoint_root or os.path.join(Const.PATH.value, Const.OUTPUT_SUBDIR.value),
            resource_id=resource_id,
            enabled=checkpoint_enabled,
        )

        # 工具描述库（全库，索引重建的输入）
        self.last_tools_description_path = tools_description_path
        self.tools_dict = load_tools_from_json(self.last_tools_description_path)

        # 测试集：{resource_id: {query: [gold_ids]}}
        self.query_good_all_dict = load_tools_from_json(test_data_path)
        self.query_good_dict = self.query_good_all_dict.get(resource_id, {})

        # GRPO / 评估超参
        self.group_size = max(1, int(group_size))
        self.k_list = sorted({int(k) for k in k_list if int(k) > 0}) or [1, 3]
        self.eval_view = eval_view
        self.embeding_term = ["description", "title_description", "merged"]
        self.max_negative_queries = max_negative_queries
        # relevance_score 低于该阈值即判定语义漂移（与 tool_summary_main 接受 2/3 分的口径一致）
        self.min_judge_score = int(min_judge_score)
        self.reward_weights = reward_weights or RewardWeights()
        self.grpo_trainer = grpo_trainer

        # 负例查询：gold 不含本 resource_id 的 query，用于边界判别奖励
        self.negative_queries = self._build_negative_queries()

        # 每轮 group 的采样结果，供 aggregate_reward 使用
        self.group_samples: List[PolicySample] = []

        # node -> model / temperature / client
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
                    llm_dict[name] = LLMClient(**cfg)

        self.llm_client = llm_client or LLMClient(**self.flow_config.config["default_client"])
        self.node_llm_clients: Dict[str, LLMClient] = {}
        if self.flow_config.node_llm_client_map:
            self.node_llm_clients.update(
                {str(k): llm_dict[v] for k, v in self.flow_config.node_llm_client_map.items() if v in llm_dict}
            )

    # ---------------- LLM 调用（沿用 tool_summary_main 的写法） ----------------
    def _client_for(self, node_name: str) -> LLMClient:
        return self.node_llm_clients.get(node_name, self.llm_client)

    def _model_for(self, node_name: str) -> str:
        return self.node_model_map.get(node_name, Const.DEFAULT_MODEL.value)

    def _temperature_for(self, node_name: str) -> float:
        return self.node_temperature_map.get(node_name, 0.8)

    def _generate_text(
        self,
        node_name: str,
        prompt: str,
        max_tokens: int = 2048,
        extra_body: Dict[str, Any] = {"top_p": 0.9},
    ) -> Any:
        """走守卫的模型调用。

        约定：失败时返回 (None, "", False) 而不抛异常。下游的 _vertify_result
        对非 Mapping 输入统一判为 "response type error"，于是节点会走
        「重试 → 放弃该样本」的既有路径，而不是把整个 resource_id 的进度带崩。
        """
        client = self._client_for(node_name)
        print(node_name, "*" * 88)
        print(node_name, "prompt= ", prompt)
        response, ok, error = self.call_guard.call(
            client.generate_text,
            prompt=prompt,
            model=self._model_for(node_name),
            temperature=self._temperature_for(node_name),
            extra_body=extra_body,
        )
        if not ok:
            print(f"[{node_name}] 模型调用失败 kind={error.get('kind')} 降级跳过本次: {error.get('message')}")
            return None, "", False

        parsed, ok_parse, parse_error = self.call_guard.call(client.parse_chat_content, response)
        if not ok_parse:
            print(f"[{node_name}] 响应解析异常，降级跳过本次: {parse_error.get('message')}")
            return None, "", False
        print(node_name, "*" * 88)
        print(node_name, "response = ", parsed)

        return parsed

    # ---------------- 中断恢复 ----------------
    def _load_group(self, state: ToolOptimizerState) -> None:
        """节点入口：把 state 里的 group 快照还原成实例属性。

        节点体内部仍然用 self.group_samples / self._sample_topk 操作（改动面最小），
        但真正的持久载体是 state——这样 LangGraph 从任意节点续跑都能拿到完整 group。
        """
        self.group_samples = [
            PolicySample.from_dict(item)
            for item in (state.get("group_samples", []) or [])
            if isinstance(item, Mapping)
        ]
        self._sample_topk = {
            int(key): dict(value or {})
            for key, value in (state.get("sample_topk", {}) or {}).items()
        }

    def _dump_group(self) -> Dict[str, Any]:
        """节点出口：把 group 序列化进 state 更新。

        sample_topk 的 key 转成 str，避免部分 checkpointer 序列化器不支持 int key。
        """
        return {
            "group_samples": [sample.to_dict() for sample in self.group_samples],
            "sample_topk": {
                str(key): value for key, value in getattr(self, "_sample_topk", {}).items()
            },
        }

    def _with_checkpoint(self, node_name: str, node_fn: Callable[[Any], Any]) -> Callable[[Any], Any]:
        """把节点包一层：跑完即落盘 state 快照，异常也先落盘再抛。

        这样中断后能知道死在哪个节点，且 state（含各 history 与 best_record）不丢。
        """

        def wrapped(state: ToolOptimizerState):
            try:
                updates = node_fn(state) or {}
            except BaseException:
                self.checkpoint.save("%s:failed" % node_name, state)
                print(f"[ckpt] 节点 {node_name} 异常，已落盘中断点: {self.checkpoint.path}")
                raise
            merged = {**dict(state), **dict(updates)}
            self.checkpoint.save(node_name, merged)
            return updates

        return wrapped

    # ---------------- 数据准备 ----------------
    def _build_negative_queries(self) -> List[str]:
        """收集 gold 不含本 resource_id 的 query，作为边界判别的负例。"""
        negatives: List[str] = []
        for other_id, query_map in self.query_good_all_dict.items():
            if other_id == self.resource_id or not isinstance(query_map, Mapping):
                continue
            for query, gold_ids in query_map.items():
                gold = gold_ids if isinstance(gold_ids, (list, tuple)) else [gold_ids]
                if self.resource_id in [str(g) for g in gold]:
                    continue
                negatives.append(str(query))
        # 去重并截断，控制单轮 Selector 调用量
        deduped = list(dict.fromkeys(negatives))
        if len(deduped) > self.max_negative_queries:
            deduped = deduped[: self.max_negative_queries]
        return deduped

    def _write_description(self, description: str) -> None:
        """把策略产出的描述写回工具描述库（替换本 resource_id 的 description）。"""
        if self.resource_id in self.tools_dict:
            self.tools_dict[self.resource_id]["description"] = description

    # ---------------- 工具描述库 → 向量索引 ----------------
    def rebuild_index(self) -> ToolPassAtKRecallEvaluator:
        """基于当前 tools_dict 重建向量索引（冻结 embedding）。"""
        eval_queries = dict(self.query_good_dict)
        for query in self.negative_queries:
            eval_queries.setdefault(query, [])

        evaluator = ToolPassAtKRecallEvaluator(
            tools=self.tools_dict,
            query_gold_ids=eval_queries,
            embeding_term=self.embeding_term,
            encode_fn=encode_fn,
            batch_size=64,
        )
        evaluator.build_vector_index()
        return evaluator

    # ---------------- ① 描述生成器 π_θ ----------------
    def _gen_policy_prompt(self, state: ToolOptimizerState) -> str:
        prompt = self.prompts.get("policy_generator", DEFAULT_POLICY_PROMPT)
        title = str(state.get("title", "") or "")
        description = str(state.get("current_description", "") or "")
        positives = list(self.query_good_dict.keys())
        return (
            prompt.replace("{{title}}", title)
            .replace("{{description}}", description)
            .replace("{{positive_queries}}", "、".join(map(str, positives)))
            .replace("{{negative_queries}}", "、".join(self.negative_queries))
        )

    def policy_generate(self, state: ToolOptimizerState):
        """采样 group_size 条候选描述，构成 GRPO 的一个 group。"""
        prompt = self._gen_policy_prompt(state)
        # 新一轮 group：实例属性与 state 里的快照一起清空
        self.group_samples = []
        self._sample_topk = {}

        for sample_index in range(self.group_size):
            structured: Dict[str, str] = {}
            flag = False
            for _ in range(3):
                response, stype, ok = self._generate_text(node_name="policy", prompt=prompt)
                structured, flag, error_type = LLMPolicyGenerator._vertify_result(response)
                if flag:
                    break
                print(f"[policy] sample={sample_index} 校验失败: {error_type}")
            if not flag:
                continue
            description = "功能：%s\n边界：%s" % (structured["功能区"], structured["边界区"])
            self.group_samples.append(
                PolicySample(sample_index=sample_index, description=description, structured=structured)
            )

        print(f"[policy] group 采样完成，有效样本 {len(self.group_samples)}/{self.group_size}")
        optimizer_record = InfoRecord(
            version_id=state.get("current_version_id", "-1"),
            stage="policy_generate",
            info={"samples": [s.to_dict() for s in self.group_samples]},
        )
        return {
            "optimizer_history": append_optimizer_history(state, optimizer_record),
            **self._dump_group(),
        }

    # ---------------- 语义保真裁判（LLMNodeCritic） ----------------
    def _gen_judge_prompt(self, state: ToolOptimizerState, description: str):
        """复用 LLMDescriptionJudge._gen_prompt 拼「原描述 vs 候选描述」的对比 prompt。

        该方法读取 optimizer_history[-1].info["optimizer_description"]，
        而本流程的 optimizer_history 存的是整个 group，所以这里为单条候选描述
        构造一个等价的临时 state，避免重复实现占位符替换逻辑。
        """
        judge_state = {
            "original_description": state.get("original_description", ""),
            "optimizer_history": [
                InfoRecord(
                    version_id=state.get("current_version_id", "-1"),
                    stage="policy_generate",
                    info={"optimizer_description": description},
                )
            ],
        }
        return LLMDescriptionJudge._gen_prompt(judge_state, self.prompts["judge"])

    def llm_description_judge(self, state: ToolOptimizerState):
        """逐条评审候选描述与原描述的语义一致性，拦截语义漂移的样本。

        作用是防止 reward hacking：策略完全可以靠编造能力、扩大适用范围来
        同时抬高检索奖励和选择奖励，但那样的描述已经不再忠于工具本身。
        relevance_score < min_judge_score 的样本标记 rejected，
        后续 retriever/selector 直接跳过其评估，奖励按 0 计入 group。
        """
        self._load_group(state)
        if not self.group_samples:
            print("[judge] group 为空，跳过")
            return {}

        for sample in self.group_samples:
            prompt_and_desc = self._gen_judge_prompt(state, sample.description)
            if not prompt_and_desc:
                print(f"[judge] sample={sample.sample_index} 描述过短或原描述缺失，跳过评审")
                sample.detail["judge"] = {"relevance_score": None, "error": "prompt unavailable"}
                continue
            prompt, _ = prompt_and_desc

            flag = False
            response: Dict[str, Any] = {}
            error_type = ""
            for _ in range(3):
                raw_response, stype, ok = self._generate_text(node_name="judge", prompt=prompt)
                # LLMDescriptionJudge._vertify_result 在响应缺 key 时会抛 KeyError，
                # 训练回路不能因为裁判返回了半截 JSON 就整轮崩掉，这里兜住
                try:
                    response, flag, error_type = LLMDescriptionJudge._vertify_result(raw_response)
                except (KeyError, TypeError, ValueError) as exc:
                    response, flag, error_type = {}, False, "%s: %s" % (type(exc).__name__, exc)
                if flag:
                    break
                print(f"[judge] sample={sample.sample_index} 校验失败: {error_type}")

            if not flag:
                # 裁判本身失败时不惩罚样本，避免把裁判的不稳定算成策略的错
                sample.detail["judge"] = {"relevance_score": None, "error": error_type}
                continue

            score = int(response["relevance_score"])
            sample.judge_score = score
            sample.rejected = score < self.min_judge_score
            sample.detail["judge"] = {
                "relevance_score": score,
                "relevance_reason": response.get("relevance_reason", ""),
                "content_quality": response.get("content_quality", ""),
                "rejected": sample.rejected,
            }
            print(
                f"[judge] sample={sample.sample_index} relevance_score={score} "
                f"{'语义漂移已拦截' if sample.rejected else '通过'}"
            )

        passed = [s for s in self.group_samples if not s.rejected]
        print(f"[judge] 通过 {len(passed)}/{len(self.group_samples)} 条候选描述")

        judge_record = InfoRecord(
            version_id=state.get("current_version_id", "-1"),
            stage="description_judge",
            info={str(s.sample_index): s.detail.get("judge", {}) for s in self.group_samples},
        )
        return {"judge_history": append_judge_history(state, judge_record), **self._dump_group()}

    # ---------------- ① Retriever（冻结）→ MRR 检索奖励 ----------------
    def retriever_reward(self, state: ToolOptimizerState):
        """逐个候选描述写入库、重建索引、跑检索，取 mrr_at_k 作为检索奖励。

        负例 query 的 gold 为空，nn_recall_passk 里 MRR 只统计有 gold 的 query，
        因此检索奖励天然只由正例决定。
        """
        self._load_group(state)
        if not self.group_samples:
            print("[retriever] group 为空，跳过")
            return {}

        max_k = max(self.k_list)
        self._sample_topk: Dict[int, Dict[str, List[str]]] = {}

        for sample in self.group_samples:
            if sample.rejected:
                print(f"[retriever] sample={sample.sample_index} 已被裁判拦截，跳过检索评估")
                self._sample_topk[sample.sample_index] = {}
                continue
            self._write_description(sample.description)
            evaluator = self.rebuild_index()
            summary_df, detail_df = evaluator.evaluate(k_list=self.k_list, eval_views=[self.eval_view])

            row = summary_df[(summary_df["view"] == self.eval_view) & (summary_df["k"] == max_k)]
            sample.retrieval_reward = float(row.iloc[0]["mrr_at_k"]) if len(row) else 0.0

            # 缓存每个 query 的 topk 候选，供 Selector 复用，避免重复检索
            topk_map: Dict[str, List[str]] = {}
            subset = detail_df[(detail_df["view"] == self.eval_view) & (detail_df["k"] == max_k)]
            for _, detail_row in subset.iterrows():
                topk_map[str(detail_row["query"])] = list(detail_row["recall_ids"])
            self._sample_topk[sample.sample_index] = topk_map

            # summary 的 pass_at_k 分母含负例（负例 gold 为空、永不 pass），会被系统性稀释，
            # 因此这里从明细里只按正例 query 求均值；MRR 本身已排除空 gold，可直接用
            sample.detail["retrieval"] = {
                "mrr_at_%d" % max_k: sample.retrieval_reward,
                "pass_at_%d" % max_k: self._positive_pass_at_k(subset),
            }

            print(f"[retriever] sample={sample.sample_index} mrr@{max_k}={sample.retrieval_reward:.4f}")

        judge_record = InfoRecord(
            version_id=state.get("current_version_id", "-1"),
            stage="retriever_reward",
            info={
                str(s.sample_index): {"retrieval_reward": s.retrieval_reward, **s.detail.get("retrieval", {})}
                for s in self.group_samples
            },
        )
        return {"retriever_history": append_retriever_history(state, judge_record), **self._dump_group()}

    def _positive_pass_at_k(self, detail_subset) -> float:
        """只按正例 query 统计 pass@k。

        rebuild_index 会把负例 query 也塞进评估集（负例 gold 为空、永远不可能 pass），
        所以 summary 里的 pass_at_k 分母被负例稀释，不能直接用。
        """
        positives = set(str(q) for q in self.query_good_dict.keys())
        if not positives or detail_subset is None or len(detail_subset) == 0:
            return 0.0
        rows = detail_subset[detail_subset["query"].astype(str).isin(positives)]
        if len(rows) == 0:
            return 0.0
        return float(rows["pass_at_k"].mean())

    # ---------------- ② LLM Selector（冻结裁判） ----------------
    def _build_candidate_cards(self, srcids: Sequence[str]) -> List[Dict[str, Any]]:
        """把 topk srcid 组装成 LLMAladdinGenerator 需要的候选卡片结构。"""
        cards: List[Dict[str, Any]] = []
        for srcid in srcids:
            tool = self.tools_dict.get(srcid, {})
            cards.append(
                {
                    "srcid": str(srcid),
                    "name": str(tool.get("title", "") or ""),
                    "description": str(tool.get("description", "") or ""),
                    "tool": tool.get("tool", {}) or {},
                }
            )
        return cards

    def _select_once(self, query: str, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """调用冻结的 LLM Selector，返回 _vertify_result 校验后的结果。"""
        prompt = LLMAladdinGenerator._gen_prompt(
            state=None,
            prompt=self.prompts.get("aladdin_select", DEFAULT_SELECTOR_PROMPT),
            query=query,
            answer="",
            aladdin_candidates=candidates,
        )
        if not prompt:
            return {"reason": "no candidate", "selected": []}

        for _ in range(3):
            response, stype, ok = self._generate_text(node_name="selector", prompt=prompt)
            result, flag, error_type = LLMAladdinGenerator._vertify_result(
                response, aladdin_candidates=candidates
            )
            if flag:
                return result
            print(f"[selector] 校验失败: {error_type}")
        return {"reason": "verify failed", "selected": []}

    def selector_reward(self, state: ToolOptimizerState):
        """正例算选择正确率，负例算边界判别率，都用 RerankScorer 打分。

        负例上 RerankScorer 的 "hit" 表示裁判误选了本工具，
        因此 boundary_reward = 1 - 负例命中率。
        """
        self._load_group(state)
        if not self.group_samples:
            print("[selector] group 为空，跳过")
            return {}

        max_k = max(self.k_list)
        positives = list(self.query_good_dict.keys())

        for sample in self.group_samples:
            if sample.rejected:
                print(f"[selector] sample={sample.sample_index} 已被裁判拦截，跳过裁判评估")
                continue
            # 让描述库回到该样本对应的状态，保证 Selector 看到的是这条候选描述
            self._write_description(sample.description)
            topk_map = getattr(self, "_sample_topk", {}).get(sample.sample_index, {})

            pos_records: List[Dict[str, Any]] = []
            neg_records: List[Dict[str, Any]] = []
            for query in positives:
                candidates = self._build_candidate_cards(topk_map.get(str(query), []))
                result = self._select_once(str(query), candidates)
                pos_records.append(
                    {
                        "query": query,
                        "resource_id": self.resource_id,
                        "selected": result,
                        "candidates": candidates,
                    }
                )
            for query in self.negative_queries:
                candidates = self._build_candidate_cards(topk_map.get(str(query), []))
                result = self._select_once(str(query), candidates)
                neg_records.append(
                    {
                        "query": query,
                        "resource_id": self.resource_id,
                        "selected": result,
                        "candidates": candidates,
                    }
                )

            pos_summary, pos_detail = RerankScorer.score_batch(pos_records, k_list=self.k_list)
            selection_reward = 0.0
            if len(pos_records) and len(pos_summary):
                pos_row = pos_summary[pos_summary["k"] == max_k].iloc[0]
                selection_reward = float(pos_row["hit_at_k"])

            boundary_reward = 1.0
            false_select_rate = 0.0
            conditional_false_select_rate = 0.0
            judgeable_negative_count = 0
            neg_detail = None
            if neg_records:
                neg_summary, _ = RerankScorer.score_batch(neg_records, k_list=self.k_list)
                if len(neg_summary):
                    neg_row = neg_summary[neg_summary["k"] == max_k].iloc[0]
                    false_select_rate = float(neg_row["hit_at_k"])
                    # 只有目标卡真的进了 topk，裁判才有"误选"的机会；
                    # 用全量负例作分母会把"检索压根没召回"也算成边界写得好，
                    # 因此奖励取 conditional 口径（RerankScorer 已算好）。
                    conditional = neg_row["conditional_hit_at_k"]
                    candidate_hit_rate = neg_row["candidate_hit_rate"]
                    if candidate_hit_rate is not None:
                        judgeable_negative_count = int(
                            round(float(candidate_hit_rate) * int(neg_row["valid_case_count"]))
                        )
                    if conditional is not None and judgeable_negative_count > 0:
                        conditional_false_select_rate = float(conditional)
                        boundary_reward = 1.0 - conditional_false_select_rate
                    else:
                        # 没有任何负例把目标卡召回进 topk，边界无从判别，不给区分度
                        boundary_reward = 1.0

            sample.selection_reward = selection_reward
            sample.boundary_reward = boundary_reward
            # 归纳三类失败案例，供 description_refine 节点做定向精修
            sample.detail["cases"] = LLMDescriptionRefiner.collect_failure_cases(
                positive_detail=pos_detail,
                negative_detail=neg_detail,
                max_k=max_k,
            )
            sample.detail["selector"] = {
                "positive_count": len(pos_records),
                "negative_count": len(neg_records),
                "judgeable_negative_count": judgeable_negative_count,
                "selection_hit_at_%d" % max_k: selection_reward,
                "negative_false_select_rate": false_select_rate,
                "negative_conditional_false_select_rate": conditional_false_select_rate,
                "boundary_reward": boundary_reward,
            }
            print(
                f"[selector] sample={sample.sample_index} "
                f"选择正确率={selection_reward:.4f} 边界判别={boundary_reward:.4f} "
                f"(可判别负例 {judgeable_negative_count}/{len(neg_records)})"
            )

        verify_record = InfoRecord(
            version_id=state.get("current_version_id", "-1"),
            stage="selector_reward",
            info={str(s.sample_index): s.detail.get("selector", {}) for s in self.group_samples},
        )
        return {"selector_history": append_selector_history(state, verify_record), **self._dump_group()}

    # ---------------- 奖励聚合 → GRPO 更新 ----------------
    def aggregate_reward(self, state: ToolOptimizerState):
        """聚合三路奖励，算组内相对优势，落盘快照并触发 GRPO 更新。"""
        self._load_group(state)
        if not self.group_samples:
            print("[aggregate] group 为空，跳过")
            version_id, next_version_id = next_version_pair(state)
            # 必须推进 iteration：否则 should_continue 会一直把流程打回 policy_generate，
            # 策略持续产出不合规输出时就成了死循环
            return {
                "current_version_id": version_id,
                "next_version_id": next_version_id,
                "iteration": state.get("iteration", 0) + 1,
            }

        for sample in self.group_samples:
            if sample.rejected:
                # 语义漂移的样本奖励清零，作为 group 内的强负样本参与优势计算
                sample.reward = 0.0
                continue
            sample.reward = self.reward_weights.aggregate(
                retrieval=sample.retrieval_reward,
                selection=sample.selection_reward,
                boundary=sample.boundary_reward,
            )

        # GRPO 组内相对优势：A_i = (r_i - mean) / (std + eps)
        rewards = [s.reward for s in self.group_samples]
        mean_reward = sum(rewards) / len(rewards)
        variance = sum((r - mean_reward) ** 2 for r in rewards) / len(rewards)
        std_reward = variance ** 0.5
        for sample in self.group_samples:
            sample.advantage = (sample.reward - mean_reward) / (std_reward + 1e-8)

        # 只从通过裁判的样本里挑最优；若全被拦截则不动描述库，避免把漂移的描述写进去
        passed = [s for s in self.group_samples if not s.rejected]
        if not passed:
            print("[aggregate] 本轮全部候选被裁判拦截，保持原描述不变")
            self._write_description(state.get("current_description", "") or "")
            version_id, next_version_id = next_version_pair(state)
            return {
                "current_version_id": version_id,
                "next_version_id": next_version_id,
                "iteration": state.get("iteration", 0) + 1,
                **self._dump_group(),
            }

        best_sample = max(passed, key=lambda s: s.reward)
        self._write_description(best_sample.description)
        version_id, next_version_id = next_version_pair(state)
        version_tag = "version_" + str(state.get("current_version_id", 0))
        output_dir = os.path.join(Const.PATH.value, Const.OUTPUT_SUBDIR.value, version_tag, self.resource_id)
        os.makedirs(output_dir, exist_ok=True)
        tool_path = os.path.join(output_dir, "tool_prompt.json")
        with open(tool_path, "w", encoding="utf-8") as writer:
            json.dump(self.tools_dict, writer, ensure_ascii=False, indent=2)

        # 复用 VersionRecord 的 4 个数值位承载奖励分解（state.py 的 dataclass 是 frozen 的，不改动它）：
        #   recall1 -> 检索奖励(MRR)  precision1 -> 选择正确率
        #   recall3 -> 边界判别奖励    precision3 -> 聚合奖励(best_record 的比较依据)
        version_info = VersionRecord(
            version_id=version_tag,
            parent_version_id=version_id,
            stage="grpo_aggregate",
            description=best_sample.description,
            tool_path=tool_path,
            top_case={"best_sample_index": best_sample.sample_index, "structured": best_sample.structured},
            case_result={"group": [s.to_dict() for s in self.group_samples], "mean_reward": mean_reward},
            recall1=best_sample.retrieval_reward,
            precision1=best_sample.selection_reward,
            recall3=best_sample.boundary_reward,
            precision3=best_sample.reward,
            accepted=True,
            reason="group best of %d, mean=%.4f, std=%.4f" % (len(rewards), mean_reward, std_reward),
        )

        self.grpo_update(output_dir=output_dir, samples=self.group_samples)

        best_record = state.get("best_record", None)
        should_update = (best_record is None) or (best_sample.reward > best_record.precision3)
        updates: Dict[str, Any] = {
            "current_version_id": version_id,
            "next_version_id": next_version_id,
            "current_description": best_sample.description,
            "iteration": state.get("iteration", 0) + 1,
            "version_history": append_version_history(state, version_info),
            # reward/advantage 必须回写：下游 refine 依赖它挑最优候选
            **self._dump_group(),
        }
        if should_update:
            updates.update(
                {
                    "best_description": best_sample.description,
                    "best_version_id": version_id,
                    "best_record": version_info,
                }
            )
        print(
            "[aggregate] best_sample=%d reward=%.4f (检索%.4f/选择%.4f/边界%.4f) mean=%.4f"
            % (
                best_sample.sample_index,
                best_sample.reward,
                best_sample.retrieval_reward,
                best_sample.selection_reward,
                best_sample.boundary_reward,
                mean_reward,
            )
        )
        return updates

    def grpo_update(self, output_dir: str, samples: List[PolicySample]) -> None:
        """GRPO 更新钩子。

        默认实现只把带 advantage 的样本导出成 JSONL，**不做任何权重更新**——
        真正的策略梯度需要 trl/verl 等训练器与可训练的 π_θ 权重。
        接入训练框架时向构造函数传入 grpo_trainer=callable(output_dir, samples)。
        """
        if self.grpo_trainer is not None:
            self.grpo_trainer(output_dir, samples)
            return

        sample_path = os.path.join(output_dir, "grpo_samples.jsonl")
        with open(sample_path, "w", encoding="utf-8") as writer:
            for sample in samples:
                writer.write(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n")
        print(f"[grpo] 未注入 trainer，仅导出训练样本（无权重更新）：{sample_path}")

    # ---------------- 基于失败案例的定向精修 ----------------
    def _score_description(self, state: ToolOptimizerState, description: str) -> PolicySample:
        """给单条描述打三路奖励。

        直接复用 retriever_reward / selector_reward 两个节点：临时把 group 换成
        只含探针样本的单元素列表，跑完再还原，避免重复实现一遍评估逻辑。
        两个节点返回的 history 更新在这里丢弃，探针的指标由 refine 节点单独记录。
        """
        saved_samples = self.group_samples
        saved_topk = getattr(self, "_sample_topk", {})
        probe = PolicySample(sample_index=-1, description=description)
        self.group_samples = [probe]
        try:
            self.retriever_reward(state)
            self.selector_reward(state)
        finally:
            self.group_samples = saved_samples
            self._sample_topk = saved_topk
        probe.reward = self.reward_weights.aggregate(
            retrieval=probe.retrieval_reward,
            selection=probe.selection_reward,
            boundary=probe.boundary_reward,
        )
        return probe

    def llm_description_refine(self, state: ToolOptimizerState):
        """拿本轮最优描述的失败案例做一次定向重写，复评后只在变好时接受。

        这一步补上了 judge 缺的环节：judge 在评估之前跑，看不到实际失败；
        这里在奖励算完之后跑，输入是具体失败 query + judge 的文字意见。
        """
        self._load_group(state)
        best_record = state.get("best_record", None)
        passed = [s for s in self.group_samples if not s.rejected]
        if not passed:
            print("[refine] 本轮无可用候选，跳过精修")
            return {}

        base = max(passed, key=lambda s: s.reward)
        cases = base.detail.get("cases", {})
        if not LLMDescriptionRefiner.has_failure(cases):
            print("[refine] 无失败案例，无需精修")
            return {}

        prompt = LLMDescriptionRefiner._gen_prompt(
            state=state,
            prompt=self.prompts["description_refiner"],
            description=base.description,
            failure_cases=cases,
            judge_info=base.detail.get("judge", {}),
        )
        if not prompt:
            print("[refine] prompt 构造失败，跳过精修")
            return {}

        structured: Dict[str, Any] = {}
        flag = False
        for _ in range(3):
            response, stype, ok = self._generate_text(node_name="refiner", prompt=prompt)
            structured, flag, error_type = LLMDescriptionRefiner._vertify_result(response)
            if flag:
                break
            print(f"[refine] 校验失败: {error_type}")
        if not flag:
            print("[refine] 精修输出始终不合规，保留原描述")
            return {}

        refined_description = LLMDescriptionRefiner.build_description(structured)

        # 精修同样要过语义保真裁判，否则"为了让失败 query 命中而编造能力"会从这里漏进来
        judge_prompt_and_desc = self._gen_judge_prompt(state, refined_description)
        judge_score: Optional[int] = None
        judge_skipped = judge_prompt_and_desc is None
        if judge_skipped:
            # 原描述过短时 LLMDescriptionJudge 无法构造对比 prompt，此时精修版拿不到
            # 语义校验。refine 的优化压力恰恰指向"让失败 query 命中"，是最容易诱发
            # 编造能力的环节，所以这里必须显式告警，不能静默放行。
            print("[refine] 警告：原描述过短，精修版跳过语义保真评审，本次结果未经漂移校验")
        else:
            judge_prompt, _ = judge_prompt_and_desc
            for _ in range(3):
                raw, stype, ok = self._generate_text(node_name="judge", prompt=judge_prompt)
                try:
                    judged, jflag, jerror = LLMDescriptionJudge._vertify_result(raw)
                except (KeyError, TypeError, ValueError) as exc:
                    judged, jflag, jerror = {}, False, "%s: %s" % (type(exc).__name__, exc)
                if jflag:
                    judge_score = int(judged["relevance_score"])
                    break
                print(f"[refine] 精修版评审校验失败: {jerror}")
            if judge_score is not None and judge_score < self.min_judge_score:
                print(f"[refine] 精修版语义漂移(score={judge_score})，丢弃")
                self._write_description(base.description)
                return self._refine_record(state, base, None, structured, judge_score,
                                           "rejected_by_judge", judge_skipped)

        probe = self._score_description(state, refined_description)
        improved = probe.reward > base.reward
        print(
            "[refine] 精修 reward=%.4f vs 原 %.4f -> %s"
            % (probe.reward, base.reward, "接受" if improved else "回退")
        )

        if not improved:
            self._write_description(base.description)
            return self._refine_record(state, base, probe, structured, judge_score,
                                       "not_improved", judge_skipped)

        self._write_description(refined_description)
        updates = self._refine_record(state, base, probe, structured, judge_score,
                                      "accepted", judge_skipped)
        updates["current_description"] = refined_description
        if best_record is None or probe.reward > best_record.precision3:
            updates["best_description"] = refined_description
        return updates

    def _refine_record(
        self,
        state: ToolOptimizerState,
        base: PolicySample,
        probe: Optional[PolicySample],
        structured: Mapping[str, Any],
        judge_score: Optional[int] = None,
        decision: str = "",
        judge_skipped: bool = False,
    ) -> Dict[str, Any]:
        """把精修过程记进 optimizer_history（它本身就是描述优化环节）。"""
        info = {
            "decision": decision,
            "诊断": structured.get("诊断", ""),
            "base_reward": base.reward,
            "base_cases": base.detail.get("cases", {}),
            "refined_judge_score": judge_score,
            "judge_skipped": judge_skipped,
        }
        if probe is not None:
            info.update(
                {
                    "refined_reward": probe.reward,
                    "refined_retrieval": probe.retrieval_reward,
                    "refined_selection": probe.selection_reward,
                    "refined_boundary": probe.boundary_reward,
                    "refined_cases": probe.detail.get("cases", {}),
                }
            )
        record = InfoRecord(
            version_id=state.get("current_version_id", "-1"),
            stage="description_refine",
            info=info,
        )
        return {"optimizer_history": append_optimizer_history(state, record)}

    # ---------------- 库级共同进化 ----------------
    def library_evolve(self, state: ToolOptimizerState):
        """epoch 边界：把本轮最优描述固化进库、重建索引并复评一次。

        注意：这里做的是「本 resource_id 的描述已进库 → 全库重建索引 → 再评估」。
        真正的"全库重生成"由 __main__ 外层遍历所有 resource_id 完成，
        每个 resource_id 的产出都写回同一份 tools_dict，从而形成库级共同进化。
        """
        best_description = state.get("best_description", "") or state.get("current_description", "")
        if best_description:
            self._write_description(best_description)

        evaluator = self.rebuild_index()
        max_k = max(self.k_list)
        summary_df, detail_df = evaluator.evaluate(k_list=self.k_list, eval_views=[self.eval_view])
        row = summary_df[(summary_df["view"] == self.eval_view) & (summary_df["k"] == max_k)]
        subset = detail_df[(detail_df["view"] == self.eval_view) & (detail_df["k"] == max_k)]

        info = {
            "iteration": state.get("iteration", 0),
            "mrr_at_%d" % max_k: float(row.iloc[0]["mrr_at_k"]) if len(row) else 0.0,
            "pass_at_%d" % max_k: self._positive_pass_at_k(subset),
        }
        print(f"[evolve] epoch 复评: {info}")
        evolve_record = InfoRecord(
            version_id=state.get("current_version_id", "-1"),
            stage="library_evolve",
            info=info,
        )
        # epoch 复评本质上也是检索侧指标，与 retriever_reward 同源，按 stage 区分。
        # 同时把 current_description 对齐到写进库的那条描述（精英保留）：
        # 否则本轮变差时，库里是全局最优、而下一轮 policy 却从更差的 current 出发，两者会脱节。
        return {
            "retriever_history": append_retriever_history(state, evolve_record),
            "current_description": best_description or state.get("current_description", ""),
            # epoch 收尾：清空 group 快照，下一轮 policy_generate 从空开始
            "group_samples": [],
            "sample_topk": {},
        }


# ---------------- 路由 ----------------
def should_continue(state: ToolOptimizerState) -> Literal["policy_generate", "__end__"]:
    """epoch 级循环：未到 max_iterations 就回到策略采样，形成 self-evolution。"""
    if state.get("iteration", 0) >= state.get("max_iterations", 1):
        return END
    return "policy_generate"


def build_optimizer_graph(dag: ToolOptimizerGraph):
    """按流程图拼装：策略采样 → 语义保真裁判 → 检索奖励 → 裁判奖励 → 奖励聚合
    → 失败案例定向精修 → 库级进化 → (循环)。"""
    graph = StateGraph(ToolOptimizerState)
    # 每个节点都包一层检查点，跑完落盘、异常也先落盘
    graph.add_node("policy_generate", dag._with_checkpoint("policy_generate", dag.policy_generate))
    graph.add_node("description_judge", dag._with_checkpoint("description_judge", dag.llm_description_judge))
    graph.add_node("retriever_reward", dag._with_checkpoint("retriever_reward", dag.retriever_reward))
    graph.add_node("selector_reward", dag._with_checkpoint("selector_reward", dag.selector_reward))
    graph.add_node("aggregate_reward", dag._with_checkpoint("aggregate_reward", dag.aggregate_reward))
    graph.add_node("description_refine", dag._with_checkpoint("description_refine", dag.llm_description_refine))
    graph.add_node("library_evolve", dag._with_checkpoint("library_evolve", dag.library_evolve))

    graph.set_entry_point("policy_generate")
    graph.add_edge("policy_generate", "description_judge")
    graph.add_edge("description_judge", "retriever_reward")
    graph.add_edge("retriever_reward", "selector_reward")
    graph.add_edge("selector_reward", "aggregate_reward")
    graph.add_edge("aggregate_reward", "description_refine")
    graph.add_edge("description_refine", "library_evolve")
    graph.add_conditional_edges(
        source="library_evolve",
        path=should_continue,
        path_map={"policy_generate": "policy_generate", END: END},
    )
    # checkpointer 落盘后，中断可从具体节点续跑；node 的 group 快照已进 state
    return graph.compile(checkpointer=dag.checkpointer)

# ---------------- init ----------------
def list_file(folder_path: str) -> List[str]:
    """列出已完成的 pkl，用于断点续跑。"""
    if not os.path.isdir(folder_path):
        return []
    all_items = os.listdir(folder_path)
    return [
        os.path.join(folder_path, item)
        for item in all_items
        if os.path.isfile(os.path.join(folder_path, item)) and item.endswith(".pkl")
    ]


if __name__ == "__main__":
    output_root = Const.PATH.value + Const.OUTPUT_SUBDIR.value
    os.makedirs(output_root, exist_ok=True)

    MAX_ITERATIONS = 2
    # 持久化 checkpointer：全部 resource_id 共用一个 db，按 thread_id 隔离
    CKPT_DB = os.path.join(output_root, "checkpoints", "langgraph.sqlite")
    finished_ids = [path.split("/")[-1].replace(".pkl", "") for path in list_file(output_root)]
    query_good_all_dict = load_tools_from_json(Const.PATH.value + "summary/query.json")

    # 外层遍历全库 resource_id：每个工具的新描述都写回同一份 tools_dict，
    # 一轮遍历完成即为一次「全库重生成 → 重建索引 → 再评估」
    for resource_id in query_good_all_dict.keys():
        if resource_id in finished_ids:
            print("skip !!!!", resource_id)
            continue
        print("开始执行：", resource_id)
        try:
            dag = ToolOptimizerGraph(
                resource_id=resource_id,
                prompt_path="../config/prompts.json",
                flow_config_path="../config/agent_config.yaml",
                tools_description_path=Const.PATH.value + "summary/tool_descriptions.json",
                test_data_path=Const.PATH.value + "summary/query.json",
                group_size=4,
                k_list=(1, 3),
                checkpointer=build_checkpointer(CKPT_DB),
                checkpoint_root=output_root,
            )
            compiled_graph = build_optimizer_graph(dag)
            run_config = {"configurable": {"thread_id": str(resource_id)}}

            initial_state = {
                "query": list(dag.query_good_dict.keys()),
                "resource_id": resource_id,
                "title": dag.tools_dict[resource_id]["title"],
                "original_description": dag.tools_dict[resource_id]["description"],
                "current_version_id": 0,
                "current_description": dag.tools_dict[resource_id]["description"],
                "next_version_id": 1,
                "inner_loop_cnt": 0,
                "max_iterations": MAX_ITERATIONS,
                "iteration": 0,
                "version_history": [],
                "judge_history": [],
                "verify_judge_history": [],
                "retriever_history": [],
                "selector_history": [],
                "optimizer_history": [],
                "group_samples": [],
                "sample_topk": {},
            }

            # ---- 恢复策略：优先节点级，退化到 epoch 级 ----
            waiting = pending_nodes(compiled_graph, run_config)
            graph_input: Optional[Dict[str, Any]] = initial_state
            if waiting:
                # LangGraph 里还有待执行节点：传 None 表示"从中断的那个节点继续"
                print(f"[ckpt] 节点级续跑，待执行节点={waiting}")
                graph_input = None
                snapshot_values = compiled_graph.get_state(run_config).values or {}
                # 描述库要对齐到中断时的状态，否则节点会拿旧描述继续算
                dag._write_description(snapshot_values.get("current_description", "") or "")
            else:
                resumed, reason = dag.checkpoint.resume_state(MAX_ITERATIONS)
                if resumed is not None:
                    print(f"[ckpt] 无节点级检查点，退化为 epoch 级续跑：{reason}")
                    graph_input = resumed
                    dag._write_description(resumed.get("current_description", "") or "")
                elif reason == "already_finished":
                    # 上次跑满轮次但没写成 .pkl（崩在 invoke 与 save_pickle 之间）：
                    # 直接用检查点补写产物，不要白跑一遍
                    payload = dag.checkpoint.load() or {}
                    recovered = payload.get("state", None)
                    if recovered:
                        print("[ckpt] 轮次已跑满、仅缺产物，用检查点补写 .pkl 并跳过")
                        save_pickle(recovered, os.path.join(output_root, resource_id + ".pkl"))
                        dag.checkpoint.clear()
                        continue
                    print(f"[ckpt] 不恢复（{reason}），从头开始")
                else:
                    print(f"[ckpt] 不恢复（{reason}），从头开始")

            state = compiled_graph.invoke(graph_input, config=run_config)
            print("-----------------start save state ... !!!")
            save_pickle(state, os.path.join(output_root, resource_id + ".pkl"))
            # 正常跑完就清掉检查点，避免下次被误判为未完成
            dag.checkpoint.clear()
            print("[guard] 本 resource_id 调用统计:", dag.call_guard.stats.to_dict())
            if dag.call_guard.circuit_open:
                print("[guard] 熔断处于打开状态，endpoint 可能已不可用，终止后续 resource_id")
                break
        except Exception as exc:
            # 检查点已在节点包装里落盘，这里只记录，不清理，便于下次续跑
            print("resource_id error:", resource_id, exc)
            continue
        break
