"""
LangGraph implementation for DeepAgent style Chinese writing agent.

Features:
1. YAML-driven routing after intent node.
2. YAML-driven flow edges, so different intents can trigger different execution paths.
3. LangGraph checkpointer memory with thread_id.
4. Optimized DeepAgentState: separates persistent memory from intermediate scratch state.
5. OpenAI-compatible LLM client for vLLM / SGLang / LMDeploy / OpenAI gateways.

Run:
    pip install openai langgraph langchain-core json-repair pyyaml

    export OPENAI_API_BASE="http://127.0.0.1:8000/v1"
    export OPENAI_API_KEY="EMPTY"
    export OPENAI_MODEL="qwen"

    python deepagent_langgraph_yaml_memory.py "帮我写一份项目进展文档" --thread-id user_001

Optional:
    export DEEPAGENT_NODE_MODELS='{"intent":"qwen3-8b","planner":"qwen3-32b","draft":"qwen3-32b"}'
    export DEEPAGENT_NODE_TEMPERATURES='{"intent":0.0,"planner":0.2,"draft":0.6}'
"""
import yaml
from json_repair import repair_json

from utils import *
from state import *
from openai import OpenAI
from llm_client import LLMClient
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Literal

try:
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import END, START, StateGraph
    from langgraph.graph.message import add_messages
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "Missing langgraph dependencies. Please run: "
        "pip install langgraph langchain-core"
    ) from exc


# 相关配置
from flow_config import FlowConfig
from prompt_registry import PromptRegistry

# 相关执行class
from nn_recall_passk import recall_passk_function


from llm_summary_optimizer import LLMDescriptionSummary
from aiapi import request_aiapi, parse_aiapi
from llm_generator import LLMDescriptionGenerator

class ToolOptimizerGraph:
    def __init__(
        self,
        resource_id: str,
        llm_client: Optional[LLMClient] = None,
        prompt_path: str = "../config/prompts.json",
        flow_config_path: str = "../config/agent_config.yaml",
        tools_description_path: str = "../data/summary/tool_descriptions.json",
        test_data_path: str = "../data/summary/query.json",
        checkpointer: Optional[Any] = None,
    ) -> None:
        # 设置通用的 参数
        self.prompts = PromptRegistry(prompt_path).get_promts()
        self.flow_config = FlowConfig(flow_config_path)
        self.checkpointer = checkpointer or InMemorySaver()
        self.last_tools_description_path = tools_description_path
        # 设置 tool resource_id
        self.resource_id = resource_id

        # 加载tools 这个列表
        self.tools_dict = load_tools_from_json(self.last_tools_description_path)

        # 加载测试集
        # test_data_path: str = "../data/summary/query.json"
        # load_tools_from_json(test_data_path)#
        self.query_good_all_dict = load_tools_from_json(test_data_path)
        self.query_good_dict = self.query_good_all_dict[resource_id]

        # node -> model
        self.node_model_map: Dict[str, str] = {} 
        if self.flow_config.node_model_map:
            self.node_model_map.update({str(k): str(v) for k, v in self.flow_config.node_model_map.items()})

        # node -> temperature
        self.node_temperature_map: Dict[str, float] =  {} 
        if self.flow_config.node_temperature_map:
            self.node_temperature_map.update({str(k): float(v) for k, v in self.flow_config.node_temperature_map.items()})
 
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
        response =  client.generate_text(
            prompt = prompt,
            model=self._model_for(node_name),
            temperature=self._temperature_for(node_name),
            extra_body = extra_body
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
        statistic_check_version = "version_" + str(state.get("current_version_id", 0))
        print("当前执行的版本：", statistic_check_version)
        statistic_output = "../recall_outputs/summary/" + statistic_check_version + "/" + self.resource_id
    
        detaildf = recall_passk_function(self.tools_dict, self.query_good_dict, self.resource_id, statistic_output)

        # 计算 top 1 的 precision
        detaildf_top1 = detaildf[(detaildf['k']==1) & (detaildf['view']=="merged")]
        relevants = detaildf_top1['gold_ids'].to_list()
        retrieveds = detaildf_top1['recall_ids'].to_list()
        precision1 = precision_at_k_batch(retrieveds, relevants, 1)
        recall1 = recall_at_k_batch(retrieveds, relevants, 1)

        # 计算 top 3 的 precision
        detaildf_top3 = detaildf[(detaildf['k']==3) & (detaildf['view']=="merged")]
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
                tools_query[recall_ids[0]] =  []
            tools_query[recall_ids[0]].append(row["query"])
            tools_case_ids.append(recall_ids[0])
        top_tools_case = {k: tools_query[k] for k in get_k_tool(tools_case_ids, 3)}
        tools_case_all = {k: tools_query[k] for k in set(tools_case_ids)}

        version_id, next_version_id = next_version_pair(state)

        versionInfo = VersionRecord(
            version_id = statistic_check_version, 
            parent_version_id = version_id,
            stage = "statistic",
            description = self.tools_dict[self.resource_id]["description"],
            case_result = tools_case_all,
            top_case = top_tools_case,
            tool_path = statistic_output + "/tool_prompt.json",
            recall1 = recall1,
            precision1 = precision1,
            recall3 = recall3,
            precision3 = precision3
        )

        # 判断是否需要更新最佳记录
        # 条件1: best_record 不存在 (即为 None)
        # 条件2: 新的指标 (recall3, precision3) 优于旧的 best_record
        should_update = (best_record is None) or \
                (recall3 > best_record.recall3 and precision3 > best_record.precision3)

        with open(statistic_output + "/tool_prompt.json","w") as w:
            json.dump(self.tools_dict, w, ensure_ascii=False, indent=2)

        if should_update:
            return {
                    "current_version_id": version_id,
                    "next_version_id": next_version_id,
                    "version_history": append_version_history(state, versionInfo),
                    "best_description": state.get("current_description", ""),
                    "best_version_id": state.get("current_version_id", 0),
                    "best_record": versionInfo,

                }
        else:
            return {
                "current_version_id": version_id,
                "next_version_id": next_version_id,
                "version_history": append_version_history(state, versionInfo)
            }

    # --------- LLMGenerator node ---------------
    def llm_description_generator(self, state: ToolOptimizerState):
        querys = state.get("query", [])
        resource_id = state.get("resource_id", "-1")
        if resource_id == "-1":
            return {}
        query_labelData_dict = {}
        for query in querys:
            response = request_aiapi(query)
            aladdin_aiapi_dict, natural_result = parse_aiapi(response)
            label_data = aladdin_aiapi_dict.get(resource_id, "")
            if len(label_data) != 0:
                query_labelData_dict[query] = label_data

        prompt = self.prompts["generator_single"]
        prompt = LLMDescriptionGenerator._gen_prompt(state = state, prompt = prompt, query_labelData_dict = query_labelData_dict)
        for _ in range(3):
            response, stype, flag = self._generate_text(node_name = "summary", prompt = prompt)
            response, flag, error_type = LLMDescriptionGenerator._vertify_result(response)
            if flag:
                break
        if flag:
            optimizer_record = InfoRecord(
                version_id = state.get("current_version_id", "-1"),
                stage = "generator",
                info = response
            )
            description = response["summary"] + response["display"] + response["interaction"]
            return {
                "original_description": description,
                "optimizer_history": append_optimizer_history(state, optimizer_record)
            }


    # ---------  LLMNodeSummary node--------------
    def llm_description_summary(self, 
                                  state: ToolOptimizerState):
        prompt = LLMDescriptionSummary._gen_prompt(state, self.prompts['summary'], list(self.query_good_dict.keys()))

        for _ in range(3):
            response, stype, flag = self._generate_text(node_name = "summary", prompt = prompt)
            response, flag, error_type = LLMDescriptionSummary._vertify_result(response)
            if flag:
                break
        if flag:
            optimizer_description = response["optimizer_description"]
            
            
            print("description: ", state.get("current_version_id", "-1"), self.tools_dict[self.resource_id]['description'])
            optimizer_record = InfoRecord(
                version_id = state.get("current_version_id", "-1"),
                stage = "summary",
                info = response
            )
            return {

                "optimizer_history": append_optimizer_history(state, optimizer_record)
            }


    # ---------- graph build ----------


def should_continue(state: ToolOptimizerState) -> Literal["summary_optimizer", END]:
    # 达到最大轮次
    max_iteration = state.get("max_iterations", 3)
    if state.get("current_version_id", 10) > max_iteration:
        return END
    return "summary_optimizer"

#------- init ----
def list_file(folder_path):
    all_items = os.listdir(folder_path)
    # 只列出文件（不包括文件夹）
    files_only = [os.path.join(folder_path, item) for item in all_items if os.path.isfile(os.path.join(folder_path, item)) and item.endswith(".pkl")]
    print("\n只列出文件:")
    return files_only



resource_ids_all = []
for path in list_file("../recall_outputs/summary/"):
    resource_ids_all.append(path.split("/")[-1].replace(".pkl", ""))

query_good_all_dict = load_tools_from_json("../data/summary/query.json")
for resource_id, value in query_good_all_dict.items():
    if resource_id in resource_ids_all or resource_id in ["47242"]:
        print(resource_id, "skip!!!")
        continue
    print("开始执行：", resource_id)
    try:
        dag = ToolOptimizerGraph(resource_id = resource_id,
            prompt_path = "../config/prompts.json",
            flow_config_path = "../config/agent_config.yaml",
            tools_description_path = "../data/summary/tool_description.json",
            test_data_path  = "../data/summary/query.json",
            checkpointer = None)

        graph = StateGraph(ToolOptimizerState)
        graph.add_node("statistic_check_tool", dag.statistic_check_tool)
        graph.add_node("description_generator", dag.llm_description_generator)
        graph.add_node("summary_optimizer", dag.llm_description_summary)
        graph.set_entry_point("description_generator")
        # graph.add_edge("statistic_check_tool", END)
        graph.add_edge("description_generator", "statistic_check_tool")
        graph.add_conditional_edges(
            "statistic_check_tool",
            should_continue,
            {
                "summary_optimizer": "summary_optimizer",
                END: END
            }
        )
        graph.add_edge("summary_optimizer", "statistic_check_tool")
        compiled_graph = graph.compile()
        querys = state.get("query", []) # 读取文件  获取 工具的应召query
        initial_state = {
                "resource_id": resource_id,
                "title": dag.tools_dict[resource_id]["title"],
                "query": querys,
                "original_description": dag.tools_dict[resource_id]["description"],
                # 当前工作基线指针（可手动/自动回滚）
                "current_version_id": 0,
                "current_description": dag.tools_dict[resource_id]["description"],
                # 下一个版本的id
                "next_version_id": 1,
                # 固定参数
                "max_iterations": 3,
                "iteration": 0,
                # 版本历史仓库：全量快照存储
                "version_history": [],
                #记录历史
                "optimizer_history": []
            }
        state = compiled_graph.invoke(initial_state)

        save_pickle(state, "../recall_outputs/summary/" + resource_id + ".pkl")  
    except:
        continue