import json
import os
import re
import json
import mimetypes

import yaml
import numpy as np
from typing import Any, Callable, Dict, List, Set, Mapping, Optional, Sequence, TypedDict, Union
from collections import Counter
import base64
from json_repair import repair_json

NO_CALL = "no_call" # 设置未召回兼容标识，标识没有召回任何工具

# ------- passk precisionk  recallk ---- 
def pass_at_k(n: int, c: int, k: int) -> float:
    """
    计算 Pass@K (通常用于代码生成或生成式检索评估)
    Args:
        n: 总生成样本数
        c: 正确样本数
        k: 评估的截断阈值 (Top-K)
    Returns:
        Pass@K 得分 (0.0 ~ 1.0)
    """
    if n - c < k:
        return 1.0
    # 使用组合数公式计算至少命中一个正确结果的概率
    # 1 - C(n-c, k) / C(n, k)
    # 为避免阶乘溢出，使用连乘计算
    prob_no_pass = 1.0
    for i in range(k):
        prob_no_pass *= (n - c - i) / (n - i)
    return 1.0 - prob_no_pass


def precision_at_k(retrieved: List[Union[str, int]], relevant: List[Union[str, int]], k: int) -> float:
    """
    计算 Precision@K (Top-K 结果中相关文档的比例)
    Args:
        retrieved: 模型检索/推荐出的结果列表 (按相关性降序排列)
        relevant: 真实相关文档的集合
        k: 评估的截断阈值
    Returns:
        Precision@K 得分 (0.0 ~ 1.0)
    """
    if k <= 0:
        return 0.0
    # 截取前 k 个结果
    top_k_results = retrieved[:k]
    if not top_k_results:
        return 0.0

    # 计算前 k 个结果中有多少是相关的
    relevant_count = sum(1 for item in top_k_results if item in relevant)
    # Precision@K = 命中数 / 实际取回条数（正常等于 k，若 retrieved 不足 k 则用实际长度）
    return relevant_count / len(top_k_results)

def precision_at_k_batch(retrieveds: List[List[Union[str, int]]], relevants: List[List[Union[str, int]]], k: int) -> float:
    """
    计算 Precision@K (Top-K 结果中相关文档的比例)
    Args:
        retrieved: 模型检索/推荐出的结果列表 (按相关性降序排列)
        relevant: 真实相关文档的集合
        k: 评估的截断阈值
    Returns:
        Precision@K 平均得分 (0.0 ~ 1.0)
    """
    
    if len(relevants) != len(retrieveds):
        return 0.0
    else:
        precision_scores = []
        for retrieved, relevant in zip(retrieveds, relevants):
            precision_scores.append(precision_at_k(retrieved, relevant, k))
    # print("precision_scores", precision_scores)
    return sum(precision_scores) / len(retrieveds)



def recall_at_k(retrieved: List[Union[str, int]], relevant: List[Union[str, int]], k: int) -> float:
    """
    计算 Recall@K (所有相关文档中，被检索到且排在 Top-K 的比例)
    Args:
        retrieved: 模型检索/推荐出的结果列表 (按相关性降序排列)
        relevant: 真实相关文档的集合
        k: 评估的截断阈值
    Returns:
        Recall@K 得分 (0.0 ~ 1.0)
    """
    if not relevant:
        return 0.0
    
    # 截取前 k 个结果
    top_k_results = retrieved[:k]
    
    # 计算前 k 个结果中命中了多少个真实相关文档
    relevant_found = sum(1 for item in top_k_results if item in relevant)
    return relevant_found / len(relevant)

def recall_at_k_batch(retrieveds: List[List[Union[str, int]]], relevants: List[List[Union[str, int]]], k: int) -> float:
    """
    计算 Recall@K (所有相关文档中，被检索到且排在 Top-K 的比例)
    Args:
        retrieved: 模型检索/推荐出的结果列表 (按相关性降序排列)
        relevant: 真实相关文档的集合
        k: 评估的截断阈值
    Returns:
        Recall@K 平均得分 (0.0 ~ 1.0)
    """
    if len(relevants) != len(retrieveds):
        return 0.0
    else:
        recall_scores = []
        for retrieved, relevant in zip(retrieveds, relevants):
            recall_scores.append(recall_at_k(retrieved, relevant, k))
    # print("recall_scores", recall_scores)
    return sum(recall_scores) / len(retrieveds)

# --------------- 
def get_k_tool(my_list, k, lest_num = None):
    # 1. 统计列表总个数
    total_count = len(my_list)

    # 2. 统计每个元素出现的次数
    counter = Counter(my_list)

    # 3. 查找出现最多的前 N 个数（例如前3个）
    top_n = counter.most_common(k) 
    if lest_num:
        return [k for k,v in top_n if v > lest_num]
    return [k for k,v in top_n]

#------------- 加载文件 ----------------------------------------------
def load_tools_from_json(path: str):
    """读取 tool 集合 JSON 文件。"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Load yaml config."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def dump_json(obj: Any, path: str):
    with open(path, "w") as w:
        json.dump(obj, w, ensure_ascii=False, indent=2)


# -------------字段校验 ------------------------------------------
def ensure_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}

def ensure_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]

def ensure_str_list(value: Any) -> List[str]:
    return [str(x).strip() for x in ensure_list(value) if str(x).strip()]


# -------------字段处理-----------------------------
NORMALIZE_PUNCT_RE = re.compile(r"[，。！？、,.!?;；:：\"'“”‘’（）()\[\]{}\s]+")
TOKEN_RE = re.compile(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+")
def normalize_text(text: str) -> str:
    return NORMALIZE_PUNCT_RE.sub("", text.lower())

def compact_text(text: Any, max_chars: int = 1000) -> str:
    value = str(text or "").strip()
    if len(value) <= max_chars:
        return value
    return value[:max_chars] + "..."

def normalize_json_output(payload: Any, need_repair: bool = True) -> Any:
    """Normalize LLM JSON output, compatible with DeepAgent safety.normalize_json_output."""
    if isinstance(payload, (dict, list)):
        return payload

    text = str(payload or "").strip()
    text = text.replace("```json", "").replace("```", "").strip()
    if not text:
        return {}

    if need_repair:
        return repair_json(text, return_objects=True)
    return json.loads(text)

def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

# --------------------pickle----------------------------
import pickle

def save_pickle(result, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(result, f)

def load_pickle(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)


# --------------------结果修改 ----------------------------
def replace_last_item(left: Optional[List[Any]], right: Optional[List[Any]]) -> List[Any]:
    """
    用 right 中的元素替换 left 中的最后一个元素。
    如果 left 为空，则直接将 right 作为新列表。
    """
    current_list = list(left or [])
    new_items = list(right or [])
    
    if not current_list:
        return new_items
        
    if not new_items:
        return current_list
        
    # 弹出最后一个元素，追加新元素
    current_list.pop()
    current_list.extend(new_items)
    
    return current_list
def append_list(left: Optional[List[Any]], right: Optional[List[Any]]) -> List[Any]:
    return list(left or []) + list(right or [])

def merge_dict(left: Optional[Dict[str, Any]], right: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    merged = dict(left or {})
    merged.update(dict(right or {}))
    return merged

def merge_node_warnings(
    left: Optional[Dict[str, List[str]]],
    right: Optional[Dict[str, List[str]]],
) -> Dict[str, List[str]]:
    merged = {k: list(v) for k, v in dict(left or {}).items()}
    for node_name, warnings in dict(right or {}).items():
        merged.setdefault(node_name, [])
        merged[node_name].extend(list(warnings or []))
    return merged

# -------------------------向量计算 --------------------------
def cosine_similarity(v1: np.ndarray, v2: np.ndarray):
    dot_product = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)

def l2_normalize(vectors: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2 归一化，归一化后点积等价于 cosine similarity。"""
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, eps)

# -------------------------读取图片 --------------------------
def encode_image_to_base64(image_path: str) -> str:
    """读取图片文件并编码为 base64 字符串"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def guess_mime_type(image_path: str) -> str:
    """根据文件后缀猜测图片的 MIME 类型"""
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        mime_type = "image/png"
    return mime_type