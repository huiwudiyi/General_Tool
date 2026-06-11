# MultiView Vector Recall Evaluator


```commandline
**广告一下**

八字藏天机，命理见人生。解析五行喜忌、性格天赋、事业财运、
感情婚姻与人生走势，帮你看清自身优势，把握关键机遇，趋吉避凶，规划更顺的人生方向
```
<p align="center">
  <img src="../Fortune_telling.png" width="388">
</p>


一个用于 **多视角向量召回评估** 的 Python 工具类，支持基于 `title`、`description`、`title + description` 三种文本视角构建向量集合，并对每条数据中的 `call_querys_list` 进行向量召回、阈值过滤和召回效果统计。

该工具适合用于评估：

- Query 与标题的语义匹配效果
- Query 与描述的语义匹配效果
- Query 与标题 + 描述组合文本的语义匹配效果
- 多路召回结果合并后的整体效果
- 不同相似度阈值下的召回率、Top1 准确率、MRR 等指标

---

## 1. 功能特性

当前代码主要实现了以下能力：

1. **多视角向量建库**
   - `title`
   - `description`
   - `title_description`

2. **批量向量编码**
   - 支持按 `batch_size` 分批调用 embedding 模型
   - 自动将向量进行 L2 归一化，方便使用点积计算余弦相似度

3. **单路召回**
   - 分别基于 `title`、`description`、`title_description` 进行 TopK 召回

4. **多路合并召回**
   - 合并三路召回结果
   - 同一个 `id` 多次出现时，保留最高相似度分数

5. **相似度阈值过滤**
   - 支持自定义多个相似度阈值
   - 对召回结果进行过滤

6. **召回效果评估**
   - Recall Rate
   - Top1 Accuracy
   - MRR
   - 平均候选数量
   - 平均 Precision
   - 命中样本的平均排名

7. **评估结果保存**
   - 支持保存为 CSV
   - 支持保存为 JSONL
   - 支持保存为 Excel

---

## 2. 环境依赖

建议使用 Python 3.8+。

安装依赖：

```bash
pip install numpy pandas sentence-transformers openpyxl
```

如果使用的是本地 embedding 模型，需要确保模型路径存在，例如：

```python
SentenceTransformer("/root/paddlejob/workspace/env_run/output/zacharychu/afs_data/model/Qwen3-Embedding-0.6B")
```

---

## 3. 数据格式

输入数据是一个 `List[Dict]`，每条数据包含：

| 字段 | 类型 | 是否必填 | 说明 |
|---|---|---|---|
| `id` | `str/int` | 是 | 数据唯一标识 |
| `title` | `str` | 否 | 标题文本 |
| `description` | `str` | 否 | 描述文本 |
| `call_querys_list` | `List[str]` | 否 | 该条数据对应的评估 query 列表 |

示例：

```python
data = [
    {
        "id": "1",
        "title": "苹果手机 iPhone 15",
        "description": "苹果公司发布的智能手机，搭载 A 系列芯片，支持高清拍照。",
        "call_querys_list": [
            "苹果手机",
            "iPhone 15 参数",
            "高清拍照手机"
        ]
    },
    {
        "id": "2",
        "title": "华为 Mate 系列手机",
        "description": "华为旗舰手机，支持影像增强、快充和鸿蒙系统。",
        "call_querys_list": [
            "华为旗舰手机",
            "鸿蒙系统手机"
        ]
    }
]
```

---

## 4. 快速开始

### 4.1 初始化模型

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "/root/paddlejob/workspace/env_run/output/zacharychu/afs_data/model/Qwen3-Embedding-0.6B"
)

def encode_fn(texts):
    return model.encode(texts, batch_size=64, show_progress_bar=False)
```

---

### 4.2 初始化评估器

```python
evaluator = MultiViewVectorRecallEvaluator(
    data=data,
    encode_fn=encode_fn,
    batch_size=64,
)
```

---

### 4.3 构建向量索引

```python
evaluator.build_vector_index()
```

该步骤会分别构建：

- `title` 向量集合
- `description` 向量集合
- `title_description` 向量集合

同时会从每条数据的 `call_querys_list` 中生成评估 query。

---

### 4.4 执行召回评估

```python
thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]

summary_results, detail_results = evaluator.evaluate(
    thresholds=thresholds,
    top_k=50,
)
```

默认会评估以下召回方式：

```python
[
    "title",
    "description",
    "title_description",
    "merged",
]
```

也可以只评估部分视角：

```python
summary_results, detail_results = evaluator.evaluate(
    thresholds=[0.4, 0.5, 0.6],
    top_k=20,
    eval_views=["title", "title_description", "merged"],
)
```

---

### 4.5 转成 DataFrame

```python
import pandas as pd

summary_df = pd.DataFrame(summary_results)
detail_df = pd.DataFrame(detail_results)
```

查看汇总结果：

```python
print(summary_df)
```

查看明细结果：

```python
print(detail_df.head())
```

---

### 4.6 保存评估结果

```python
saved_files = evaluator.save_recall_results(
    summary_df=summary_df,
    detail_df=detail_df,
    output_dir="./recall_outputs",
    prefix="recall_eval",
    save_csv=True,
    save_jsonl=True,
    save_excel=True,
)
```

默认会输出：

```text
./recall_outputs/recall_eval_summary.csv
./recall_outputs/recall_eval_detail.csv
./recall_outputs/recall_eval_summary.jsonl
./recall_outputs/recall_eval_detail.jsonl
./recall_outputs/recall_eval.xlsx
```

---

## 5. 完整使用示例

```python
import pandas as pd
from sentence_transformers import SentenceTransformer

# 1. 准备数据
data = [
    {
        "id": "1",
        "title": "苹果手机 iPhone 15",
        "description": "苹果公司发布的智能手机，搭载 A 系列芯片，支持高清拍照。",
        "call_querys_list": [
            "苹果手机",
            "iPhone 15 参数",
            "高清拍照手机"
        ]
    },
    {
        "id": "2",
        "title": "华为 Mate 系列手机",
        "description": "华为旗舰手机，支持影像增强、快充和鸿蒙系统。",
        "call_querys_list": [
            "华为旗舰手机",
            "鸿蒙系统手机"
        ]
    }
]

# 2. 加载向量模型
model = SentenceTransformer(
    "/root/paddlejob/workspace/env_run/output/zacharychu/afs_data/model/Qwen3-Embedding-0.6B"
)

def encode_fn(texts):
    return model.encode(texts, batch_size=64, show_progress_bar=False)

# 3. 初始化评估器
evaluator = MultiViewVectorRecallEvaluator(
    data=data,
    encode_fn=encode_fn,
    batch_size=64,
)

# 4. 构建向量集合
evaluator.build_vector_index()

# 5. 评估召回效果
summary_results, detail_results = evaluator.evaluate(
    thresholds=[0.3, 0.4, 0.5, 0.6, 0.7],
    top_k=50,
)

# 6. 转成 DataFrame
summary_df = pd.DataFrame(summary_results)
detail_df = pd.DataFrame(detail_results)

# 7. 保存结果
evaluator.save_recall_results(
    summary_df=summary_df,
    detail_df=detail_df,
    output_dir="./recall_outputs",
    prefix="recall_eval",
)
```

---

## 6. 输出结果说明

### 6.1 summary 汇总结果

`summary_results` 用于展示不同召回视角、不同阈值下的整体指标。

字段说明：

| 字段 | 说明 |
|---|---|
| `view` | 召回视角：`title`、`description`、`title_description`、`merged` |
| `threshold` | 相似度过滤阈值 |
| `query_count` | 评估 query 总数 |
| `hit_count` | 命中的 query 数量 |
| `recall_rate` | 召回率，命中数量 / query 总数 |
| `top1_acc` | Top1 命中率 |
| `mrr` | Mean Reciprocal Rank |
| `avg_candidate_count` | 平均过滤后候选数量 |
| `avg_precision` | 平均 Precision |
| `avg_hit_rank` | 命中样本的平均排名 |

---

### 6.2 detail 明细结果

`detail_results` 用于记录每个 query 在每种召回视角、每个阈值下的召回详情。

字段说明：

| 字段 | 说明 |
|---|---|
| `query` | 当前评估 query |
| `target_id` | query 对应的正确目标 id |
| `view` | 当前召回视角 |
| `threshold` | 当前相似度阈值 |
| `hit` | 是否命中目标 id |
| `top1_hit` | Top1 是否命中 |
| `rank` | 目标 id 在过滤结果中的排名 |
| `candidate_count` | 过滤后的候选数量 |
| `precision` | 单 query 精度 |
| `recall_ids` | 过滤后的召回 id 列表 |
| `recall_scores` | 过滤后的召回分数列表 |
| `recall_views` | 每个召回结果来源的视角 |

---

## 7. 指标解释

### 7.1 Recall Rate

表示目标 `id` 是否出现在过滤后的召回结果中。

```text
recall_rate = hit_count / query_count
```

适合观察不同阈值下的整体召回能力。

---

### 7.2 Top1 Accuracy

表示目标 `id` 是否排在召回结果第一位。

```text
top1_acc = top1_hit_count / query_count
```

适合衡量召回排序的首位准确性。

---

### 7.3 MRR

MRR 即 Mean Reciprocal Rank，用于衡量目标结果的排序位置。

```text
MRR = mean(1 / rank)
```

目标结果越靠前，MRR 越高。

---

### 7.4 Avg Candidate Count

表示每个 query 在阈值过滤后平均保留多少个候选结果。

该指标可以用于观察阈值对候选规模的影响。

---

### 7.5 Avg Precision

当前代码中，每个 query 默认只有一个正确目标 id。

因此，单 query 的 precision 计算逻辑是：

```text
如果命中：
    precision = 1 / candidate_count
否则：
    precision = 0
```

该指标更适合观察过滤后候选集合的纯度。

---

## 8. 核心类与方法说明

### 8.1 `l2_normalize`

```python
def l2_normalize(vectors: np.ndarray, eps: float = 1e-12) -> np.ndarray:
```

对向量进行 L2 归一化。

归一化后，可以使用点积近似计算余弦相似度。

---

### 8.2 `MultiViewVectorRecallEvaluator`

多视角向量召回评估器。

核心职责：

- 构建多视角向量集合
- 执行向量召回
- 合并多路召回
- 阈值过滤
- 指标统计
- 保存评估结果

---

### 8.3 `build_vector_index`

```python
def build_vector_index(self):
```

构建三路向量集合：

- `title`
- `description`
- `title_description`

并从 `call_querys_list` 中生成评估样本。

---

### 8.4 `recall_one_view`

```python
def recall_one_view(
    self,
    query_embedding: np.ndarray,
    view_name: str,
    top_k: int = 50,
) -> List[Dict[str, Any]]:
```

基于某一路文本视角进行 TopK 召回。

---

### 8.5 `recall_merged`

```python
def recall_merged(
    self,
    query_embedding: np.ndarray,
    top_k: int = 50,
) -> List[Dict[str, Any]]:
```

合并多路召回结果。

合并逻辑：

- 分别从 `title`、`description`、`title_description` 召回
- 如果同一个 `id` 在多路结果中出现，只保留最高分
- 最终按照分数从高到低排序

---

### 8.6 `filter_by_threshold`

```python
def filter_by_threshold(
    results: List[Dict[str, Any]],
    threshold: float,
) -> List[Dict[str, Any]]:
```

根据相似度阈值过滤召回结果。

---

### 8.7 `calc_one_query_metrics`

```python
def calc_one_query_metrics(
    filtered_results: List[Dict[str, Any]],
    target_id: str,
) -> Dict[str, Any]:
```

计算单个 query 的指标，包括：

- 是否命中
- 是否 Top1 命中
- 命中排名
- MRR 分数
- 候选数量
- Precision

---

### 8.8 `evaluate`

```python
def evaluate(
    self,
    thresholds: List[float],
    top_k: int = 50,
    eval_views: List[str] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
```

执行整体评估。

返回：

```python
summary_results, detail_results
```

其中：

- `summary_results`：整体汇总指标
- `detail_results`：每个 query 的召回明细

---

### 8.9 `save_recall_results`

```python
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
```

保存评估结果。

支持格式：

- CSV
- JSONL
- Excel

---

## 9. 推荐目录结构

```text
project/
├── README.md
├── evaluator.py
├── run_eval.py
├── data/
│   └── recall_data.json
└── recall_outputs/
    ├── recall_eval_summary.csv
    ├── recall_eval_detail.csv
    ├── recall_eval_summary.jsonl
    ├── recall_eval_detail.jsonl
    └── recall_eval.xlsx
```

---

## 10. 注意事项

1. 在调用 `evaluate()` 之前，必须先调用：

```python
evaluator.build_vector_index()
```

否则会抛出异常：

```text
Please call build_vector_index() first.
```

2. 当前代码使用 `np.dot(doc_embeddings, query_embedding)` 计算相似度，因此需要确保文档向量和 query 向量都已经归一化。

3. 当前评估默认是 **单正样本评估**，即每个 query 只有一个 `target_id`。

4. 如果同一个 query 可能对应多个正确 id，需要修改 `target_id` 为 `target_ids`，并相应调整命中与 precision 的计算方式。

5. 当前实现使用 NumPy 暴力检索，适合中小规模数据评估。如果数据量较大，建议接入 FAISS、HNSW 或 Milvus 等向量检索引擎。

---

## 11. 后续可优化方向

### 11.1 接入 FAISS

当前召回逻辑是：

```python
scores = np.dot(doc_embeddings, query_embedding)
top_indices = np.argsort(-scores)[:top_k]
```

当数据量较大时，建议替换为 FAISS IndexFlatIP 或 HNSW。

---

### 11.2 支持多正样本

当前每个 query 只有一个目标 `target_id`。

可以扩展为：

```python
{
    "query": "苹果手机",
    "target_ids": ["1", "3", "5"]
}
```

然后计算：

- Recall@K
- Precision@K
- MAP
- NDCG

---

### 11.3 增加错误分析

可以基于 `detail_df` 增加错误分析字段：

- 未命中 query
- 低分命中 query
- Top1 误召 query
- 命中但排名靠后的 query
- 不同 view 的召回差异

---

### 11.4 支持更多文本视角

除了当前三种视角，还可以扩展：

- `title + tag`
- `description + tag`
- `title + description + tag`
- `category`
- `brand`
- `metadata`

---

### 11.5 支持线上 Case 回放

可以将线上 query 和人工标注的目标 id 组织成评估集，周期性运行该评估器，用于观察 embedding 模型迭代收益。

---

## 12. 适用场景

该工具适用于以下场景：

- 向量召回模型评估
- Query-Doc 匹配效果分析
- 多字段向量建库效果对比
- 不同相似度阈值调参
- 召回结果离线评估
- embedding 模型升级前后对比
- 检索系统一阶段召回实验

---

## 13. License

可根据项目实际情况补充 License，例如：

```text
MIT License
```
