
# MultiView Vector Recall pass@k Evaluator

```commandline
**广告一下**

八字藏天机，命理见人生。解析五行喜忌、性格天赋、事业财运、
感情婚姻与人生走势，帮你看清自身优势，把握关键机遇，趋吉避凶，规划更顺的人生方向
```
<p align="center">
  <img src="../Fortune_telling.png" width="388">
</p>


## 1. 项目简介

本项目用于评估「工具集合」在向量召回场景下的召回效果。

脚本会将工具集合中的 `title`、`description`、`title + description` 分别构建为向量底库，然后使用测试 Query 生成向量，在工具集合中进行相似度召回，并计算不同召回视角下的 `pass@k` 指标。

适用场景包括：

- 工具调用 / Function Calling 工具召回评估
- MCP 工具召回评估
- Agent 工具选择前的候选工具召回
- 向量模型召回效果对比
- 不同工具文本字段的召回效果分析

---

## 2. 核心功能

当前脚本支持以下能力：

1. **多视角构建工具向量库**

   对每个工具分别构建三路向量：

   - `title`
   - `description`
   - `title_description`

2. **Query 向量召回**

   使用向量模型对测试 Query 编码，并在工具向量库中召回最相似的工具。

3. **多路融合召回**

   `merged` 模式会对同一个工具在多路向量中的最高相似度进行融合排序。

4. **Pass@K 评估**

   支持自定义多个 `k` 值，例如：

   ```python
   k_list=[1, 3, 5, 10]
   ```

5. **结果明细保存**

   支持保存：

   - summary 汇总结果
   - detail 明细结果
   - CSV
   - JSONL
   - Excel

---

## 3. 代码文件说明

核心脚本：

```text
nn_recall_passk.py.py
```

主要模块说明：

| 模块 / 函数 | 作用 |
|---|---|
| `l2_normalize` | 对向量做 L2 归一化，使点积等价于 cosine similarity |
| `encode_fn` | 调用 SentenceTransformer 对文本编码 |
| `ToolPassAtKRecallEvaluator` | 工具召回评估主类 |
| `build_vector_index` | 构建工具集合三路向量底库 |
| `recall_one_view` | 单路向量召回 |
| `recall_merged` | 多路融合召回 |
| `calc_pass_at_k` | 计算单条 Query 的 pass@k |
| `evaluate` | 批量评估所有 Query |
| `save_recall_results` | 保存评估结果 |
| `load_tools_from_json` | 从 JSON 文件加载工具集合 |
| `load_query_gold_from_json` | 从 JSON 文件加载 Query 标注集合 |

---

## 4. 输入数据格式

### 4.1 工具集合格式

工具集合是一个 JSON List，每个元素代表一个工具。

示例文件：

```text
data/tool_descriptions.json
```

格式如下：

```json
[
  {
    "id": "idx1",
    "title": "工具1",
    "description": "这是工具1的描述"
  },
  {
    "id": "idx2",
    "title": "工具2",
    "description": "这是工具2的描述"
  }
]
```

字段说明：

| 字段 | 类型 | 是否必需 | 说明 |
|---|---|---|---|
| `id` | string | 是 | 工具唯一 ID |
| `title` | string | 是 | 工具标题 |
| `description` | string | 是 | 工具描述 |

---

### 4.2 Query 测试集合格式

Query 测试集合是一个 JSON Dict。

示例文件：

```text
data/high_pv_query.json
```

格式如下：

```json
{
  "query2": ["idx1", "idx4"],
  "query3": ["idx2"]
}
```

含义：

- Key：测试 Query 文本
- Value：该 Query 对应的正确工具 ID 列表

例如：

```json
{
  "帮我查天气": ["weather_tool"],
  "帮我订机票": ["flight_tool", "travel_tool"]
}
```

表示：

- Query `帮我查天气` 的正确工具是 `weather_tool`
- Query `帮我订机票` 的正确工具是 `flight_tool` 或 `travel_tool`

---

## 5. 环境依赖

建议使用 Python 3.9+。

安装依赖：

```bash
pip install numpy pandas sentence-transformers openpyxl
```

如果模型依赖 PyTorch，也需要根据 CUDA 环境安装对应版本的 `torch`。

例如：

```bash
pip install torch
```

---

## 6. 模型配置

脚本中默认使用本地 Qwen3-Embedding 模型：

```python
MODEL_PATH = "zacharychu/afs_data/model/Qwen3-Embedding-8B"
model = SentenceTransformer(MODEL_PATH)
```

如果模型路径不同，需要修改为自己的本地模型路径：

```python
MODEL_PATH = "/your/local/path/Qwen3-Embedding-8B"
```

也可以替换为其他 SentenceTransformer 兼容的 embedding 模型。

---

## 7. 快速开始

### 7.1 准备目录结构

推荐目录结构：

```text
project/
├── nn_recall_passk.py.py
├── data/
│   ├── tool_descriptions.json
│   └── high_pv_query.json
└── recall_outputs/
```

### 7.2 准备工具集合

创建文件：

```text
data/tool_descriptions.json
```

示例：

```json
[
  {
    "id": "idx1",
    "title": "天气查询",
    "description": "查询城市天气、温度、降雨概率、空气质量"
  },
  {
    "id": "idx2",
    "title": "机票预订",
    "description": "查询航班信息并完成机票预订"
  }
]
```

### 7.3 准备 Query 标注集合

创建文件：

```text
data/high_pv_query.json
```

示例：

```json
{
  "北京明天天气怎么样": ["idx1"],
  "帮我订一张去上海的机票": ["idx2"]
}
```

### 7.4 运行脚本

```bash
python nn_recall_passk.py.py
```

运行后会：

1. 加载工具集合
2. 加载 Query 测试集合
3. 构建三路工具向量
4. 对 Query 进行召回
5. 计算不同视角下的 pass@k
6. 保存评估结果

---

## 8. 评估指标说明

### 8.1 pass@k

`pass@k` 表示：

> 对于一个 Query，如果召回 Top K 结果中至少命中一个正确工具 ID，则该 Query 的 pass@k = 1，否则为 0。

例如：

```text
gold_ids = ["idx1", "idx4"]
recall_top_3 = ["idx5", "idx1", "idx8"]
```

因为 Top 3 中包含 `idx1`，所以：

```text
pass@3 = 1
```

如果：

```text
gold_ids = ["idx1", "idx4"]
recall_top_3 = ["idx5", "idx6", "idx8"]
```

则：

```text
pass@3 = 0
```

---

### 8.2 first_hit_rank

`first_hit_rank` 表示第一个命中的正确工具在召回结果中的排名。

例如：

```text
gold_ids = ["idx1", "idx4"]
recall_top_5 = ["idx5", "idx6", "idx4", "idx1", "idx8"]
```

第一个命中的是 `idx4`，排名第 3，因此：

```text
first_hit_rank = 3
```

---

### 8.3 avg_first_hit_rank

`avg_first_hit_rank` 表示所有成功命中的 Query 的平均首次命中排名。

该指标越小，说明正确工具越靠前。

---

### 8.4 missing_gold_query_count

`missing_gold_query_count` 表示：

> gold_ids 中存在不在工具集合中的 ID 的 Query 数量。

例如：

工具集合中只有：

```text
idx1, idx2, idx3
```

但是 Query 标注为：

```json
{
  "query2": ["idx1", "idx4"]
}
```

其中 `idx4` 不存在于工具集合，因此该 Query 会被统计到 `missing_gold_query_count`。

---

## 9. 召回视角说明

脚本默认评估以下四种召回视角：

```python
eval_views=["title", "description", "title_description", "merged"]
```

| 视角 | 说明 |
|---|---|
| `title` | 只使用工具标题构建向量库 |
| `description` | 只使用工具描述构建向量库 |
| `title_description` | 使用标题 + 描述构建向量库 |
| `merged` | 多路融合，取每个工具在三路向量中的最高相似度 |

---

## 10. 输出结果说明

默认输出目录：

```text
./recall_outputs
```

默认输出文件：

```text
recall_outputs/
├── tool_pass_at_k_recall_summary.csv
├── tool_pass_at_k_recall_detail.csv
├── tool_pass_at_k_recall_summary.jsonl
├── tool_pass_at_k_recall_detail.jsonl
└── tool_pass_at_k_recall.xlsx
```

---

### 10.1 summary 结果

`summary` 是汇总指标，每一行表示一个召回视角和一个 k 值的整体效果。

核心字段：

| 字段 | 说明 |
|---|---|
| `view` | 召回视角 |
| `k` | Top K |
| `query_count` | Query 总数 |
| `pass_count` | 命中的 Query 数量 |
| `pass_at_k` | pass_count / query_count |
| `avg_first_hit_rank` | 平均首次命中排名 |
| `missing_gold_query_count` | 存在缺失 gold id 的 Query 数量 |

示例：

```text
view                 k   query_count   pass_count   pass_at_k
title                1   1000          720          0.720
title                3   1000          850          0.850
description          1   1000          760          0.760
merged               5   1000          920          0.920
```

---

### 10.2 detail 结果

`detail` 是每条 Query 的召回明细。

核心字段：

| 字段 | 说明 |
|---|---|
| `query` | 测试 Query |
| `gold_ids` | 正确工具 ID 列表 |
| `missing_gold_ids` | 不在工具集合中的 gold id |
| `view` | 召回视角 |
| `k` | Top K |
| `pass_at_k` | 当前 Query 是否命中 |
| `hit_ids` | 命中的工具 ID |
| `first_hit_rank` | 首次命中排名 |
| `recall_ids` | Top K 召回工具 ID |
| `recall_scores` | Top K 召回分数 |
| `recall_views` | 召回结果来自哪个视角 |

---

## 11. 自定义配置

### 11.1 修改 Top K

在 `main()` 中修改：

```python
summary_df, detail_df = evaluator.evaluate(
    k_list=[1, 3, 5, 10],
    eval_views=["title", "description", "title_description", "merged"],
)
```

例如增加 Top 20：

```python
k_list=[1, 3, 5, 10, 20]
```

---

### 11.2 修改评估视角

如果只想评估 `title_description` 和 `merged`：

```python
eval_views=["title_description", "merged"]
```

---

### 11.3 修改输出目录

```python
evaluator.save_recall_results(
    summary_df=summary_df,
    detail_df=detail_df,
    output_dir="./recall_outputs",
    prefix="tool_pass_at_k_recall",
)
```

可以修改为：

```python
output_dir="./outputs/qwen3_embedding_eval"
prefix="qwen3_tool_recall"
```

---

### 11.4 替换编码函数

当前编码函数：

```python
def encode_fn(texts: List[str]) -> np.ndarray:
    embeddings = model.encode(
        texts,
        batch_size=64,
        normalize_embeddings=False,
        show_progress_bar=False,
    )
    return np.asarray(embeddings)
```

如果要接入远程 embedding 服务，可以保持接口不变：

```python
def encode_fn(texts: List[str]) -> np.ndarray:
    # 请求远程 embedding API
    # 返回 shape = [len(texts), dim] 的 np.ndarray
    return embeddings
```

只要输入是 `List[str]`，输出是 `np.ndarray` 即可。

---

## 12. 运行流程

整体流程如下：

```text
读取 tool_descriptions.json
        ↓
读取 high_pv_query.json
        ↓
构建 title / description / title_description 三路文本
        ↓
调用 embedding 模型构建工具向量底库
        ↓
对测试 Query 生成 Query 向量
        ↓
分别进行 title / description / title_description / merged 召回
        ↓
计算 pass@1 / pass@3 / pass@5 / pass@10
        ↓
生成 summary_df 和 detail_df
        ↓
保存 CSV / JSONL / Excel
```

---

## 13. 注意事项

### 13.1 工具 ID 必须唯一

如果工具集合中存在重复 ID，脚本会跳过重复项，并打印警告：

```text
Warning: skipped duplicate tool ids
```

建议在数据准备阶段保证 `id` 唯一。

---

### 13.2 gold_ids 必须和工具集合 ID 对齐

如果 Query 标注中的 gold id 不存在于工具集合中，会影响评估结果。

可以通过输出字段查看：

```text
missing_gold_ids
missing_gold_query_count
```

---

### 13.3 空文本会影响召回效果

如果工具的 `title` 或 `description` 为空，对应视角下的向量质量会较差。

建议在构建数据时保证：

- `title` 简洁准确
- `description` 覆盖工具能力、适用场景、输入输出信息

---

### 13.4 当前实现适合中小规模数据

当前召回使用的是：

```python
np.dot(doc_embeddings, query_embedding)
```

这属于全量暴力检索。

如果工具数量达到几十万或百万级，建议改为 FAISS / HNSW / Milvus 等向量索引。

---

## 14. 常见问题

### Q1：为什么使用 L2 normalize？

因为向量归一化后，两个向量的点积等价于 cosine similarity。

脚本中：

```python
embeddings = l2_normalize(embeddings)
scores = np.dot(doc_embeddings, query_embedding)
```

这样可以使用点积快速计算语义相似度。

---

### Q2：merged 是怎么融合的？

`merged` 会遍历三路向量：

- title
- description
- title_description

对每个工具 ID 取三路中的最高相似度，最后按照最高分排序。

也就是：

```text
tool_score = max(title_score, description_score, title_description_score)
```

---

### Q3：pass@k 和 recall@k 有什么区别？

在当前脚本中，`pass@k` 判断的是：

> Top K 中是否至少命中一个正确答案。

如果一个 Query 有多个正确工具，只要命中任意一个，就算通过。

更严格的多标签召回指标可以扩展为：

- recall@k
- precision@k
- mAP
- MRR
- nDCG

---

### Q4：为什么 summary 中 avg_first_hit_rank 为空？

如果某个视角和 k 下没有任何 Query 命中，则没有首次命中排名，因此 `avg_first_hit_rank` 为空。

---

## 15. 可优化方向

后续可以继续增强以下能力：

1. **接入 FAISS 加速召回**

   当前是 numpy 暴力检索，适合小规模评估。大规模工具库建议接入 FAISS。

2. **增加阈值过滤评估**

   在 Top K 基础上增加相似度阈值，例如：

   ```python
   score >= 0.3
   score >= 0.5
   score >= 0.7
   ```

3. **增加更多排序指标**

   可扩展：

   - MRR
   - nDCG@k
   - Precision@k
   - Recall@k

4. **支持多模型对比**

   可以将不同 embedding 模型的结果分别保存，横向比较不同模型的召回效果。

5. **支持召回结果错误分析**

   对未命中 Query 自动输出：

   - Query
   - Gold Tool
   - Top K 误召工具
   - 相似度分数
   - 误召原因分析

6. **支持多字段权重融合**

   当前 `merged` 是取最大分，可以扩展为加权融合：

   ```text
   score = 0.3 * title_score + 0.5 * description_score + 0.2 * title_description_score
   ```

---

## 16. 推荐改造建议

当前脚本已经可以完成基础评估，但建议进一步做两点工程化改造：

### 16.1 将模型路径改为命令行参数

当前模型路径是硬编码：

```python
MODEL_PATH = "/root/paddlejob/workspace/env_run/output/zacharychu/afs_data/model/Qwen3-Embedding-8B"
```

建议改为：

```bash
python nn_recall_passk.py.py \
  --model_path /your/model/path \
  --tools_path data/tool_descriptions.json \
  --query_path data/high_pv_query.json \
  --output_dir recall_outputs
```

---

### 16.2 将 main 中的数据读取逻辑整理干净

当前 `main()` 中同时包含了示例数据和文件读取逻辑，建议整理成：

```python
tools = load_tools_from_json("data/tool_descriptions.json")
query_gold_ids = load_query_gold_from_json("data/high_pv_query.json")
```