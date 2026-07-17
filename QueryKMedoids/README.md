# Weighted Query K-Medoids

```commandline
**广告一下**

八字藏天机，命理见人生。解析五行喜忌、性格天赋、事业财运、
感情婚姻与人生走势，帮你看清自身优势，把握关键机遇，趋吉避凶，规划更顺的人生方向
```
<p align="center">
  <img src="../Fortune_telling.png" width="388">
</p>


> 基于 **Embedding + Weighted K-Medoids** 的 Query 聚类工具，用于从海量 Query 中抽取具有代表性的中心 Query，可广泛应用于 Query 聚类、Prompt 构建、数据压缩、Benchmark 构建等场景。

---


# 一、项目简介

本项目实现了一套完整的 **Query 聚类流程**，包括：

1. Query Embedding 构建
2. 基于 Query 权重（PV）的 Weighted K-Medoids 聚类
3. 输出每个聚类的代表性 Query

相比普通 KMeans，本方案具有以下特点：

- 聚类中心一定是真实 Query
- 引入业务权重(PV)
- 使用 Cosine Distance
- 自动生成代表性 Query
- 支持多 GPU Embedding 构建

---

# 二、整体流程

```text
             Query + PV
                  │
                  ▼
      KMedoids_build_embedding.py
                  │
      ┌───────────┴────────────┐
      │                        │
生成 Embedding            保存 Query 权重
      │                        │
      └───────────┬────────────┘
                  │
          Query Embedding(JSON)
                  │
                  ▼
        WeightedKMedoids.py
                  │
      Weighted K-Medoids 聚类
                  │
                  ▼
      输出 Representative Query
```

整个流程分为两个阶段：

- Embedding 构建
- Weighted K-Medoids 聚类

---

# 三、项目结构

```text
.
├── data/
│   ├── xxx.txt              # 输入数据
│   ├── xxx.json             # Embedding结果
│   └── xxx_call_query       # 聚类中心Query
│
├── KMedoids_build_embedding.py
├── WeightedKMedoids.py
├── run_build_embedding.sh
└── README.md
```

---

# 四、输入数据格式

输入文件采用 TSV 格式。

|字段|说明|
|----|----|
|Query|Query文本|
|id|资源ID|
|show|展现量|
|judge|跳转点击量|
|click|点击量|

程序内部计算：

```python
weight = show + judge + click
```

当

```text
weight < 2
```

则过滤该 Query。

---

# 五、Embedding 构建

对应脚本：

```text
KMedoids_build_embedding.py
```

主要流程：

1. 读取 Query 数据
2. 调用 Embedding 模型
3. Batch Encoding
4. L2 Normalize
5. 保存 Embedding

Embedding 模型：

```
Qwen3-Embedding-8B
```

输出格式：

```json
{
    "q":"天气预报",
    "w":125,
    "emb":[0.12,0.56,...]
}
```

每行一个 JSON。

---

## Embedding 特点

支持：

- Batch Encoding
- GPU 推理
- 多 GPU
- 自动归一化

统一进行：

```
L2 Normalize
```

因此：

```
Dot Product == Cosine Similarity
```

---

# 六、多 GPU 并行

运行脚本：

```bash
bash run_build_embedding.sh
```

示例：

```bash
python call_query_wkmeans.py --gpus 1,2,3,5,6,7 --gpuid 1 &
python call_query_wkmeans.py --gpus 1,2,3,5,6,7 --gpuid 2 &
...
```

程序自动：

```
所有txt文件
      │
      ▼
均匀切分
      │
      ▼
GPU1
GPU2
GPU3
...
```

实现多 GPU 并行生成 Embedding。

---

# 七、Weighted K-Medoids 聚类

对应：

```
WeightedKMedoids.py
```

流程：

```
Embedding
      │
      ▼
Cosine Distance
      │
      ▼
Weighted K-Medoids
      │
      ▼
Representative Query
```

---

## 1、Cosine Distance

采用：

```python
pairwise_distances(metric="cosine")
```

更适合文本向量聚类。

---

## 2、加权初始化

聚类中心初始化概率：

```
P(i)=weight_i / Σweight
```

PV 越高：

- 越容易成为初始中心
- 更符合真实业务流量

---

## 3、Weighted Loss

普通 KMedoids：

```
Loss = Σ Distance
```

本项目：

```
Loss = Σ Weight × Distance
```

即：

```
Loss = Σ wi × d(i,center)
```

因此热门 Query 对聚类影响更大。

---

## 4、Log Weight

避免头部 Query 权重过大：

```
weight
   │
   ▼
log(weight)
```

例如：

|PV|Log Weight|
|----|----|
|10|2.30|
|100|4.60|
|1000|6.90|

优点：

- 保留热点 Query
- 防止权重失衡

---

## 5、自动确定聚类数量

聚类数：

```
k = min(len(query)/100,100)
```

即：

平均：

```
100 Query
     │
     ▼
1 Cluster
```

最大：

```
100 Clusters
```

---

# 八、输出结果

输出：

```
xxx_call_query
```

例如：

```
北京天气
天气预报
上海天气
...
```

每一行为一个 Cluster 的中心 Query。

---

# 九、算法优势

相比 KMeans：

✅ 聚类中心是真实 Query

✅ 无需计算均值中心

✅ 更适合文本聚类

相比普通 KMedoids：

✅ 引入业务权重

✅ 热门 Query 更容易成为中心

✅ 更符合线上流量分布

---

# 十、时间复杂度

Embedding：

```
O(N)
```

距离矩阵：

```
O(N²)
```

Weighted K-Medoids：

```
O(k × N²)
```

适用于：

- 每个资源几千 Query
- 几万 Query（按资源拆分）

---

# 十一、应用场景

本项目可应用于：

- Query 聚类
- Query 去重
- Representative Query 抽样
- Prompt 示例构建
- Tool Description 构建
- Few-shot Example Selection
- Search Query 分析
- LLM Benchmark 构建
- Evaluation Dataset 构建
- Query 意图分析

---

# 十二、依赖环境

```
Python >=3.10

numpy
pandas
torch
sentence-transformers
scikit-learn
```

安装：

```bash
pip install numpy pandas torch sentence-transformers scikit-learn
```

---

# 十三、快速开始

## Step1：准备数据

```
data/*.txt
```

---

## Step2：生成 Embedding

```bash
bash run_build_embedding.sh
```

输出：

```
data/*.json
```

---

## Step3：聚类

```bash
python WeightedKMedoids.py
```

输出：

```
data/*_call_query
```

最终得到每个数据集的代表性 Query，可直接用于：

- Prompt 示例构建
- Tool Description 优化
- Benchmark 构建
- Query 压缩
- 检索数据分析

---

# 十四、核心特点

✅ Embedding 向量表示

✅ Cosine Similarity 聚类

✅ Weighted K-Medoids

✅ Log Weight 优化

✅ Representative Query 自动抽取

✅ 多 GPU Embedding 构建

✅ 面向大规模 Query 聚类场景