# Static_And_Label


```commandline
**广告一下**

八字藏天机，命理见人生。解析五行喜忌、性格天赋、事业财运、
感情婚姻与人生走势，帮你看清自身优势，把握关键机遇，趋吉避凶，规划更顺的人生方向
```
<p align="center">
  <img src="../Fortune_telling.png" width="388">
</p>

`Static_And_Label` 是一个面向中文 Query 标注分析、静态规则挖掘、Embedding 相似召回和文本分类预测的工具目录。

该项目适用于文创、办公写作、成语、诗词等中文 Query 场景，可以辅助完成：

- 标注数据分析；
- 关键词和静态特征统计；
- 相似 bad-case 召回；
- Query 质量评估；
- 多模型文本分类预测；
- 人工验收辅助判断。

---

## 核心能力

| 能力 | 说明 |
|---|---|
| 静态特征分析 | 对 Query、关键词、token、词性、长度、标签分布等进行统计分析 |
| Embedding 召回 | 使用 SentenceTransformer 生成 Query 向量，并通过 Faiss 进行相似样本召回 |
| bad-case 投票 | 根据召回样本中的标签分布计算 `vote` 分数，辅助判断是否需要返修 |
| 文本分类预测 | 基于 jieba 分词、TF-IDF 特征和多个机器学习模型进行分类 |
| 集成预测 | 使用多个模型的 majority vote 生成最终预测结果 |
| 结果解释 | 输出关键特征、局部证据、分类报告和预测结果文件 |

---

## 目录结构

```text
Static_And_Label/
├── 0_keyWord_regrex.ipynb          # 关键词 / 正则 / 标注数据初步分析
├── 1_static_key_word.ipynb         # 静态关键词、token、词性及统计特征分析
├── 2_embedding_search.ipynb        # Embedding 召回 Notebook 版本
├── 2_embedding_search.py           # Embedding + Faiss 相似召回脚本
├── 3_predicting_search.ipynb       # 分类预测 Notebook 版本
├── 3_sample_search.ipynb           # 样本分类实验 Notebook
├── 3_sample_search.py              # 样本分类脚本
├── 3_predict_search.py             # 多模型分类预测脚本
└── README.md                       # 项目说明文档
```

---

## 整体流程

```text
原始标注数据
    ↓
关键词 / 正则 / 静态特征分析
    ↓
Embedding 相似召回
    ↓
bad-case 投票分数计算
    ↓
jieba + TF-IDF + 多模型分类预测
    ↓
输出预测结果、评估指标和解释文件
```

---

## 模块说明

### 1. 关键词与静态特征分析

相关文件：

```text
0_keyWord_regrex.ipynb
1_static_key_word.ipynb
```

主要用于对标注数据进行初步分析，包括：

- Query 文本分析；
- 关键词和正则匹配分析；
- token 和词性统计；
- 不同标签下的关键词分布；
- query_count / occurrence_count 统计；
- 具有明显标签倾向的关键词挖掘。

常见分析字段包括：

```text
Query
标注
分值
备注
token
tokens
pos
query_count_0
query_count_1
query_count_2
occurrence_count_0
occurrence_count_1
occurrence_count_2
length
```

---

### 2. Embedding 相似召回

相关文件：

```text
2_embedding_search.py
2_embedding_search.ipynb
```

该模块会读取训练标注数据，使用 SentenceTransformer 加载本地 Embedding 模型，对 Query 生成向量，然后使用 Faiss 构建向量索引。

核心流程：

```text
训练 Query → Embedding → Faiss Index
待预测 Query → Embedding → TopK 相似召回
相似样本标签 → bad-case 比例 → vote 分数
```

默认输入：

```text
data/工作写作标注结果.xlsx
output/验收数据/文创线上数据_清洗数据.xlsx
```

默认输出：

```text
output/验收数据/文创线上数据_清洗数据_vote.xlsx
```

核心字段：

```text
训练文本列: 随机Q
训练标签列: 返修
待预测文本列: Query
输出字段: embedding_sim, vote
```

运行方式：

```bash
python 2_embedding_search.py
```

注意：脚本中默认模型路径为本地路径，使用前需要改成自己的模型路径。

```python
model_name = "/root/paddlejob/workspace/env_run/output/zacharychu/afs_data/model/Qwen3-1.7B"
```

代码中使用：

```python
faiss.IndexFlatIP(dimension)
```

`IndexFlatIP` 是内积检索。由于脚本中使用了 `normalize_embeddings=True`，因此内积相似度可以近似理解为 cosine similarity。

---

### 3. jieba + TF-IDF + 多模型分类

相关文件：

```text
3_predict_search.py
3_sample_search.py
3_predicting_search.ipynb
3_sample_search.ipynb
```

该模块使用传统机器学习方法完成中文 Query 分类预测。

核心流程：

```text
Query → jieba 分词 → TF-IDF 特征 → 多模型分类 → 集成投票 → 输出预测结果
```

默认支持的模型包括：

```text
logistic_regression
linear_svm
multinomial_nb
decision_tree
random_forest
```

主要能力：

- jieba 中文分词；
- TF-IDF 特征提取；
- class weight 处理类别不均衡；
- class 0 阈值调优；
- 多模型训练和预测；
- majority vote 集成预测；
- 输出分类报告、重要特征和局部解释证据。

---

## 默认参数说明

### `3_predict_search.py`

```text
训练文件: data/工作写作标注结果.xlsx
测试文件: output/验收数据/文创线上数据_清洗数据_vote.xlsx
训练文本列: 随机Q
训练标签列: 返修
测试文本列: Query
测试标签列: 评分
输出目录: outputs/jieba_cls
```

### `3_sample_search.py`

```text
训练文件: data/工作写作标注结果.xlsx
测试文件: output/验收数据/文创线上数据_清洗数据_vote.xlsx
训练文本列: 随机Q
训练标签列: 评分（0、1、2）
测试文本列: Query
测试标签列: 评分
输出目录: outputs/jieba_cls
```

---

## 标签处理逻辑

脚本默认支持标签：

```python
[0, 1, 2]
```

并会将标签 `2` 合并为 `1`：

```python
train_df[args.train_label] = train_df[args.train_label].replace(2, 1)
test_df[args.test_label] = test_df[args.test_label].replace(2, 1)
```

因此最终通常会变成二分类任务：

```text
0: 负向 / bad / 需返修
1: 正向 / normal / 可通过
```

实际标签含义需要结合业务标注规范确认。

---

## 安装依赖

建议使用 Python 3.9+。

```bash
pip install pandas numpy faiss-cpu sentence-transformers jieba scikit-learn openpyxl
```

如果使用 GPU 版 Faiss，可根据 CUDA 环境安装：

```bash
pip install faiss-gpu
```

如果 GPU 版本安装失败，可以先使用 `faiss-cpu` 跑通整体流程。

---

## 数据准备

建议准备如下目录结构：

```text
data/
├── 工作写作标注结果.xlsx
├── 创意文案汇总.xlsx
├── 办公写作_汇总.xlsx
├── 成语标注数据.xlsx
└── 诗词标注数据.xlsx

output/
└── 验收数据/
    └── 文创线上数据_清洗数据.xlsx
```

不同脚本依赖的字段名不同，运行前需要确认 Excel 中存在对应字段。

常用字段：

```text
随机Q
返修
Query
评分
评分（0、1、2）
备注
```

---

## 运行方式

### 1. Embedding 召回

```bash
python 2_embedding_search.py
```

生成结果：

```text
output/验收数据/文创线上数据_清洗数据_vote.xlsx
```

新增字段：

```text
embedding_sim
vote
```

### 2. 多模型分类预测

```bash
python 3_predict_search.py
```

或者自定义参数：

```bash
python 3_predict_search.py \
  --train data/工作写作标注结果.xlsx \
  --test output/验收数据/文创线上数据_清洗数据_vote.xlsx \
  --train_col 随机Q \
  --train_label 返修 \
  --test_col Query \
  --test_label 评分 \
  --output-dir outputs/jieba_cls
```

### 3. 样本分类实验

```bash
python 3_sample_search.py \
  --train data/工作写作标注结果.xlsx \
  --test output/验收数据/文创线上数据_清洗数据_vote.xlsx \
  --train_col 随机Q \
  --train_label "评分（0、1、2）" \
  --test_col Query \
  --test_label 评分
```

---

## 输出结果

分类脚本默认输出目录：

```text
outputs/jieba_cls/
```

主要输出文件：

```text
metrics_summary.csv          # 各模型 accuracy / macro_f1 汇总
key_features.json            # 各模型重要特征
local_evidence.json          # 单样本局部解释特征
ensemble_report.json         # 集成模型分类报告
```

同时会在测试 Excel 文件同目录下生成预测结果文件：

```text
*_predictions.xlsx
```

预测结果文件会新增以下字段：

```text
pred_logistic_regression
score_logistic_regression
pred_linear_svm
score_linear_svm
pred_multinomial_nb
score_multinomial_nb
pred_decision_tree
score_decision_tree
pred_random_forest
score_random_forest
pred_ensemble_majority_vote
score_ensemble_majority_vote
```

---

## 推荐使用流程

### Step 1：静态分析

先通过 Notebook 分析标注数据中的关键词、词性、分布和标签倾向：

```text
0_keyWord_regrex.ipynb
1_static_key_word.ipynb
```

### Step 2：Embedding 召回

运行：

```bash
python 2_embedding_search.py
```

得到带相似召回和 vote 分数的文件：

```text
output/验收数据/文创线上数据_清洗数据_vote.xlsx
```

### Step 3：分类预测

运行：

```bash
python 3_predict_search.py
```

得到模型预测结果和评估文件：

```text
outputs/jieba_cls/
*_predictions.xlsx
```

### Step 4：人工复核

结合以下信息综合判断 Query 是否需要返修：

- 模型预测标签；
- 模型预测分数；
- Embedding 相似召回结果；
- vote 分数；
- local evidence 局部解释特征；
- 原始 Query 内容。

---

## 注意事项

1. 当前代码中部分路径是本地绝对路径或固定业务路径，迁移环境时需要修改。
2. `2_embedding_search.py` 中的模型路径需要替换为实际可用的 Embedding 模型路径。
3. `3_predict_search.py` 中测试集标签会被统一覆盖，更适合“无真实标签线上预测”场景。
4. 如果测试集本身有真实标签，建议去掉测试标签覆盖逻辑。
5. `argparse` 中 `--labels` 默认值是 Python list，但命令行传参时不会自动解析成 list，正式工程化时建议改成字符串再解析。
6. `IndexFlatIP` 是内积索引，配合 normalize embedding 后可以作为 cosine similarity 使用。
7. 当前部分脚本没有封装 `main()`，导入脚本时会直接执行，建议后续增加：

```python
if __name__ == "__main__":
    main()
```

---

## 后续优化建议

1. 将固定路径改成命令行参数；
2. 将 Notebook 中的实验代码沉淀为可复用 Python 模块；
3. 增加统一配置文件，例如 `config.yaml`；
4. 增加日志模块，替代直接 `print`；
5. 增加模型保存和加载能力；
6. 增加预测阈值配置；
7. 增加 bad-case 样本导出；
8. 增加人工复核优先级字段；
9. 将 Embedding 召回和分类预测合并成统一 pipeline；
10. 增加 README 示例数据格式，方便新用户快速跑通。

---

## 适用场景

该工具适合以下场景：

- 中文 Query 质量评估；
- 文创 Query 标注分析；
- 办公写作 Query 返修预测；
- 低质 Query 识别；
- 静态关键词规则挖掘；
- 相似 bad-case 召回；
- 传统机器学习文本分类 baseline；
- 人工验收数据辅助判别。

---

## 一句话总结

`Static_And_Label` 提供了一套从静态规则分析、Embedding 相似召回到传统机器学习分类预测的中文 Query 质量分析工具链，适合作为文创、办公写作等场景下的标注分析、bad-case 发现和线上验收辅助工具。
