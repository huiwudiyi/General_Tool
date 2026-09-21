# 工具描述自改进生成方案（参照 SAGE：Skill-Augmented RL）

> 参照论文《Reinforcement Learning for Self-Improving Agent with Skill Library》（SAGE = Skill Augmented GRPO for self-Evolution, arXiv:2512.17102）的核心思想，为「工具描述生成」任务设计的强化学习自改进方案。
>
> 版本：v1.0 ｜ 日期：2026-09-15

---

## 目录

1. [背景与目标](#1-背景与目标)
2. [核心思想：SAGE → 本任务的映射](#2-核心思想sage--本任务的映射)
3. [整体架构](#3-整体架构)
4. [数据准备](#4-数据准备)
5. [描述生成器（策略模型）](#5-描述生成器策略模型)
6. [奖励函数设计（核心，MRR 版）](#6-奖励函数设计核心mrr-版)
7. [训练流程（GRPO + 库级自进化）](#7-训练流程grpo--库级自进化)
8. [冷启动 SFT](#8-冷启动-sft)
9. [评估指标](#9-评估指标)
10. [消融实验设计](#10-消融实验设计)
11. [工程实现与伪代码](#11-工程实现与伪代码)
12. [落地路线图](#12-落地路线图)
13. [风险与对策](#13-风险与对策)

## 1. 背景与目标

### 1.1 任务背景

给定一个工具（tool），我们拥有三类原始信息：

- **应召 query 集合 $Q^+$**：本应该召回该工具的真实用户查询集合（正样本）。
- **结果数据**：工具执行后返回的结构化数据（字段、示例值等）。
- **结果界面**：工具结果的前端呈现（截图或 DOM，可文本化 / OCR）。

目标是**自动生成高质量的工具描述**，描述由两部分构成：

- **功能区（Functional Area）**：说明工具"能做什么"——功能、输入输出、典型使用场景。
- **边界区（Boundary Area）**：说明工具"何时用 / 不用"——适用范围、前置条件、不适用场景、与相似工具的区别。

### 1.2 优化目标（描述"好"的定义）

一条描述好不好，由两个**可验证的下游任务**决定：

1. **可被检索**：用应召 query 集合按相似度召回 top-k 工具时，目标工具排名靠前（用 **MRR** 衡量）。
2. **可被正确选择**：在召回的 top-k 工具里，大模型能正确地把该 query 路由到目标工具。

这两点分别由描述的**功能区**（决定语义可召回性）和**边界区**（决定可区分性 / 抗误选）驱动。

---

## 2. 核心思想：SAGE → 本任务的映射

SAGE 的核心是：**把"技能"当作一个可自我改进的库组件，用可验证的下游成败作为奖励，通过 GRPO 让"生成技能的策略"和"技能库"共同进化。** 其精髓在于奖励的**条件性**——技能只有在被下游任务成功复用时才追认奖励。

本任务与 SAGE 高度同构：

| SAGE 概念 | 本任务对应物 | 说明 |
|-----------|--------------|------|
| Skill（可执行函数技能） | **工具描述**（功能区 + 边界区） | 待自我改进的产物 |
| Skill Library | **工具描述库 / 向量索引** | 所有工具描述构成检索空间 |
| Policy $\pi_\theta$（Agent 模型） | **描述生成器 LLM** | 待训练的策略 |
| Task Success Reward | 检索命中 + 选择正确 | 可验证的下游奖励 |
| Skill Generation Reward（被下游成功复用才给分） | **选择奖励**（先召回、再被正确选中才给分） | 同样的条件耦合结构 |
| Sequential Rollout（技能沿任务链累积） | **库级共同进化**（改一个描述会改变全库检索环境） | 全库描述协同优化 |
| GRPO | GRPO | 同一套 critic-free 组相对优化 |

> **关键洞察**：SAGE 里"技能好不好，看它是否被后续任务成功复用"是事后追认的。本任务里"描述好不好，看它是否让工具被 query 正确召回并选中"——天然是可验证、可追认的下游奖励，无需人工打分。

---

## 3. 整体架构

```
输入: [应召query集合 Q⁺] + [结果数据] + [结果界面(截图/DOM文本化)]
                       │
                       ▼
        ┌──────────────────────────────────┐
        │  描述生成器 π_θ (待训练 LLM)         │  ← 策略
        │  输出结构化描述 d = {功能区, 边界区} │
        └──────────────┬───────────────────┘
                       │ 写入 / 替换工具描述
                       ▼
        ┌──────── 工具描述库 (向量索引) ────────┐
        └──────────────┬────────────────────────┘
          ┌────────────┴────────────┐
          ▼                          ▼
   ① Retriever (embedding, 冻结)   ② LLM Selector (冻结裁判)
   query → topk 相似度检索          给定 query+topk 描述, 选出工具
          │                          │
          ▼                          ▼
   MRR 检索奖励               选择正确率 + 边界判别奖励
                     │
                     ▼
              奖励聚合 → GRPO 更新 π_θ
                     │
                     ▼   (每 epoch 全库重生成 → 重建索引 → 再评估)
              库级共同进化 (self-evolution)
```

### 组件清单

| 模块 | 角色 | 是否训练 |
|------|------|----------|
| 描述生成器 $\pi_\theta$ | 策略，生成结构化描述 | ✅ 训练 |
| Retriever（embedding 模型） | query→描述相似度 top-k 召回 | ❌ 冻结 |
| LLM Selector | 模拟真实 Agent，从 top-k 选工具 | ❌ 冻结（裁判） |
| Reward Evaluator | 计算 MRR / 选择正确率 / 边界误选率 | — |
| GRPO Trainer | 组相对策略优化 | — |

---

## 4. 数据准备

### 4.1 每个工具需要的样本

对工具 $t$：

- **正样本 $Q^+_t$**：应召 query 集合（真实日志挖掘 / 人工标注）。建议每工具 ≥ 20 条，覆盖不同表述。
- **结果数据**：字段 schema + 若干示例值。过长时截断 / 摘要。
- **结果界面**：截图（走多模态生成器）或 DOM 文本化 + OCR（走纯文本生成器）。

### 4.2 难负例挖掘 $Q^-_t$（边界区训练的关键）

只用正样本会让模型把描述写得过于宽泛（越宽越易召回），导致边界模糊、与相似工具打架。需构造**难负例**：

1. 对所有工具的 $Q^+$ 做 embedding，聚类 / 近邻搜索；
2. 取与 $t$ 语义相邻（易混）的兄弟工具的 query 作为 $Q^-_t$；
3. 语义定义：这些 query 可能把 $t$ 召回进 top-k，但**正确答案不是 $t$**。

$Q^-_t$ 专门考核边界区能否阻止"越界误选"。

### 4.3 数据划分

- train / valid / test 按工具划分（避免同一工具泄漏）；
- 另留一批 test 期新增工具，评估"库级进化"对动态新增工具的鲁棒性。

---

## 5. 描述生成器（策略模型）

### 5.1 输入构造

将三类信息拼成生成器输入（多模态则界面走图像通道）：

```
[工具名] tool_name
[应召query示例] q1; q2; ...; qn   (采样若干条, 防过长)
[结果数据schema] {字段: 类型, 示例值}
[结果界面] <截图> 或 <DOM文本化摘要>
[生成要求] 输出 JSON, 含 功能区 与 边界区 两字段
```

### 5.2 结构化输出（强约束）

固定 JSON 输出，不可解析直接判 0 分（进 Penalty）：

```json
{
  "功能区": "该工具用于……；输入为……；输出为……；典型场景：……",
  "边界区": "适用范围：……；前置条件：……；不适用于：……；与工具X的区别：……"
}
```

- **功能区**主要服务检索（语义对齐 query）；
- **边界区**主要服务选择（消歧、抗误选）。

### 5.3 生成器 Prompt 模板（示意）

```
你是工具描述专家。请根据以下信息为该工具生成描述。
描述必须严格输出 JSON，包含「功能区」与「边界区」两个字段：
- 功能区：清晰说明工具能做什么、输入输出、典型使用场景，用词应与用户可能的提问方式对齐。
- 边界区：说明适用范围、前置条件、明确不适用的场景，以及与相似工具的区分点。
要求：准确、精炼，避免冗余，避免夸大适用范围。

工具名：{tool_name}
应召查询示例：{sampled_queries}
结果数据：{result_schema}
结果界面：{ui_text_or_image}

请输出 JSON：
```

---

## 6. 奖励函数设计（核心，MRR 版）

对工具 $t$、生成描述 $d$、应召 query 集合 $Q^+_t=\{q_1,\dots,q_n\}$、难负例集合 $Q^-_t$。

### 6.1 检索奖励 $R_{\text{ret}}$ —— 用 MRR

将 $d$ 写入索引，对每个应召 query $q_i$ 做全库相似度排序，得到目标工具 $t$ 的排名 $\text{rank}(t\,|\,q_i)$（1 为最优）。检索奖励为该 query 集合上的 **MRR（Mean Reciprocal Rank）**：

$$R_{\text{ret}}(d) = \text{MRR} = \frac{1}{|Q^+_t|}\sum_{i=1}^{n}\frac{1}{\text{rank}(t\,|\,q_i)}$$

工程约定：

- 只在 top-K 截断窗口内计算，若 $t$ 未进 top-K，则该项 $\frac{1}{\text{rank}} = 0$（硬截断）；
- 或用**软截断** $\frac{1}{K+1}$ 给未命中一点残差信号，缓解早期稀疏奖励；
- MRR ∈ [0, 1]，天然归一化，比 pass@k 提供更平滑的排名梯度（排名从 5→2 也有正反馈，而 pass@k 只看进不进 top-k）。

> **为什么用 MRR 替代 pass@k**：pass@k 是 0/1 阶跃，"第 3 名"和"第 10 名"（都在 top-10 内）奖励相同，梯度稀疏；MRR 对排名连续敏感，能持续奖励"把工具往前推"的描述改进，训练更稳、收敛更快。

### 6.2 选择奖励 $R_{\text{sel}}$（以召回为前提，SAGE 式条件耦合）

选择必须先召回——没进 top-k 不可能被选中。对每个 query：

$$R_{\text{sel}}(d) = \frac{1}{|Q^+_t|}\sum_{i=1}^{n}\underbrace{\mathbf{1}\big[t \in \text{TopK}(q_i)\big]}_{\text{先召回}}\cdot \underbrace{\mathbf{1}\big[\text{LLMSelect}(q_i, \text{TopK}(q_i)) = t\big]}_{\text{再选中}}$$

这对应 SAGE 的 **Skill Generation Reward**：描述"生成得好 → 下游被正确使用"才追认。

### 6.3 边界判别奖励 $R_{\text{bnd}}$（专门驱动边界区）

对难负例 $Q^-_t$，即便 $t$ 被召回进 top-k，Selector 也**不应**选中 $t$：

$$R_{\text{bnd}}(d) = \frac{1}{|Q^-_t|}\sum_{q \in Q^-_t}\mathbf{1}\big[\text{LLMSelect}(q, \text{TopK}(q)) \neq t\big]$$

惩罚"越界误选"，直接考核边界区质量。

### 6.4 总奖励

$$R(d) = \alpha\, R_{\text{ret}} + \beta\, R_{\text{sel}} + \gamma\, R_{\text{bnd}} - \delta\, \text{Penalty}_{\text{format/length}}$$

- $\text{Penalty}$：JSON 不可解析 / 缺字段 / 超长，按程度扣分（呼应 SAGE 省 token 目标）；
- 建议初始 $\beta > \alpha$（选对比召回更重要）；$\gamma$ 随训练课程式增大（先学被召回，再学守边界）；
- 参考初值：$\alpha=1.0,\ \beta=1.5,\ \gamma$ 从 0.3 线性升到 1.0，$\delta=0.2$。

---

## 7. 训练流程（GRPO + 库级自进化）

### 7.1 GRPO Rollout（单步）

策略是描述生成器，对每个工具采样一组描述做组内相对优化：

```
for each training step:
  1. 采样工具 t，取输入 (Q⁺_t, 结果数据, 结果界面)
  2. 从 π_θ_old 采样 G 条候选描述 {d_1,...,d_G}         # group, 如 G=8
  3. for each d_j:
       - 将 d_j 临时替换进索引中 t 的描述条目
       - 在 Q⁺_t 上跑检索 → 计算 MRR = R_ret
       - 在召回结果上跑 LLMSelect → 计算 R_sel
       - 在 Q⁻_t 上跑检索+选择 → 计算 R_bnd
       - R_j = α·R_ret + β·R_sel + γ·R_bnd − δ·Penalty
  4. 组内相对优势(GRPO, critic-free):
       A_j = (R_j − mean({R})) / (std({R}) + ε)
  5. 以 A_j 更新 π_θ，加 KL 正则约束对参考策略的偏离
```

GRPO 目标（沿用 DeepSeekMath / SAGE）：

$$\mathcal{J}(\theta) = \mathbb{E}\Big[\frac{1}{G}\sum_{j=1}^{G}\min\big(\rho_j A_j,\ \text{clip}(\rho_j,1-\epsilon,1+\epsilon)A_j\big) - \eta\,\text{KL}(\pi_\theta\Vert\pi_{\text{ref}})\Big]$$

其中 $\rho_j = \pi_\theta(d_j)/\pi_{\theta_{old}}(d_j)$。

### 7.2 库级共同进化（对应 SAGE 的 Sequential Rollout / 自进化）

检索结果依赖**全库其它工具的描述**——改 A 的描述会改变 B 的 top-k 环境。因此做多轮库级协同进化：

```
for epoch in 1..E:
    1. 用当前 π_θ 重新生成 / 精炼「全部」工具描述
    2. 重建向量索引 (rebuild SkillBank)
    3. 在 valid 集上评估 (MRR / 选择正确率 / 边界误选率)
    4. 分析失败模式:
       - 被互相误召回的工具对 → 针对性加大其 Q⁻ 权重
       - 长期低 MRR 的工具 → 加大采样频率
    5. 继续 GRPO 训练下一轮
```

这一步复刻 SAGE"技能库与策略协同进化"，并自然覆盖"新增工具引入混淆"的动态场景。

---

## 8. 冷启动 SFT

对应 SAGE 先用专家经验 SFT 再 RL，避免早期奖励全 0（工具压根召不回）导致 GRPO 优势坍缩：

1. 收集一批人工 / 强模型（如 Claude / GPT）撰写的高质量「功能区+边界区」描述；
2. 对生成器做 SFT，得到较好起点策略 $\pi_{\text{ref}}$；
3. 再以 $\pi_{\text{ref}}$ 为参考策略启动 GRPO。

---

## 9. 评估指标

| 维度 | 指标 | 说明 |
|------|------|------|
| 检索 | **MRR**（主）、Recall@k | query→工具排名质量 |
| 选择 | top-k 选择正确率 | 召回后能否被正确路由 |
| 边界质量 | 难负例误选率（越低越好） | 边界区是否守得住 |
| 效率 | 描述平均 token 数 | 越短越好（呼应 SAGE） |
| 端到端 | 召回+选择联合正确率 | 真实链路指标 |

---

## 10. 消融实验设计

验证各设计的独立贡献：

| 消融项 | 目的 |
|--------|------|
| 去掉边界区（只生成功能区） | 验证边界区对选择正确率 / 误选率的贡献 |
| 去掉 $R_{\text{bnd}}$ 与难负例 | 验证边界奖励的作用 |
| MRR vs pass@k | 验证 MRR 带来的收敛速度 / 稳定性优势 |
| 去掉库级共同进化（仅单工具优化） | 验证协同进化对全库一致性的作用 |
| 去掉 SFT 冷启动 | 验证冷启动对训练稳定性的作用 |

---

## 11. 工程实现与伪代码

### 11.1 奖励评估器（核心逻辑）

```python
def evaluate_description(tool_id, desc, index, retriever, selector,
                         Q_pos, Q_neg, K=10,
                         alpha=1.0, beta=1.5, gamma=0.5, delta=0.2):
    # 0) 结构/格式校验
    penalty = format_length_penalty(desc)   # JSON可解析、字段齐全、长度

    # 1) 临时把 desc 写入索引中 tool_id 的条目
    index.upsert(tool_id, retriever.embed(desc))

    # 2) 检索奖励: MRR
    rr_sum = 0.0
    hit_topk = {}
    for q in Q_pos:
        ranked = index.search(retriever.embed(q), topk=K)   # 返回工具id有序列表
        rank = position_of(tool_id, ranked)                 # 1-based, 未命中=None
        rr_sum += (1.0 / rank) if rank else 0.0             # 硬截断
        hit_topk[q] = (rank is not None)
    R_ret = rr_sum / len(Q_pos)

    # 3) 选择奖励 (以召回为前提)
    sel_hit = 0
    for q in Q_pos:
        if not hit_topk[q]:
            continue
        cand = index.search(retriever.embed(q), topk=K)
        chosen = selector.select(q, [index.desc_of(c) for c in cand], cand)
        sel_hit += int(chosen == tool_id)
    R_sel = sel_hit / len(Q_pos)

    # 4) 边界判别奖励 (难负例上不应被选中)
    bnd_ok = 0
    for q in Q_neg:
        cand = index.search(retriever.embed(q), topk=K)
        chosen = selector.select(q, [index.desc_of(c) for c in cand], cand)
        bnd_ok += int(chosen != tool_id)
    R_bnd = bnd_ok / max(len(Q_neg), 1)

    return alpha*R_ret + beta*R_sel + gamma*R_bnd - delta*penalty
```

### 11.2 GRPO 训练骨架（对接 trl / verl）

```python
for step in range(num_steps):
    tool = sample_tool()
    x = build_input(tool)                        # 应召query+结果数据+界面
    group = policy.generate(x, n=G)              # 采样 G 条描述
    rewards = [evaluate_description(tool.id, d, ...) for d in group]
    adv = (rewards - mean(rewards)) / (std(rewards) + 1e-6)
    loss = grpo_loss(policy, group, adv, ref_policy, kl_coef=eta)
    loss.backward(); optimizer.step()

    if step % epoch_len == 0:                     # 库级共同进化
        regenerate_all_descriptions(policy)
        rebuild_index()
        report_metrics_on_valid()
```

### 11.3 技术选型建议

- **生成器**：Qwen2.5-7B/14B-Instruct（纯文本）或 Qwen2.5-VL（需读界面截图）。
- **Retriever**：bge-m3 / gte-large 等中文友好 embedding；索引用 FAISS / Milvus。
- **Selector**：中等规模指令模型即可（如 Qwen2.5-7B-Instruct），冻结作裁判，prompt 要求"从候选中选一个或返回无"。
- **RL 框架**：verl（agent RL 成熟）或 trl 的 GRPOTrainer。

---

## 12. 落地路线图

| 阶段 | 内容 | 产出 |
|------|------|------|
| P0 数据 | 挖掘 $Q^+$、构造 $Q^-$、准备结果数据/界面 | 训练/验证/测试集 |
| P1 冷启动 | 强模型写高质量描述 → SFT 生成器 | $\pi_{\text{ref}}$ |
| P2 奖励闭环 | 实现 Retriever+Selector+Reward Evaluator | 可跑的评估器 + 离线指标 |
| P3 RL 训练 | GRPO 单工具优化 | 首版 RL 生成器 |
| P4 库级进化 | 全库重生成+重建索引的多轮协同进化 | 稳定的自进化流程 |
| P5 评估上线 | 完整指标 + 消融，灰度替换线上描述 | 报告 + 上线 |

---

## 13. 风险与对策

| 风险 | 对策 |
|------|------|
| 早期奖励稀疏（工具压根召不回） | SFT 冷启动 + MRR 软截断残差信号 |
| 描述"越宽泛越易召回"导致边界崩塌 | 难负例 + $R_{\text{bnd}}$ + $\gamma$ 课程式增大 |
| Selector 裁判不稳定 / 有偏 | 固定 prompt、固定模型版本、必要时多次投票取多数 |
| 库级进化震荡（改一个影响一片） | 小步长、限制单轮重生成比例、KL 正则约束偏离 |
| 奖励被 hack（钻裁判空子） | 定期人工抽检、加入格式/长度正则、多裁判交叉 |
| 描述过长吃 token | 长度 Penalty，鼓励精炼（呼应 SAGE 省 token） |

---

## 附：与 SAGE 的对应关系速查

- **技能库 → 工具描述库**：都是可自我改进的结构化知识库。
- **可验证结果奖励 → MRR + 选择正确率**：都用下游可验证信号，不靠人工打分。
- **Skill Generation Reward 的条件性 → 选择奖励"先召回才可能被选"**：同构的条件耦合。
- **Sequential Rollout 技能累积 → 库级共同进化**：都让库与策略协同演化。
- **GRPO + SFT 冷启动**：训练范式完全一致。

本方案针对"功能区/边界区"的结构，额外引入了**边界判别奖励 $R_{\text{bnd}}$** 与**难负例**，这是相对 SAGE 的针对性扩展。
