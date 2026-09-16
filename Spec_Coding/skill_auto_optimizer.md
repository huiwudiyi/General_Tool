# Skill 自动优化系统 — 需求文档

## 1. 背景与目标

在 [Hermes Agent](https://github.com/NousResearch/hermes-agent) 中，skill 以 `SKILL.md`
（Markdown + YAML frontmatter）的形式定义，`description` 字段直接决定该 skill 能否被
准确触发。Hermes 官方自带的 curator 只做**确定性的统计与归档**（`use_count`/`stale`/
`archived`），不涉及内容质量评估，也不会改写 skill 描述。

本项目的目标：在每次 skill 被调用完成后，自动采集调用数据，通过 LLM judge 评估这次
调用的效果、诊断当前 `description` 是否存在描述模糊等问题，并生成改进建议，经人工审核
后写回 `SKILL.md`，形成一个"使用 → 评估 → 优化"的闭环，持续提升 skill 的触发准确率。

## 2. 范围边界（In / Out of Scope）

明确边界是为了防止"vibe coding"式开发中范围随对话自然膨胀，导致实现偏离最初确认过的
需求。以下边界已在开发前与需求方逐条确认：

**In Scope**
- 单个 skill 的 `description` 字段的评估与改进建议生成。
- 事件触发（skill 调用完成后）→ LLM judge → 人工审核 → 写回，这一条完整链路。
- Deepseek、豆包两种 LLM provider 的可配置切换。

**Out of Scope（明确不做，避免范围蔓延）**
- 不修改 `SKILL.md` 正文（Procedure/Pitfalls/Verification 等章节），仅动 `description`。
- 不做全自动无人审核的落盘（`auto_apply` 默认关闭，作为后续迭代项）。
- 不涉及 Hermes 官方 curator 的统计/归档逻辑本身，只做补充，不改动其代码。
- 不涉及 skill 的新增/删除，只优化已存在 skill 的描述质量。
- 不接管认证、密钥分发、多租户权限等基础设施能力（见第 5 节安全护栏）。

## 3. 需求澄清过程（关键决策）

| # | 问题 | 结论 |
|---|------|------|
| 1 | 目标代码库 | 全新项目 |
| 2 | 优化对象 | Skill 的内容/描述（`SKILL.md` 的 `description`） |
| 3 | 触发方式 | 事件触发：某个 skill 被调用完成后 |
| 4 | 优化依据 | 日志（query + 执行过程 + 结果） + LLM judge 评估 |
| 5 | 技术栈 | Python + Hermes Agent + LangGraph；模型用 Deepseek 和豆包 |
| 6 | 与 Hermes 现有能力的关系 | 调研确认 Hermes 自带 curator 无 LLM 评估/改写能力，本项目是**对 curator 的补充**，采用"基于 Hermes 现有机制做定制"（方案 A），而非另起一套独立系统 |
| 7 | 事件采集的接入点 | Hermes 的 `post_tool_call`、`on_skill_lifecycle` 两个 hook 都是**观察型 hook，返回值被忽略**，因此只能用于日志采集，无法在 hook 内同步完成评估与改写 |
| 8 | 日志内容 | query、执行过程、执行结果 |
| 9 | LLM judge 输出 | 不仅打分，还要给出改进 `description` 的具体建议 |
| 10 | 改写落地方式 | 默认**不自动覆盖**，写入待审核队列，人工 approve 后才写回 `SKILL.md` |

## 4. 系统架构

```
Hermes Agent
  └─ Hermes Plugin（事件采集）
        │ 挂载 post_tool_call / on_skill_lifecycle 两个观察型 hook
        │ 异步写入本地 JSONL 日志，不阻塞 Hermes 主流程
        ▼
   事件日志（JSONL，按天滚动）
        │ 独立服务轮询 + 去抖（同一 skill 短时间内多次触发只处理一次）
        ▼
   事件监听服务 → LangGraph 评估优化图
        │ persist_event（落库）
        │   → load_current_skill（读取当前 SKILL.md 的 description）
        │     → judge（调用 Deepseek/豆包，打分 + 生成改进建议）
        │       → enqueue_review（写入审核队列）
        ▼
   SQLite 存储（events 表 + review_queue 表）
        │
        ▼
   审核 CLI（list / show / approve / reject）
        │ approve 时才真正写回
        ▼
   目标 SKILL.md（description 被更新）
```

**关键设计约束**：由于 Hermes 的相关 hook 是观察型、不能同步介入主流程，因此架构上
必须拆成两部分——「轻量的 Plugin（仅采集）」与「独立运行的评估服务（重逻辑）」，
不能把 LLM 调用直接放进 hook 回调。

## 5. 功能需求

### 4.1 事件采集（Hermes Plugin）
- 以 Hermes Plugin 形式实现（`register(ctx)` 入口），部署到 Hermes 的插件目录。
- 挂载 `post_tool_call`、`on_skill_lifecycle` 两个 hook。
- 采集内容：skill 名称、调用参数、执行结果、耗时、use_count、是否复用等。
- 写入过程异步化（独立写线程 + 队列），采集失败不能影响 Hermes 主流程，需静默降级。
- 日志格式：JSONL，按天滚动。

### 4.2 事件监听与调度
- 独立 Python 服务，轮询事件日志目录，读取新增行。
- 支持去抖：同一 skill 短时间内多次触发只触发一次评估，避免重复调用 LLM。
- 将符合"skill 调用完成"语义的事件转换为标准化的调用摘要（skill_name / query / process / result），送入优化图。

### 4.3 评估优化图（LangGraph）
- 节点：`persist_event`（落库原始事件）→ `load_current_skill`（读取目标 skill 当前
  description，若 skill 不存在则跳过）→ `judge`（调用 LLM）→ `enqueue_review`（写入
  审核队列）。
- LLM judge 输出结构化 JSON：`score`（0-1 打分）、`reasoning`（诊断依据）、
  `suggested_description`（改进建议文本）。

### 4.4 LLM 客户端
- 统一封装 Deepseek、豆包（火山方舟）两种 provider，均为 OpenAI 兼容协议。
- 通过配置文件切换 provider 和模型，API Key 从环境变量读取，不落盘、不写入日志。
- 支持超时与重试配置。

### 4.5 存储层
- SQLite，两张表：
  - `events`：原始事件（skill_name / query / process / result / raw / created_at）。
  - `review_queue`：judge 产出的建议（score / reasoning / current_description /
    suggested_description / status / created_at / reviewed_at），`status` 含
    `pending` / `approved` / `rejected`。

### 4.6 审核 CLI
- `list`：查看指定状态（默认 `pending`）的建议列表。
- `show <id>`：查看某条建议的完整详情（打分、理由、新旧描述对比）。
- `approve <id>`：定位对应的 `SKILL.md`，将 `suggested_description` 写回 frontmatter
  的 `description` 字段，仅替换该字段，不影响正文与其他 frontmatter 字段；写回后更新
  状态为 `approved`。
- `reject <id>`：仅更新状态为 `rejected`，不做文件改动。

## 6. 非功能需求与安全护栏

参考行业对 vibe coding 的护栏建议（人机边界、密钥管理、审计留痕、变更可控），本项目
的非功能需求均落到具体机制，而非停留在原则层面：

| 护栏原则 | 在本项目中的落地 |
|---|---|
| 划清 AI 自主与人工负责的边界 | Skill 描述改写默认**不自动生效**，`suggested_description` 必须经人工 `approve` 才写回 `SKILL.md`；`auto_apply` 开关默认为 `false` |
| 密钥不落盘、不进日志 | Deepseek/豆包 API Key 仅通过环境变量（`DEEPSEEK_API_KEY`/`DOUBAO_API_KEY`）注入，配置文件只存 env var 名称，事件日志/数据库均不记录 Key |
| AI 生成代码需人工审查 | 交付时所有代码变更需过评审，本文档第 3 节明确的 Out of Scope 项禁止未经确认扩展 |
| 采集侧故障隔离 | Hermes Plugin 的 hook 回调捕获所有异常、静默降级，任何采集失败不能影响 Hermes 主流程（对应观察型 hook 的既有约束） |
| 变更可追溯 | `review_queue` 表记录每条建议的 `created_at`/`reviewed_at`/`status`，构成审计轨迹 |
| 先小范围验证再推广 | 交付先提供离线示例（`examples/`，见第 7 节）跑通全链路，验证后才对接真实 Hermes 环境和真实 API Key |

其余非功能约束：

- **稳定性**：hook 回调保持轻量，重逻辑（LLM 调用）放在独立服务里异步处理。
- **可扩展性**：LLM provider 可通过配置切换，不改代码。

## 7. 验收方式

- 单元测试覆盖：存储层的增删查改、`SKILL.md` frontmatter 解析与写回、LLM 响应 JSON
  解析容错、LangGraph 图的正常与跳过路径（打桩 LLM，不依赖真实网络）。
- 提供一个**离线可运行的完整示例**（`examples/`）：模拟事件日志 + 示例 `SKILL.md` +
  打桩 LLM 客户端，跑通"采集 → 评估 → 审核 → 写回"全链路，验证架构闭环成立，且不需要
  真实 API Key 或网络访问。这一步对应"沙盒试点"原则：先在隔离环境验证机制成立，再接入
  真实 Hermes 环境和真实 LLM 调用。

## 8. 交付物

代码位于 `repositories/skill-auto-optimizer/`：

```
plugin/                 Hermes Plugin（事件采集）
optimizer/
  config.py             配置加载
  skill_doc.py           SKILL.md frontmatter 解析/写回
  watcher.py             日志轮询 + 触发优化图
  llm/                   Deepseek/豆包客户端 + judge 逻辑
  graph/                 LangGraph 评估优化图
  storage/               SQLite 存储层
cli/review.py            审核 CLI
config/config.yaml       配置文件
examples/                离线可跑的完整示例
tests/                   单元测试（9 项，全部通过）
README.md                安装、配置、部署、运行说明
```

## 9. 后续可能的迭代方向（超出当前范围，仅记录）

- `auto_apply` 全自动落盘模式（跳过人工审核，适合置信度阈值以上的建议）。
- 除 `description` 外，扩展到对 skill 正文（Procedure/Pitfalls）的优化建议。
- 接入 Hermes 的 curator 统计数据（`use_count`/`stale` 状态）作为 judge 的辅助上下文。
