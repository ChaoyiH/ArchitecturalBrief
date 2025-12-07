# 项目架构文档 (ARCHITECTURE.md)

> ⚠️ **AI 助手提示**：分析本项目代码前，请先阅读此文件以理解模块职责边界。

## 系统概述

本项目提供两种运行模式，共享底层 RAG 基础设施：

| 模式 | 入口文件 | 功能 |
|------|---------|------|
| **问答模式** | `main.py` | 交互式建筑规范问答，基于 GB 标准检索 |
| **任务书生成 / 渲染模式** | `design_generator.py` | 生成或仅渲染《建筑设计任务书》（full 并行，单步顺序） |

两种模式的架构兼容性设计：
- 共享 `core/` 底层基础设施（embedding、检索、LLM 调用）
- 共享 `utils/` 数据处理工具
- 任务书生成扩展了 `pipelines/` 领域专属生成器

---

## 目录结构

```
code/
├── config.py                 # 全局配置与各模块默认参数
├── main.py                   # 📖 问答模式入口
├── design_generator.py       # 📝 任务书生成 / 渲染入口（full 并行，单步顺序）
│
├── core/                     # 🔧 底层基础设施 (两种模式共享)
│   ├── __init__.py
│   ├── embedding_manager.py       # 共享 Embedding 实例管理 (线程安全单例)
│   ├── index_construction.py      # FAISS 向量索引构建
│   ├── retrieval_optimization.py  # 混合检索 + RRF 重排
│   └── generation_integration.py  # LLM 调用封装 (MiniMax/Moonshot)
│
├── pipelines/                # 🚀 业务流水线 (任务书生成模式专用)
│   ├── __init__.py
│   │
│   ├── domains/              # 领域专属生成器 (7 个模块)
│   │   ├── design_concept_pipeline.py    # 设计理念
│   │   ├── central_hub_pipeline.py       # 综合大厅/中庭
│   │   ├── exhibition_pipeline.py        # 展览空间
│   │   ├── special_theater_pipeline.py   # 特效影院
│   │   ├── science_education_pipeline.py # 科教活动
│   │   ├── public_service_pipeline.py    # 公共服务区
│   │   └── business_research_pipeline.py # 业务科研区
│   │
│   ├── orchestration/        # 编排与整合
│   │   ├── brief_assembly_pipeline.py    # 确定性 Markdown 拼接 (无 LLM)
│   │   └── json_renderer.py              # 策略化 JSON→Markdown 渲染
│   │
│   └── graph_engine/         # LangGraph 并行执行引擎
│       ├── __init__.py
│       ├── state.py          # TypedDict 全局状态定义
│       ├── nodes.py          # 异步节点包装器 (适配器模式)
│       └── graph.py          # StateGraph 图构建与编译
│
├── utils/                    # 🛠️ 工具模块 (两种模式共享)
│   ├── __init__.py
│   ├── data_preparation.py       # 数据加载、清洗、分块、领域数据抽取
│   ├── indicator_analyzer.py     # 经济技术指标分析 (面积分级 + 案例统计)
│   └── log_setup.py              # 日志与消息记录
│
├── vector_index/             # 持久化向量索引存储
│   ├── index.faiss           # 问答模式向量索引
│   ├── design_concepts/      # 设计理念索引
│   ├── exhibition_space/     # 展览空间索引
│   ├── central_hub/          # 综合大厅索引
│   ├── special_theater/      # 特效影院索引
│   ├── science_education/    # 科教活动索引
│   ├── public_service/       # 公共服务索引
│   └── business_research/    # 业务科研索引
│
└── log/                      # 运行日志存档
```

---

## 模块职责说明

### 底层基础设施 (`core/`)

| 模块 | 职责 |
|------|------|
| `embedding_manager.py` | 线程安全的 HuggingFace Embedding 单例管理器，避免并发初始化冲突 |
| `index_construction.py` | FAISS 向量索引的构建、加载、持久化 |
| `retrieval_optimization.py` | 向量检索 + BM25 混合检索 → RRF 重排序 |
| `generation_integration.py` | LLM 调用封装，支持 MiniMax 和 Moonshot |

### 业务流水线 (`pipelines/`)

| 模块 | 职责 |
|------|------|
| `domains/*_pipeline.py` | 各领域专属生成器：数据抽取 → Prompt 构建 → LLM 调用 → JSON 输出 |
| `orchestration/brief_assembly_pipeline.py` | 确定性模板拼接器：将各模块 JSON 交给 JSONRenderer 渲染后组装为 Markdown 任务书 |
| `orchestration/json_renderer.py` | 策略化 JSON→Markdown 渲染：功能面积配比、案例列表、适配矩阵等表格/卡片化，通用列表自动表格化 |
| `graph_engine/` | LangGraph 并行执行引擎，支持多模块并行（含指标分析）+ 组装汇聚 |

### 工具模块 (`utils/`)

| 模块 | 职责 |
|------|------|
| `data_preparation.py` | Markdown/JSON 加载、语义分块、领域数据抽取器 |
| `indicator_analyzer.py` | 经济技术指标分析：面积解析、分级判定、案例统计与相似项目匹配 |
| `log_setup.py` | 终端日志捕获、对话消息记录 |

---

## 数据流示意

### 问答模式 (`main.py`)

```
用户问题
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ core/retrieval_optimization.py                               │
│  • 向量检索 + BM25 → RRF 混合重排                              │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ core/generation_integration.py                               │
│  • 构建 Prompt → 调用 LLM → 返回回答                           │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
回答输出
```

### 任务书生成 / 渲染模式 (`design_generator.py`)

```
项目参数 (名称、特征)
    │
    ├── 生成模式 (step=full 或逗号分隔步骤)
    │   │
    │   ▼
    │   ┌─────────────────────────────────────────────────────┐
    │   │ pipelines/graph_engine/                              │
    │   │  • LangGraph StateGraph 并行执行多模块 (7 领域 + 指标) │
    │   │  • 所有模块完成后汇聚到 assembly 节点                  │
    │   └─────────────────────────────────────────────────────┘
    │       │
    │       ▼
    │   ┌─────────────────────────────────────────────────────┐
    │   │ pipelines/orchestration/brief_assembly_pipeline.py   │
    │   │  • 调用 JSONRenderer 进行表格/卡片化渲染               │
    │   │  • 确定性模板拼接 (无 LLM，毫秒级)                     │
    │   └─────────────────────────────────────────────────────┘
    │       │
    │       ▼
    │   《项目名_设计任务书.md》 + 同名 .json
    │
    ├── 单模块/少量步骤 (step=design 等)
    │   │  顺序执行对应 pipeline，同步返回 JSON
    │   └── 由 brief_assembly_pipeline 调用 JSONRenderer 输出 Markdown（若需要组装）
    │
    └── 渲染模式 (step=render)
        │  输入：已有模块 JSON（不调用 LLM）
        ▼
    ┌──────────────────────────────────────────────────────┐
    │ pipelines/orchestration/json_renderer.py             │
    │  • 策略化 JSON→Markdown 渲染（表格/卡片/自动表）        │
    └──────────────────────────────────────────────────────┘
        │
        ▼
    《项目名_设计任务书.md》
```

---

## LangGraph 图结构 (默认并行)

```
                    ┌─────────┐
                    │  START  │
                    └────┬────┘
                         │
            ┌───────┬───────┬───────┬───────┬───────┬───────┬───────┐
            ▼       ▼       ▼       ▼       ▼       ▼       ▼       ▼
        ┌────────┐┌────────┐┌────────┐┌────────┐┌────────┐┌────────┐┌────────┐┌────────┐
        │design  ││indica- ││central ││exhibi- ││special ││science ││public  ││business│
        │        ││tors    ││_hub    ││tion    ││_theater││_edu    ││_service││_research│
        └────┬───┘└────┬───┘└────┬───┘└────┬───┘└────┬───┘└────┬───┘└────┬───┘└────┬───┘
            │         │         │         │         │         │         │         │
            └─────────┴─────────┴─────────┴────┬────┴─────────┴─────────┴─────────┘
                                       ▼
                           ┌──────────┐
                           │ assembly │
                           └────┬─────┘
                                ▼
                           ┌─────────┐
                           │   END   │
                           └─────────┘
```

**状态定义** (`graph_engine/state.py`)：
- `BriefGenerationInput`: 项目参数
- `ModuleOutput`: 各模块输出 (prompt, contexts, response)
- `BriefGenerationState`: TypedDict 全局状态，作为"数据总线"

---

## 关键设计决策

### 1. 共享 Embedding 实例

**问题**：并行执行时多个线程同时初始化 HuggingFace Embedding 导致 PyTorch meta tensor 错误。

**解决方案**：`core/embedding_manager.py` 提供线程安全的单例 Embedding 实例，所有 pipeline 共享。

### 2. 确定性任务书组装 + 策略化渲染

**问题**：使用 LLM 重写任务书会丢失上游模块产生的详细数据（面积指标、设备参数）。

**解决方案**：`brief_assembly_pipeline.py` 调用 `JSONRenderer` 做策略化渲染（面积配比、案例卡片、适配矩阵自动表格化），随后确定性拼接为 Markdown，渲染器也支持 `step=render` 复用已有 JSON。

### 3. 适配器模式节点包装

**原则**：不修改原有 Pipeline 内部逻辑。

**实现**：`graph_engine/nodes.py` 使用 `asyncio.to_thread()` 将同步 Pipeline 调用包装为异步节点。

---

## 配置说明

主配置文件：`config.py`

| 配置项 | 说明 |
|--------|------|
| `llm_provider` | LLM 提供商 (`minimax` / `moonshot`) |
| `llm_model` | 模型名称 |
| `embedding_model` | Embedding 模型名称 |
| `top_k` | 检索返回的文档数量 |
| `DEFAULT_*_CONFIG` | 各领域模块的专属配置 |

环境变量：
```bash
MINIMAX_API_KEY=...
MINIMAX_GROUP_ID=...
# 或
MOONSHOT_API_KEY=...
```

---

## 扩展指南

### 添加新的领域模块

1. 在 `pipelines/domains/` 创建 `xxx_pipeline.py`
2. 实现 `XxxGenerator` 类，提供 `.generate()` 方法
3. 在 `config.py` 添加 `XxxConfig` 和 `DEFAULT_XXX_CONFIG`
4. 在 `graph_engine/nodes.py` 添加对应的节点包装器
5. 在 `graph_engine/graph.py` 注册新节点并添加边
6. 在 `brief_assembly_pipeline.py` 的 `SECTION_ORDER` 添加新章节

### 修改图执行顺序

编辑 `graph_engine/graph.py` 的 `add_edge()` 调用即可。LangGraph 支持条件分支（`add_conditional_edges`）。
