# 项目架构文档 (ARCHITECTURE.md)

> ⚠️ **AI 助手提示**：分析本项目代码前，请先阅读此文件以理解模块职责边界。

## 目录结构

```
code/
├── config.py                 # 全局配置与各模块默认参数
├── main.py                   # 传统 RAG 问答系统入口
├── design_generator.py       # 设计任务书生成 CLI 入口
│
├── core/                     # 🔧 底层基础设施 (Infrastructure)
│   ├── __init__.py
│   ├── index_construction.py      # FAISS 向量索引构建
│   ├── retrieval_optimization.py  # 混合检索 + RRF 重排
│   └── generation_integration.py  # LLM 调用封装 (MiniMax/Moonshot)
│
├── pipelines/                # 🚀 业务流水线 (Business Pipelines)
│   ├── __init__.py
│   ├── domains/              # 领域专属生成器
│   │   ├── __init__.py
│   │   ├── design_concept_pipeline.py    # 设计理念
│   │   ├── central_hub_pipeline.py       # 综合大厅/中庭
│   │   ├── exhibition_pipeline.py        # 展览空间
│   │   ├── special_theater_pipeline.py   # 特效影院
│   │   ├── science_education_pipeline.py # 科教活动
│   │   ├── public_service_pipeline.py    # 公共服务区
│   │   └── business_research_pipeline.py # 业务科研区
│   │
│   └── orchestration/        # 编排与整合
│       ├── __init__.py
│       └── brief_assembly_pipeline.py    # 全案任务书整合
│
├── utils/                    # 🛠️ 工具模块 (Utilities)
│   ├── __init__.py
│   ├── data_preparation.py   # 数据加载、清洗、分块、领域数据抽取
│   └── log_setup.py          # 日志与消息记录
│
├── rag_modules/              # ⚠️ [DEPRECATED] 旧模块目录，仅保留兼容导入
│   └── __init__.py
│
├── vector_index/             # 持久化向量索引存储
│   ├── design_concepts/
│   ├── exhibition_space/
│   ├── central_hub/
│   ├── special_theater/
│   ├── science_education/
│   ├── public_service/
│   └── business_research/
│
└── log/                      # 运行日志存档
```

## 模块职责说明

| 目录 | 职责 (Responsibility) |
|------|----------------------|
| `core/` | 提供与具体业务无关的底层能力：向量索引构建、混合检索算法、LLM 调用封装。 |
| `pipelines/domains/` | 每个文件对应一个建筑设计模块（如展览空间、特效影院），包含专属的数据抽取、Prompt 构建和生成逻辑。 |
| `pipelines/orchestration/` | 将多个领域模块的输出聚合，生成完整的《建筑设计任务书》Markdown 文档。 |
| `utils/` | 数据预处理（Markdown 分块、JSON 解析）、日志捕获等可复用工具。 |
| `rag_modules/` | **已废弃**，仅作为旧代码的兼容层，内部重新导出新路径模块。 |

## 数据流示意

```
┌─────────────┐
│  data/      │  原始语料（规范文档、案例 JSON、知识库 MD）
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│ utils/data_preparation.py                                    │
│  • 加载文档 → Markdown 分块 → 语义单元                         │
│  • 领域数据抽取器 (Exhibition/CentralHub/Theater/...)          │
└─────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│ core/index_construction.py                                   │
│  • FAISS 向量索引构建与持久化                                   │
└─────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│ core/retrieval_optimization.py                               │
│  • 向量检索 + BM25 混合 → RRF 重排                             │
└─────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│ pipelines/domains/*_pipeline.py                              │
│  • 构建领域专属 Prompt → 调用 LLM → 返回 JSON                   │
└─────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│ pipelines/orchestration/brief_assembly_pipeline.py           │
│  • 聚合各模块 JSON → 生成 Markdown 任务书                       │
└─────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────┐
│  输出文件    │  {项目名}_设计任务书.md
└─────────────┘
```

## 快速入口

| 命令 | 用途 |
|------|------|
| `python main.py` | 传统 RAG 问答交互 |
| `python design_generator.py --step full ...` | 生成完整设计任务书 |
| `python design_generator.py --step design ...` | 仅生成设计理念 |

---
*Last updated: 2025-11-27*
