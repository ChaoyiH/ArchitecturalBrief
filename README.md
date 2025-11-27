# ArchitecturalBrief - 建筑设计规范与任务书 AI 助手

## 📖 项目简介

**ArchitecturalBrief** 是一个基于 RAG（检索增强生成）技术的 AI 系统，专门服务于建筑设计领域。该项目旨在帮助建筑师、规划师及相关从业人员：
- **智能问答**：快速检索国家建筑标准（GB 规范）并通过 AI 进行智能解答
- **任务书生成**：基于案例库自动生成完整的《建筑设计任务书》

本项目整合了向量数据库（FAISS）与大语言模型（MiniMax/Moonshot），能够理解自然语言提问，并从本地的规范库和案例库中提取准确依据。

## ✨ 主要功能

| 功能 | 说明 |
|------|------|
| **智能问答** | 支持针对建筑规范、设计要求的自然语言提问 |
| **规范检索** | 覆盖建筑设计防火规范、无障碍设计规范、博物馆/科技馆建设标准等 |
| **全案任务书生成** | 一键生成完整的设计任务书 Markdown 文档 |
| **分模块生成** | 支持设计理念、展览空间、影院、科教活动等 7 个专项模块 |

## 📂 目录结构

```text
ArchitecturalBrief/
├── code/                           # 核心源代码
│   ├── main.py                     # 传统 RAG 问答系统入口
│   ├── design_generator.py         # 设计任务书生成 CLI 入口
│   ├── config.py                   # 全局配置与模块默认参数
│   │
│   ├── core/                       # 🔧 底层基础设施
│   │   ├── index_construction.py   # FAISS 向量索引构建
│   │   ├── retrieval_optimization.py # 混合检索 + RRF 重排
│   │   └── generation_integration.py # LLM 调用封装
│   │
│   ├── pipelines/                  # 🚀 业务流水线
│   │   ├── domains/                # 领域专属生成器 (7 个模块)
│   │   │   ├── design_concept_pipeline.py
│   │   │   ├── exhibition_pipeline.py
│   │   │   ├── central_hub_pipeline.py
│   │   │   ├── special_theater_pipeline.py
│   │   │   ├── science_education_pipeline.py
│   │   │   ├── public_service_pipeline.py
│   │   │   └── business_research_pipeline.py
│   │   └── orchestration/          # 编排与整合
│   │       └── brief_assembly_pipeline.py
│   │
│   ├── utils/                      # 🛠️ 工具模块
│   │   ├── data_preparation.py     # 数据加载、清洗、分块
│   │   └── log_setup.py            # 日志与消息记录
│   │
│   ├── vector_index/               # 持久化向量索引存储
│   └── log/                        # 运行日志存档
│
├── data/                           # 知识库数据
│   ├── archdaily/                  # ArchDaily 项目数据 (JSON)
│   ├── china/                      # 国内博物馆数据 (JSON)
│   ├── world/                      # 国际博物馆数据 (JSON)
│   ├── gb/                         # 国家标准规范 (Markdown)
│   └── zlj/                        # 设计资料集 (Markdown + Images)
│
├── QUICKSTART.md                   # 快速开始指南
└── README.md                       # 项目说明文档
```

> 📖 更详细的架构说明请参阅 [`code/ARCHITECTURE.md`](code/ARCHITECTURE.md)

## 🚀 快速开始

### 1. 环境准备

确保您的系统已安装 Python 3.10 或更高版本。

```bash
cd code
pip install -r requirements.txt
```

### 2. 配置环境变量

在 `code/` 目录下创建 `.env` 文件或设置系统环境变量：

```bash
# MiniMax (推荐)
MINIMAX_API_KEY=your-api-key
MINIMAX_GROUP_ID=your-group-id

# 或 Moonshot
MOONSHOT_API_KEY=your-api-key
```

### 3. 运行系统

**传统问答模式：**
```bash
python main.py
```

**生成完整设计任务书：**
```bash
python design_generator.py \
  --project-name "海洋科技馆" \
  --project-features "滨海选址，强调生态教育与沉浸式互动" \
  --step full
```

**仅生成单个模块（如设计理念）：**
```bash
python design_generator.py \
  --project-name "科普馆" \
  --project-features "山区，低碳建筑" \
  --step design \
  --dry-run  # 预览 Prompt，不调用 LLM
```

### 4. 常用参数

| 参数 | 说明 |
|------|------|
| `--project-name` | 项目名称 |
| `--project-features` | 项目特征描述（选址、规模、体验重点等） |
| `--step` | 生成阶段：`design` / `exhibition` / `central_hub` / `special_theater` / `science_education` / `public_service` / `business_research` / `full` |
| `--top-k` | 检索案例数量 |
| `--rebuild-index` | 强制重建向量索引 |
| `--dry-run` | 仅输出 Prompt，不调用 LLM |
| `--show-contexts` | 打印检索到的案例摘要 |

## 🛠️ 开发与维护

### 数据更新

如果您需要添加新的规范或案例：

1. 将 Markdown/JSON 文件放入对应的 `data/` 子目录
2. 运行时添加 `--rebuild-index` 参数以更新向量索引

### 核心模块说明

| 模块 | 职责 |
|------|------|
| `core/` | 底层基础设施：向量索引构建、混合检索、LLM 调用封装 |
| `pipelines/domains/` | 各领域专属生成器（展览、影院、科教等） |
| `pipelines/orchestration/` | 全案任务书整合 |
| `utils/` | 数据预处理、日志工具 |

## 📊 数据格式说明

### 结构化数据 (JSON)
- **archdaily/**: ArchDaily 项目数据，包含设计理念、功能分区描述
- **china/ & world/**: 博物馆建设指标（面积、层数、高度）及功能介绍

### 非结构化数据 (Markdown)
- **gb/** (标准规范): 层级结构 H1→H2→H3→H4，包含 LaTeX 公式和表格
- **zlj/** (设计资料集): 知识百科，包含 HTML/Markdown 表格及图片引用

---

*Last updated: 2025-11-27*
