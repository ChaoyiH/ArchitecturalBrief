# ArchitecturalBrief - 建筑设计规范与任务书 AI 助手

## 📖 项目简介

**ArchitecturalBrief** 是一个基于 RAG（检索增强生成）技术的 AI 问答系统，专门服务于建筑设计领域。该项目旨在帮助建筑师、规划师及相关从业人员快速检索国家建筑标准（GB 规范）、设计任务书及相关资料，并通过 AI 进行智能解答。

本项目整合了向量数据库（FAISS）与大语言模型，能够理解自然语言提问，并从本地的 Markdown 格式规范库中提取准确依据进行回答。

## ✨ 主要功能

* **智能问答**：支持针对建筑规范、设计要求的自然语言提问。
* **规范检索**：覆盖各类建筑设计防火规范、无障碍设计规范、博物馆/科技馆建设标准等（位于 `data/gb/`）。
* **项目资料库**：支持特定项目的资料查询（位于 `data/zlj/`）。
* **RAG 架构**：
    * **数据处理**：自动清洗和切分 Markdown 格式的建筑文档。
    * **向量索引**：使用 FAISS 构建本地向量索引，实现高效检索。
    * **生成整合**：结合检索到的上下文，利用 LLM 生成专业回答。

## 📂 目录结构

```text
ArchitecturalBrief/
├── code/                       # 核心源代码
│   ├── main.py                 # 程序主入口
│   ├── config.py               # 配置文件
│   ├── log_setup.py            # 日志配置
│   ├── requirements.txt        # 项目依赖文件
│   ├── rag_modules/            # RAG 核心模块
│   │   ├── data_preparation.py # 数据预处理
│   │   ├── index_construction.py # 向量索引构建
│   │   ├── retrieval_optimization.py # 检索优化
│   │   └── generation_integration.py # 生成模块
│   └── vector_index/           # 存储生成的 FAISS 索引文件
├── data/                       # 知识库数据
│   ├── gb/                     # 国家标准 (Markdown 格式)
│   └── zlj/                    # 项目资料及图片资源
├── MIGRATION_SUMMARY.md        # 迁移总结文档
├── COMPARISON.md               # 版本对比文档
└── QUICKSTART.md               # 快速开始指南
````

## 🚀 快速开始 (Quick Start)

### 1\. 环境准备

确保您的系统已安装 Python 3.10 或更高版本。

```bash
# 克隆项目到本地 (如果尚未下载)
# git clone [repository_url]

# 进入代码目录
cd code
```

### 2\. 安装依赖

建议使用虚拟环境（venv 或 conda）来管理依赖。

```bash
# 创建并激活虚拟环境 (可选)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 安装项目依赖
pip install -r requirements.txt
```

### 3\. 配置环境变量

在 `code/` 目录下或系统环境变量中配置必要的 API Key（通常是 OpenAI 或其他 LLM 提供商的 Key）。请检查 `code/config.py` 了解具体的变量名称（通常为 `OPENAI_API_KEY`）。

### 4\. 构建索引 (如果是首次运行或数据有更新)

如果 `code/vector_index/` 目录下没有索引文件，或者您添加了新的 Markdown 规范文件到 `data/` 目录，需要重新构建索引：

*注：具体运行方式取决于 `main.py` 的参数设置，以下为通用示例，请根据实际代码逻辑调整。*

```python
# 假设代码中包含构建索引的逻辑
python main.py --build_index
```

*(如果没有命令行参数，请查看 `code/rag_modules/index_construction.py` 直接运行构建脚本)*

### 5\. 启动程序

运行主程序开始进行问答：

```bash
python main.py
```

## 🛠️ 开发与维护

### 数据更新

如果您需要添加新的规范：

1.  将 Markdown 文件放入 `data/gb/` 或相关子目录。
2.  确保文件中图片链接（如有）指向正确路径。
3.  重新运行索引构建脚本以更新向量库。

### 核心模块说明

  * **Data Preparation**: 负责读取 Markdown，去除无关字符，按语义或段落切分文本。
  * **Index Construction**: 将切分后的文本转换为 Embedding 向量并存入 FAISS。
  * **Retrieval**: 计算用户 Query 与本地向量的相似度，召回 Top-K 相关片段。
  * **Generation**: 构建 Prompt，将 Query 和 Context 发送给 LLM。
