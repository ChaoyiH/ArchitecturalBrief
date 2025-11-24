# ArchitecturalBrief - 建筑设计规范与任务书 AI 助手

## 📖 项目简介

**ArchitecturalBrief** 是一个基于 RAG（检索增强生成）技术的 AI 问答系统，专门服务于建筑设计领域。该项目旨在帮助建筑师、规划师及相关从业人员快速检索国家建筑标准（GB 规范）、设计任务书及相关资料，并通过 AI 进行智能解答。

本项目整合了向量数据库（FAISS）与大语言模型，能够理解自然语言提问，并从本地的 Markdown 格式规范库中提取准确依据进行回答。

## ✨ 主要功能

* **智能问答**：支持针对建筑规范、设计要求的自然语言提问。
* **规范检索**：覆盖各类建筑设计防火规范、无障碍设计规范、博物馆/科技馆建设标准等（位于 `data/gb/`）。
* **项目资料库**：支持特定项目的资料查询（位于 `data/zlj/`）。
* **任务书生成**：新增 “Design Concepts” 入口，可基于案例库自动生成设计理念方向。
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
│   ├── archdaily/              # ArchDaily 项目数据 (JSON)
│   ├── china/                  # 国内博物馆数据 (JSON)
│   ├── world/                  # 国际博物馆数据 (JSON)
│   ├── gb/                     # 国家标准规范 (Markdown)
│   └── zlj/                    # 设计资料集与图片资源 (Markdown + Images)
├── MIGRATION_SUMMARY.md        # 迁移总结文档
├── COMPARISON.md               # 版本对比文档
├── QUICKSTART.md               # 快速开始指南
└── README.md                   # 项目说明文档
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

### 6. 任务书生成（Design Concepts 第一阶段）

设计理念模块拥有独立入口（与 `main.py` 平行）并内置语义字段抽取、向量检索与 JSON Prompt。

```bash
# 进入 code 目录后运行
python design_generator.py \
  --project-name "海洋科技馆" \
  --project-features "滨海选址，强调生态教育与沉浸式互动" \
  --show-contexts

# 首次运行（或更新资料后）可强制重建索引
python design_generator.py --project-name "科普馆" --project-features "山区，低碳" --rebuild-index --dry-run
```

常用参数：

- `--project-name` / `--project-features`：描述当前任务书需求。
- `--query`：自定义检索语句（默认使用特征描述）。
- `--top-k`、`--min-area`、`--max-area`、`--category`：语料筛选器。
- `--dry-run`：只输出 Prompt 与参考案例，不调用 LLM。
- `--show-contexts`：打印检索到的案例摘要，便于调试。
- `--rebuild-index`：重建 `code/vector_index/design_concepts/` 中的设计理念索引。

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


## 数据格式说明

以下是 `data` 文件夹中各子目录数据文件的数据模式（Schema）。本项目包含结构化（JSON）和非结构化（Markdown）两类数据。在处理 RAG 检索或代码生成时，请参考以下数据模式：

-----

### 结构化数据 (JSON)
* **archdaily/**: 包含项目 ID、标题、描述、关键区域（如设计理念、展览区）的文本提取。
* **china/ & world/**: 包含博物馆的基础建设指标（面积、层数、高度）及各功能区的详细介绍。

### 非结构化数据 (Markdown)
* **gb/** (标准规范):
    * 层级严格：H1(标题) -> H2(章) -> H3(节) -> H4(条文号)。
    * 包含 LaTeX 公式和 Markdown 表格。
* **zlj/** (设计资料集):
    * 层级结构：H1(主题) -> H2(模块) -> H3(知识点) -> H4(图表)。
    * 包含 HTML/Markdown 混合表格及本地图片引用。

-----

### 1\. `data/archdaily/` (ArchDaily 项目数据)

该目录下的文件为 `.json` 格式，主要包含从 ArchDaily 爬取的博物馆建筑项目详情，侧重于建筑设计说明及功能分区的提取。

**JSON 数据模式:**

```json
{
  "Project ID": "String (项目唯一标识ID)",
  "Project Title": "String (项目名称)",
  "Categories": "List[String] (项目类别，如 Museum, Research Center)",
  "City": "String (城市)",
  "Country": "String (国家)",
  "Architects": "List[String] (建筑师/事务所列表)",
  "Area": "String (建筑面积，包含单位)",
  "Year": "String (年份)",
  "Project URL": "String (原文链接)",
  "Description": "List[String] (项目描述，每一项为一段文本)",
  "设计理念": "String (提取出的设计理念描述)",
  "陈列展览区": "String (提取出的展览空间相关描述)",
  "公共服务区": "String (提取出的公共服务空间描述，如入口、休息区)",
  "业务科研用房": "String (提取出的办公与科研空间描述)",
  "藏品库区": "String (提取出的藏品存储空间描述)",
  "综合大厅/中庭": "String (提取出的中庭/大厅空间描述)",
  "特效影厅": "String (提取出的影院/剧场相关描述)",
  "科教活动": "String (提取出的教育活动空间描述)",
  "建筑形态特征": "String (提取出的建筑外观与形态特征描述)",
  "images": [
    {
      "filename": "String (图片文件名)",
      "tags": "List[String] (图片标签，如 Interior Photography, Facade)",
      "caption": "String (图片版权或说明)"
    }
  ]
}
```

### 2\. `data/china/` (国内博物馆数据)

该目录下的文件为 `.json` 格式，主要包含中国国内博物馆的详细建设指标与功能介绍。

**JSON 数据模式:**

```json
{
  "name": "String (博物馆中文名称)",
  "english_name": "String (博物馆英文名称)",
  "opening_date": "String (开馆日期/分期开放时间)",
  "total_construction_area_sqm": "String (总建筑面积，通常为数字字符串)",
  "floors_above_ground": "String (地上层数)",
  "floors_under_ground": "String (地下层数)",
  "building_height_meters": "String (建筑高度，单位：米)",
  "concept&appearance": "String (设计理念与建筑外观造型描述)",
  "permanent_exhibitions": "String (常设展览内容介绍，通常包含各楼层展厅详情)",
  "central_hall": "String (中央大厅/序厅的空间描述)",
  "special_effects_theaters": "String (特效影厅配置，如IMAX、4D影院等)",
  "science_popularization_activities": "String (科普活动与教育项目介绍)"
}
```

### 3\. `data/world/` (国际博物馆数据)

该目录下的文件为 `.json` 格式，主要包含世界其他国家博物馆的建设指标与功能介绍，结构与国内数据高度相似，但字段名略有不同（如面积单位后缀）。

**JSON 数据模式:**

```json
{
  "name": "String (博物馆名称)",
  "opening_date": "String (开馆日期)",
  "total_construction_area": "String (总建筑面积，通常包含数值和单位文本)",
  "floors_above_ground": "Number/String (地上层数)",
  "floors_under_ground": "Number/String (地下层数)",
  "building_height": "String (建筑高度，通常包含数值和单位文本)",
  "concept&appearance": "String (设计概念与建筑外观描述)",
  "permanent_exhibitions": "String (常设展览详情)",
  "central_hall": "String (中央大厅/中庭描述)",
  "special_effects_theaters": "String (特效影厅设施描述)",
  "science_popularization_activities": "String (科普与教育活动描述)"
}
```

### 4\. `data/gb/` (国家标准规范数据)

该目录下的文件为 `.md` (Markdown) 格式，存储了建筑设计相关的国家标准和规范文本。这些文件遵循严格的层级结构，以模拟标准文档的章节条款。

**Markdown 数据模式 (结构规则):**

  * **文档标题 (H1)**:
      * 格式: `# <标准名称> <标准编号>`
      * 示例: `# 《博物馆建筑设计规范》 JGJ 66-2015`
  * **章 (H2)**:
      * 格式: `## <章号> <章标题>`
      * 示例: `## 1 总 则`
  * **节 (H3)**:
      * 格式: `### <节号> <节标题>`
      * 示例: `### 3.1 选 址`
  * **条 (H4)**:
      * 格式: `#### <条号>`
      * 示例: `#### 3.1.1`
  * **内容元素**:
      * **正文**: 标准条款的具体文本。
      * **列表**: 使用有序列表 (`1.`, `2.`) 或无序列表表示款、项内容。
      * **表格**: 使用 Markdown 标准表格语法 (`| Header | ... |`) 展示数据指标（如面积指标、参数限值）。
      * **公式**: 使用 LaTeX 语法 (`$$...$$`) 表示计算公式（如疏散人数计算）。

### 5\. `data/zlj/` (资料集/知识库数据)

该目录下的文件为 `.md` (Markdown) 格式，整理了不同类型建筑（如博物馆、科技馆等）的通用设计知识、流线组织、功能构成等百科类信息。

**Markdown 数据模式 (结构规则):**

  * **主题标题 (H1)**:
      * 格式: `# <建筑类型名称>`
      * 示例: `# 博物馆`
  * **一级分类 (H2)**:
      * 格式: `## <主要知识模块>`
      * 示例: `## 基本概念`, `## 布局与要求`, `## 陈列展览区`
  * **二级分类 (H3)**:
      * 格式: `### <具体知识点>`
      * 示例: `### 定义`, `### 流线组织`, `### 空间尺度`
  * **三级分类 (H4)**
      * 格式: `### <图表>`
  * **内容元素**:
      * **正文**: 知识点的详细描述。
      * **表格**: 混合使用了 Markdown 表格和 HTML 表格 (`<table>`)，用于展示复杂的分类统计或对比数据（如“博物馆建筑规模分类表”）。
      * **图片**: 使用 Markdown 图片语法 (`![](images/...)`) 引用相关示意图或案例图，图片文件通常位于同级目录的 `images/` 文件夹中。