# 🚀 快速启动指南

## 环境配置

### 1. 激活 Conda 环境

```powershell
conda activate cook-rag-1
cd d:\016_RAG\project\code
```

### 2. 配置 API 密钥

```powershell
# MiniMax (推荐)
$env:MINIMAX_API_KEY = "your-api-key"
$env:MINIMAX_GROUP_ID = "your-group-id"

# 或 Moonshot
$env:MOONSHOT_API_KEY = "your-api-key"
```

---

## 使用方式

### 方式一：传统问答系统

```powershell
python main.py
```

支持的问题示例：
- "科技馆的疏散宽度要求是什么？"
- "博物馆建筑设计规范"
- "防火分区面积限制"
- "展厅层高标准"

### 方式二：生成设计任务书

#### 2.1 使用 YAML 配置（推荐）

1. 在 `code` 目录下复制示例配置：

```powershell
cd d:\016_RAG\project\code
copy design_config.example.yaml design_config.yaml
```

2. 按项目修改 `design_config.yaml` 中的：

- `project.name` / `project.features` / `project.target_area`
- `llm.provider` / `llm.model`
- `pipeline.step` / `pipeline.mode` / `pipeline.top_k` 等

3. 使用配置文件生成任务书：

```powershell
python design_generator.py --config design_config.yaml
```

> 命令行参数会覆盖 YAML 中的同名配置，例如：
>
> ```powershell
> python design_generator.py --config design_config.yaml --step full
> ```
>
> 会以 YAML 为基础，仅把步骤强制改为 `full`（默认并行执行）。

#### 2.2 仅渲染已有 JSON（跳过生成）

1. 确认已有 JSON（例如 `济南科技馆_设计任务书.json`）路径。
2. 更新或直接使用 `render_from_json_config.yaml`（`pipeline.step: render`，`pipeline.render_json` 指向 JSON）。

```powershell
python design_generator.py --config render_from_json_config.yaml
```

输出：`code/{项目名}_设计任务书.md`，不在终端打印全文。

#### 2.3 仍使用命令行参数

**生成完整任务书（全案整合）：**
```powershell
python design_generator.py `
  --project-name "海洋科技馆" `
  --project-features "滨海选址，强调生态教育与沉浸式互动" `
  --step full
```

**仅生成单个模块：**
```powershell
# 设计理念
python design_generator.py --project-name "科普馆" --project-features "山区低碳" --step design

# 展览空间
python design_generator.py --project-name "科普馆" --project-features "山区低碳" --step exhibition

# 综合大厅
python design_generator.py --project-name "科普馆" --project-features "山区低碳" --step central_hub

# 仅渲染已有 JSON
python design_generator.py --project-name "科普馆" --project-features "山区低碳" --step render --render-json .\科普馆_设计任务书.json
```

**预览模式（不调用 LLM）：**
```powershell
python design_generator.py `
  --project-name "科普馆" `
  --project-features "山区低碳" `
  --step design `
  --dry-run `
  --show-contexts
```

---

## 常用参数

| 参数 | 说明 |
|------|------|
| `--project-name` | 项目名称（必填） |
| `--project-features` | 项目特征描述（必填） |
| `--step` | 生成阶段，可选值见下表 |
| `--top-k` | 检索案例数量（默认 4） |
| `--rebuild-index` | 强制重建向量索引 |
| `--dry-run` | 仅输出 Prompt，不调用 LLM |
| `--show-contexts` | 打印检索到的案例摘要 |
| `--render-json` | `step=render` 时指定现有 JSON 路径 |

### `--step` 可选值

| 值 | 对应模块 |
|----|---------|
| `design` | 设计理念建议书 |
| `central_hub` | 综合大厅与核心空间策划书 |
| `exhibition` | 展览空间设计要求 |
| `special_theater` | 特效影院区空间设计策划书 |
| `science_education` | 科教活动与空间融合策划书 |
| `public_service` | 公共服务区空间设计策划书 |
| `business_research` | 业务科研区空间设计策划书 |
| `render` | 仅渲染现有 JSON（不调用 LLM） |
| `full` / `all` | 全案整合（生成完整任务书，默认并行） |

### LLM 提供商与模型组合

当前代码支持以下 `--llm-provider` 与模型组合（也可在 `design_config.yaml` 的 `llm` 块中配置）：

| provider | 示例 model | 说明 |
|----------|------------|------|
| `minimax` | `Minimax-M2` | 默认配置，需设置 `MINIMAX_API_KEY` 和 `MINIMAX_GROUP_ID` 环境变量 |
| `moonshot` | `kimi-k2-0711-preview` | 使用 Moonshot/Kimi，需设置 `MOONSHOT_API_KEY` 环境变量 |
| `gemini` | `gemini-3-pro-preview` / `gemini-2.5-pro` | 使用 Google Gemini，需设置 `GEMINI_API_KEY` 环境变量 |

在命令行中使用示例：

```powershell
# MiniMax
python design_generator.py --project-name "科普馆" --project-features "山区低碳" `
  --llm-provider minimax --llm-model Minimax-M2 --step design

# Moonshot
python design_generator.py --project-name "科普馆" --project-features "山区低碳" `
  --llm-provider moonshot --llm-model kimi-k2-0711-preview --step design

# Gemini
python design_generator.py --project-name "济南科技馆" --project-features "在黄河边，以黄河为概念" `
  --llm-provider gemini --llm-model gemini-3-pro-preview --step design
```

---

## 故障排查

### 问题 1：API 密钥错误
**症状**: `请设置 MINIMAX_API_KEY 环境变量`

**解决**:
```powershell
$env:MINIMAX_API_KEY = "your-api-key"
$env:MINIMAX_GROUP_ID = "your-group-id"
```

### 问题 2：依赖包缺失
**症状**: `ModuleNotFoundError`

**解决**:
```powershell
conda activate cook-rag-1
pip install -r requirements.txt
```

### 问题 3：索引文件过旧
**症状**: 检索结果不准确

**解决**:
```powershell
python design_generator.py --project-name "测试" --project-features "测试" --step design --rebuild-index
```

---

## 输出文件

- **任务书**：生成于 `code/{项目名}_设计任务书.md`
- **日志**：存储于 `code/log/<时间戳>/`

---

*更多信息请参阅 [README.md](README.md) 和 [code/ARCHITECTURE.md](code/ARCHITECTURE.md)*
