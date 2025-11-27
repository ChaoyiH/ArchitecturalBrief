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
| `full` / `all` | 全案整合（生成完整任务书） |

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
