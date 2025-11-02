# 📊 系统架构对比：菜单系统 vs 建筑规范系统

## 1. 数据源对比

| 项目 | 原系统（菜单） | 新系统（建筑规范） |
|------|---------------|-------------------|
| 数据路径 | `data/cook/` | `data/gb/` + `data/zlj/` |
| 数据类型 | 菜谱Markdown文件 | 建筑规范Markdown文件 |
| 文档结构 | 菜品名、食材、步骤 | 规范条文、技术要求 |
| 元数据 | 分类、难度、菜品名 | 文档名、来源路径 |

## 2. 配置对比 (config.py)

### 原配置
```python
class RAGConfig:
    data_path: str = "../data/cook"        # 单一路径
    top_k: int = 3                          # 返回3个结果
```

### 新配置
```python
class RAGConfig:
    data_paths: list = None                 # 多路径列表
    top_k: int = 5                          # 返回5个结果
    
    def __post_init__(self):
        if self.data_paths is None:
            self.data_paths = ["../data/gb", "../data/zlj"]
```

## 3. 数据准备对比 (data_preparation.py)

### 移除的功能
```python
# ❌ 删除：菜品分类配置
CATEGORY_MAPPING = {
    'meat_dish': '荤菜',
    'vegetable_dish': '素菜',
    ...
}

# ❌ 删除：难度标签
DIFFICULTY_LABELS = ['非常简单', '简单', ...]

# ❌ 删除：元数据增强
def _enhance_metadata(self, doc):
    doc.metadata['category'] = ...
    doc.metadata['difficulty'] = ...

# ❌ 删除：过滤方法
def filter_documents_by_category(...)
def filter_documents_by_difficulty(...)
```

### 保留的核心功能
```python
# ✅ 保留：智能分块
def chunk_documents(self):
    chunks = self._markdown_header_split()
    # Markdown结构感知分块
    
# ✅ 保留：父子映射
def get_parent_documents(self, child_chunks):
    # 从子块获取完整文档
```

### 新增的适配
```python
# ✅ 新增：多路径支持
def __init__(self, data_paths: list):
    self.data_paths = data_paths

# ✅ 新增：doc_name元数据
doc.metadata['doc_name'] = md_file.stem
```

## 4. 检索优化对比 (retrieval_optimization.py)

### 删除的唯一功能
```python
# ❌ 删除：元数据过滤检索
def metadata_filtered_search(self, query: str, filters: Dict):
    docs = self.hybrid_search(query, top_k * 3)
    # 应用元数据过滤...
    return filtered_docs
```

### 完全保留的功能
```python
# ✅ 保留：混合检索
def hybrid_search(self, query: str, top_k: int):
    vector_docs = self.vector_retriever.get_relevant_documents(query)
    bm25_docs = self.bm25_retriever.get_relevant_documents(query)
    reranked_docs = self._rrf_rerank(vector_docs, bm25_docs)
    return reranked_docs[:top_k]

# ✅ 保留：RRF重排
def _rrf_rerank(self, vector_docs, bm25_docs, k=60):
    # RRF算法融合两种检索结果
    doc_scores = {}
    for rank, doc in enumerate(vector_docs):
        rrf_score = 1.0 / (k + rank + 1)
        doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score
    # 按分数排序...
```

## 5. 提示词对比 (generation_integration.py)

### 原提示词（菜单系统）
```python
prompt = """
你是一位专业的烹饪助手。请根据以下食谱信息回答用户的问题。

用户问题: {question}

相关食谱信息:
{context}

请提供详细、实用的回答。如果信息不足，请诚实说明。
"""
```

### 新提示词（建筑规范系统）
```python
prompt = """
你是一个专业的中国建筑设计顾问。请严格根据以下提供的【建筑设计规范条文和技术资料】来回答用户的问题。

你的回答必须：
1. 严谨、准确，完全基于所提供的上下文
2. 引用具体的规范条文编号
3. 如果上下文中找不到答案，请明确指出

用户问题: {question}

【建筑设计规范条文和技术资料】:
{context}
"""
```

### 查询重写示例对比

| 原系统（菜单） | 新系统（建筑规范） |
|---------------|-------------------|
| "做菜" → "简单易做的家常菜谱" | "科技馆" → "科技馆建筑设计规范要求" |
| "川菜" → "经典川菜菜谱" | "防火" → "建筑防火设计规范" |
| "宫保鸡丁怎么做" → 保持不变 | "博物馆疏散宽度" → 保持不变 |

## 6. 主程序对比 (main.py)

### 删除的方法
```python
# ❌ 删除：元数据过滤提取
def _extract_filters_from_query(self, query: str):
    filters = {}
    for cat in category_keywords:
        if cat in query:
            filters['category'] = cat
    return filters

# ❌ 删除：分类搜索
def search_by_category(self, category: str, query: str = ""):
    filters = {"category": category}
    docs = self.retrieval_module.metadata_filtered_search(...)

# ❌ 删除：食材列表
def get_ingredients_list(self, dish_name: str):
    ...
```

### 简化的检索逻辑

**原逻辑（复杂）：**
```python
# 1. 提取过滤条件
filters = self._extract_filters_from_query(question)

# 2. 根据过滤条件选择检索方式
if filters:
    relevant_chunks = self.retrieval_module.metadata_filtered_search(
        rewritten_query, filters, top_k=self.config.top_k
    )
else:
    relevant_chunks = self.retrieval_module.hybrid_search(
        rewritten_query, top_k=self.config.top_k
    )
```

**新逻辑（简洁）：**
```python
# 始终使用混合检索（V1.5 简化版）
relevant_chunks = self.retrieval_module.hybrid_search(
    rewritten_query, top_k=self.config.top_k
)
```

### 新增功能
```python
# ✅ 新增：命令行参数支持
def main():
    parser = argparse.ArgumentParser(description='建筑规范RAG系统')
    parser.add_argument('--build_index', action='store_true', 
                       help='重建向量索引')
    args = parser.parse_args()
    
    if args.build_index:
        # 删除旧索引并重建
        ...
```

## 7. 用户界面对比

### 欢迎信息

**原系统：**
```
============================================================
🍽️  尝尝咸淡RAG系统 - 交互式问答  🍽️
============================================================
💡 解决您的选择困难症，告别'今天吃什么'的世纪难题！
```

**新系统：**
```
============================================================
🏛️  建筑规范RAG系统 - 交互式问答  🏛️
============================================================
💡 为您提供专业、准确的建筑设计规范查询服务！
```

### 典型查询流程对比

| 步骤 | 原系统（菜单） | 新系统（建筑规范） |
|------|---------------|-------------------|
| 查询 | "宫保鸡丁怎么做" | "科技馆疏散宽度要求" |
| 重写 | 保持不变 | 保持不变（已明确） |
| 过滤 | 尝试提取"川菜"分类 | **跳过**（V1.5简化） |
| 检索 | 混合检索或过滤检索 | **始终混合检索** |
| RRF | ✅ 应用 | ✅ 应用 |
| 结果 | 找到3个菜品块 | 找到5个规范块 |
| 回答 | 烹饪步骤 | 规范条文引用 |

## 8. 代码统计对比

| 文件 | 原代码行数 | 新代码行数 | 变化 |
|------|----------|----------|------|
| config.py | 52 | 52 | 重构配置结构 |
| data_preparation.py | 364 | ~290 | -70行（移除元数据处理） |
| retrieval_optimization.py | ~170 | ~120 | -50行（移除过滤方法） |
| generation_integration.py | 402 | 402 | 仅修改提示词 |
| main.py | 370 | ~315 | -55行（移除过滤逻辑） |
| **总计** | **~1358** | **~1179** | **-179行** |

**代码简化率**: 约 13.2%

## 9. 核心架构图

### 原系统架构（复杂）
```
用户查询
    ↓
查询路由
    ↓
查询重写
    ↓
提取过滤条件 ←── 分类、难度关键词
    ↓
┌───────────────┐
│ 有过滤条件？   │
└───────┬───────┘
        ├── 是 → 元数据过滤检索 → 混合检索 → 应用过滤
        └── 否 → 混合检索（Vector + BM25）
                    ↓
                RRF重排
                    ↓
                获取父文档
                    ↓
                生成回答
```

### 新系统架构（简洁）
```
用户查询
    ↓
查询路由
    ↓
查询重写
    ↓
混合检索（Vector + BM25）
    ↓
RRF重排
    ↓
获取父文档
    ↓
生成回答（引用规范条文）
```

## 10. 性能与可维护性

| 指标 | 原系统 | 新系统 | 改进 |
|------|--------|--------|------|
| 检索路径 | 2条（过滤/非过滤） | 1条（统一混合） | ✅ 简化50% |
| 代码复杂度 | 高（多条件判断） | 低（单一路径） | ✅ 更易维护 |
| 调试难度 | 高（需判断走哪条路径） | 低（路径确定） | ✅ 易于调试 |
| 扩展性 | 中（需考虑过滤逻辑） | 高（专注检索质量） | ✅ 更易扩展 |
| 数据依赖 | 强（需正确分类） | 弱（仅需内容） | ✅ 降低维护成本 |

## 总结

### ✅ 成功保留的核心价值
1. **混合检索** - Vector + BM25 双重保障
2. **RRF重排** - 智能融合多源结果
3. **智能分块** - Markdown结构感知
4. **父子映射** - 完整上下文检索

### ✅ 成功简化的复杂度
1. **移除元数据过滤** - 从2条检索路径简化为1条
2. **移除分类依赖** - 不再需要人工标注和维护
3. **统一检索接口** - 所有查询统一处理

### ✅ 成功适配的新场景
1. **多数据源** - 支持 gb/ 和 zlj/ 同时检索
2. **专业提示词** - 针对建筑规范优化
3. **规范引用** - 强调条文编号和准确性

**迁移策略**: "最小改动，最大效果" ✅
