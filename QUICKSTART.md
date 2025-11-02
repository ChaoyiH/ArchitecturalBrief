# 🚀 快速启动指南

## 第一步：删除旧索引

```powershell
# 删除旧的向量索引文件
Remove-Item -Recurse -Force "d:\016_RAG\project\code\vector_index"
```

## 第二步：激活环境并重建索引

```powershell
# 激活conda环境
conda activate cook-rag-1

# 进入代码目录
cd d:\016_RAG\project\code

# 重建索引（这会读取 data/gb/ 和 data/zlj/ 中的所有规范文件）
python main.py --build_index
```

**预期输出：**
```
🔨 重建索引模式
删除旧索引: code\vector_index
🚀 正在初始化RAG系统...
初始化数据准备模块...
...
📊 知识库统计:
   文档总数: X
   文本块数: Y
   文档类型数: Z
✅ 索引构建完成！
```

## 第三步：运行系统

```powershell
# 启动交互式问答系统
python main.py
```

**预期输出：**
```
============================================================
🏛️  建筑规范RAG系统 - 交互式问答  🏛️
============================================================
💡 为您提供专业、准确的建筑设计规范查询服务！
...
```

## 第四步：测试查询

尝试以下问题：

1. **具体规范查询**
   ```
   科技馆的疏散宽度要求是什么？
   ```

2. **规范列表查询**
   ```
   有哪些关于博物馆的规范？
   ```

3. **技术参数查询**
   ```
   防火分区的面积限制
   ```

4. **设计要求查询**
   ```
   展厅层高有什么标准？
   ```

## 🔧 故障排查

### 问题1：找不到数据文件
**症状**: `数据路径不存在: ../data/gb`

**解决**:
```powershell
# 检查数据目录是否存在
ls d:\016_RAG\project\data\gb
ls d:\016_RAG\project\data\zlj
```

### 问题2：API密钥错误
**症状**: `请设置 MOONSHOT_API_KEY 环境变量`

**解决**:
```powershell
# 设置环境变量
$env:MOONSHOT_API_KEY = "your-api-key-here"

# 或者在 .env 文件中设置
# MOONSHOT_API_KEY=your-api-key-here
```

### 问题3：依赖包缺失
**症状**: `Import "langchain_xxx" could not be resolved`

**解决**:
```powershell
# 确保在正确的环境中
conda activate cook-rag-1

# 如果需要，重新安装依赖
pip install -r requirements.txt
```

## ✅ 验证清单

运行后，确认以下功能正常：

- [ ] 能成功加载 `data/gb/` 和 `data/zlj/` 的文档
- [ ] 索引构建没有错误
- [ ] 能正常进行查询
- [ ] 返回的答案引用了正确的规范条文
- [ ] 流式输出工作正常（看到逐字输出）
- [ ] 退出系统正常（输入"退出"或Ctrl+C）

## 🎯 系统特性

### 保留的高级功能：
✅ 混合检索（Vector + BM25）  
✅ RRF重排（智能融合结果）  
✅ Markdown智能分块（保留文档结构）  
✅ 父子文档映射（完整上下文）  
✅ 查询智能重写  
✅ 流式输出  

### 移除的功能：
❌ 元数据过滤（菜品分类、难度）  
❌ 分类搜索  

---

**如有问题，请参考完整文档**: `MIGRATION_SUMMARY.md`
