# ARCHITECTURE REPORT

## Part 1: 项目目录结构 (Directory Tree)

```text
code/
├── core
│   ├── __init__.py
│   ├── embedding_manager.py
│   ├── generation_integration.py
│   ├── index_construction.py
│   └── retrieval_optimization.py
├── pipelines
│   ├── domains
│   │   ├── __init__.py
│   │   ├── business_research_pipeline.py
│   │   ├── central_hub_pipeline.py
│   │   ├── design_concept_pipeline.py
│   │   ├── exhibition_pipeline.py
│   │   ├── public_service_pipeline.py
│   │   ├── science_education_pipeline.py
│   │   └── special_theater_pipeline.py
│   ├── graph_engine
│   │   ├── __init__.py
│   │   ├── graph.py
│   │   ├── nodes.py
│   │   └── state.py
│   ├── orchestration
│   │   ├── __init__.py
│   │   └── brief_assembly_pipeline.py
│   └── __init__.py
├── utils
│   ├── __init__.py
│   ├── data_preparation.py
│   ├── generate_arch_report.py
│   ├── indicator_analyzer.py
│   └── log_setup.py
├── ARCHITECTURE.md
├── config.py
├── design_generator.py
├── main.py
├── prompts.py
├── requirements.txt
└── 某科技馆项目_设计任务书.md
```

## Part 2: 核心架构层 (Orchestration Layer)

### `code\config.py`

```python
"""RAG系统配置文件"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parent
DATA_ROOT = PROJECT_ROOT / "data"

@dataclass
class RAGConfig:
    """RAG系统配置类"""

    # 路径配置 - 建筑规范数据源
    data_paths: list = None
    index_save_path: str = "./vector_index"
    
    def __post_init__(self):
        """初始化后的处理"""
        if self.data_paths is None:
            gb_path = DATA_ROOT / "gb"
            zlj_path = DATA_ROOT / "zlj"
            self.data_paths = [str(gb_path), str(zlj_path)]

        # 统一索引保存路径为绝对路径，避免因cwd不同导致的问题
        index_path = Path(self.index_save_path)
        if not index_path.is_absolute():
            self.index_save_path = str((CODE_DIR / index_path).resolve())
        else:
            self.index_save_path = str(index_path)

    # 模型配置
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    llm_provider: str = "minimax"
    llm_model: str = "Minimax-M2"
    # llm_provider: str = "moonshot"
    # llm_model: str = "kimi-k2-0711-preview"

    # 检索配置
    top_k: int = 5

    # 生成配置
    temperature: float = 0.1
    max_tokens: int = 8192
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'RAGConfig':
        """从字典创建配置对象"""
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'data_paths': self.data_paths,
            'index_save_path': self.index_save_path,
            'embedding_model': self.embedding_model,
            'llm_model': self.llm_model,
            'top_k': self.top_k,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens
        }

# 默认配置实例
DEFAULT_CONFIG = RAGConfig()


@dataclass
class ExhibitionConfig:
    """陈列展览区生成模块配置"""

    china_data_path: Optional[str] = None
    world_data_path: Optional[str] = None
    archdaily_data_path: Optional[str] = None
    gb_data_path: Optional[str] = None
    zlj_data_path: Optional[str] = None
    index_save_path: Optional[str] = None
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    llm_provider: str = DEFAULT_CONFIG.llm_provider
    llm_model: str = DEFAULT_CONFIG.llm_model
    temperature: float = 0.1
    max_tokens: int = 4096
    top_k: int = 6

    def __post_init__(self):
        if self.china_data_path is None:
            self.china_data_path = str(DATA_ROOT / "china")
        if self.world_data_path is None:
            self.world_data_path = str(DATA_ROOT / "world")
        if self.archdaily_data_path is None:
            self.archdaily_data_path = str(DATA_ROOT / "archdaily")
        if self.gb_data_path is None:
            self.gb_data_path = str(DATA_ROOT / "gb")
        if self.zlj_data_path is None:
            self.zlj_data_path = str(DATA_ROOT / "zlj")
        if self.index_save_path is None:
            self.index_save_path = str((CODE_DIR / "vector_index" / "exhibition_space").resolve())
        else:
            index_path = Path(self.index_save_path)
            if not index_path.is_absolute():
                self.index_save_path = str((CODE_DIR / index_path).resolve())
            else:
                self.index_save_path = str(index_path)


DEFAULT_EXHIBITION_CONFIG = ExhibitionConfig()


@dataclass
class PublicServiceConfig:
    """公共服务区生成模块配置"""

    china_data_path: Optional[str] = None
    world_data_path: Optional[str] = None
    archdaily_data_path: Optional[str] = None
    gb_data_path: Optional[str] = None
    zlj_data_path: Optional[str] = None
    index_save_path: Optional[str] = None
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    llm_provider: str = DEFAULT_CONFIG.llm_provider
    llm_model: str = DEFAULT_CONFIG.llm_model
    temperature: float = 0.1
    max_tokens: int = 4096
    top_k: int = 6

    def __post_init__(self):
        if self.china_data_path is None:
            self.china_data_path = str(DATA_ROOT / "china")
        if self.world_data_path is None:
            self.world_data_path = str(DATA_ROOT / "world")
        if self.archdaily_data_path is None:
            self.archdaily_data_path = str(DATA_ROOT / "archdaily")
        if self.gb_data_path is None:
            self.gb_data_path = str(DATA_ROOT / "gb")
        if self.zlj_data_path is None:
            self.zlj_data_path = str(DATA_ROOT / "zlj")
        if self.index_save_path is None:
            self.index_save_path = str((CODE_DIR / "vector_index" / "public_service").resolve())
        else:
            index_path = Path(self.index_save_path)
            if not index_path.is_absolute():
                self.index_save_path = str((CODE_DIR / index_path).resolve())
            else:
                self.index_save_path = str(index_path)


DEFAULT_PUBLIC_SERVICE_CONFIG = PublicServiceConfig()


@dataclass
class CentralHubConfig:
    """综合大厅/交通枢纽模块配置"""

    china_data_path: Optional[str] = None
    world_data_path: Optional[str] = None
    archdaily_data_path: Optional[str] = None
    gb_data_path: Optional[str] = None
    zlj_data_path: Optional[str] = None
    index_save_path: Optional[str] = None
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    llm_provider: str = DEFAULT_CONFIG.llm_provider
    llm_model: str = DEFAULT_CONFIG.llm_model
    temperature: float = 0.1
    max_tokens: int = 4096
    top_k: int = 6

    def __post_init__(self):
        if self.china_data_path is None:
            self.china_data_path = str(DATA_ROOT / "china")
        if self.world_data_path is None:
            self.world_data_path = str(DATA_ROOT / "world")
        if self.archdaily_data_path is None:
            self.archdaily_data_path = str(DATA_ROOT / "archdaily")
        if self.gb_data_path is None:
            self.gb_data_path = str(DATA_ROOT / "gb")
        if self.zlj_data_path is None:
            self.zlj_data_path = str(DATA_ROOT / "zlj")
        if self.index_save_path is None:
            self.index_save_path = str((CODE_DIR / "vector_index" / "central_hub").resolve())
        else:
            index_path = Path(self.index_save_path)
            if not index_path.is_absolute():
                self.index_save_path = str((CODE_DIR / index_path).resolve())
            else:
                self.index_save_path = str(index_path)


DEFAULT_CENTRAL_HUB_CONFIG = CentralHubConfig()


@dataclass
class SpecialTheaterConfig:
    """特效影院模块配置"""

    china_data_path: Optional[str] = None
    world_data_path: Optional[str] = None
    archdaily_data_path: Optional[str] = None
    gb_data_path: Optional[str] = None
    zlj_data_path: Optional[str] = None
    index_save_path: Optional[str] = None
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    llm_provider: str = DEFAULT_CONFIG.llm_provider
    llm_model: str = DEFAULT_CONFIG.llm_model
    temperature: float = 0.1
    max_tokens: int = 4096
    top_k: int = 6

    def __post_init__(self):
        if self.china_data_path is None:
            self.china_data_path = str(DATA_ROOT / "china")
        if self.world_data_path is None:
            self.world_data_path = str(DATA_ROOT / "world")
        if self.archdaily_data_path is None:
            self.archdaily_data_path = str(DATA_ROOT / "archdaily")
        if self.gb_data_path is None:
            self.gb_data_path = str(DATA_ROOT / "gb")
        if self.zlj_data_path is None:
            self.zlj_data_path = str(DATA_ROOT / "zlj")
        if self.index_save_path is None:
            self.index_save_path = str((CODE_DIR / "vector_index" / "special_theater").resolve())
        else:
            index_path = Path(self.index_save_path)
            if not index_path.is_absolute():
                self.index_save_path = str((CODE_DIR / index_path).resolve())
            else:
                self.index_save_path = str(index_path)


DEFAULT_SPECIAL_THEATER_CONFIG = SpecialTheaterConfig()


@dataclass
class ScienceEducationConfig:
    """科教活动模块配置"""

    china_data_path: Optional[str] = None
    world_data_path: Optional[str] = None
    archdaily_data_path: Optional[str] = None
    gb_data_path: Optional[str] = None
    zlj_data_path: Optional[str] = None
    index_save_path: Optional[str] = None
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    llm_provider: str = DEFAULT_CONFIG.llm_provider
    llm_model: str = DEFAULT_CONFIG.llm_model
    temperature: float = 0.1
    max_tokens: int = 4096
    top_k: int = 6

    def __post_init__(self):
        if self.china_data_path is None:
            self.china_data_path = str(DATA_ROOT / "china")
        if self.world_data_path is None:
            self.world_data_path = str(DATA_ROOT / "world")
        if self.archdaily_data_path is None:
            self.archdaily_data_path = str(DATA_ROOT / "archdaily")
        if self.gb_data_path is None:
            self.gb_data_path = str(DATA_ROOT / "gb")
        if self.zlj_data_path is None:
            self.zlj_data_path = str(DATA_ROOT / "zlj")
        if self.index_save_path is None:
            self.index_save_path = str((CODE_DIR / "vector_index" / "science_education").resolve())
        else:
            index_path = Path(self.index_save_path)
            if not index_path.is_absolute():
                self.index_save_path = str((CODE_DIR / index_path).resolve())
            else:
                self.index_save_path = str(index_path)


DEFAULT_SCIENCE_EDUCATION_CONFIG = ScienceEducationConfig()


@dataclass
class BusinessResearchConfig:
    """业务科研区生成模块配置"""

    china_data_path: Optional[str] = None
    world_data_path: Optional[str] = None
    archdaily_data_path: Optional[str] = None
    gb_data_path: Optional[str] = None
    zlj_data_path: Optional[str] = None
    index_save_path: Optional[str] = None
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    llm_provider: str = DEFAULT_CONFIG.llm_provider
    llm_model: str = DEFAULT_CONFIG.llm_model
    temperature: float = 0.1
    max_tokens: int = 4096
    top_k: int = 6

    def __post_init__(self):
        if self.china_data_path is None:
            self.china_data_path = str(DATA_ROOT / "china")
        if self.world_data_path is None:
            self.world_data_path = str(DATA_ROOT / "world")
        if self.archdaily_data_path is None:
            self.archdaily_data_path = str(DATA_ROOT / "archdaily")
        if self.gb_data_path is None:
            self.gb_data_path = str(DATA_ROOT / "gb")
        if self.zlj_data_path is None:
            self.zlj_data_path = str(DATA_ROOT / "zlj")
        if self.index_save_path is None:
            self.index_save_path = str((CODE_DIR / "vector_index" / "business_research").resolve())
        else:
            index_path = Path(self.index_save_path)
            if not index_path.is_absolute():
                self.index_save_path = str((CODE_DIR / index_path).resolve())
            else:
                self.index_save_path = str(index_path)


DEFAULT_BUSINESS_RESEARCH_CONFIG = BusinessResearchConfig()

@dataclass
class DesignConceptConfig:
    """设计理念生成模块配置"""

    china_data_path: Optional[str] = None
    world_data_path: Optional[str] = None
    archdaily_data_path: Optional[str] = None
    knowledge_bwg_path: Optional[str] = None
    index_save_path: Optional[str] = None
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    llm_provider: str = DEFAULT_CONFIG.llm_provider
    llm_model: str = DEFAULT_CONFIG.llm_model
    temperature: float = 0.1
    max_tokens: int = 2048
    top_k: int = 4

    def __post_init__(self):
        if self.china_data_path is None:
            self.china_data_path = str(DATA_ROOT / "china")
        if self.world_data_path is None:
            self.world_data_path = str(DATA_ROOT / "world")
        if self.archdaily_data_path is None:
            self.archdaily_data_path = str(DATA_ROOT / "archdaily")
        if self.knowledge_bwg_path is None:
            self.knowledge_bwg_path = str(DATA_ROOT / "zlj" / "bwg.md")
        if self.index_save_path is None:
            self.index_save_path = str((CODE_DIR / "vector_index" / "design_concepts").resolve())
        else:
            index_path = Path(self.index_save_path)
            if not index_path.is_absolute():
                self.index_save_path = str((CODE_DIR / index_path).resolve())
            else:
                self.index_save_path = str(index_path)


DEFAULT_DESIGN_CONCEPT_CONFIG = DesignConceptConfig()

```

### `code\pipelines\graph_engine\state.py`

```python
"""
全局状态定义 (State Definition)

此模块定义了 LangGraph 图中流转的全局状态对象。
所有节点都从此状态读取输入、写入输出。

关于并行写入：
- LangGraph 默认对字典进行浅合并 (shallow merge)
- 每个节点只返回增量字段（如 {"exhibition": result}），框架自动合并
- 不同节点写入不同字段，不存在覆盖冲突
- 若多个节点写入同一字段，需自定义 Reducer（本架构已避免此情况）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TypedDict


class BriefGenerationInput(TypedDict, total=False):
    """用户输入的初始参数，用于启动图执行。"""

    project_name: str
    project_features: str
    query: Optional[str]
    top_k: Optional[int]
    rebuild_index: bool
    dry_run: bool
    filters: Optional[Dict[str, Any]]


class ModuleOutput(TypedDict, total=False):
    """单个模块的输出结构。"""

    prompt: Optional[Dict[str, str]]
    contexts: Optional[List[Any]]
    response: Optional[str]
    error: Optional[str]


class BriefGenerationState(TypedDict, total=False):
    """
    全局状态对象 - 贯穿整个图执行的数据总线。

    Attributes:
        input: 用户输入参数（不可变）
        design: 设计理念模块输出
        indicators: 经济技术指标分析模块输出
        central_hub: 综合大厅模块输出
        exhibition: 展览空间模块输出
        special_theater: 特效影院模块输出
        science_education: 科教活动模块输出
        public_service: 公共服务模块输出
        business_research: 业务科研模块输出
        assembled_brief: 最终组装的任务书 Markdown
        errors: 各模块运行错误记录
        execution_log: 执行日志（可选，用于调试）
    """

    # ========== 输入层 ==========
    input: BriefGenerationInput

    # ========== 模块输出层（第一阶段并行） ==========
    design: ModuleOutput
    indicators: ModuleOutput
    central_hub: ModuleOutput
    exhibition: ModuleOutput
    special_theater: ModuleOutput
    science_education: ModuleOutput
    public_service: ModuleOutput
    business_research: ModuleOutput

    # ========== 组装输出层（第二阶段） ==========
    assembled_brief: Optional[str]

    # ========== 元信息 ==========
    errors: Dict[str, str]
    execution_log: List[str]


def create_initial_state(
    project_name: str,
    project_features: str,
    query: Optional[str] = None,
    top_k: Optional[int] = None,
    rebuild_index: bool = False,
    dry_run: bool = False,
    filters: Optional[Dict[str, Any]] = None,
) -> BriefGenerationState:
    """
    工厂函数：创建初始状态对象。

    Args:
        project_name: 项目名称
        project_features: 项目特征描述
        query: 自定义检索查询
        top_k: 检索数量
        rebuild_index: 是否重建索引
        dry_run: 是否仅预览 Prompt
        filters: 检索过滤条件

    Returns:
        初始化好的 BriefGenerationState
    """
    return BriefGenerationState(
        input=BriefGenerationInput(
            project_name=project_name,
            project_features=project_features,
            query=query,
            top_k=top_k,
            rebuild_index=rebuild_index,
            dry_run=dry_run,
            filters=filters,
        ),
        design={},
        indicators={},
        central_hub={},
        exhibition={},
        special_theater={},
        science_education={},
        public_service={},
        business_research={},
        assembled_brief=None,
        errors={},
        execution_log=[],
    )


# 模块名到状态字段的映射（便于动态访问）
MODULE_STATE_KEYS = [
    "design",
    "indicators",
    "central_hub",
    "exhibition",
    "special_theater",
    "science_education",
    "public_service",
    "business_research",
]

# 模块名到任务书章节键的映射
SECTION_KEY_MAP = {
    "design": "concept",
    "indicators": "indicators",
    "central_hub": "central_hub",
    "exhibition": "exhibition",
    "special_theater": "special_theater",
    "science_education": "science_education",
    "public_service": "public_service",
    "business_research": "business_research",
}

```

### `code\pipelines\graph_engine\graph.py`

```python
"""
图构建模块 (Graph Construction)

使用 LangGraph StateGraph 构建任务书生成的异步执行图。

图结构：
    START
      │
      ├──> design ─────────────┐
      ├──> central_hub ────────┤
      ├──> exhibition ─────────┤
      ├──> special_theater ────┼──> assembly ──> END
      ├──> science_education ──┤
      ├──> public_service ─────┤
      └──> business_research ──┘

设计说明：
- 所有领域模块从 START 并行启动
- 所有领域模块完成后，汇聚到 assembly 节点
- assembly 完成后流向 END

使用 LangGraph 的 fanout/fanin 模式实现并行执行。
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from langgraph.graph import END, START, StateGraph

from .nodes import (
    NODE_REGISTRY,
    PARALLEL_NODES,
    assembly_node,
    business_research_node,
    central_hub_node,
    design_node,
    exhibition_node,
    indicator_node,
    public_service_node,
    science_education_node,
    special_theater_node,
)
from .state import BriefGenerationState

logger = logging.getLogger(__name__)


def build_brief_generation_graph() -> StateGraph:
    """
    构建任务书生成的 StateGraph。

    Returns:
        未编译的 StateGraph 实例（需调用 .compile() 获取可执行对象）
    """
    # 创建图，指定状态类型
    graph = StateGraph(BriefGenerationState)

    # 添加所有领域节点
    graph.add_node("design", design_node)
    graph.add_node("indicators", indicator_node)
    graph.add_node("central_hub", central_hub_node)
    graph.add_node("exhibition", exhibition_node)
    graph.add_node("special_theater", special_theater_node)
    graph.add_node("science_education", science_education_node)
    graph.add_node("public_service", public_service_node)
    graph.add_node("business_research", business_research_node)

    # 添加组装节点
    graph.add_node("assembly", assembly_node)

    # === 边定义 ===
    # 从 START 并行分发到所有领域节点
    for node_name in PARALLEL_NODES:
        graph.add_edge(START, node_name)

    # 从所有领域节点汇聚到 assembly
    for node_name in PARALLEL_NODES:
        graph.add_edge(node_name, "assembly")

    # assembly 完成后结束
    graph.add_edge("assembly", END)

    logger.debug("任务书生成图构建完成")
    return graph


def compile_graph(checkpointer=None):
    """
    编译图为可执行应用。

    Args:
        checkpointer: 可选的检查点器，用于状态持久化和恢复

    Returns:
        编译后的 LangGraph 应用对象
    """
    graph = build_brief_generation_graph()

    compile_kwargs: Dict[str, Any] = {}
    if checkpointer is not None:
        compile_kwargs["checkpointer"] = checkpointer

    app = graph.compile(**compile_kwargs)
    logger.info("LangGraph 应用编译完成")
    return app


# 预编译的默认应用实例（无检查点）
# 使用方式: from pipelines.graph_engine.graph import app; await app.ainvoke(state)
_default_app = None


def get_app():
    """
    获取默认的编译后应用实例（懒加载单例）。

    Returns:
        编译后的 LangGraph 应用
    """
    global _default_app
    if _default_app is None:
        _default_app = compile_graph()
    return _default_app


async def run_graph(initial_state: BriefGenerationState) -> BriefGenerationState:
    """
    便捷函数：运行图并返回最终状态。

    Args:
        initial_state: 初始状态字典

    Returns:
        执行完成后的最终状态
    """
    # 在并行任务启动前预加载 embedding 模型
    from .nodes import ensure_embedding_loaded
    ensure_embedding_loaded()
    
    app = get_app()
    result = await app.ainvoke(initial_state)
    return result

```

### `code\pipelines\graph_engine\nodes.py`

```python
"""
节点包装器 (Node Wrappers)

采用适配器模式，将现有的同步 Pipeline 类包装为 LangGraph 兼容的异步节点函数。
每个节点函数：
- 输入：全局状态 (BriefGenerationState)
- 输出：增量字典，仅包含该模块产生的数据

设计原则：
- 不修改原有 Pipeline 内部逻辑
- 使用 asyncio.to_thread 将同步调用转为异步，实现非阻塞并发
- 所有异常被捕获并记录到 errors 字段
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict

from config import (
    DEFAULT_BUSINESS_RESEARCH_CONFIG,
    DEFAULT_CENTRAL_HUB_CONFIG,
    DEFAULT_DESIGN_CONCEPT_CONFIG,
    DEFAULT_EXHIBITION_CONFIG,
    DEFAULT_PUBLIC_SERVICE_CONFIG,
    DEFAULT_SCIENCE_EDUCATION_CONFIG,
    DEFAULT_SPECIAL_THEATER_CONFIG,
)
from pipelines.domains.business_research_pipeline import BusinessResearchGenerator
from pipelines.domains.central_hub_pipeline import CentralHubGenerator
from pipelines.domains.design_concept_pipeline import DesignConceptGenerator
from pipelines.domains.exhibition_pipeline import ExhibitionGenerator
from pipelines.domains.public_service_pipeline import PublicServiceGenerator
from pipelines.domains.science_education_pipeline import ScienceEducationGenerator
from pipelines.domains.special_theater_pipeline import SpecialTheaterGenerator
from pipelines.orchestration.brief_assembly_pipeline import BriefAssemblyPipeline
from utils.indicator_analyzer import analyze_indicators

from .state import BriefGenerationState, ModuleOutput, SECTION_KEY_MAP

logger = logging.getLogger(__name__)

# 预加载标记
_embedding_preloaded = False


def ensure_embedding_loaded() -> None:
    """
    确保 embedding 模型已预加载（在并行任务启动前调用）。
    
    这可以避免多个线程同时初始化 HuggingFaceEmbeddings 导致的
    PyTorch meta tensor 错误。
    """
    global _embedding_preloaded
    if _embedding_preloaded:
        return
    
    from core.embedding_manager import preload_embedding
    logger.info("预加载共享 embedding 模型...")
    preload_embedding()
    _embedding_preloaded = True
    logger.info("embedding 模型预加载完成")


def _parse_json_response(response_text: str | None) -> Any:
    """尝试将模块输出解析为 JSON，失败则返回原始文本。"""
    if not response_text:
        return {}
    text = response_text.strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.debug("模块输出非 JSON，按原始文本处理")
        return text


# =============================================================================
# 设计理念节点
# =============================================================================


def _run_design_sync(state: BriefGenerationState) -> ModuleOutput:
    """同步执行设计理念生成。"""
    inp = state.get("input", {})
    try:
        generator = DesignConceptGenerator(DEFAULT_DESIGN_CONCEPT_CONFIG)
        generator.ensure_index(rebuild=inp.get("rebuild_index", False))
        result = generator.generate(
            project_name=inp.get("project_name", ""),
            project_features=inp.get("project_features", ""),
            query=inp.get("query"),
            top_k=inp.get("top_k"),
            filters=inp.get("filters"),
            dry_run=inp.get("dry_run", False),
        )
        return ModuleOutput(
            prompt=result.get("prompt"),
            contexts=result.get("contexts"),
            response=result.get("response"),
        )
    except Exception as exc:
        logger.exception("设计理念模块执行失败")
        return ModuleOutput(error=str(exc))


async def design_node(state: BriefGenerationState) -> Dict[str, Any]:
    """异步设计理念节点。"""
    logger.info("🎨 开始执行: 设计理念模块")
    output = await asyncio.to_thread(_run_design_sync, state)
    logger.info("🎨 完成: 设计理念模块")
    return {"design": output}


# =============================================================================
# 综合大厅节点
# =============================================================================


def _run_central_hub_sync(state: BriefGenerationState) -> ModuleOutput:
    """同步执行综合大厅生成。"""
    inp = state.get("input", {})
    try:
        generator = CentralHubGenerator(DEFAULT_CENTRAL_HUB_CONFIG)
        result = generator.generate(
            project_name=inp.get("project_name", ""),
            project_features=inp.get("project_features", ""),
            query=inp.get("query"),
            top_k=inp.get("top_k"),
            rebuild_index=inp.get("rebuild_index", False),
            dry_run=inp.get("dry_run", False),
        )
        return ModuleOutput(
            prompt=result.get("prompt"),
            contexts=result.get("contexts"),
            response=result.get("response"),
        )
    except Exception as exc:
        logger.exception("综合大厅模块执行失败")
        return ModuleOutput(error=str(exc))


async def central_hub_node(state: BriefGenerationState) -> Dict[str, Any]:
    """异步综合大厅节点。"""
    logger.info("🏛️ 开始执行: 综合大厅模块")
    output = await asyncio.to_thread(_run_central_hub_sync, state)
    logger.info("🏛️ 完成: 综合大厅模块")
    return {"central_hub": output}


# =============================================================================
# 展览空间节点
# =============================================================================


def _run_exhibition_sync(state: BriefGenerationState) -> ModuleOutput:
    """同步执行展览空间生成。"""
    inp = state.get("input", {})
    try:
        generator = ExhibitionGenerator(DEFAULT_EXHIBITION_CONFIG)
        result = generator.generate(
            project_name=inp.get("project_name", ""),
            project_features=inp.get("project_features", ""),
            query=inp.get("query"),
            top_k=inp.get("top_k"),
            rebuild_index=inp.get("rebuild_index", False),
            dry_run=inp.get("dry_run", False),
        )
        return ModuleOutput(
            prompt=result.get("prompt"),
            contexts=result.get("contexts"),
            response=result.get("response"),
        )
    except Exception as exc:
        logger.exception("展览空间模块执行失败")
        return ModuleOutput(error=str(exc))


async def exhibition_node(state: BriefGenerationState) -> Dict[str, Any]:
    """异步展览空间节点。"""
    logger.info("🖼️ 开始执行: 展览空间模块")
    output = await asyncio.to_thread(_run_exhibition_sync, state)
    logger.info("🖼️ 完成: 展览空间模块")
    return {"exhibition": output}


# =============================================================================
# 特效影院节点
# =============================================================================


def _run_special_theater_sync(state: BriefGenerationState) -> ModuleOutput:
    """同步执行特效影院生成。"""
    inp = state.get("input", {})
    try:
        generator = SpecialTheaterGenerator(DEFAULT_SPECIAL_THEATER_CONFIG)
        result = generator.generate(
            project_name=inp.get("project_name", ""),
            project_features=inp.get("project_features", ""),
            query=inp.get("query"),
            top_k=inp.get("top_k"),
            rebuild_index=inp.get("rebuild_index", False),
            dry_run=inp.get("dry_run", False),
        )
        return ModuleOutput(
            prompt=result.get("prompt"),
            contexts=result.get("contexts"),
            response=result.get("response"),
        )
    except Exception as exc:
        logger.exception("特效影院模块执行失败")
        return ModuleOutput(error=str(exc))


async def special_theater_node(state: BriefGenerationState) -> Dict[str, Any]:
    """异步特效影院节点。"""
    logger.info("🎬 开始执行: 特效影院模块")
    output = await asyncio.to_thread(_run_special_theater_sync, state)
    logger.info("🎬 完成: 特效影院模块")
    return {"special_theater": output}


# =============================================================================
# 科教活动节点
# =============================================================================


def _run_science_education_sync(state: BriefGenerationState) -> ModuleOutput:
    """同步执行科教活动生成。"""
    inp = state.get("input", {})
    try:
        generator = ScienceEducationGenerator(DEFAULT_SCIENCE_EDUCATION_CONFIG)
        result = generator.generate(
            project_name=inp.get("project_name", ""),
            project_features=inp.get("project_features", ""),
            query=inp.get("query"),
            top_k=inp.get("top_k"),
            rebuild_index=inp.get("rebuild_index", False),
            dry_run=inp.get("dry_run", False),
        )
        return ModuleOutput(
            prompt=result.get("prompt"),
            contexts=result.get("contexts"),
            response=result.get("response"),
        )
    except Exception as exc:
        logger.exception("科教活动模块执行失败")
        return ModuleOutput(error=str(exc))


async def science_education_node(state: BriefGenerationState) -> Dict[str, Any]:
    """异步科教活动节点。"""
    logger.info("🔬 开始执行: 科教活动模块")
    output = await asyncio.to_thread(_run_science_education_sync, state)
    logger.info("🔬 完成: 科教活动模块")
    return {"science_education": output}


# =============================================================================
# 公共服务节点
# =============================================================================


def _run_public_service_sync(state: BriefGenerationState) -> ModuleOutput:
    """同步执行公共服务生成。"""
    inp = state.get("input", {})
    try:
        generator = PublicServiceGenerator(DEFAULT_PUBLIC_SERVICE_CONFIG)
        result = generator.generate(
            project_name=inp.get("project_name", ""),
            project_features=inp.get("project_features", ""),
            query=inp.get("query"),
            top_k=inp.get("top_k"),
            rebuild_index=inp.get("rebuild_index", False),
            dry_run=inp.get("dry_run", False),
        )
        return ModuleOutput(
            prompt=result.get("prompt"),
            contexts=result.get("contexts"),
            response=result.get("response"),
        )
    except Exception as exc:
        logger.exception("公共服务模块执行失败")
        return ModuleOutput(error=str(exc))


async def public_service_node(state: BriefGenerationState) -> Dict[str, Any]:
    """异步公共服务节点。"""
    logger.info("🚻 开始执行: 公共服务模块")
    output = await asyncio.to_thread(_run_public_service_sync, state)
    logger.info("🚻 完成: 公共服务模块")
    return {"public_service": output}


# =============================================================================
# 业务科研节点
# =============================================================================


def _run_business_research_sync(state: BriefGenerationState) -> ModuleOutput:
    """同步执行业务科研生成。"""
    inp = state.get("input", {})
    try:
        generator = BusinessResearchGenerator(DEFAULT_BUSINESS_RESEARCH_CONFIG)
        result = generator.generate(
            project_name=inp.get("project_name", ""),
            project_features=inp.get("project_features", ""),
            query=inp.get("query"),
            top_k=inp.get("top_k"),
            rebuild_index=inp.get("rebuild_index", False),
            dry_run=inp.get("dry_run", False),
        )
        return ModuleOutput(
            prompt=result.get("prompt"),
            contexts=result.get("contexts"),
            response=result.get("response"),
        )
    except Exception as exc:
        logger.exception("业务科研模块执行失败")
        return ModuleOutput(error=str(exc))


async def business_research_node(state: BriefGenerationState) -> Dict[str, Any]:
    """异步业务科研节点。"""
    logger.info("📊 开始执行: 业务科研模块")
    output = await asyncio.to_thread(_run_business_research_sync, state)
    logger.info("📊 完成: 业务科研模块")
    return {"business_research": output}


# =============================================================================
# 经济技术指标节点
# =============================================================================


def _extract_target_area(input_data: Dict[str, Any]) -> float:
    """从输入中提取建筑面积，支持多种字段名与类型。"""

    candidates = [
        "target_area",
        "building_area",
        "gross_floor_area",
    ]

    value = None
    for key in candidates:
        if key in input_data and input_data[key] is not None:
            value = input_data[key]
            break

    if value is None:
        return 0.0

    # 兼容字符串/数字
    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, str):
        text = value.strip().replace(",", "")
        try:
            return float(text)
        except ValueError:
            logger.warning("无法解析输入面积字段为数值: %r", value)
            return 0.0

    logger.warning("未知的面积字段类型: %r", type(value))
    return 0.0


def _run_indicators_sync(state: BriefGenerationState) -> ModuleOutput:
    """同步执行经济技术指标分析。"""

    inp = state.get("input", {})
    try:
        target_area = _extract_target_area(inp)
        if target_area <= 0:
            raise ValueError("未提供有效的建筑面积参数（target_area/building_area/gross_floor_area）。")

        result = analyze_indicators(target_area)
        # 将字典结果序列化为 JSON 字符串，方便下游统一处理
        response_text = json.dumps(result, ensure_ascii=False, indent=2)
        return ModuleOutput(response=response_text)
    except Exception as exc:
        logger.exception("经济技术指标模块执行失败")
        return ModuleOutput(error=str(exc))


async def indicator_node(state: BriefGenerationState) -> Dict[str, Any]:
    """异步经济技术指标节点。"""

    logger.info("📐 开始执行: 经济技术指标模块")
    output = await asyncio.to_thread(_run_indicators_sync, state)
    logger.info("📐 完成: 经济技术指标模块")
    return {"indicators": output}


# =============================================================================
# 任务书组装节点
# =============================================================================


def _run_assembly_sync(state: BriefGenerationState) -> str:
    """同步执行任务书组装。"""
    inp = state.get("input", {})

    # 收集各模块输出
    sections: Dict[str, Any] = {}
    for module_key in [
        "design",
        "indicators",
        "central_hub",
        "exhibition",
        "special_theater",
        "science_education",
        "public_service",
        "business_research",
    ]:
        output: ModuleOutput = state.get(module_key, {})
        section_key = SECTION_KEY_MAP.get(module_key, module_key)

        if output.get("error"):
            sections[section_key] = f"生成失败: {output['error']}"
        elif output.get("response"):
            parsed = _parse_json_response(output["response"])
            sections[section_key] = parsed or "(暂无内容)"
        else:
            sections[section_key] = "(暂无内容)"

    assembler = BriefAssemblyPipeline()
    result = assembler.generate_brief(
        project_name=inp.get("project_name", ""),
        project_features=inp.get("project_features", ""),
        sections=sections,
    )
    return result.get("response", "")


async def assembly_node(state: BriefGenerationState) -> Dict[str, Any]:
    """异步任务书组装节点。"""
    logger.info("📝 开始执行: 任务书组装")
    inp = state.get("input", {})

    # 检查是否为 dry_run 模式
    if inp.get("dry_run", False):
        logger.info("📝 Dry-run 模式，跳过组装")
        return {"assembled_brief": "(dry-run 模式，跳过组装)"}

    markdown = await asyncio.to_thread(_run_assembly_sync, state)
    logger.info("📝 完成: 任务书组装")
    return {"assembled_brief": markdown}


# =============================================================================
# 节点注册表（便于图构建时动态引用）
# =============================================================================

NODE_REGISTRY = {
    "design": design_node,
    "indicators": indicator_node,
    "central_hub": central_hub_node,
    "exhibition": exhibition_node,
    "special_theater": special_theater_node,
    "science_education": science_education_node,
    "public_service": public_service_node,
    "business_research": business_research_node,
    "assembly": assembly_node,
}

# 第一阶段并行节点（不依赖其他模块输出）
PARALLEL_NODES = [
    "design",
    "indicators",
    "central_hub",
    "exhibition",
    "special_theater",
    "science_education",
    "public_service",
    "business_research",
]

```

### `code\design_generator.py`

```python
"""Entry point for design concept & exhibition-space generators.

支持两种执行模式：
- sequential (默认): 顺序执行各模块
- parallel: 使用 LangGraph 异步图引擎并行执行
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config import (
    DEFAULT_BUSINESS_RESEARCH_CONFIG,
    DEFAULT_CENTRAL_HUB_CONFIG,
    DEFAULT_DESIGN_CONCEPT_CONFIG,
    DEFAULT_EXHIBITION_CONFIG,
    DEFAULT_SPECIAL_THEATER_CONFIG,
    DEFAULT_SCIENCE_EDUCATION_CONFIG,
    DEFAULT_PUBLIC_SERVICE_CONFIG,
    BusinessResearchConfig,
    CentralHubConfig,
    DesignConceptConfig,
    ExhibitionConfig,
    SpecialTheaterConfig,
    ScienceEducationConfig,
    PublicServiceConfig,
)
from pipelines.orchestration.brief_assembly_pipeline import BriefAssemblyPipeline
from pipelines.domains.business_research_pipeline import BusinessResearchGenerator
from pipelines.domains.central_hub_pipeline import CentralHubGenerator
from pipelines.domains.design_concept_pipeline import DesignConceptGenerator
from pipelines.domains.exhibition_pipeline import ExhibitionGenerator
from pipelines.domains.public_service_pipeline import PublicServiceGenerator
from pipelines.domains.special_theater_pipeline import SpecialTheaterGenerator
from pipelines.domains.science_education_pipeline import ScienceEducationGenerator
from utils.indicator_analyzer import analyze_indicators
from langchain_core.documents import Document

# LangGraph 图引擎（延迟导入以保持向后兼容）
_graph_engine = None

logger = logging.getLogger(__name__)


def _get_graph_engine():
    """延迟加载 LangGraph 图引擎模块。"""
    global _graph_engine
    if _graph_engine is None:
        from pipelines.graph_engine import create_initial_state, run_graph
        _graph_engine = {"create_initial_state": create_initial_state, "run_graph": run_graph}
    return _graph_engine

# 尝试启用统一日志/消息捕获
try:  # pragma: no cover
    from utils.log_setup import setup as _log_setup, record_message as _record_msg

    _log_setup()
except Exception:  # noqa: BLE001
    _record_msg = None  # type: ignore[assignment]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Task brief generator (设计理念 / 展览空间)"
    )
    parser.add_argument("--project-name", required=True, help="项目名称或类型")
    parser.add_argument(
        "--project-features",
        required=True,
        help="项目关键特征或背景描述（例如选址、规模、体验重点等）",
    )
    parser.add_argument("--query", help="自定义检索查询语句（默认使用项目特征）")
    parser.add_argument("--top-k", type=int, default=None, help="检索案例数量")
    parser.add_argument("--min-area", type=float, help="按建筑面积下限过滤")
    parser.add_argument("--max-area", type=float, help="按建筑面积上限过滤")
    parser.add_argument("--category", help="按项目类别过滤")
    parser.add_argument(
        "--target-area",
        type=float,
        default=None,
        help="目标建筑面积 (m²)，用于经济技术指标分析",
    )
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="强制重建所选步骤的向量索引",
    )
    parser.add_argument(
        "--step",
        default="design",
        help=(
            "指定生成阶段，可选 design / central_hub / exhibition / special_theater / science_education / public_service / business_research / both / all / full（全案整合），"
            "或以逗号分隔组合"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅输出提示词与上下文，不调用模型",
    )
    parser.add_argument(
        "--show-contexts",
        action="store_true",
        help="打印检索到的上下文摘要",
    )
    parser.add_argument(
        "--mode",
        default="sequential",
        choices=["sequential", "parallel"],
        help="执行模式：sequential (顺序) 或 parallel (并行，使用 LangGraph)",
    )
    return parser.parse_args()


def _build_filters(args: argparse.Namespace) -> Dict[str, object]:
    filters: Dict[str, object] = {}
    if args.min_area is not None:
        filters["min_area"] = args.min_area
    if args.max_area is not None:
        filters["max_area"] = args.max_area
    if args.category:
        filters["category"] = args.category
    return filters


def _resolve_steps(step_arg: str) -> List[str]:
    if not step_arg:
        return ["design"]

    lowered = step_arg.lower()
    alias_map = {
        "design": ["design"],
        "concept": ["design"],
        "exhibition": ["exhibition"],
        "public_service": ["public_service"],
        "public-service": ["public_service"],
        "service": ["public_service"],
        "central_hub": ["central_hub"],
        "central-hub": ["central_hub"],
        "central": ["central_hub"],
        "atrium": ["central_hub"],
        "hub": ["central_hub"],
        "indicators": ["indicators"],
        "indicator": ["indicators"],
        "area": ["indicators"],
        "data": ["indicators"],
        "special_theater": ["special_theater"],
        "special-theater": ["special_theater"],
        "theater": ["special_theater"],
        "cinema": ["special_theater"],
        "full": ["full"],
        "all": ["full"],
        "science_education": ["science_education"],
        "science-education": ["science_education"],
        "education": ["science_education"],
        "science": ["science_education"],
        "business_research": ["business_research"],
        "business-research": ["business_research"],
        "business": ["business_research"],
        "research": ["business_research"],
        "both": ["design", "exhibition"],
    }

    if lowered in alias_map:
        return alias_map[lowered]

    tokens = [token.strip() for token in lowered.replace("+", ",").split(",") if token.strip()]
    resolved: List[str] = []
    valid_steps = {
        "design",
        "central_hub",
        "exhibition",
        "special_theater",
        "science_education",
        "public_service",
        "business_research",
        "full",
    }
    for token in tokens:
        mapped = alias_map.get(token, [token])
        for item in mapped:
            if item not in valid_steps:
                continue
            if item not in resolved:
                resolved.append(item)
    return resolved or ["design"]


EXECUTION_ORDER = [
    "design",
    "indicators",
    "central_hub",
    "exhibition",
    "special_theater",
    "science_education",
    "public_service",
    "business_research",
]

SECTION_KEY_MAP = {
    "design": "concept",
    "indicators": "indicators",
    "central_hub": "central_hub",
    "exhibition": "exhibition",
    "special_theater": "special_theater",
    "science_education": "science_education",
    "public_service": "public_service",
    "business_research": "business_research",
}

STEP_TITLES = {
    "design": "设计理念建议书",
    "indicators": "经济技术指标分析报告",
    "central_hub": "综合大厅与核心空间策划书",
    "exhibition": "展览空间设计要求",
    "special_theater": "特效影院区空间设计策划书",
    "science_education": "科教活动与空间融合策划书",
    "public_service": "公共服务区空间设计策划书",
    "business_research": "业务科研区空间设计策划书",
    "full": "建筑设计任务书",
}


def _log_request(step: str, project_name: str, project_features: str, query: Optional[str]):
    if '_record_msg' not in globals() or _record_msg is None:
        return
    try:
        _record_msg(
            "user",
            f"{step.title()} brief: {project_name}",
            meta={"features": project_features, "query": query},
        )
    except Exception:  # noqa: BLE001
        pass


def _log_response(step: str, response: str):
    if '_record_msg' not in globals() or _record_msg is None:
        return
    try:
        _record_msg("assistant", f"[{step}] {response}")
    except Exception:  # noqa: BLE001
        pass


def _print_contexts(contexts: List[Document]):
    print("📚 检索上下文摘要:")
    for idx, doc in enumerate(contexts, 1):
        meta = doc.metadata or {}
        area_meta = meta.get('total_area') or meta.get('total_area_num')
        print(
            f"[{idx}] {meta.get('project_name', meta.get('doc_name', '片段'))} | 来源: {meta.get('source_type')} | 面积: {area_meta}"
        )
        snippet = doc.page_content.strip()
        snippet = snippet[:300] + "..." if len(snippet) > 300 else snippet
        print(snippet)
        print("-" * 40)


def _execute_step(step_name: str, args: argparse.Namespace, filters: Dict[str, object]) -> Dict[str, object]:
    if step_name == "design":
        design_config: DesignConceptConfig = DEFAULT_DESIGN_CONCEPT_CONFIG
        design_generator = DesignConceptGenerator(design_config)
        design_generator.ensure_index(rebuild=args.rebuild_index)
        _log_request("design", args.project_name, args.project_features, args.query)
        return design_generator.generate(
            project_name=args.project_name,
            project_features=args.project_features,
            query=args.query,
            top_k=args.top_k,
            filters=filters if filters else None,
            dry_run=args.dry_run,
        )

    if step_name == "indicators":
        # 推导目标面积：优先使用显式传入的 target_area，其次尝试从最小/最大面积取中值
        target_area: Optional[float] = args.target_area
        if target_area is None:
            if args.min_area is not None and args.max_area is not None and args.max_area >= args.min_area:
                target_area = (args.min_area + args.max_area) / 2.0
            elif args.min_area is not None:
                target_area = args.min_area
            elif args.max_area is not None:
                target_area = args.max_area
            else:
                target_area = 0.0

        _log_request("indicators", args.project_name, args.project_features, args.query)
        result = analyze_indicators(float(target_area)) if target_area is not None else {}
        return {"response": json.dumps(result, ensure_ascii=False)}

    if step_name == "central_hub":
        hub_config: CentralHubConfig = DEFAULT_CENTRAL_HUB_CONFIG
        hub_generator = CentralHubGenerator(hub_config)
        _log_request("central_hub", args.project_name, args.project_features, args.query)
        return hub_generator.generate(
            project_name=args.project_name,
            project_features=args.project_features,
            query=args.query,
            top_k=args.top_k,
            rebuild_index=args.rebuild_index,
            dry_run=args.dry_run,
        )

    if step_name == "exhibition":
        exhibition_config: ExhibitionConfig = DEFAULT_EXHIBITION_CONFIG
        exhibition_generator = ExhibitionGenerator(exhibition_config)
        _log_request("exhibition", args.project_name, args.project_features, args.query)
        return exhibition_generator.generate(
            project_name=args.project_name,
            project_features=args.project_features,
            query=args.query,
            top_k=args.top_k,
            rebuild_index=args.rebuild_index,
            dry_run=args.dry_run,
        )

    if step_name == "special_theater":
        theater_config: SpecialTheaterConfig = DEFAULT_SPECIAL_THEATER_CONFIG
        theater_generator = SpecialTheaterGenerator(theater_config)
        _log_request("special_theater", args.project_name, args.project_features, args.query)
        return theater_generator.generate(
            project_name=args.project_name,
            project_features=args.project_features,
            query=args.query,
            top_k=args.top_k,
            rebuild_index=args.rebuild_index,
            dry_run=args.dry_run,
        )

    if step_name == "science_education":
        science_config: ScienceEducationConfig = DEFAULT_SCIENCE_EDUCATION_CONFIG
        science_generator = ScienceEducationGenerator(science_config)
        _log_request("science_education", args.project_name, args.project_features, args.query)
        return science_generator.generate(
            project_name=args.project_name,
            project_features=args.project_features,
            query=args.query,
            top_k=args.top_k,
            rebuild_index=args.rebuild_index,
            dry_run=args.dry_run,
        )

    if step_name == "public_service":
        service_config: PublicServiceConfig = DEFAULT_PUBLIC_SERVICE_CONFIG
        service_generator = PublicServiceGenerator(service_config)
        _log_request("public_service", args.project_name, args.project_features, args.query)
        return service_generator.generate(
            project_name=args.project_name,
            project_features=args.project_features,
            query=args.query,
            top_k=args.top_k,
            rebuild_index=args.rebuild_index,
            dry_run=args.dry_run,
        )

    if step_name == "business_research":
        business_config: BusinessResearchConfig = DEFAULT_BUSINESS_RESEARCH_CONFIG
        business_generator = BusinessResearchGenerator(business_config)
        _log_request("business_research", args.project_name, args.project_features, args.query)
        return business_generator.generate(
            project_name=args.project_name,
            project_features=args.project_features,
            query=args.query,
            top_k=args.top_k,
            rebuild_index=args.rebuild_index,
            dry_run=args.dry_run,
        )

    raise ValueError(f"未知的步骤: {step_name}")


def _parse_json_response(response_text: Optional[str]) -> Any:
    if not response_text:
        return {}
    text = response_text.strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.warning("模块输出非 JSON，按原始文本返回 (前200字): %s", text[:200])
        return text


def _resolve_output_path(project_name: str) -> Path:
    sanitized = "".join((ch if ch not in '<>:"/\\|?*' else "_") for ch in project_name).strip()
    if not sanitized:
        sanitized = "项目"
    filename = f"{sanitized}_设计任务书.md"
    return Path(__file__).resolve().parent / filename


def generate_full_brief(args: argparse.Namespace, filters: Dict[str, object]) -> None:
    """顺序模式生成完整任务书。"""
    if args.dry_run:
        print("⚠️ 全案整合（full）暂不支持 --dry-run，请移除该参数后重试。")
        return

    context_data: Dict[str, Any] = {}
    step_errors: List[Tuple[str, str]] = []

    for step_name in EXECUTION_ORDER:
        title = STEP_TITLES.get(step_name, step_name)
        print(f"➡️ 正在生成 {title}...")
        try:
            result = _execute_step(step_name, args, filters)
        except Exception as exc:  # noqa: BLE001
            logger.exception("模块 %s 生成失败", step_name)
            step_errors.append((title, str(exc)))
            context_data[SECTION_KEY_MAP.get(step_name, step_name)] = f"生成失败: {exc}"
            continue

        response_payload = result.get("response")
        parsed_payload = _parse_json_response(response_payload)
        context_data[SECTION_KEY_MAP.get(step_name, step_name)] = parsed_payload or "(暂无内容)"

    if not context_data:
        print("⚠️ 未获取到任何模块输出，无法整合任务书。")
        return

    assembler = BriefAssemblyPipeline()
    assembly = assembler.generate_brief(args.project_name, args.project_features, context_data)
    markdown = assembly.get("response")
    if not markdown:
        print("⚠️ 整合器未返回内容，请稍后重试。")
        return

    output_path = _resolve_output_path(args.project_name)
    output_path.write_text(markdown, encoding="utf-8")
    print(f"✅ 《{args.project_name} 建筑设计任务书》已生成 -> {output_path}")
    _log_response("full", markdown[:2000])

    if step_errors:
        print("⚠️ 以下模块生成失败，已在任务书中标注：")
        for title, err in step_errors:
            print(f"   - {title}: {err}")


async def generate_full_brief_parallel(args: argparse.Namespace) -> None:
    """
    并行模式生成完整任务书（使用 LangGraph 图引擎）。

    所有领域模块（7个）并行执行，完成后汇聚到组装节点。
    相比顺序模式，可显著减少总执行时间。
    """
    engine = _get_graph_engine()
    create_initial_state = engine["create_initial_state"]
    run_graph = engine["run_graph"]

    print("🚀 启动并行模式（LangGraph 图引擎）")
    start_time = time.time()

    # 创建初始状态（保持签名兼容，Graph 内部从 state.input 中读取 target_area）
    initial_state = create_initial_state(
        project_name=args.project_name,
        project_features=args.project_features,
        query=args.query,
        top_k=args.top_k,
        rebuild_index=args.rebuild_index,
        dry_run=args.dry_run,
    )

    # 注入 target_area 到初始状态的 input 字段，供 indicators 节点使用
    try:
        if isinstance(initial_state, dict):
            input_payload = initial_state.get("input") or {}
            if not isinstance(input_payload, dict):
                input_payload = {}
            input_payload.setdefault("project_name", args.project_name)
            input_payload.setdefault("project_features", args.project_features)
            if args.query is not None:
                input_payload.setdefault("query", args.query)
            input_payload["target_area"] = args.target_area
            initial_state["input"] = input_payload
    except Exception:  # noqa: BLE001
        logger.exception("无法在初始状态中注入 target_area，将继续使用默认图配置")

    # 执行图
    print("⏳ 并行生成所有模块内容...")
    final_state = await run_graph(initial_state)
    elapsed = time.time() - start_time

    # 检查错误
    step_errors: List[Tuple[str, str]] = []
    for module_key in EXECUTION_ORDER:
        output = final_state.get(module_key, {})
        if output.get("error"):
            title = STEP_TITLES.get(module_key, module_key)
            step_errors.append((title, output["error"]))

    # 获取组装结果
    markdown = final_state.get("assembled_brief", "")
    if not markdown:
        print("⚠️ 整合器未返回内容，请稍后重试。")
        return

    output_path = _resolve_output_path(args.project_name)
    output_path.write_text(markdown, encoding="utf-8")
    print(f"✅ 《{args.project_name} 建筑设计任务书》已生成 -> {output_path}")
    print(f"⏱️ 并行执行总耗时: {elapsed:.1f} 秒")
    _log_response("full", markdown[:2000])

    if step_errors:
        print("⚠️ 以下模块生成失败，已在任务书中标注：")
        for title, err in step_errors:
            print(f"   - {title}: {err}")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    args = _parse_args()
    filters = _build_filters(args)

    requested_steps: List[str] = _resolve_steps(args.step)

    if "full" in requested_steps:
        # 全案整合模式：根据 --mode 选择执行方式
        if args.mode == "parallel":
            asyncio.run(generate_full_brief_parallel(args))
        else:
            generate_full_brief(args, filters)
        return

    # 单步/多步模式：暂不支持并行，使用顺序执行
    if args.mode == "parallel":
        print("ℹ️ 并行模式仅支持 --step full，当前自动降级为顺序模式")

    steps: List[str] = [step for step in EXECUTION_ORDER if step in requested_steps]
    if not steps:
        steps = [step for step in requested_steps if step != "full"] or ["design"]

    results: Dict[str, Dict[str, object]] = {}

    for step_name in steps:
        results[step_name] = _execute_step(step_name, args, filters)

    for step_name in steps:
        result = results.get(step_name)
        if not result:
            continue

        contexts = result.get("contexts", [])
        prompt = result.get("prompt")
        response = result.get("response")
        title = STEP_TITLES.get(step_name, step_name)

        print("=" * 80)
        print(f"🎯 项目: {args.project_name} | 步骤: {title}")
        print(f"🧭 特征: {args.project_features}")
        print("=" * 80)

        if args.show_contexts and contexts:
            _print_contexts(contexts)

        if args.dry_run:
            print("🧱 DRY RUN (未调用模型)")
            if prompt:
                print("System Prompt:\n", prompt["system_prompt"])
                print("\nUser Prompt:\n", prompt["user_prompt"])
            continue

        if not response:
            print("⚠️ 未获得模型输出")
            continue

        print(f"🧠 {title} (JSON):\n")
        print(response)
        _log_response(step_name, str(response))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as exc:  # noqa: BLE001
        logger.exception("设计理念生成模块运行失败: %s", exc)
        print(f"系统错误: {exc}")
        sys.exit(1)
```

### `code\pipelines\orchestration\brief_assembly_pipeline.py`

```python
"""
确定性任务书组装模块

将各模块的 JSON/字典输出按固定模板拼接为完整的 Markdown 设计任务书。
不使用 LLM，执行速度为毫秒级，完整保留上游模块产生的所有数据细节。
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class BriefAssemblyPipeline:
    """确定性模板拼接器，将模块输出组装为 Markdown 设计任务书。"""

    # 章节顺序与 key 映射
    SECTION_ORDER = [
        ("concept", "设计理念 (Design Concept)"),
        ("indicators", "经济技术指标 (Technical Indicators)"),
        ("central_hub", "核心枢纽 (Central Hub)"),
        ("exhibition", "展陈体系 (Exhibition)"),
        ("special_theater", "特效影院 (Special Theater)"),
        ("science_education", "科教研学 (Science Education)"),
        ("public_service", "公共服务 (Public Service)"),
        ("business_research", "业务科研与后勤 (Business & Research)"),
    ]

    def __init__(self) -> None:
        """初始化（无需 LLM 配置）。"""
        pass

    @staticmethod
    def _dict_to_markdown(data: Any, level: int = 3) -> str:
        """
        递归将字典/列表/字符串转换为 Markdown 格式文本。

        Args:
            data: 待转换的数据（dict, list, str, 或其他）
            level: 当前标题层级（默认 3，即 ###）

        Returns:
            格式化后的 Markdown 字符串
        """
        if data is None:
            return "_（暂无数据）_\n"

        if isinstance(data, str):
            text = data.strip()
            return f"{text}\n" if text else "_（暂无数据）_\n"

        if isinstance(data, bool):
            return f"{'是' if data else '否'}\n"

        if isinstance(data, (int, float)):
            return f"{data}\n"

        if isinstance(data, list):
            if not data:
                return "_（暂无数据）_\n"
            lines = []
            for item in data:
                if isinstance(item, dict):
                    # 列表中的字典，展开为子项
                    item_md = BriefAssemblyPipeline._dict_to_markdown(item, level + 1)
                    lines.append(item_md)
                elif isinstance(item, str):
                    lines.append(f"- {item.strip()}")
                else:
                    lines.append(f"- {item}")
            return "\n".join(lines) + "\n"

        if isinstance(data, dict):
            if not data:
                return "_（暂无数据）_\n"

            lines = []
            header_prefix = "#" * min(level, 6)  # 最多 6 级标题

            for key, value in data.items():
                # 将 key 格式化为更友好的标题
                title = BriefAssemblyPipeline._format_key_as_title(key)

                if isinstance(value, dict):
                    lines.append(f"{header_prefix} {title}\n")
                    lines.append(BriefAssemblyPipeline._dict_to_markdown(value, level + 1))
                elif isinstance(value, list):
                    lines.append(f"{header_prefix} {title}\n")
                    lines.append(BriefAssemblyPipeline._dict_to_markdown(value, level + 1))
                elif isinstance(value, str) and len(value) > 100:
                    # 长文本单独作为段落
                    lines.append(f"{header_prefix} {title}\n")
                    lines.append(f"{value.strip()}\n")
                else:
                    # 短文本/数字/布尔值作为行内描述
                    formatted_value = BriefAssemblyPipeline._dict_to_markdown(value, level + 1).strip()
                    lines.append(f"**{title}**: {formatted_value}\n")

            return "\n".join(lines) + "\n"

        # 其他类型直接转字符串
        return f"{data}\n"

    @staticmethod
    def _format_key_as_title(key: str) -> str:
        """
        将字典的 key 格式化为更友好的标题。

        Args:
            key: 原始 key（如 snake_case 或 camelCase）

        Returns:
            格式化后的标题
        """
        title = key.replace("_", " ")
        title = title.title()
        return title

    def assemble_generated_content(
        self,
        project_name: str,
        project_features: str,
        context: Dict[str, Any],
    ) -> str:
        """
        将各模块输出按模板拼接为完整的 Markdown 任务书。

        Args:
            project_name: 项目名称
            project_features: 项目特征描述
            context: 各模块输出字典，key 为模块名，value 为该模块生成的内容

        Returns:
            完整的 Markdown 文档字符串
        """
        lines: List[str] = []

        # 标题
        lines.append(f"# {project_name} 建筑设计任务书\n")
        lines.append(f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        lines.append("---\n")

        # 1. 项目概况
        lines.append("## 1. 项目概况\n")
        lines.append(f"{project_features}\n")
        lines.append("")

        # 2-8. 各专业章节
        section_num = 2
        for key, title in self.SECTION_ORDER:
            lines.append(f"## {section_num}. {title}\n")

            section_data = context.get(key)
            if section_data is None:
                lines.append("_（该模块暂无输出）_\n")
            elif isinstance(section_data, str):
                if section_data.startswith("生成失败"):
                    lines.append(f"> ⚠️ {section_data}\n")
                else:
                    lines.append(f"{section_data.strip()}\n")
            else:
                lines.append(self._dict_to_markdown(section_data, level=3))

            lines.append("")
            section_num += 1

        # 尾部
        lines.append("---\n")
        lines.append("*本任务书由 RAG 系统自动生成，仅供参考。*\n")

        return "\n".join(lines)

    def generate_brief(
        self,
        project_name: str,
        project_features: str,
        sections: Dict[str, Any],
    ) -> Dict[str, str]:
        """
        生成设计任务书（接口兼容）。

        Args:
            project_name: 项目名称
            project_features: 项目特征
            sections: 各模块输出字典

        Returns:
            包含 "response" 键的字典，值为生成的 Markdown 文本
        """
        logger.info("开始组装任务书（确定性模板拼接）...")
        markdown = self.assemble_generated_content(project_name, project_features, sections)
        logger.info("任务书组装完成，长度: %d 字符", len(markdown))

        return {
            "response": markdown,
        }

```

## Part 3: 领域业务层 (Domain Layer) - Signatures Only

### `code\pipelines\domains\__init__.py`

(no public classes or functions found)

### `code\pipelines\domains\business_research_pipeline.py`

```python
class BusinessResearchVectorStore:
    # Vector store for business & research knowledge.

  def __init__(...):

  def load(...):

  def build(...):

  def ensure_ready(...):

  def search(...):

class BusinessResearchPromptBuilder:

  def build_prompt(...):

  def _format_context(...):

class BusinessResearchGenerator:
    # Facade for business & research task generation.

  def __init__(...):

  def ensure_index(...):

  def _ensure_llm(...):

  def _expand_queries(...):

  def retrieve_contexts(...):

  def generate(...):

```

### `code\pipelines\domains\central_hub_pipeline.py`

```python
class CentralHubVectorStore:
    # Vector store handler for central hub knowledge.

  def __init__(...):

  def load(...):

  def build(...):

  def ensure_ready(...):

  def search(...):

class CentralHubPromptBuilder:
    # Builds prompts for central hub design using centralized prompt registry.

  def build_prompt(...):

  def _format_context(...):

class CentralHubGenerator:
    # Facade for central hub task generation.

  def __init__(...):

  def ensure_index(...):

  def _ensure_llm(...):

  def _expand_queries(...):

  def retrieve_contexts(...):

  def generate(...):

```

### `code\pipelines\domains\design_concept_pipeline.py`

```python
class DesignConceptDocument:
    # Lightweight container describing extracted concept text and metadata.

  def to_document(...):

class DesignConceptETL:
    # Extracts semantic fields for design concepts from heterogeneous data sources.

  def __init__(...):

  def load_documents(...):

  def _load_structured_json(...):

  def _load_archdaily(...):

  def _load_zlj_sections(...):

  def _extract_section(...):

  def _split_subsections(...):

  def _parse_numeric(...):

class DesignConceptVectorStore:
    # FAISS-backed semantic index for design concept documents.

  def __init__(...):

  def load(...):

  def build(...):

  def ensure_ready(...):

  def search(...):

  def _match_filters(...):

class DesignConceptPromptBuilder:
    # Builds prompts aligned with the design brief requirements.

  def build_prompt(...):

  def _format_context(...):

class DesignConceptGenerator:
    # High-level facade that orchestrates ETL, vector search, and prompt-driven generation.

  def __init__(...):

  def ensure_index(...):

  def _ensure_llm(...):

  def retrieve_contexts(...):

  def generate(...):

```

### `code\pipelines\domains\exhibition_pipeline.py`

```python
class ExhibitionVectorStore:
    # Vector index dedicated to exhibition space knowledge.

  def __init__(...):

  def load(...):

  def build(...):

  def ensure_ready(...):

  def search(...):

class ExhibitionPromptBuilder:
    # Builds prompts for exhibition space design using centralized prompt registry.

  def build_prompt(...):

  def _format_context(...):

class ExhibitionGenerator:
    # High-level facade for exhibition space requirement generation.

  def __init__(...):

  def ensure_index(...):

  def _ensure_llm(...):

  def _expand_queries(...):

  def retrieve_contexts(...):

  def generate(...):

```

### `code\pipelines\domains\public_service_pipeline.py`

```python
class PublicServiceVectorStore:
    # Vector store dedicated to public service area knowledge.

  def __init__(...):

  def load(...):

  def build(...):

  def ensure_ready(...):

  def search(...):

class PublicServicePromptBuilder:

  def build_prompt(...):

  def _format_context(...):

class PublicServiceGenerator:
    # High-level facade for public service area planning.

  def __init__(...):

  def ensure_index(...):

  def _ensure_llm(...):

  def _expand_queries(...):

  def retrieve_contexts(...):

  def generate(...):

```

### `code\pipelines\domains\science_education_pipeline.py`

```python
class ScienceEducationVectorStore:
    # Vector store for science & education knowledge.

  def __init__(...):

  def load(...):

  def build(...):

  def ensure_ready(...):

  def search(...):

class ScienceEducationPromptBuilder:

  def build_prompt(...):

  def _format_context(...):

class ScienceEducationGenerator:
    # Facade for science education planning.

  def __init__(...):

  def ensure_index(...):

  def _ensure_llm(...):

  def _expand_queries(...):

  def retrieve_contexts(...):

  def generate(...):

```

### `code\pipelines\domains\special_theater_pipeline.py`

```python
class SpecialTheaterVectorStore:
    # Vector store wrapper for special theater knowledge.

  def __init__(...):

  def load(...):

  def build(...):

  def ensure_ready(...):

  def search(...):

class SpecialTheaterPromptBuilder:
    # Builds prompts for special theater design using centralized prompt registry.

  def build_prompt(...):

  def _format_context(...):

class SpecialTheaterGenerator:
    # Facade for special theater generation.

  def __init__(...):

  def ensure_index(...):

  def _ensure_llm(...):

  def _expand_queries(...):

  def retrieve_contexts(...):

  def generate(...):

```

### `code\utils\__init__.py`

(no public classes or functions found)

### `code\utils\data_preparation.py`

```python
class DataPreparationModule:
    # 数据准备模块 - 负责数据加载、清洗和预处理

  def __init__(...):
    # 初始化数据准备模块
    # 
    # Args:
    #     data_paths: 数据文件夹路径列表

  def load_documents(...):
    # 加载文档数据
    # 
    # Returns:
    #     加载的文档列表

  def chunk_documents(...):
    # Markdown结构感知分块
    # 
    # Returns:
    #     分块后的文档列表

  def _markdown_header_split(...):
    # 使用Markdown标题分割器进行结构化分割
    # 
    # Returns:
    #     按标题结构分割的文档列表

  def get_statistics(...):
    # 获取数据统计信息
    # 
    # Returns:
    #     统计信息字典

  def export_metadata(...):
    # 导出元数据到JSON文件
    # 
    # Args:
    #     output_path: 输出文件路径

  def get_parent_documents(...):
    # 根据子块获取对应的父文档（智能去重）
    # 
    # Args:
    #     child_chunks: 检索到的子块列表
    # 
    # Returns:
    #     对应的父文档列表（去重，按相关性排序）

class ExhibitionDataExtractor:
    # 基于语义字段提取展览空间相关片段的处理器 (KnowledgeBaseProcessor)。

  def __init__(...):

  def load_documents(...):

  def _load_structured_json(...):

  def _collect_json_fields(...):

  def _load_archdaily(...):

  def _load_gb_markdown(...):

  def _load_zlj_markdown(...):

  def _iter_sections(...):

class PublicServiceDataExtractor:
    # 公共服务区数据抽取器。

  def __init__(...):

  def load_documents(...):

  def _load_structured_json(...):

  def _collect_public_sections(...):

  def _extract_description_snippets(...):

  def _load_archdaily(...):

  def _load_gb_markdown(...):

  def _load_zlj_markdown(...):

  def _iter_public_sections(...):

class CentralHubDataExtractor:
    # 综合大厅 / 中庭数据抽取器。

  def __init__(...):

  def load_documents(...):

  def _load_structured_json(...):

  def _collect_field_lines(...):

  def _extract_keyword_snippets(...):

  def _normalize_value(...):

  def _load_archdaily(...):

  def _load_gb_markdown(...):

  def _extract_gb_sections(...):

  def _load_zlj_markdown(...):

  def _iter_bwg_sections(...):

class SpecialTheaterDataExtractor:
    # 特效影院数据抽取器。

  def __init__(...):

  def load_documents(...):

  def _load_structured_json(...):

  def _collect_field_lines(...):

  def _extract_snippets(...):

  def _normalize(...):

  def _load_archdaily(...):

  def _load_gb_markdown(...):

  def _extract_gb_sections(...):

  def _load_zlj_markdown(...):

class ScienceEducationDataExtractor:
    # 科教活动数据抽取器。

  def __init__(...):

  def load_documents(...):

  def _load_structured_json(...):

  def _collect_field_lines(...):

  def _extract_description_snippets(...):

  def _normalize(...):

  def _load_markdown(...):

  def _iter_markdown_blocks(...):

class BusinessResearchDataExtractor:
    # 业务科研区数据抽取器。

  def __init__(...):

  def load_documents(...):

  def _load_structured_json(...):

  def _collect_structured_sections(...):

  def _extract_description_snippets(...):

  def _load_archdaily(...):

  def _load_gb_markdown(...):

  def _extract_section(...):

  def _load_zlj_markdown(...):

  def _iter_keyword_sections(...):

```

### `code\utils\indicator_analyzer.py`

```python
def parse_area_value(...):
    # 从各种格式的输入中提取面积数值（单位：平方米）。
    # 
    # 支持的输入格式：
    # - int / float: 直接返回
    # - str: "20,000 sqm", "1.5 hectares", "3500m²", "21367 m²", "100600" 等
    # 
    # 核心约束：
    # - 若包含 "m²" 或 "m2"，严禁将单位中的 "2" 识别为数值的一部分
    # - 使用正则表达式提取第一个连续的数值序列（支持千分位逗号和小数点）
    # - 若无法提取有效数值，返回 None
    # 
    # Args:
    #     value: 任意类型的输入值
    #     
    # Returns:
    #     float: 面积数值（平方米），或 None（无法解析时）

def get_size_class(...):
    # 根据面积判定科技馆等级（依据《科学技术馆建设标准》）。
    # 
    # 分级标准（硬编码）：
    # - 特大型馆：面积 > 40,000 m² (设计使用年限: 100年)
    # - 大型馆：20,000 m² ≤ 面积 ≤ 40,000 m² (设计使用年限: 100年)
    # - 中型馆：8,000 m² ≤ 面积 < 20,000 m² (设计使用年限: 50年)
    # - 小型馆：面积 < 8,000 m² (设计使用年限: 50年)
    # 
    # Args:
    #     area_sqm: 建筑面积（平方米）
    #     
    # Returns:
    #     dict: {
    #         "class_name": str,      # 等级名称
    #         "design_life": int,     # 设计使用年限（年）
    #         "area_range": str,      # 面积范围描述
    #     }

class BuildingCase:
    # 建筑案例数据结构

  def __post_init__(...):
    # 初始化后自动计算等级

def _extract_year(...):
    # 从 JSON 数据中提取年份

def _extract_area(...):
    # 从 JSON 数据中提取面积

def _extract_height(...):
    # 从 JSON 数据中提取建筑总高度（米）。
    # 
    # 支持字段名：Height, height, 建筑高度, building_height 等；
    # 支持单位：m, meter(s), 米；若包含 ft/feet，则自动换算为米 (1 ft = 0.3048 m)。
    # 若解析出的高度 < 3m 或 > 800m，则视为异常值并丢弃。

def _extract_name(...):
    # 从 JSON 数据中提取项目名称

def load_all_cases(...):
    # 加载所有建筑案例数据。
    # 
    # Args:
    #     data_dirs: 数据目录列表，默认为 ["archdaily", "china", "world"]
    #     base_path: 数据根目录，默认为 code/../data
    #     
    # Returns:
    #     List[BuildingCase]: 解析成功的案例列表

def analyze_indicators(...):
    # 根据目标建筑面积执行经济技术指标分析。
    # 
    # 分析内容：
    # 1. 定级：根据《科学技术馆建设标准》判定目标馆等级
    # 2. 同级统计：计算同等级案例的面积均值、最大值、最小值、近年趋势
    # 3. 相似案例匹配：寻找面积最接近的前 K 个案例
    # 
    # Args:
    #     target_area: 目标建筑面积（平方米）
    #     data_dirs: 数据目录列表，默认为 ["archdaily", "china", "world"]
    #     base_path: 数据根目录
    #     top_k_similar: 返回相似案例数量
    #     recent_year_threshold: "近年"定义的起始年份
    #     
    # Returns:
    #     dict: {
    #         "target_area": float,
    #         "target_classification": {
    #             "class_name": str,
    #             "design_life": int,
    #             "area_range": str,
    #         },
    #         "statistics": {
    #             "same_class_count": int,
    #             "mean_area": float,
    #             "max_area": float,
    #             "min_area": float,
    #             "recent_count": int,
    #             "recent_mean_area": float | None,
    #         },
    #         "similar_cases": [
    #             {
    #                 "name": str,
    #                 "area": float,
    #                 "year": int | None,
    #                 "source": str,
    #                 "filepath": str,
    #                 "area_diff": float,
    #             }
    #         ],
    #         "all_cases_count": int,
    #     }

def format_analysis_report(...):
    # 将分析结果格式化为可读的文本报告

def query_building_indicators(...):
    # Run building indicator analysis and return a formatted report.
    # 
    # Useful for retrieving building cases and technical indicators based on area.
    # 
    # This is the primary tool-style entrypoint for agents/graphs:
    # - Takes a target gross floor area (m²)
    # - Classifies the building size per national standards
    # - Computes statistics over similar-scale reference projects
    # - Returns a human-readable Markdown-like text report.

def query_building_indicators(...):
    # 通过建筑面积查询国家标准等级及相似案例的工具。
    # 
    # 该函数作为对外暴露的 Agent / Graph 工具入口：
    # 
    # - 输入目标建筑面积（单位：平方米）
    # - 基于《科学技术馆建设标准》自动判定馆舍等级与设计使用年限
    # - 结合内置案例库，统计同等级项目的面积分布与近年趋势
    # - 返回人类可读的文本报告，便于 LLM 直接引用与总结
    # 
    # Args:
    #     target_area: 目标建筑面积（平方米）。
    # 
    # Returns:
    #     已格式化的经济技术指标分析报告文本。

def _configure_sys_path(...):
    # 确保以模块方式运行时可以正确导入项目内模块。
    # 
    # 支持在项目根目录下执行：
    # 
    #     python -m code.utils.indicator_analyzer 35000

def main(...):
    # 命令行入口：在终端中快速运行指标分析。

```

### `code\utils\log_setup.py`

```python
class _Tee:

  def __init__(...):

  def write(...):

  def flush(...):

  def encoding(...):

  def isatty(...):

  def fileno(...):

def _env_truthy(...):

def setup(...):
    # Start capturing terminal output to a timestamped file.
    # 
    # Args:
    #     enable: Force enable/disable. If None, read from env LOG_CAPTURE. Default: enabled.
    #     log_dir: Target directory for logs. Default: "code/log" beside this file.
    # Returns:
    #     Path to the created log file if active, else None.

def stop(...):
    # Stop capturing and restore original streams.

def is_active(...):
    # Return True if capture is active.

def is_message_active(...):
    # Return True if message capture is active.

def record_message(...):
    # Append an OpenAI-style message to the JSON file if enabled.
    # 
    # Message format: {"role": "user|assistant|system", "content": str, "timestamp": iso, "meta": {...}}

def current_session_dir(...):

def current_log_path(...):

def current_message_path(...):

```
