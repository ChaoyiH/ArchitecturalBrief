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
