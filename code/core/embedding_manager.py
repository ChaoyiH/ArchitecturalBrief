"""
Embedding 模型管理器

提供线程安全的单例 embedding 模型实例，避免并发初始化导致的 PyTorch meta tensor 错误。
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

try:
    from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore

logger = logging.getLogger(__name__)

# 线程锁保证单例
_lock = threading.Lock()
_embedding_instances: dict[str, HuggingFaceEmbeddings] = {}


def get_embedding(
    model_name: str = "BAAI/bge-small-zh-v1.5",
    model_kwargs: Optional[dict] = None,
    encode_kwargs: Optional[dict] = None,
) -> HuggingFaceEmbeddings:
    """
    获取共享的 HuggingFaceEmbeddings 实例（线程安全单例）。

    Args:
        model_name: 模型名称
        model_kwargs: 模型参数
        encode_kwargs: 编码参数

    Returns:
        HuggingFaceEmbeddings 实例
    """
    global _embedding_instances

    # 生成缓存键
    cache_key = model_name

    # 先检查是否已存在（无锁快速路径）
    if cache_key in _embedding_instances:
        return _embedding_instances[cache_key]

    # 加锁创建
    with _lock:
        # 双重检查
        if cache_key in _embedding_instances:
            return _embedding_instances[cache_key]

        logger.info("正在初始化共享 embedding 模型: %s", model_name)

        _model_kwargs = model_kwargs or {"device": "cpu"}
        _encode_kwargs = encode_kwargs or {"normalize_embeddings": True}

        embedding = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=_model_kwargs,
            encode_kwargs=_encode_kwargs,
        )

        _embedding_instances[cache_key] = embedding
        logger.info("共享 embedding 模型初始化完成: %s", model_name)

        return embedding


def preload_embedding(model_name: str = "BAAI/bge-small-zh-v1.5") -> None:
    """
    预加载 embedding 模型（在并行任务启动前调用）。

    Args:
        model_name: 模型名称
    """
    get_embedding(model_name)


def clear_cache() -> None:
    """清除所有缓存的 embedding 实例。"""
    global _embedding_instances
    with _lock:
        _embedding_instances.clear()
        logger.info("已清除所有 embedding 缓存")
