"""
生成集成模块
"""

import asyncio
import importlib
import os
import logging
import threading
import time
from typing import Any, List, Optional, Sequence

from google import genai
from google.genai import types

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_community.chat_models.moonshot import MoonshotChat
from langchain_community.chat_models import MiniMaxChat
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
try:  # LangChain >=0.2
    from langchain_core.pydantic_v1 import PrivateAttr  # type: ignore
except ImportError:  # pragma: no cover - fallback for其他版本
    from pydantic import PrivateAttr  # type: ignore

logger = logging.getLogger(__name__)


# ==============================================================================
# Rate Limiter - 令牌桶算法实现
# ==============================================================================

class RateLimiter:
    """
    线程安全的令牌桶速率限制器。
    
    用于控制 API 请求频率，确保不超过 RPM (requests per minute) 限制。
    采用激进策略：允许突发请求以最大化吞吐量。
    """
    
    # 全局单例缓存：provider -> RateLimiter
    _instances: dict = {}
    _lock = threading.Lock()
    
    # 各 provider 的 RPM 限制
    PROVIDER_RPM = {
        "moonshot": 20,   # Kimi
        "minimax": 20,
        "gemini": 25,
        "google": 30,
    }
    
    def __init__(self, provider: str, rpm: Optional[int] = None):
        """
        初始化速率限制器。
        
        Args:
            provider: LLM 提供商名称
            rpm: 每分钟请求数限制，若不指定则使用默认值
        """
        self.provider = provider
        self.rpm = rpm or self.PROVIDER_RPM.get(provider, 20)
        
        # 令牌桶参数 - 使用激进策略
        self.tokens = float(self.rpm)  # 初始满桶，允许突发
        self.max_tokens = float(self.rpm)  # 桶容量 = RPM
        self.refill_rate = self.rpm / 60.0  # 每秒补充令牌数
        self.last_refill = time.monotonic()
        
        # 最小请求间隔（秒）= 60 / RPM，略微放宽 5% 以避免边界问题
        self.min_interval = 60.0 / self.rpm * 0.95
        self.last_request_time = 0.0
        
        self._token_lock = threading.Lock()
        
        logger.info(f"速率限制器初始化: provider={provider}, rpm={self.rpm}, min_interval={self.min_interval:.2f}s")
    
    @classmethod
    def get_instance(cls, provider: str, rpm: Optional[int] = None) -> "RateLimiter":
        """获取或创建指定 provider 的单例速率限制器。"""
        with cls._lock:
            if provider not in cls._instances:
                cls._instances[provider] = cls(provider, rpm)
            return cls._instances[provider]
    
    def _refill(self) -> None:
        """补充令牌桶。"""
        now = time.monotonic()
        elapsed = now - self.last_refill
        tokens_to_add = elapsed * self.refill_rate
        self.tokens = min(self.max_tokens, self.tokens + tokens_to_add)
        self.last_refill = now
    
    def acquire(self, timeout: float = 120.0) -> bool:
        """
        获取一个令牌（阻塞直到可用或超时）。
        
        使用激进策略：
        1. 如果桶中有令牌，立即获取
        2. 确保请求间隔不小于 min_interval
        
        Args:
            timeout: 最大等待时间（秒）
            
        Returns:
            是否成功获取令牌
        """
        deadline = time.monotonic() + timeout
        
        while True:
            with self._token_lock:
                self._refill()
                now = time.monotonic()
                
                # 检查是否需要等待最小间隔
                time_since_last = now - self.last_request_time
                if time_since_last < self.min_interval:
                    wait_for_interval = self.min_interval - time_since_last
                else:
                    wait_for_interval = 0.0
                
                if self.tokens >= 1.0 and wait_for_interval <= 0:
                    self.tokens -= 1.0
                    self.last_request_time = now
                    remaining_tokens = self.tokens
                    logger.debug(f"速率令牌获取成功: provider={self.provider}, 剩余令牌={remaining_tokens:.1f}")
                    return True
                
                # 计算需要等待的时间
                if self.tokens < 1.0:
                    wait_for_token = (1.0 - self.tokens) / self.refill_rate
                else:
                    wait_for_token = 0.0
                
                wait_time = max(wait_for_interval, wait_for_token)
            
            # 检查是否超时
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                logger.warning(f"速率限制器超时: provider={self.provider}")
                return False
            
            # 等待一段时间后重试
            sleep_time = min(wait_time, remaining, 0.1)  # 更频繁检查以响应并发
            time.sleep(sleep_time)
    
    async def acquire_async(self, timeout: float = 120.0) -> bool:
        """异步版本的令牌获取。"""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.acquire(timeout))


# ==============================================================================
# Rate-Limited LLM Wrapper
# ==============================================================================

class RateLimitedChatModel(BaseChatModel):
    """
    带速率限制的 ChatModel 包装器。
    
    在每次调用前获取令牌，确保不超过 RPM 限制。
    """
    
    _inner_llm: BaseChatModel = PrivateAttr()
    _rate_limiter: RateLimiter = PrivateAttr()
    
    def __init__(self, inner_llm: BaseChatModel, rate_limiter: RateLimiter):
        super().__init__()
        self._inner_llm = inner_llm
        self._rate_limiter = rate_limiter
    
    @property
    def _llm_type(self) -> str:
        return f"rate_limited_{self._inner_llm._llm_type}"
    
    def _generate(
        self,
        messages: Sequence[BaseMessage],
        stop: Sequence[str] | None = None,
        **kwargs
    ) -> ChatResult:
        # 获取令牌（会阻塞直到可用）
        if not self._rate_limiter.acquire(timeout=120.0):
            raise RuntimeError(f"速率限制器超时，无法获取调用令牌")
        
        logger.debug(f"已获取速率令牌: provider={self._rate_limiter.provider}")
        return self._inner_llm._generate(messages, stop=stop, **kwargs)
    
    async def _agenerate(
        self,
        messages: Sequence[BaseMessage],
        stop: Sequence[str] | None = None,
        **kwargs
    ) -> ChatResult:
        # 异步获取令牌
        if not await self._rate_limiter.acquire_async(timeout=120.0):
            raise RuntimeError(f"速率限制器超时，无法获取调用令牌")
        
        logger.debug(f"已获取速率令牌 (async): provider={self._rate_limiter.provider}")
        return await self._inner_llm._agenerate(messages, stop=stop, **kwargs)


class GeminiChatModel(BaseChatModel):
    """轻量封装的 Gemini 2.5 Pro ChatModel，兼容 LangChain Runnable 接口。"""

    model_name: str
    temperature: float
    max_tokens: int
    timeout: int

    _client: Any = PrivateAttr()
    _types: Any = PrivateAttr()

    def __init__(
        self,
        model_name: str,
        temperature: float,
        max_tokens: int,
        timeout: int,
    ) -> None:
        super().__init__(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        try:  # 延迟导入，避免在未使用 Gemini 时强依赖
            genai_module = importlib.import_module("google.genai")
            types_module = importlib.import_module("google.genai.types")
        except ImportError as exc:  # pragma: no cover - 运行期才会触发
            raise ImportError(
                "缺少 google-genai 依赖，请先执行 `pip install google-genai`"
            ) from exc

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("请设置 GEMINI_API_KEY 环境变量以使用 Gemini 模式")

        # Gemini API 至少需要 10s deadline；直接使用默认 http 配置以避免 1s 限制
        self._client = genai_module.Client(api_key=api_key)
        self._types = types_module

    @property
    def _llm_type(self) -> str:  # noqa: D401
        return "gemini"

    def _build_config(self):
        return self._types.GenerateContentConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
        )

    def _convert_messages(self, messages: Sequence[BaseMessage]):
        contents = []
        for msg in messages:
            role = "user"
            if msg.type == "ai":
                role = "model"

            parts = []
            payload = msg.content
            if isinstance(payload, str):
                parts.append(self._types.Part.from_text(text=payload))
            elif isinstance(payload, list):
                for chunk in payload:
                    if isinstance(chunk, str):
                        parts.append(self._types.Part.from_text(text=chunk))
                    elif isinstance(chunk, dict) and chunk.get("type") == "text":
                        parts.append(self._types.Part.from_text(text=str(chunk.get("text", ""))))
                    elif isinstance(chunk, dict) and "text" in chunk:
                        parts.append(self._types.Part.from_text(text=str(chunk["text"])) )
            if not parts:
                parts.append(self._types.Part.from_text(text=str(payload)))

            contents.append(self._types.Content(role=role, parts=parts))
        return contents

    def _run_completion(self, messages: Sequence[BaseMessage]) -> str:
        if not messages:
            raise ValueError("提示词为空，无法调用 Gemini")

        contents = self._convert_messages(messages)
        response = self._client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=self._build_config(),
        )
        text = self._extract_text(response)
        if not text:
            raise ValueError("Gemini API 返回空响应，请稍后重试或检查提示词格式")
        return text

    def _extract_text(self, response: Any) -> str:
        return response.text

    def _generate(self, messages: Sequence[BaseMessage], stop: Sequence[str] | None = None, **kwargs) -> ChatResult:
        text = self._run_completion(messages)
        message = AIMessage(content=text)
        generation = ChatGeneration(message=message, text=text)
        return ChatResult(generations=[generation])

    async def _agenerate(self, messages: Sequence[BaseMessage], stop: Sequence[str] | None = None, **kwargs) -> ChatResult:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self._generate(messages, stop=stop, **kwargs))


class GoogleGenAIWrapper:
    """轻量包装 Google GenAI v2 客户端，提供 invoke 接口。"""

    def __init__(self, client: genai.Client, model_name: str, temperature: float, max_tokens: int) -> None:
        self.client = client
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

    def invoke(self, prompt_text: str, config: Optional[types.GenerateContentConfig] = None) -> str:
        cfg = config or types.GenerateContentConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        )
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[prompt_text],
            config=cfg,
        )
        text = getattr(response, "text", None) or ""
        if not text:
            raise ValueError("Google GenAI 返回空响应")
        return text

class GenerationIntegrationModule:
    """生成集成模块 - 负责LLM集成和回答生成"""
    
    def __init__(
        self,
        provider: str,
        model_name: str,
        temperature: float = 0.1,
        max_tokens: int = 16384,
        timeout: int = 300,
        enable_rate_limit: bool = True,
    ) -> None:
        """
        初始化生成集成模块
        
        Args:
            provider: LLM 提供商 (moonshot / minimax / gemini)
            model_name: 模型名称
            temperature: 生成温度
            max_tokens: 最大token数
            timeout: 请求超时时间（秒），默认 300 秒
            enable_rate_limit: 是否启用速率限制，默认启用
        """
        self.provider = provider
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.enable_rate_limit = enable_rate_limit
        self.llm = None
        self.setup_llm()
    
    def setup_llm(self):
        """根据provider配置初始化大语言模型 (工厂)"""
        logger.info(f"正在初始化LLM: provider={self.provider}, model={self.model_name}")

        inner_llm: BaseChatModel

        if self.provider == "moonshot":
            api_key = os.getenv("MOONSHOT_API_KEY")
            if not api_key:
                raise ValueError("请为 'moonshot' 设置 MOONSHOT_API_KEY 环境变量")

            inner_llm = MoonshotChat(
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                moonshot_api_key=api_key
            )
        
        elif self.provider == "minimax":
            group_id = os.getenv("MINIMAX_GROUP_ID")
            api_key = os.getenv("MINIMAX_API_KEY")
            if not group_id or not api_key:
                raise ValueError("请为 'minimax' 设置 MINIMAX_GROUP_ID 和 MINIMAX_API_KEY 环境变量")

            import httpx
            inner_llm = MiniMaxChat(
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                minimax_api_key=api_key,
                minimax_group_id=group_id,
                base_url="https://api.minimax.chat/v1/text/chatcompletion_v2",
                timeout=httpx.Timeout(self.timeout, connect=30.0),
            )

        elif self.provider == "gemini":
            inner_llm = GeminiChatModel(
                model_name=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=self.timeout,
            )

        elif self.provider == "google":
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("请设置 GEMINI_API_KEY 环境变量以使用 Google Gemini")

            client = genai.Client(api_key=api_key)
            # 直接使用原生 SDK，绕过 LangChain 以降低时延
            self.llm = GoogleGenAIWrapper(client=client, model_name=self.model_name, temperature=self.temperature, max_tokens=self.max_tokens)
            logger.info("LLM (google) 初始化完成，已使用原生 genai SDK")
            return

        else:
            raise ValueError(f"不支持的LLM provider: {self.provider}")
        
        # 包装速率限制器
        if self.enable_rate_limit:
            rate_limiter = RateLimiter.get_instance(self.provider)
            self.llm = RateLimitedChatModel(inner_llm, rate_limiter)
            logger.info(f"LLM ({self.provider}) 初始化完成，已启用速率限制 (RPM={rate_limiter.rpm})")
        else:
            self.llm = inner_llm
            logger.info(f"LLM ({self.provider}) 初始化完成")

    
    def generate_basic_answer(self, query: str, context_docs: List[Document]):
        """
        生成基础回答

        Args:
            query: 用户查询s
            context_docs: 上下文文档列表

        Yields:
            生成的回答片段
        """
        context = self._build_context(context_docs)
        logger.debug(f"[RAG] docs={len(context_docs)}, context_len={len(context)}")

        template_text = """
你是一个专业的中国建筑设计顾问。请根据以下提供的【建筑设计规范条文和技术资料】来回答用户的问题。

你的回答必须：
1. 现在提供的上下文中寻找答案。
2. 如果上下文中找不到答案，请明确指出，并根据你的知识作答。

用户问题: {question}

【建筑设计规范条文和技术资料】:
{context}

回答:
"""

        if self.provider == "google":
            prompt_text = template_text.format(question=query, context=context)
            return self._generate_google_answer(prompt_text.strip())

        prompt = ChatPromptTemplate.from_template(template_text)

        chain = (
            {"question": RunnablePassthrough(), "context": lambda _: context}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        response = chain.invoke(query)
        return response

    def _generate_google_answer(self, prompt_text: str) -> str:
        config = types.GenerateContentConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        )
        return self.llm.invoke(prompt_text, config=config)


    def _build_context(self, docs: List[Document], max_length: int = 4000) -> str:
        """
        构建上下文字符串
        
        Args:
            docs: 文档列表
            max_length: 最大长度
            
        Returns:
            格式化的上下文字符串
        """
        if not docs:
            return "暂无相关规范信息。"
        
        context_parts = []
        current_length = 0
        
        for i, doc in enumerate(docs, 1):
            # 添加元数据信息
            metadata_info = f"【文档 {i}】"
            if 'doc_name' in doc.metadata:
                metadata_info += f" {doc.metadata['doc_name']}"
            if 'source' in doc.metadata:
                source_path = doc.metadata['source']
                metadata_info += f" | 来源: {source_path}"
            
            # 构建文档文本
            doc_text = f"{metadata_info}\n{doc.page_content}\n"
            
            # 检查长度限制
            if current_length + len(doc_text) > max_length:
                break
            
            context_parts.append(doc_text)
            current_length += len(doc_text)
        
        return "\n" + "="*50 + "\n".join(context_parts)
