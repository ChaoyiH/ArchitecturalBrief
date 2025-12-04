"""
生成集成模块
"""

import asyncio
import importlib
import os
import logging
from typing import Any, List, Sequence

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

class GenerationIntegrationModule:
    """生成集成模块 - 负责LLM集成和回答生成"""
    
    def __init__(
        self,
        provider: str,
        model_name: str,
        temperature: float = 0.1,
        max_tokens: int = 16384,
        timeout: int = 300,
    ) -> None:
        """
        初始化生成集成模块
        
        Args:
            provider: LLM 提供商 (moonshot / minimax / gemini)
            model_name: 模型名称
            temperature: 生成温度
            max_tokens: 最大token数
            timeout: 请求超时时间（秒），默认 300 秒
        """
        self.provider = provider
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.llm = None
        self.setup_llm()
    
    def setup_llm(self):
        """根据provider配置初始化大语言模型 (工厂)"""
        logger.info(f"正在初始化LLM: provider={self.provider}, model={self.model_name}")

        if self.provider == "moonshot":
            api_key = os.getenv("MOONSHOT_API_KEY")
            if not api_key:
                raise ValueError("请为 'moonshot' 设置 MOONSHOT_API_KEY 环境变量")

            self.llm = MoonshotChat(
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
            self.llm = MiniMaxChat(
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                minimax_api_key=api_key,
                minimax_group_id=group_id,
                base_url="https://api.minimax.chat/v1/text/chatcompletion_v2",
                timeout=httpx.Timeout(self.timeout, connect=30.0),
            )

        elif self.provider == "gemini":
            self.llm = GeminiChatModel(
                model_name=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=self.timeout,
            )
            
        else:
            raise ValueError(f"不支持的LLM provider: {self.provider}")
        
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
        prompt = ChatPromptTemplate.from_template("""
你是一个专业的中国建筑设计顾问。请根据以下提供的【建筑设计规范条文和技术资料】来回答用户的问题。

你的回答必须：
1. 现在提供的上下文中寻找答案。
2. 如果上下文中找不到答案，请明确指出，并根据你的知识作答。

用户问题: {question}

【建筑设计规范条文和技术资料】:
{context}

回答:""")

        chain = (
            {"question": RunnablePassthrough(), "context": lambda _: context}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        response = chain.invoke(query)
        return response


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
