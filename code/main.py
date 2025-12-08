"""
RAG系统主程序
"""

import os
import sys
import logging
from pathlib import Path
from typing import List

# 添加模块路径
sys.path.append(str(Path(__file__).parent))

from dotenv import load_dotenv
# 终端输出捕获与消息记录（独立模块，可用环境变量 LOG_CAPTURE/MSG_CAPTURE 开关，默认开启）
try:
    from utils.log_setup import setup as _log_setup, record_message as _record_msg
    _log_setup()
except Exception:
    # 捕获失败不影响系统运行
    pass
from config import DEFAULT_CONFIG, RAGConfig
from utils.data_preparation import DataPreparationModule
from core import (
    IndexConstructionModule,
    RetrievalOptimizationModule,
    GenerationIntegrationModule
)
# 新增QA入口
from pipelines.qa.main_qa import QAPipeline

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BuildingRegulationRAGSystem:
    """建筑规范RAG系统主类"""

    def __init__(self, config: RAGConfig = None):
        """
        初始化RAG系统

        Args:
            config: RAG系统配置，默认使用DEFAULT_CONFIG
        """
        self.config = config or DEFAULT_CONFIG
        self.data_module = None
        self.index_module = None
        self.retrieval_module = None
        self.generation_module = None

        # 检查数据路径
        for data_path in self.config.data_paths:
            if not Path(data_path).exists():
                logger.warning(f"数据路径不存在: {data_path}")

        # 检查API密钥
        if not os.getenv("MOONSHOT_API_KEY"):
            raise ValueError("请设置 MOONSHOT_API_KEY 环境变量")
    
    def initialize_system(self):
        """初始化所有模块"""
        print("🚀 正在初始化RAG系统...")

        # 1. 初始化数据准备模块
        print("初始化数据准备模块...")
        self.data_module = DataPreparationModule(self.config.data_paths)

        # 2. 初始化索引构建模块
        print("初始化索引构建模块...")
        self.index_module = IndexConstructionModule(
            model_name=self.config.embedding_model,
            index_save_path=self.config.index_save_path
        )

        # 3. 初始化生成集成模块
        print("🤖 初始化生成集成模块...")
        self.generation_module = GenerationIntegrationModule(
            provider=self.config.llm_provider,
            model_name=self.config.llm_model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )

        print("✅ 系统初始化完成！")
    
    def build_knowledge_base(self):
        """构建知识库"""
        print("\n正在构建知识库...")

        # 1. 尝试加载已保存的索引
        vectorstore = self.index_module.load_index()

        if vectorstore is not None:
            print("✅ 成功加载已保存的向量索引！")
            # 仍需要加载文档和分块用于检索模块
            print("加载规范文档...")
            self.data_module.load_documents()
            print("进行文本分块...")
            chunks = self.data_module.chunk_documents()
        else:
            print("未找到已保存的索引，开始构建新索引...")

            # 2. 加载文档
            print("加载规范文档...")
            self.data_module.load_documents()

            # 3. 文本分块
            print("进行文本分块...")
            chunks = self.data_module.chunk_documents()

            # 4. 构建向量索引
            print("构建向量索引...")
            vectorstore = self.index_module.build_vector_index(chunks)

            # 5. 保存索引
            print("保存向量索引...")
            self.index_module.save_index()

        # 6. 初始化检索优化模块
        print("初始化检索优化...")
        self.retrieval_module = RetrievalOptimizationModule(vectorstore, chunks)

        # 7. 显示统计信息
        stats = self.data_module.get_statistics()
        print(f"\n📊 知识库统计:")
        print(f"   文档总数: {stats['total_documents']}")
        print(f"   文本块数: {stats['total_chunks']}")
        print(f"   文档类型数: {stats.get('doc_count', 0)}")

        print("✅ 知识库构建完成！")
    
    def ask_question(self, question: str):
        """
        回答用户问题

        Args:
            question: 用户问题

        Returns:
            生成的回答或生成器
        """
        if not all([self.retrieval_module, self.generation_module]):
            raise ValueError("请先构建知识库")
        
        print(f"\n❓ 用户问题: {question}")
        
        # 3. 混合检索相关子块
        print("🔍 检索相关文档...")
        relevant_chunks = self.retrieval_module.hybrid_search(question, top_k=self.config.top_k)

        # 显示检索到的子块信息
        if relevant_chunks:
            chunk_info = []
            for chunk in relevant_chunks:
                doc_name = chunk.metadata.get('doc_name', '未知文档')
                # 尝试从内容中提取章节标题
                content_preview = chunk.page_content[:50].replace('\n', ' ').strip()
                if content_preview.startswith('#'):
                    # 如果是标题开头，提取标题
                    title_end = content_preview.find('\n') if '\n' in chunk.page_content[:100] else len(content_preview)
                    section_title = chunk.page_content[:title_end].strip('#').strip()
                    chunk_info.append(f"{doc_name}({section_title})")
                else:
                    chunk_info.append(f"{doc_name}(内容片段)")

            print(f"找到 {len(relevant_chunks)} 个相关文档块: {', '.join(chunk_info)}")
        else:
            print(f"找到 {len(relevant_chunks)} 个相关文档块")

        # 4. 检查是否找到相关内容
        if not relevant_chunks:
            return "抱歉，没有找到相关的信息。"

        print("获取完整文档...")
        relevant_docs = self.data_module.get_parent_documents(relevant_chunks)

        # 显示找到的文档名称
        doc_names = []
        for doc in relevant_docs:
            doc_name = doc.metadata.get('doc_name', '未知文档')
            doc_names.append(doc_name)

        if doc_names:
            print(f"找到文档: {', '.join(doc_names)}")
        else:
            print(f"对应 {len(relevant_docs)} 个完整文档")

        print("✍️ 生成详细回答...")


        return self.generation_module.generate_basic_answer(question, relevant_chunks)


    
    def run_interactive(self):
        """运行交互式问答"""
        print("=" * 60)
        print("�️  建筑规范RAG系统 - 交互式问答  �️")
        print("=" * 60)
        print("💡 为您提供专业、准确的建筑设计规范查询服务！")
        
        # 初始化系统
        self.initialize_system()
        
        # 构建知识库
        self.build_knowledge_base()
        
        print("\n交互式问答 (输入'退出'结束):")
        
        while True:
            try:
                user_input = input("\n您的问题: ").strip()
                if user_input.lower() in ['退出', 'quit', 'exit', '']:
                    break
            

                # 记录用户消息（若启用）
                try:
                    _record_msg("user", user_input)
                except Exception:
                    pass

                print("\n回答:")
                answer = self.ask_question(user_input)
                # 记录助手消息（若启用）
                try:
                    _record_msg("assistant", str(answer))
                except Exception:
                    pass
                print(f"{answer}\n")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"处理问题时出错: {e}")
        
        print("\n感谢使用建筑规范RAG系统！")



def run_interactive_qa():
    """多轮 QA 入口，基于 QAPipeline。"""
    qa = QAPipeline()
    print("=" * 60)
    print("🧠  双路混合检索 QA")
    print("=" * 60)
    print("输入'退出'或'quit'结束。\n")

    while True:
        try:
            user_input = input("你的问题: ").strip()
            if user_input.lower() in ["退出", "quit", "exit", "q", "bye", ""]:
                break
            answer = qa.answer(user_input)
            print(f"\n答: {answer}\n")
        except KeyboardInterrupt:
            break
        except Exception as exc:  # noqa: BLE001
            logger.exception("处理 QA 失败: %s", exc)
            print(f"发生错误: {exc}")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='建筑规范RAG系统')
    parser.add_argument('--build_index', action='store_true', help='重建向量索引（传统规范RAG路径）')
    parser.add_argument('--qa', action='store_true', help='启动双路混合检索 QA（默认行为）')
    args = parser.parse_args()

    # 默认进入QA模式，除非显式要求构建索引或运行旧版交互
    if args.qa or not args.build_index:
        run_interactive_qa()
        return

    try:
        # 创建RAG系统
        rag_system = BuildingRegulationRAGSystem()

        if args.build_index:
            # 仅构建索引
            print("🔨 重建索引模式")
            rag_system.initialize_system()

            # 删除旧索引
            index_path = Path(rag_system.config.index_save_path)
            if index_path.exists():
                import shutil
                print(f"删除旧索引: {index_path}")
                shutil.rmtree(index_path)

            rag_system.build_knowledge_base()
            print("✅ 索引构建完成！")
    except Exception as e:
        logger.error(f"系统运行出错: {e}")
        print(f"系统错误: {e}")

if __name__ == "__main__":
    main()
