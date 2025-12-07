"""Business & Research (Back-of-House) pipeline using LangGraph."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, TypedDict

from langchain_core.documents import Document
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph

from config import BusinessResearchConfig
from core.generation_integration import GenerationIntegrationModule
from utils.data_preparation import BusinessResearchDataExtractor
from utils.indicator_analyzer import analyze_indicators

logger = logging.getLogger(__name__)


class BusinessResearchState(TypedDict, total=False):
	"""Graph state for the business research pipeline."""

	normative_data: Dict[str, Any]
	empirical_data: List[Dict[str, Any]]
	final_json: Dict[str, Any]


class BusinessResearchGenerator:
	"""Generates back-of-house program guidance (业务研究用房)."""

	# Branch-B keyword filters
	EMPIRICAL_KEYWORDS = [
		"office",
		"administration",
		"admin",
		"lab",
		"laboratory",
		"laboratories",
		"storage",
		"visible",
		"workshop",
		"research",
		"collection",
		"curation",
		"conservation",
		"repair",
		"restoration",
		"开放式库房",
		"科研",
		"实验",
		"库房",
		"办公",
	]

	def __init__(self, config: BusinessResearchConfig):
		self.config = config
		self.extractor = BusinessResearchDataExtractor(config)
		self._llm_module: Optional[GenerationIntegrationModule] = None
		self._cached_docs: Optional[List[Document]] = None

	# ------------------------------------------------------------------
	# Helpers
	# ------------------------------------------------------------------
	def _ensure_llm(self) -> None:
		if self._llm_module is None:
			self._llm_module = GenerationIntegrationModule(
				provider=self.config.llm_provider,
				model_name=self.config.llm_model,
				temperature=self.config.temperature,
				max_tokens=self.config.max_tokens,
			)

	def _load_normative_corpus(self) -> str:
		"""Load GB + 专栏讲解文本用于规范抽取。"""

		parts: List[str] = []
		gb_root = Path(self.config.gb_data_path)
		zlj_root = Path(self.config.zlj_data_path)
		gb_files = ["科学技术馆建设标准.md", "科学技术馆建设标准 条文说明.md"]
		for name in gb_files:
			path = gb_root / name
			if path.exists():
				try:
					parts.append(path.read_text(encoding="utf-8"))
				except Exception as exc:  # noqa: BLE001
					logger.warning("读取规范文件失败 %s: %s", path, exc)
		kjg_path = zlj_root / "kjg.md"
		if kjg_path.exists():
			try:
				parts.append(kjg_path.read_text(encoding="utf-8"))
			except Exception as exc:  # noqa: BLE001
				logger.warning("读取专栏讲解文件失败 %s: %s", kjg_path, exc)
		return "\n\n".join(parts).strip()

	def _ensure_documents(self, rebuild: bool = False) -> List[Document]:
		if self._cached_docs is not None and not rebuild:
			return self._cached_docs
		self._cached_docs = self.extractor.load_documents()
		return self._cached_docs

	# ------------------------------------------------------------------
	# Branch A: Normative track
	# ------------------------------------------------------------------
	def _summarize_indicators(self, target_area: float) -> Dict[str, Any]:
		indicators = analyze_indicators(target_area)
		classification = indicators.get("target_classification", {}) or {}
		compliance = indicators.get("compliance", {}) or {}
		ratios = compliance.get("function_area_ranges", {}) or {}

		br = ratios.get("business_research") or {}
		mgmt = ratios.get("management") or {}

		def _fmt_range(entry: Dict[str, Any]) -> str:
			if not entry:
				return ""
			min_area = entry.get("area_min_sqm")
			max_area = entry.get("area_max_sqm")
			if min_area is None or max_area is None:
				return ""
			return f"{min_area:,.0f}–{max_area:,.0f} sqm"

		ratio_text = []
		if br:
			ratio_text.append(
				f"业务研究 {br.get('percent_min', '-')}-{br.get('percent_max', '-')}%"
			)
		if mgmt:
			ratio_text.append(
				f"管理保障 {mgmt.get('percent_min', '-')}-{mgmt.get('percent_max', '-')}%"
			)

		suggested_area_range = _fmt_range(br)
		return {
			"raw": indicators,
			"classification": classification,
			"compliance": compliance,
			"ratio_text": " / ".join(ratio_text),
			"suggested_area_range": suggested_area_range,
		}

	def _run_normative_branch(
		self,
		project_name: str,
		target_area: float,
		corpus: str,
		dry_run: bool,
	) -> Dict[str, Any]:
		indicator_pack = self._summarize_indicators(target_area)
		classification = indicator_pack.get("classification", {})
		class_name = classification.get("class_name", "未知")
		class_en_map = {
			"特大型馆": "Extra Large",
			"大型馆": "Large",
			"中型馆": "Medium",
			"小型馆": "Small",
		}
		target_level = f"{class_en_map.get(class_name, 'Unknown')} ({class_name})"
		suggested_area_range = indicator_pack.get("suggested_area_range") or ""
		ratio_text = indicator_pack.get("ratio_text") or "业务研究 10%-15% / 管理保障 10%-15%"

		base_normative = {
			"target_level": target_level,
			"compliance_source": "《科学技术馆建设标准》建标101-2007",
			"suggested_area_range": suggested_area_range,
			"standard_ratio_text": ratio_text,
			"mandatory_sub_functions": [
				"行政办公",
				"科研与实验室",
				"藏品库房/库前区",
				"展品/展陈修复与制作",
			],
		}

		if dry_run:
			return {"normative_data": base_normative, "prompt": None}

		self._ensure_llm()
		parser = JsonOutputParser()
		fmt = parser.get_format_instructions()
		prompt = ChatPromptTemplate.from_messages(
			[
				(
					"system",
					"你是科技馆后勤/业务研究用房的规范解析助手。基于法规原文与指标计算，输出 JSON，不要遗漏必备功能。",
				),
				(
					"human",
					"项目: {project_name}\n目标面积: {target_area} m²\n等级判定: {target_level}\n功能配比: {ratio_text}\n法规文本:\n{corpus}\n\n请抽取: mandatory_sub_functions (列表), ratio_requirements (字符串，可包含百分比), target_level (可重写), compliance_source。\n{format_instructions}",
				),
			]
		)

		chain = prompt | self._llm_module.llm | parser
		llm_output: Dict[str, Any] = {}
		try:
			llm_output = chain.invoke(
				{
					"project_name": project_name,
					"target_area": target_area,
					"target_level": target_level,
					"ratio_text": ratio_text,
					"corpus": corpus[:6000],
					"format_instructions": fmt,
				}
			) or {}
		except Exception as exc:  # noqa: BLE001
			logger.exception("规范抽取 LLM 调用失败，将使用基础数据: %s", exc)

		merged = {
			"target_level": llm_output.get("target_level") or base_normative["target_level"],
			"compliance_source": llm_output.get("compliance_source")
			or base_normative["compliance_source"],
			"suggested_area_range": base_normative["suggested_area_range"],
			"standard_ratio_text": llm_output.get("ratio_requirements")
			or base_normative["standard_ratio_text"],
			"mandatory_sub_functions": llm_output.get("mandatory_sub_functions")
			or base_normative["mandatory_sub_functions"],
		}

		return {"normative_data": merged, "prompt": prompt}

	# ------------------------------------------------------------------
	# Branch B: Empirical track
	# ------------------------------------------------------------------
	def _filter_empirical_docs(self, docs: Sequence[Document]) -> List[Document]:
		scored: List[tuple[int, Document]] = []
		for doc in docs:
			text = (doc.page_content or "").lower()
			score = sum(1 for kw in self.EMPIRICAL_KEYWORDS if kw.lower() in text)
			if score > 0:
				scored.append((score, doc))
		scored.sort(key=lambda x: x[0], reverse=True)
		return [doc for _, doc in scored]

	def _format_empirical_context(self, docs: Sequence[Document]) -> str:
		lines: List[str] = []
		for idx, doc in enumerate(docs, 1):
			meta = doc.metadata or {}
			name = meta.get("project_name") or meta.get("doc_name") or meta.get("source")
			src = meta.get("source_type", "unknown")
			snippet = (doc.page_content or "").strip()
			snippet = snippet[:800] + "..." if len(snippet) > 800 else snippet
			lines.append(f"[{idx}] {name} ({src})\n{snippet}")
		return "\n\n".join(lines)

	def _run_empirical_branch(
		self,
		project_name: str,
		project_features: str,
		docs: Sequence[Document],
		top_k: int,
		dry_run: bool,
	) -> Dict[str, Any]:
		filtered = self._filter_empirical_docs(docs)[:top_k]
		context_text = self._format_empirical_context(filtered)

		if dry_run:
			return {"empirical_data": [], "prompt": None, "contexts": filtered}

		self._ensure_llm()
		parser = JsonOutputParser()
		fmt = parser.get_format_instructions()
		prompt = ChatPromptTemplate.from_messages(
			[
				(
					"system",
					"你是科技馆后勤与业务研究区的案例挖掘助手，专注办公/科研/库房/工作坊创新做法，忽略常规办公室。"
					"CRITICAL: empirical_highlights 中的 feature_name 与 description 必须输出为简体中文，若检索文本为英文需译为专业中文建筑术语，禁止输出英文。",
				),
				(
					"human",
					"项目: {project_name}\n特征: {project_features}\n案例片段:\n{context}\n\n请输出 JSON 列表，每项包含 feature_name, case_source, description。强调可视化库房、开放实验室、公众可视化修复等。\n{format_instructions}",
				),
			]
		)

		chain = prompt | self._llm_module.llm | parser
		llm_output: List[Dict[str, Any]] = []
		try:
			llm_output = chain.invoke(
				{
					"project_name": project_name,
					"project_features": project_features,
					"context": context_text[:6000],
					"format_instructions": fmt,
				}
			) or []
		except Exception as exc:  # noqa: BLE001
			logger.exception("实证抽取 LLM 调用失败，返回空列表: %s", exc)

		if not isinstance(llm_output, list):
			llm_output = []

		return {"empirical_data": llm_output, "prompt": prompt, "contexts": filtered}

	# ------------------------------------------------------------------
	# Node C: Deterministic merge (no LLM)
	# ------------------------------------------------------------------
	@staticmethod
	def _merge_structures(normative: Dict[str, Any], empirical: List[Dict[str, Any]]) -> Dict[str, Any]:
		core_zones = [
			"行政办公区",
			"科研与实验区",
			"藏品库房与库前区",
			"展陈修复与制作工坊",
			"设备与后勤支撑",
		]
		adjacency = (
			"科研/修复与库房保持同层邻近；库房需靠近货梯与卸货口；"
			"安保/监控与库房同轴布局；对公众开放的可视化库房需设置导览与防火分隔。"
		)
		final_json = {
			"normative_analysis": normative or {},
			"empirical_highlights": empirical or [],
			"space_program_suggestion": {
				"core_zones": core_zones,
				"adjacency_requirements": adjacency,
			},
		}
		return final_json

	# ------------------------------------------------------------------
	# Public API
	# ------------------------------------------------------------------
	def generate(
		self,
		project_name: str,
		project_features: str,
		*,
		target_area: Optional[float] = None,
		top_k: Optional[int] = None,
		rebuild_index: bool = False,
		dry_run: bool = False,
	) -> Dict[str, Any]:
		"""Run the LangGraph pipeline and return JSON text payload."""

		target_area = float(target_area) if target_area is not None else 20000.0
		top_k = top_k or self.config.top_k

		corpus_text = self._load_normative_corpus()
		docs = self._ensure_documents(rebuild=rebuild_index)

		# Build LangGraph
		graph = StateGraph(BusinessResearchState)

		def normative_node(state: BusinessResearchState) -> Dict[str, Any]:
			result = self._run_normative_branch(project_name, target_area, corpus_text, dry_run)
			return {"normative_data": result.get("normative_data", {}), "_normative_prompt": result.get("prompt")}

		def empirical_node(state: BusinessResearchState) -> Dict[str, Any]:
			result = self._run_empirical_branch(project_name, project_features, docs, top_k, dry_run)
			return {
				"empirical_data": result.get("empirical_data", []),
				"_empirical_prompt": result.get("prompt"),
				"_empirical_contexts": result.get("contexts"),
			}

		def merge_node(state: BusinessResearchState) -> Dict[str, Any]:
			normative = state.get("normative_data") or {}
			empirical = state.get("empirical_data") or []
			merged = self._merge_structures(normative, empirical)
			return {"final_json": merged}

		graph.add_node("normative", normative_node)
		graph.add_node("empirical", empirical_node)
		graph.add_node("merge", merge_node)

		graph.add_edge(START, "normative")
		graph.add_edge(START, "empirical")
		graph.add_edge("normative", "merge")
		graph.add_edge("empirical", "merge")
		graph.add_edge("merge", END)

		app = graph.compile()
		final_state = app.invoke(BusinessResearchState())

		final_json = final_state.get("final_json", {}) or {}

		# dry-run should still expose prompts/contexts for debugging
		response_text = None if dry_run else json.dumps(final_json, ensure_ascii=False, indent=2)
		return {
			"response": response_text,
			"normative_data": final_state.get("normative_data"),
			"empirical_data": final_state.get("empirical_data"),
			"final_json": final_json,
			"prompts": {
				"normative": final_state.get("_normative_prompt"),
				"empirical": final_state.get("_empirical_prompt"),
			},
			"contexts": {
				"normative_corpus_preview": corpus_text[:2000],
				"empirical_docs": final_state.get("_empirical_contexts"),
			},
		}


__all__ = ["BusinessResearchGenerator"]
