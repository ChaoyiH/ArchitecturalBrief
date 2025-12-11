"""Virtual Hearing Engine - Brief Reviewer.

A lightweight, standalone review orchestrator that simulates multiple personas
critiquing a design brief. It intentionally avoids dependencies on the existing
LangGraph or design generator pipelines.
"""

from __future__ import annotations

import logging
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union

import importlib.util
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from config.review_matrix import Stakeholder, Scale, Lifecycle, get_persona_matrix
from config.review_definitions import SCALE_DEFINITIONS, LIFECYCLE_DEFINITIONS
from core.generation_integration import GenerationIntegrationModule
from prompts_review import REVIEW_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Load DEFAULT_CONFIG from the legacy config.py without clashing with the
# new config package namespace.
# ---------------------------------------------------------------------------
_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.py"
_spec = importlib.util.spec_from_file_location("base_config", _CONFIG_PATH)
if _spec and _spec.loader:
    _base_cfg = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_base_cfg)  # type: ignore[arg-type]
    DEFAULT_CONFIG = getattr(_base_cfg, "DEFAULT_CONFIG", None)
else:  # pragma: no cover
    DEFAULT_CONFIG = None

logger = logging.getLogger(__name__)

PersonaTask = Tuple[Stakeholder, Scale, Lifecycle]


class PersonaFactory:
    """Create persona tasks from the sparse matrix."""

    def __init__(self) -> None:
        self.matrix = get_persona_matrix()

    def sample(self, sample_size: Union[int, str] = "all") -> List[PersonaTask]:
        tasks: List[PersonaTask] = []
        for role, pairs in self.matrix.items():
            for scale, phase in pairs:
                tasks.append((role, scale, phase))

        if sample_size == "all":
            return tasks

        if isinstance(sample_size, int) and sample_size > 0:
            return random.sample(tasks, k=min(sample_size, len(tasks)))

        logger.warning("Invalid sample_size %s, defaulting to all personas", sample_size)
        return tasks


@dataclass
class ReviewResult:
    role: str
    scale: str
    phase: str
    feedback: str


class ReviewAgent:
    """LLM caller reusing the shared GenerationIntegrationModule."""

    def __init__(self, provider: str | None = None, model: str | None = None) -> None:
        provider = provider or (getattr(DEFAULT_CONFIG, "llm_provider", "minimax") if DEFAULT_CONFIG else "minimax")
        model = model or (getattr(DEFAULT_CONFIG, "llm_model", "Minimax-M2") if DEFAULT_CONFIG else "Minimax-M2")
        max_tokens = getattr(DEFAULT_CONFIG, "max_tokens", 16384) if DEFAULT_CONFIG else 16384
        self.module = GenerationIntegrationModule(
            provider=provider,
            model_name=model,
            temperature=0.3,
            max_tokens=max_tokens,
        )
        self.provider = provider
        self.max_tokens = max_tokens

    def call_llm(self, system_prompt: str, user_content: str, retries: int = 3, delay: float = 1.5) -> str:
        prompt_text = f"""{system_prompt}\n\n【任务书内容】\n{user_content}"""

        for attempt in range(1, retries + 1):
            try:
                if self.provider == "google":
                    return self.module._generate_google_answer(prompt_text)

                prompt = ChatPromptTemplate.from_template("{text}")
                chain = prompt | self.module.llm | StrOutputParser()
                return chain.invoke({"text": prompt_text})
            except Exception as exc:  # noqa: BLE001
                if attempt >= retries:
                    logger.warning("LLM 调用重试耗尽 (attempt=%s): %s", attempt, exc)
                    raise
                sleep_for = delay * attempt
                logger.warning("LLM 调用失败 (attempt=%s/%s): %s，%.1fs 后重试", attempt, retries, exc, sleep_for)
                time.sleep(sleep_for)


class ReviewEngine:
    """Coordinate persona sampling, LLM calls, and aggregation."""

    def __init__(self, provider: str | None = None, model: str | None = None) -> None:
        self.factory = PersonaFactory()
        self.agent = ReviewAgent(provider=provider, model=model)

    def _build_system_prompt(self, role: Stakeholder, scale: Scale, phase: Lifecycle) -> str:
        scale_desc = SCALE_DEFINITIONS.get(scale.name.capitalize(), "")
        phase_desc = LIFECYCLE_DEFINITIONS.get(phase.name.capitalize(), "")
        return REVIEW_SYSTEM_PROMPT.format(
            role=role.value,
            scale=scale.value,
            scale_desc=scale_desc,
            phase=phase.value,
            phase_desc=phase_desc,
        )

    def _run_single(self, md_content: str, persona: PersonaTask) -> ReviewResult:
        role, scale, phase = persona
        logger.info("Reviewer [%s] working on [%s/%s]...", role.value, scale.value, phase.value)
        system_prompt = self._build_system_prompt(role, scale, phase)
        feedback = self.agent.call_llm(system_prompt, md_content)
        feedback = self._postprocess_feedback(feedback)
        return ReviewResult(role=role.value, scale=scale.value, phase=phase.value, feedback=feedback)

    @staticmethod
    def _postprocess_feedback(text: str) -> str:
        if not text or not text.strip():
            return "PASS"

        cleaned = text.strip()
        # Heuristic: if the reply doesn't end with a typical sentence terminator, warn about possible truncation.
        terminators = ("。", "！", "!", "?", "？", "\n", ".")
        if not cleaned.endswith(terminators):
            logger.warning("反馈可能被截断，请复查或重试。")
            cleaned = cleaned + "\n\n（提示：回复可能被截断，请复查。）"
        return cleaned

    def run(self, md_content: str, n: Union[int, str] = "all") -> List[Dict[str, str]]:
        personas = self.factory.sample(n)
        results: List[ReviewResult] = []

        # Simple parallelism; fallback to sequential if any issue arises
        with ThreadPoolExecutor(max_workers=min(len(personas), 8)) as executor:
            future_map = {executor.submit(self._run_single, md_content, p): p for p in personas}
            for future in as_completed(future_map):
                try:
                    res = future.result()
                    results.append(res)
                except Exception as exc:  # noqa: BLE001
                    role, scale, phase = future_map[future]
                    logger.warning(
                        "Reviewer [%s/%s/%s] failed: %s", role.value, scale.value, phase.value, exc
                    )

        # Keep output as list of dicts for JSON serialization
        return [
            {
                "role": r.role,
                "scale": r.scale,
                "phase": r.phase,
                "feedback": r.feedback,
            }
            for r in results
        ]


__all__ = [
    "PersonaFactory",
    "ReviewAgent",
    "ReviewEngine",
    "ReviewResult",
]
