from __future__ import annotations

import ast
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
OUTPUT_PATH = CODE_ROOT / "ARCHITECTURE_REPORT.md"


IGNORED_DIR_NAMES = {"__pycache__", ".git", "logs", "log", "vector_index"}
IGNORED_FILE_SUFFIXES = {".pyc"}


def build_directory_tree(root: Path) -> str:
    """Recursively build a directory tree for the given root.

    Only includes files under CODE_ROOT, skipping cache/log/index folders.
    """

    lines: List[str] = [f"code/"]

    def _walk(dir_path: Path, prefix: str) -> None:
        try:
            entries = sorted(dir_path.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        except FileNotFoundError:
            return

        for idx, entry in enumerate(entries):
            if entry.name in IGNORED_DIR_NAMES:
                continue
            if entry.is_file() and entry.suffix in IGNORED_FILE_SUFFIXES:
                continue

            connector = "└── " if idx == len(entries) - 1 else "├── "
            line = f"{prefix}{connector}{entry.name}"
            lines.append(line)

            if entry.is_dir():
                child_prefix = prefix + ("    " if idx == len(entries) - 1 else "│   ")
                _walk(entry, child_prefix)

    _walk(root, "")
    return "\n".join(lines)


def read_file_if_exists(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError:
        return None


CORE_FILES = [
    CODE_ROOT / "config.py",
    CODE_ROOT / "pipelines" / "graph_engine" / "state.py",
    CODE_ROOT / "pipelines" / "graph_engine" / "graph.py",
    CODE_ROOT / "pipelines" / "graph_engine" / "nodes.py",
    CODE_ROOT / "design_generator.py",
    CODE_ROOT / "pipelines" / "orchestration" / "brief_assembly_pipeline.py",
]


def extract_signatures_from_source(source: str) -> List[Tuple[str, str, Optional[str]]]:
    """Extract class and function signatures with docstrings using ast.

    Returns a list of tuples: (kind, signature, docstring).
    kind is either "class" or "def".
    """

    results: List[Tuple[str, str, Optional[str]]] = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return results

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            signature = f"class {node.name}:"
            doc = ast.get_docstring(node)
            results.append(("class", signature, doc))
            # also capture methods on the class
            for body_node in node.body:
                if isinstance(body_node, ast.FunctionDef):
                    func_sig = f"def {body_node.name}(...):"
                    func_doc = ast.get_docstring(body_node)
                    results.append(("def", f"  {func_sig}", func_doc))
        elif isinstance(node, ast.FunctionDef):
            signature = f"def {node.name}(...):"
            doc = ast.get_docstring(node)
            results.append(("def", signature, doc))
    return results


def collect_domain_and_utils_files() -> Dict[str, List[Tuple[str, str, Optional[str]]]]:
    """Collect signatures for files under pipelines/domains and utils.

    Only includes .py files excluding this script itself.
    """

    result: Dict[str, List[Tuple[str, str, Optional[str]]]] = {}

    target_dirs = [
        CODE_ROOT / "pipelines" / "domains",
        CODE_ROOT / "utils",
    ]

    for base in target_dirs:
        if not base.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            # filter ignored dirs in-place
            dirnames[:] = [
                d
                for d in dirnames
                if d not in IGNORED_DIR_NAMES
            ]
            for filename in filenames:
                if not filename.endswith(".py"):
                    continue
                if filename == Path(__file__).name:
                    continue
                file_path = Path(dirpath) / filename
                try:
                    src = file_path.read_text(encoding="utf-8")
                except OSError:
                    continue
                sigs = extract_signatures_from_source(src)
                rel = os.path.relpath(file_path, PROJECT_ROOT)
                result[rel] = sigs

    return result


def generate_report() -> None:
    lines: List[str] = []

    # Part 1: Directory tree
    lines.append("# ARCHITECTURE REPORT")
    lines.append("")
    lines.append("## Part 1: 项目目录结构 (Directory Tree)")
    lines.append("")
    tree = build_directory_tree(CODE_ROOT)
    lines.append("```text")
    lines.append(tree)
    lines.append("```")
    lines.append("")

    # Part 2: 核心架构层
    lines.append("## Part 2: 核心架构层 (Orchestration Layer)")
    lines.append("")
    for path in CORE_FILES:
        rel = os.path.relpath(path, PROJECT_ROOT)
        content = read_file_if_exists(path)
        if content is None:
            continue
        lines.append(f"### `{rel}`")
        lines.append("")
        lines.append("```python")
        lines.append(content)
        lines.append("```")
        lines.append("")

    # Part 3: 领域业务层
    lines.append("## Part 3: 领域业务层 (Domain Layer) - Signatures Only")
    lines.append("")

    domain_utils_map = collect_domain_and_utils_files()
    for rel, sigs in sorted(domain_utils_map.items()):
        lines.append(f"### `{rel}`")
        lines.append("")
        if not sigs:
            lines.append("(no public classes or functions found)")
            lines.append("")
            continue
        lines.append("```python")
        for kind, sig, doc in sigs:
            lines.append(sig)
            if doc:
                # indent docstring for readability
                for doc_line in doc.splitlines():
                    lines.append(f"    # {doc_line}")
            lines.append("")
        lines.append("```")
        lines.append("")

    OUTPUT_PATH.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":  # pragma: no cover
    generate_report()
