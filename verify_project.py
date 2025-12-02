#!/usr/bin/env python3
"""
Verification script to check if all project files exist.
Run this after creating your project structure.

Usage: python verify_project.py
"""

from pathlib import Path
from typing import List, Tuple

# Color codes for terminal output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"


# All required files
REQUIRED_FILES = [
    # Root config
    "pyproject.toml",
    ".env.example",
    ".gitignore",
    "README.md",
    "CONTRIBUTING.md",
    "LICENSE",
    "Makefile",
    ".pre-commit-config.yaml",
    "Dockerfile",
    "docker-compose.yml",
    # Source - main
    "src/ai_agent/__init__.py",
    "src/ai_agent/config.py",
    "src/ai_agent/main.py",
    # Source - core
    "src/ai_agent/core/__init__.py",
    "src/ai_agent/core/llm.py",
    "src/ai_agent/core/embeddings.py",
    "src/ai_agent/core/vector_store.py",
    # Source - rag
    "src/ai_agent/rag/__init__.py",
    "src/ai_agent/rag/document_loader.py",
    "src/ai_agent/rag/text_splitter.py",
    "src/ai_agent/rag/indexer.py",
    "src/ai_agent/rag/retriever.py",
    # Source - agents
    "src/ai_agent/agents/__init__.py",
    "src/ai_agent/agents/base_agent.py",
    "src/ai_agent/agents/rag_agent.py",
    "src/ai_agent/agents/tool_agent.py",
    # Source - tools
    "src/ai_agent/tools/__init__.py",
    "src/ai_agent/tools/base_tool.py",
    "src/ai_agent/tools/calculator_tool.py",
    "src/ai_agent/tools/search_tool.py",
    "src/ai_agent/tools/datetime_tool.py",
    # Source - api
    "src/ai_agent/api/__init__.py",
    "src/ai_agent/api/schemas.py",
    "src/ai_agent/api/routes.py",
    # Source - utils
    "src/ai_agent/utils/__init__.py",
    "src/ai_agent/utils/logger.py",
    "src/ai_agent/utils/helpers.py",
    "src/ai_agent/utils/timing.py",
    # Tests
    "tests/__init__.py",
    "tests/conftest.py",
    "tests/test_rag.py",
    # Scripts
    "scripts/setup.sh",
    "scripts/index_documents.py",
]

REQUIRED_DIRS = [
    "src/ai_agent",
    "src/ai_agent/core",
    "src/ai_agent/rag",
    "src/ai_agent/agents",
    "src/ai_agent/tools",
    "src/ai_agent/api",
    "src/ai_agent/utils",
    "tests",
    "scripts",
    "data",
    "data/documents",
    "data/indices",
    "data/uploads",
    "data/chroma",
    "data/lancedb",
]


def check_files() -> Tuple[List[str], List[str]]:
    """Check which files exist and which are missing.

    Returns:
        Tuple of (existing_files, missing_files)
    """
    existing = []
    missing = []

    for file_path in REQUIRED_FILES:
        path = Path(file_path)
        if path.exists():
            existing.append(file_path)
        else:
            missing.append(file_path)

    return existing, missing


def check_dirs() -> Tuple[List[str], List[str]]:
    """Check which directories exist and which are missing.

    Returns:
        Tuple of (existing_dirs, missing_dirs)
    """
    existing = []
    missing = []

    for dir_path in REQUIRED_DIRS:
        path = Path(dir_path)
        if path.is_dir():
            existing.append(dir_path)
        else:
            missing.append(dir_path)

    return existing, missing


def check_file_content(file_path: str, min_size: int = 10) -> bool:
    """Check if file has content.

    Args:
        file_path: Path to file
        min_size: Minimum file size in bytes

    Returns:
        True if file has content, False otherwise
    """
    path = Path(file_path)
    if not path.exists():
        return False
    return path.stat().st_size >= min_size


def main():
    """Run verification."""
    print("=" * 60)
    print("Local AI Agent - Project Verification")
    print("=" * 60)
    print()

    # Check directories
    print("Checking directories...")
    existing_dirs, missing_dirs = check_dirs()

    if missing_dirs:
        print(f"{RED}✗ Missing {len(missing_dirs)} directories:{RESET}")
        for dir_path in missing_dirs:
            print(f"  - {dir_path}")
        print()
    else:
        print(f"{GREEN}✓ All {len(REQUIRED_DIRS)} directories exist{RESET}")
        print()

    # Check files
    print("Checking files...")
    existing_files, missing_files = check_files()

    print(f"{GREEN}✓ Found {len(existing_files)}/{len(REQUIRED_FILES)} files{RESET}")

    if missing_files:
        print(f"{RED}✗ Missing {len(missing_files)} files:{RESET}")
        for file_path in missing_files:
            print(f"  - {file_path}")
        print()

    # Check file content
    print("Checking file content...")
    empty_files = []
    for file_path in existing_files:
        if not check_file_content(file_path):
            empty_files.append(file_path)

    if empty_files:
        print(f"{YELLOW}⚠ Found {len(empty_files)} empty or nearly empty files:{RESET}")
        for file_path in empty_files:
            print(f"  - {file_path}")
        print()
    else:
        print(f"{GREEN}✓ All existing files have content{RESET}")
        print()

    # Summary
    print("=" * 60)
    print("Summary:")
    print(f"  Directories: {len(existing_dirs)}/{len(REQUIRED_DIRS)}")
    print(f"  Files: {len(existing_files)}/{len(REQUIRED_FILES)}")
    print(f"  Empty files: {len(empty_files)}")
    print()

    # Final verdict
    if not missing_dirs and not missing_files and not empty_files:
        print(f"{GREEN}✓ PROJECT SETUP COMPLETE!{RESET}")
        print()
        print("Next steps:")
        print("  1. python3 -m venv venv")
        print("  2. source venv/bin/activate")
        print("  3. pip install -e '.[dev]'")
        print("  4. cp .env.example .env")
        print("  5. ollama pull llama3.2")
        print("  6. python -m ai_agent.main")
        return 0
    else:
        print(f"{RED}✗ PROJECT SETUP INCOMPLETE{RESET}")
        print()
        print("Please create the missing files/directories.")
        print("Refer to the COMPLETE FILE CHECKLIST artifact.")
        return 1


if __name__ == "__main__":
    exit(main())
