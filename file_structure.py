from __future__ import annotations

from pathlib import Path

# Directories or files to ignore during tree traversal
EXCLUDED_DIRS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".venv",
    "venv",
    "build",
    "dist",
}


def display_directory_tree(root_path: str | Path, indent: str = "") -> None:
    """Recursively displays files and subdirectories, ignoring system and version control folders."""
    root = Path(root_path)
    if not root.exists():
        print(f"Path does not exist: {root}")
        return

    # Filter out excluded directories/files
    items = sorted(
        [item for item in root.iterdir() if item.name not in EXCLUDED_DIRS],
        key=lambda x: (not x.is_dir(), x.name.lower()),
    )

    for index, path in enumerate(items):
        is_last = index == len(items) - 1
        connector = "└── " if is_last else "├── "
        print(f"{indent}{connector}{path.name}")

        if path.is_dir():
            extension = "    " if is_last else "│   "
            display_directory_tree(path, indent + extension)


if __name__ == "__main__":
    target_dir = Path(".")
    print(f"Directory Tree for: {target_dir.resolve()} (Excluding .git)")
    display_directory_tree(target_dir)
