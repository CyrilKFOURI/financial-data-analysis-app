"""Architecture rules from TRD §2.1 that must hold for the whole repo."""

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CORE_SRC = REPO_ROOT / "packages" / "fda_core" / "src" / "fda_core"
OTHER_WORKSPACE_PACKAGES = {"fda_data", "fda_contracts", "fda_api", "fda_worker"}


def _imported_top_level_modules(source_file: Path) -> set[str]:
    tree = ast.parse(source_file.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            modules.add(node.module.split(".")[0])
    return modules


def test_fda_core_imports_no_other_workspace_package() -> None:
    violations = {
        str(path.relative_to(REPO_ROOT)): sorted(
            _imported_top_level_modules(path) & OTHER_WORKSPACE_PACKAGES
        )
        for path in CORE_SRC.rglob("*.py")
    }
    assert {file: mods for file, mods in violations.items() if mods} == {}
