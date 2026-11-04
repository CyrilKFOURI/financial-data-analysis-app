"""The web formula drawer's catalog.json is generated from docs/ANALYTICS_CATALOG.md: no drift."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    "import('./apps/web/scripts/gen-catalog.mjs').then(m => process.stdout.write("
    "JSON.stringify(m.parseCatalog("
    "require('fs').readFileSync('docs/ANALYTICS_CATALOG.md', 'utf8')))))"
)


@pytest.mark.skipif(shutil.which("node") is None, reason="node is needed to run the generator")
def test_catalog_json_matches_the_catalog() -> None:
    done = subprocess.run(
        ["node", "-e", SCRIPT],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    generated = json.loads(done.stdout)
    stored = json.loads((ROOT / "apps/web/src/shell/catalog.json").read_text("utf-8"))
    assert generated == stored, "run `corepack pnpm --dir apps/web gen:catalog`"
    assert len(stored) >= 280
