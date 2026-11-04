"""Assemble the static online demo under apps/web/public/demo (run in the demo environment).

    python -m fda_api.demo_build data/demo/subjects.json apps/web

1. wheels of fda_core, fda_contracts and fda_demo for the in-browser engine (+ index.json);
2. inputs of every analysis of every demo subject (fda_api.demo_export);
3. the "data as of" date in apps/web/.env.demo.
The read API responses are recorded separately (playwright.demo.config.ts).
"""

import json
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

from fda_api.demo_export import export_subject
from fda_contracts.envelopes import SubjectRef

PACKAGES = ["packages/fda_core", "packages/fda_contracts", "packages/fda_demo"]


def build_wheels(root: Path, out: Path) -> list[str]:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for package in PACKAGES:
        subprocess.run(
            ["uv", "build", "--wheel", "--out-dir", str(out), str(root / package)], check=True
        )
    wheels = sorted(p.name for p in out.glob("*.whl"))
    (out / "index.json").write_text(json.dumps(wheels), encoding="utf-8")
    return wheels


def main() -> None:
    subjects_file, web = Path(sys.argv[1]), Path(sys.argv[2])
    root = Path(__file__).resolve().parents[4]
    demo = web / "public" / "demo"
    print("wheels:", build_wheels(root, demo / "wheels"))
    inputs = demo / "inputs"
    if inputs.exists():
        shutil.rmtree(inputs)
    written: set[str] = set()
    subjects = json.loads(subjects_file.read_text(encoding="utf-8"))
    for kind, ids in subjects.items():
        for ident in ids:
            counts = export_subject(SubjectRef(type=kind, id=ident), demo, written)
            print(f"{kind}:{ident}: {counts}")
    (web / ".env.demo").write_text(
        f"VITE_DEMO=1\nVITE_DEMO_AS_OF={date.today().isoformat()}\n", encoding="utf-8"
    )
    size = sum(p.stat().st_size for p in demo.rglob("*") if p.is_file())
    print(f"demo bundle: {size / 1e6:.1f} MB, {len(written)} input blobs")


if __name__ == "__main__":
    main()
