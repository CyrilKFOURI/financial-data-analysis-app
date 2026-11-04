"""Export the inputs of every analysis of the demo subjects for the static online demo.

    python -m fda_api.demo_export --out apps/web/public/demo asset:SPY portfolio:1 market:global

For each subject and each analysis that applies to it, the real resolver runs once (default
parameters, full history) and every input it would hand to the engine is written as a
content-addressed JSON blob; a manifest per subject maps analysis → conventions + blobs, or the
resolver's own problem detail. The browser engine then recomputes live from these inputs.
`--verify` re-runs each analysis on the decoded inputs and checks the payload is identical.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

from fda_demo.codec import decode, encode

from fda_api.resolve import Memo, resolve
from fda_contracts.envelopes import RunContext, RunRequest, SubjectRef
from fda_core.registry import REGISTRY, AnalysisInputs, InsufficientDataError, NotApplicableError
from fda_data.db import lake_root, make_engine
from fda_data.lake import PriceLake

FULL_RANGE = "MAX"


def _dump(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"), allow_nan=False, sort_keys=True)


def _write_blob(blobs: Path, value: Any, written: set[str]) -> str:
    text = _dump(encode(value))
    name = hashlib.sha256(text.encode()).hexdigest()[:20]
    if name not in written:
        (blobs / f"{name}.json").write_text(text, encoding="utf-8")
        written.add(name)
    return name


def _same(a: Any, b: Any) -> bool:
    return _dump(encode(a)) == _dump(encode(b))


def export_subject(
    subject: SubjectRef, out: Path, written: set[str], verify: bool = False
) -> dict[str, int]:
    engine, lake = make_engine(), PriceLake(lake_root())
    blobs = out / "inputs" / "blobs"
    blobs.mkdir(parents=True, exist_ok=True)
    request = RunRequest(
        subject=subject, context=RunContext(range=FULL_RANGE), analyses=[{"id": "DEV-001"}]
    )
    memo: Memo = {}
    analyses: dict[str, Any] = {}
    counts = {"ok": 0, "problem": 0, "mismatch": 0}
    for spec in REGISTRY.all():
        if subject.type not in spec.applies_to:
            continue
        params = spec.params_model()
        try:
            resolved = resolve(spec, request, engine, lake, params, memo)
            data = {name: load() for name, load in resolved.loaders.items()}
        except InsufficientDataError as exc:
            analyses[spec.id] = {
                "error": {"type": "/errors/insufficient-data", "status": 422, "detail": str(exc)}
            }
            counts["problem"] += 1
            continue
        try:
            entry = {
                "conventions": encode(resolved.conventions),
                "data_version": resolved.data_version,
                "inputs": {
                    name: _write_blob(blobs, value, written) for name, value in data.items()
                },
            }
        except TypeError as exc:  # an input type the codec does not know: report, skip
            print(f"  {spec.id}: cannot export ({exc})", file=sys.stderr)
            counts["problem"] += 1
            continue
        if verify:
            counts["mismatch"] += 0 if _verify(spec, resolved.conventions, data, params) else 1
        analyses[spec.id] = entry
        counts["ok"] += 1
    key = subject.id.upper() if subject.type == "asset" else subject.id
    path = out / "inputs" / subject.type / f"{key}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        _dump({"subject": subject.model_dump(), "analyses": analyses}), encoding="utf-8"
    )
    return counts


def _verify(spec: Any, conventions: dict[str, Any], data: dict[str, Any], params: Any) -> bool:
    """The analysis gives the same payload on the original and on the round-tripped inputs."""

    def run(d: dict[str, Any]) -> Any:
        try:
            out = spec.fn(AnalysisInputs(data=d, conventions=dict(conventions)), params)
            return dict(out.payload)
        except (InsufficientDataError, NotApplicableError) as exc:
            return f"{type(exc).__name__}: {exc}"

    roundtrip = {k: decode(json.loads(_dump(encode(v)))) for k, v in data.items()}
    same = _same(run(data), run(roundtrip))
    if not same:
        print(f"  {spec.id}: payload differs after the round trip", file=sys.stderr)
    return same


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("subjects", nargs="+", help="type:id, e.g. asset:SPY portfolio:1")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)
    written = {p.stem for p in (args.out / "inputs" / "blobs").glob("*.json")}
    for item in args.subjects:
        kind, _, ident = item.partition(":")
        started = time.perf_counter()
        counts = export_subject(SubjectRef(type=kind, id=ident), args.out, written, args.verify)  # type: ignore[arg-type]
        print(f"{item}: {counts} in {time.perf_counter() - started:.0f} s")


if __name__ == "__main__":
    main()
