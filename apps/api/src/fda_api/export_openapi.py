"""Write the OpenAPI document used to generate the web client types.

Usage: `uv run python -m fda_api.export_openapi apps/web/src/api/openapi.json`
"""

import json
import sys
from pathlib import Path

from fda_api.main import app


def main() -> None:
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("openapi.json")
    target.write_text(json.dumps(app.openapi(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
