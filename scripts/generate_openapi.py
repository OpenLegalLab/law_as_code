"""Export the FastAPI OpenAPI schema to docs/swagger.json."""

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from oll_api_server.app import app

spec = app.openapi()

out = pathlib.Path(__file__).resolve().parents[1] / "docs" / "swagger.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(spec, indent=2, ensure_ascii=False) + "\n")
print(f"Wrote {out}")
