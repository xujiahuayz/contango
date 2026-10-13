import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent


def _load_dotenv(path: Path) -> None:
    """Read KEY=VALUE lines from .env into os.environ; variables already set win."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


_load_dotenv(PROJECT_ROOT / ".env")

COINGLASS_SECRET = os.environ.get("COINGLASS_SECRET")
KAIKO_API_KEY = os.environ.get("KAIKO_API_KEY")
COINGLASS_API_KEY = os.environ.get("COINGLASS_API_KEY")
COINALYZE_API_KEY = os.environ.get("COINALYZE_API_KEY")
THEGRAPH_API_KEY = os.environ.get("THEGRAPH_API_KEY")
