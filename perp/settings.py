import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent

COINGLASS_SECRET = os.environ.get("COINGLASS_SECRET")
KAIKO_API_KEY = os.environ.get("KAIKO_API_KEY")
