from perp.settings import PROJECT_ROOT

DATA_PATH = PROJECT_ROOT / "data"
FIGURE_PATH = PROJECT_ROOT / "figures"
CACHE_PATH = PROJECT_ROOT / ".cache"

INTEREST_TOKEN_PREFIX = "interest-"
DEBT_TOKEN_PREFIX = "debt-"

CONTANGO_TOKEN_PREFIX = "c-"
AAVE_TOKEN_PREFIX = "a-"


COINGLASS_PATH = DATA_PATH / "coinglass_ethusd.json"
AAVE_V3_PARAM_PATH = DATA_PATH / "reservepara_v3.jsonl.gz"
