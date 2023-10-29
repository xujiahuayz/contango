from perp.settings import PROJECT_ROOT

DATA_PATH = PROJECT_ROOT / "data"
FIGURE_PATH = PROJECT_ROOT / "figures"
CACHE_PATH = PROJECT_ROOT / ".cache"
TABLE_PATH = PROJECT_ROOT / "tables"

INTEREST_TOKEN_PREFIX = "interest-"
DEBT_TOKEN_PREFIX = "debt-"

CONTANGO_TOKEN_PREFIX = "c-"
AAVE_TOKEN_PREFIX = "a-"


COINGLASS_PATH = DATA_PATH / "coinglass.jsonl.gz"
AAVE_V3_PARAM_PATH = DATA_PATH / "reservepara_v3.jsonl.gz"
KAIKO_EXCHANGE_PATH = DATA_PATH / "exchange_df.pkl"
KAIKO_SLIPPAGE_PATH = DATA_PATH / "slippage.jsonl.gz"


SYMBOL_LIST = [
    "ETH",
    "BTC",
    "LINK",
    "UNI",
    "AAVE",
    "CRV",
    "MKR",
    # "BAL",
    "SNX",
    # "LDO",
    # "1INCH",
    # "ENS",
    # "RPL",
]

PRODUCT_LIST = [
    # "Bitmex",
    "Binance",
    "Bybit",
    "OKX",
    "Huobi",
    "Gate",
    "Bitget",
    "dYdX",
    "CoinEx",
    # "BingX",
    "Contango",
]
