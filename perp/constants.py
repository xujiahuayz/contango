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
UNISWAP_PATH = DATA_PATH / "uniswap.jsonl.gz"
TIME_BLOCK_PATH = DATA_PATH / "time_block.json"
UNISWAP_TIME_SERIES_PATH = DATA_PATH / "uniswap_ts.jsonl.gz"

USD_STABLECOIN = "DAI"


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

KAIKO_EXCHANGES = ["Binance", "Huobi", "OkEX", "CoinEx"]

TOKEN_DICT = {
    "WETH": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
    "WBTC": "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599",
    "LINK": "0x514910771AF9Ca656af840dff83E8264EcF986CA",
    "UNI": "0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984",
    "AAVE": "0x7Fc66500c84A76Ad7e9c93437bFc5Ac33E2DDaE9",
    "CRV": "0xD533a949740bb3306d119CC777fa900bA034cd52",
    "MKR": "0x9f8F72aA9304c8B593d555F12eF6589cC3A579A2",
    "SNX": "0xC011a73ee8576Fb46F5E1c5751cA3B9Fe0af2a6F",
    "USDC": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
    "DAI": "0x6B175474E89094C44Da98b954EedeAC495271d0F",
}

TRADE_SIZE_LIST = [1e2, 1e4, 1e6]
