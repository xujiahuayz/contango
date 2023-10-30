import gzip
import json

import requests

from perp.constants import (
    SYMBOL_LIST,
    TOKEN_DICT,
    TRADE_SIZE_LIST,
    UNISWAP_PATH,
    USD_STABLECOIN,
)

UNISWAP_END_POINT = "https://api.uniswap.org/v2/quote"

headers = {
    "origin": "https://app.uniswap.org",
}


with gzip.open(UNISWAP_PATH, "wt") as f:
    for risk_asset in SYMBOL_LIST:
        risk_asset = ("W" if risk_asset in ["ETH", "BTC"] else "") + risk_asset

        for buy_risk_asset in [True, False]:
            if buy_risk_asset:
                data_entry = {
                    "tokenIn": TOKEN_DICT[USD_STABLECOIN],
                    "tokenOut": TOKEN_DICT[risk_asset],
                    "type": "EXACT_INPUT",
                }
            else:
                data_entry = {
                    "tokenIn": TOKEN_DICT[risk_asset],
                    "tokenOut": TOKEN_DICT[USD_STABLECOIN],
                    "type": "EXACT_OUTPUT",
                }

            for s in TRADE_SIZE_LIST:
                data = {
                    "tokenInChainId": 1,
                    "tokenOutChainId": 1,
                    "amount": str(
                        int(s * 10 ** (18 if USD_STABLECOIN == "DAI" else 6))
                    ),
                    "sendPortionEnabled": True,
                    "configs": [
                        {"useSyntheticQuotes": False, "routingType": "DUTCH_LIMIT"},
                        {
                            "protocols": ["V2", "V3", "MIXED"],
                            "enableUniversalRouter": True,
                            "routingType": "CLASSIC",
                        },
                    ],
                }
                data.update(data_entry)

                response = requests.post(
                    UNISWAP_END_POINT, headers=headers, json=data, timeout=500
                )
                if response.status_code != 200:
                    print(response.status_code)
                    print(response.text)
                    continue
                print(data)
                result = response.json()
                result.update(
                    {
                        "buy_risk_asset": buy_risk_asset,
                        "risk_asset": risk_asset,
                        "trade_size": s,
                    }
                )
                f.write(json.dumps(result) + "\n")
