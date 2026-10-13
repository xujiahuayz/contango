import gzip
import json

import pandas as pd

from perp.constants import COINGLASS_PATH, OUT_PRICE_PATH

results = []
with gzip.open(COINGLASS_PATH, "rt", encoding="utf-8") as f:
    for line in f:
        results.append(json.loads(line))

prices = []
with gzip.open(OUT_PRICE_PATH, "rt", encoding="utf-8") as f:
    for line in f:
        # reservepara_list.update(json.loads(line))
        prices.append(json.loads(line))
prices = pd.DataFrame(prices)
def coinglass_fr_df(risk_asset: str) -> pd.DataFrame:
    long_df = pd.DataFrame(results)
    dataMap = long_df[long_df['base'] == risk_asset]
    dataMap['t'] = (
        pd.to_datetime(dataMap['t'], unit='s', utc=True)
    )
    dataMap = dataMap.set_index(['t', 'exchange_name'])['funding_rate']
    price = prices[prices['base'] == risk_asset].set_index(['t'])['price']
    price.index = pd.to_datetime(price.index, unit="s", utc=True).tz_convert("UTC")
    df = dataMap.groupby(level=[0, 1]).last().unstack()
    df.index = pd.to_datetime(df.index, unit="s", utc=True).tz_convert("UTC")

    # change the percentage to decimal
    df = df / 100
    df["price"] = price
    return df.iloc[:-1]
#
#
if __name__ == "__main__":
    coinglass_df = coinglass_fr_df(risk_asset="ETH")
    print(coinglass_df)
