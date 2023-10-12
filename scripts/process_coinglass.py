import json

import pandas as pd

from perp.constants import COINGLASS_PATH

with open(COINGLASS_PATH, "r") as f:
    results = json.load(f)
dataMap = pd.DataFrame(results["dataMap"])
frDataMap = pd.DataFrame(results["frDataMap"])
dataMap == frDataMap
# merge the two dataframes
coinglass_df = dataMap.merge(
    frDataMap, left_index=True, right_index=True, suffixes=("", "_fr")
)
coinglass_df["timestamp"] = pd.to_datetime(results["dateList"], unit="ms")
coinglass_df.set_index("timestamp", inplace=True)

# change the percentage to decimal
coinglass_df = coinglass_df / 100
coinglass_df["price"] = results["priceList"]
