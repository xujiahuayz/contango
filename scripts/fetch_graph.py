import gzip
import json

from perp.constants import AAVE_V3_PARAM_PATH
from perp.graphql import graphdata, query_structurer

BATCH_SIZE = 1000
AAVE_V3_ENDPOINT = "https://api.thegraph.com/subgraphs/name/aave/protocol-v3"

series = "reserveParamsHistoryItems"
specs = """
    reserve{
      symbol
    }
    variableBorrowRate
    variableBorrowIndex
    utilizationRate
    stableBorrowRate
    averageStableBorrowRate
    liquidityIndex
    liquidityRate
    totalLiquidity
    totalLiquidityAsCollateral
    availableLiquidity
    timestamp
"""

last_ts = 0
with gzip.open(AAVE_V3_PARAM_PATH, "wt") as f:
    while True:
        reservepara_query = query_structurer(
            series,
            specs,
            arg=f"first: {BATCH_SIZE}, orderBy: timestamp, orderDirection: asc, where: {{ timestamp_gt: {last_ts}}}",
        )
        res = graphdata(reservepara_query, url=AAVE_V3_ENDPOINT)
        if "data" in set(res) and res["data"][series]:
            rows = res["data"][series]
            f.write("\n".join([json.dumps(row) for row in rows]) + "\n")
            last_ts = rows[-1]["timestamp"]
        else:
            break
