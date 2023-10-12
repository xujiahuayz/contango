from pickle import dump


from perp.graphql import graphdata, query_structurer
from perp.constants import DATA_PATH

BATCH_SIZE = 1000
AAVE_V1_ENDPOINT = "https://api.thegraph.com/subgraphs/name/aave/protocol-multy-raw"

series3 = "reserveParamsHistoryItems"
specs3 = """
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
    totalBorrows
    totalBorrowsVariable
    totalBorrowsStable
    timestamp
"""

# "The `first` argument must be between 0 and 1000, but is 2000
if __name__ == "__main__":
    last_ts = 0
    data_reservepara = []
    while True:
        reservepara_query = query_structurer(
            series3,
            specs3,
            arg=f"first: {BATCH_SIZE}, orderBy: timestamp, orderDirection: asc, where: {{ timestamp_gt: {last_ts}}}",
        )
        res = graphdata(reservepara_query, url=AAVE_V1_ENDPOINT)
        if "data" in set(res) and res["data"][series3]:
            rows = res["data"][series3]
            data_reservepara.extend(rows)
            last_ts = rows[-1]["timestamp"]
        else:
            break

    dump(data_reservepara, open(DATA_PATH / "reservepara.pkl", "wb"))
