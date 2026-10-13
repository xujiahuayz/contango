import datetime as dt
import json
import gzip
import random
from decimal import Decimal, getcontext
import requests

from perp.constants import AAVE_V3_PARAM_PATH, DAI_PATH, SYMBOL_LIST
from perp.settings import THEGRAPH_API_KEY

# ====== CONFIG ======
API_KEY = THEGRAPH_API_KEY
DAYS = 180
STEP_HOURS = 8

# Multi-chain config
CHAINS = ["mainnet-v3", "mainnet-v2", "polygon-v3", "arbitrum-v3", "optimism-v3"]
SYMBOLS = ['W' + s if s in ('ETH', 'BTC') else s for s in SYMBOL_LIST]
MULTI_CHAIN_PATH = AAVE_V3_PARAM_PATH

# DAI-specific config
DAI_ARB = "0xda10009cbd5d07dd0cecc66161fc93d7c9000da1"
DAI_PATH_OUT = DAI_PATH

getcontext().prec = 40

# ====== SUBGRAPH IDS ======
GATEWAY = "https://gateway.thegraph.com/api/{}/subgraphs/id/{}"

AAVE_IDS = {
    "mainnet-v3": "Cd2gEDVeqnjBn1hSeqFMitw8Q1iiyV9FYUZkLNRcL87g",
    "mainnet-v2": "8wR23o1zkS4gpLqLNU4kG3JHYVucqGyopL5utGxP2q1N",
    "polygon-v3": "6yuf1C49aWEscgk5n9D1DekeG1BCk5Z9imJYJT3sVmAT",
    "arbitrum-v3": "4xyasjQeREe7PxnF6wVdobZvCw5mhoHZq3T7guRpuNPf",
    "optimism-v3": "3RWFxWNstn4nP3dXiDfKi9GgBoHx7xzc7APkXs1MLEgi",
}

BLOCKS_IDS = {
    "mainnet": "9A6bkprqEG2XsZUYJ5B2XXp6ymz9fNcn4tVPxMWDztYC",
    "polygon": "DMnXZnphMTkcFiK5NHm6LzwhJ7yUy7seVnZuNkNsXLHp",
    "arbitrum": "64DCU8nq48qdDABnobpDafsg7RF75Rx5soKrHiGA8mqp",
    "optimism": "HsWM1oAXHGWdkH8bK98UrW38PvyPx6Q4waRow2LT8mcp",
}

# DAI-specific Aave V3 Arbitrum
AAVE_ARB_ID = "DLuE98kEb5pQNXAcKFQGQgfSQ57Xdou4jnVbAEqMfy3B"


# ====== HELPER FUNCTIONS ======
def gql(api_key, subgraph_id, query, variables):
    """Simple GraphQL query"""
    url = GATEWAY.format(api_key, subgraph_id)
    r = requests.post(url, json={"query": query, "variables": variables}, timeout=30)
    r.raise_for_status()
    data = r.json()
    if "errors" in data:
        raise RuntimeError(f"GraphQL error: {data['errors']}")
    return data["data"]


def base_chain(chain):
    """Extract base chain from chain-version key"""
    return chain.split("-")[0]


def build_timestamps(days, step_hours):
    """Build timestamp grid aligned to step_hours intervals"""
    now = dt.datetime.now(dt.timezone.utc)
    end = now.replace(minute=0, second=0, microsecond=0)
    end -= dt.timedelta(hours=end.hour % step_hours)
    start = end - dt.timedelta(days=days)
    stamps = []
    cur = start
    while cur <= end:
        stamps.append((cur.isoformat().replace("+00:00", "Z"), int(cur.timestamp())))
        cur += dt.timedelta(hours=step_hours)
    return stamps


def to_dec(x):
    """Convert to Decimal safely"""
    return Decimal(str(x)) if x is not None else Decimal(0)


def ray_to_pct(ray):
    """Convert ray (27 decimals) to percentage"""
    return (Decimal(str(ray)) / Decimal(10) ** 27) * Decimal(100)


def compute_util(total_liq, avail, stable, variable):
    """Calculate utilization rate"""
    borrows = stable + variable
    if total_liq and total_liq > 0:
        return min(borrows / total_liq, Decimal(1))
    denom = avail + borrows
    return borrows / denom if denom > 0 else None


# ====== MULTI-CHAIN FUNCTIONS ======
def get_block(api_key, base, ts):
    """Get block number at or before timestamp"""
    q = """query($ts: BigInt!) {
        blocks(first: 1, orderBy: timestamp, orderDirection: desc, where: {timestamp_lte: $ts}) {
            number timestamp
        }
    }"""
    try:
        data = gql(api_key, BLOCKS_IDS[base], q, {"ts": ts})
        if data["blocks"]:
            b = data["blocks"][0]
            return int(b["timestamp"]), int(b["number"])
    except:
        pass
    return None


def get_reserves(api_key, aave_id, block, symbols):
    """Get reserve data at specific block"""
    q = """query($symbols: [String!], $block: Block_height) {
        reserves(block: $block, where: {symbol_in: $symbols}) {
            id symbol underlyingAsset decimals lastUpdateTimestamp
            liquidityRate variableBorrowRate availableLiquidity totalLiquidity
            totalPrincipalStableDebt totalCurrentVariableDebt
        }
    }"""
    data = gql(api_key, aave_id, q, {"symbols": symbols, "block": {"number": block}})
    return data.get("reserves", [])


def make_row(chain, iso, ts, block, reserve):
    """Create output row from reserve data"""
    avail = to_dec(reserve.get("availableLiquidity"))
    stable = to_dec(reserve.get("totalPrincipalStableDebt"))
    var = to_dec(reserve.get("totalCurrentVariableDebt"))
    totliq = to_dec(reserve.get("totalLiquidity")) if reserve.get("totalLiquidity") is not None else None
    util = compute_util(totliq, avail, stable, var)
    liq_ray = to_dec(reserve.get("liquidityRate"))
    bor_ray = to_dec(reserve.get("variableBorrowRate"))

    return {
        "chain": chain,
        "date": iso,
        "ts": ts,
        "block": block,
        "symbol": reserve["symbol"],
        "underlyingAsset": reserve["underlyingAsset"],
        "decimals": int(reserve["decimals"]) if reserve.get("decimals") is not None else None,
        "timestamp": int(reserve["lastUpdateTimestamp"]) if reserve.get("lastUpdateTimestamp") else None,
        "liquidityRateRay": str(liq_ray),
        "variableBorrowRateRay": str(bor_ray),
        "availableLiquidity": str(avail),
        "totalLiquidity": None if totliq is None else str(totliq),
        "totalPrincipalStableDebt": str(stable),
        "totalCurrentVariableDebt": str(var),
        "utilization": None if util is None else str(util),
        "liquidityRatePct": None if liq_ray == 0 else str(ray_to_pct(liq_ray)),
        "variableBorrowRatePct": None if bor_ray == 0 else str(ray_to_pct(bor_ray)),
    }


# ====== DAI-SPECIFIC FUNCTIONS ======
def get_dai_reserve():
    """Get DAI reserve metadata"""
    q = """query($addr: Bytes!) { 
        reserves(where: { underlyingAsset: $addr }) { 
            id symbol decimals underlyingAsset
        } 
    }"""
    data = gql(API_KEY, AAVE_ARB_ID, q, {"addr": DAI_ARB})
    return data["reserves"][0]


def get_dai_history(reserve_id, from_ts):
    """Get DAI historical rate data"""
    out, skip = [], 0
    while True:
        q = """query($rid: String!, $from: Int!, $skip: Int!) {
            reserveParamsHistoryItems(where: { reserve: $rid, timestamp_gte: $from }
            orderBy: timestamp orderDirection: asc first: 1000 skip: $skip) {
                timestamp liquidityRate variableBorrowRate
            }
        }"""
        data = gql(API_KEY, AAVE_ARB_ID, q, {"rid": reserve_id, "from": int(from_ts), "skip": skip})
        batch = data["reserveParamsHistoryItems"]
        if not batch:
            break
        out.extend(batch)
        skip += 1000
    return sorted(out, key=lambda x: int(x["timestamp"]))


def get_dai_snapshot(ts):
    """Get DAI snapshot at timestamp with jitter for missing blocks"""
    jitters = [0, 15, -15, 30, -30, 60, -60, 90, -90, 120, -120]
    random.shuffle(jitters)

    for m in jitters:
        t_try = ts + m * 60
        block_info = get_block(API_KEY, "arbitrum", t_try)
        if not block_info:
            continue

        _, block = block_info
        try:
            q = """query($addr: Bytes!, $block: Block_height) {
                reserves(block: $block, where: { underlyingAsset: $addr }) {
                    lastUpdateTimestamp availableLiquidity totalLiquidity
                    totalPrincipalStableDebt totalCurrentVariableDebt
                }
            }"""
            data = gql(API_KEY, AAVE_ARB_ID, q, {"addr": DAI_ARB, "block": {"number": block}})
            if data.get("reserves"):
                r = data["reserves"][0]
                return {
                    "lastUpdateTimestamp": int(r["lastUpdateTimestamp"]) if r.get("lastUpdateTimestamp") else None,
                    "availableLiquidity": r.get("availableLiquidity", ""),
                    "totalLiquidity": r.get("totalLiquidity", ""),
                    "totalPrincipalStableDebt": r.get("totalPrincipalStableDebt", ""),
                    "totalCurrentVariableDebt": r.get("totalCurrentVariableDebt", ""),
                }
        except:
            continue
    return None


# ====== MAIN FETCH FUNCTIONS ======
def fetch_multi_chain():
    """Fetch data for all chains and symbols"""
    print("\n=== FETCHING MULTI-CHAIN DATA ===")
    stamps = build_timestamps(DAYS, STEP_HOURS)
    print(f"[INFO] Built {len(stamps)} sample timestamps with step={STEP_HOURS}h over last {DAYS} days")

    rows = []
    for chain in CHAINS:
        base = base_chain(chain)
        print(f"[INFO] Processing {chain}...")

        for iso, ts in stamps:
            block_info = get_block(API_KEY, base, ts)
            if not block_info:
                continue

            block_ts, block = block_info
            try:
                reserves = get_reserves(API_KEY, AAVE_IDS[chain], block, SYMBOLS)
                for r in reserves:
                    rows.append(make_row(chain, iso, block_ts, block, r))
            except Exception as e:
                print(f"[ERROR] {chain} at block {block}: {e}")

    with gzip.open(MULTI_CHAIN_PATH, "wt", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[INFO] Saved {len(rows)} rows → {MULTI_CHAIN_PATH}")


def fetch_dai_detailed():
    """Fetch DAI-specific data with detailed snapshots"""
    print("\n=== FETCHING DAI DETAILED DATA ===")
    now = int(dt.datetime.now(dt.timezone.utc).timestamp())
    start = now - DAYS * 86400
    grid = list(range(start - (start % (STEP_HOURS * 3600)), now, STEP_HOURS * 3600))

    reserve = get_dai_reserve()
    print(f"[INFO] Found DAI reserve: {reserve['symbol']}")

    history = get_dai_history(reserve["id"], grid[0])
    print(f"[INFO] Got {len(history)} history items")

    # Resample history to grid
    rows, i = [], 0
    for ts in grid:
        while i + 1 < len(history) and int(history[i + 1]["timestamp"]) <= ts:
            i += 1
        h = history[i] if history and int(history[i]["timestamp"]) <= ts else {}

        row = {
            "sample_timestamp": ts,
            "sample_datetime_utc": dt.datetime.fromtimestamp(ts, tz=dt.timezone.utc).isoformat(),
            "id": reserve["id"],
            "symbol": reserve["symbol"],
            "underlyingAsset": reserve["underlyingAsset"],
            "decimals": int(reserve["decimals"]) if reserve.get("decimals") is not None else None,
            "timestamp": h.get("timestamp", "") if h else "",
            "liquidityRate": h.get("liquidityRate", "") if h else "",
            "variableBorrowRate": h.get("variableBorrowRate", "") if h else "",
            "availableLiquidity": h.get("availableLiquidity", "") if h else "",
            "totalLiquidity": h.get("totalLiquidity", "") if h else "",
            "totalPrincipalStableDebt": h.get("totalPrincipalStableDebt", "") if h else "",
            "totalCurrentVariableDebt": h.get("totalCurrentVariableDebt", "") if h else "",
        }

        # Try to get snapshot data
        snap = get_dai_snapshot(ts)
        if snap:
            row.update(snap)

        rows.append(row)

    with gzip.open(DAI_PATH_OUT, "wt") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    print(f"[INFO] Saved {len(rows)} rows → {DAI_PATH_OUT}")


def main():
    """Run all fetch operations"""
    print(f"Starting data fetch: {DAYS} days, {STEP_HOURS}h intervals")

    # Fetch multi-chain data
    fetch_multi_chain()

    # Fetch DAI detailed data
    fetch_dai_detailed()

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()