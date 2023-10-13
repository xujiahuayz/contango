import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from perp.constants import DATA_PATH
from perp.env import DefiEnv, PlfPool, User, cPerp, cPool
from perp.utils import PriceDict
from scripts.process_coinglass import coinglass_fr_df
from scripts.process_graph import aave_rates_df


def c_perp_position_change(
    risk_asset: str,
    usd_asset: str,
    long_risk: bool,
) -> pd.DataFrame:
    coinglass_df = coinglass_fr_df(risk_asset=risk_asset)
    aave_df = aave_rates_df(
        risk_asset=("W" if risk_asset in ["ETH", "BTC"] else "") + risk_asset,
        usd_asset=usd_asset,
        long_risk=long_risk,
    )

    df = coinglass_df.merge(aave_df, how="left", left_index=True, right_index=True)

    if long_risk:
        env = DefiEnv(
            prices=PriceDict({usd_asset: 1.0, risk_asset: df.iloc[0]["price"]}),
        )
    else:
        df["price"] = 1 / df["price"]  # always use the short side as numeraire
        env = DefiEnv(
            prices=PriceDict({usd_asset: df.iloc[0]["price"], risk_asset: 1.0}),
        )

    market_user = User(
        env=env,
        name="MarketUser",
        funds_available={usd_asset: 999_999_999_999, risk_asset: 999_999_999_999},
    )

    INITIAL_FUNDS = 1_000_000
    PERIODIC_EXPONENT = 1 / (3 * 365)
    charlie = User(
        env=env,
        name="Charlie",
        funds_available={usd_asset if long_risk else risk_asset: INITIAL_FUNDS},
    )
    plf_risk = PlfPool(
        env=env,
        initiator=market_user,
        initial_starting_funds=1_000_000,
        initial_borrowing_funds=0,
        asset_name=risk_asset,
        collateral_factor=0.7,
        liquidation_threshold=0.8,
        flashloan_fee=0,
        periodic_exponent=PERIODIC_EXPONENT,
    )
    plf_usd = PlfPool(
        env=env,
        initiator=market_user,
        initial_starting_funds=1_000_000,
        initial_borrowing_funds=0,
        asset_name=usd_asset,
        collateral_factor=0.7,
        liquidation_threshold=0.8,
        flashloan_fee=0,
        periodic_exponent=PERIODIC_EXPONENT,
    )
    c_risk_pool = cPool(
        env=env,
        asset_name=risk_asset,
        funds_available=1_000_000,
        c_ratio=0.2,
        periodic_exponent=PERIODIC_EXPONENT,
    )
    c_usd_pool = cPool(
        env=env,
        asset_name=usd_asset,
        funds_available=1_000_000,
        c_ratio=0.2,
        periodic_exponent=PERIODIC_EXPONENT,
    )

    cperp1 = cPerp(
        env=env,
        position_name="cperp1",
        initiator_name="Charlie",
        init_asset=usd_asset if long_risk else risk_asset,
        target_asset=risk_asset if long_risk else usd_asset,
        target_quantity=1,
        target_collateral_factor=0.8,
        trading_slippage=0,
    )

    # initiate a new float column in coinglass_aave_df
    df["cperp_health"] = np.nan
    df["cperp_pnl"] = np.nan
    df["cperp_value"] = np.nan
    df["cperp_principal"] = np.nan
    df["cperp_debt"] = np.nan

    # get index and row values in coinglass_aave_df
    for index, row_values in df.iterrows():
        if long_risk:
            plf_risk.supply_apy = row_values["liquidityRate"]
            plf_usd.borrow_apy = row_values["variableBorrowRate"]
            env.prices[risk_asset] = row_values["price"]
        else:
            plf_usd.supply_apy = row_values["liquidityRate"]
            plf_risk.borrow_apy = row_values["variableBorrowRate"]
            env.prices[usd_asset] = row_values["price"]
        # assign values to the row
        df.at[index, "cperp_health"] = cperp1.plf_health
        df.at[index, "cperp_pnl"] = charlie.wealth - INITIAL_FUNDS
        df.at[index, "cperp_value"] = cperp1.value
        df.at[index, "cperp_principal"] = cperp1.funds_available[
            (plf_risk if long_risk else plf_usd).interest_token_name
        ]
        df.at[index, "cperp_debt"] = cperp1.funds_available[
            (plf_usd if long_risk else plf_risk).borrow_token_name
        ]

        env.accrue_interest()

    # rolling diff of cperp_value
    df["cperp_value_diff"] = df["cperp_value"].diff()

    df["cperp_principal_value_change"] = df["cperp_principal"].shift(1) * df[
        "price"
    ].diff(1)

    df["cperp_funding_payment"] = (
        df["cperp_value_diff"] - df["cperp_principal_value_change"]
    )

    df["Contango"] = df["cperp_funding_payment"] / (df["price"] * df["cperp_principal"])

    return df


if __name__ == "__main__":
    coinglass_aave_df = c_perp_position_change(
        risk_asset="ETH", usd_asset="USDC", long_risk=False
    )
    plt.plot(coinglass_aave_df["Binance"], label="Binance")
    plt.plot(coinglass_aave_df["OKX"], label="OKX")

    plt.plot(coinglass_aave_df["dYdX"], label="dYdX")
    plt.plot(coinglass_aave_df["Contango"], label="Contango")
    plt.legend()

    coinglass_aave_df.to_excel(DATA_PATH / "coinglass_aave_df.xlsx")
