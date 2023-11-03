import asyncio
import gzip
import json
from itertools import product

from perp.constants import (
    SYMBOL_LIST,
    TIME_BLOCK_PATH,
    TOKEN_DICT,
    TRADE_SIZE_LIST,
    UNISWAP_TIME_SERIES_PATH,
    USD_STABLECOIN,
)
from perp.settings import PROJECT_ROOT

cli_path = PROJECT_ROOT.parent / "smart-order-router"

# get timestamp from TIME_BLOCK_PATH

time_block_dict = json.load(open(TIME_BLOCK_PATH, "r"))


async def run_command(
    token_in: str,
    token_out: str,
    amount: float,
    exact_in: bool,
    block_number: int,
    protocols: str = "v2,v3",
    recipient: str | None = None,
) -> str | None:
    command = (
        f"./bin/cli quote --tokenIn {token_in} --tokenOut {token_out} "
        f"--amount {amount} --protocols {protocols} --blockNumber {block_number} "
    )
    command += " --exactIn" if exact_in else " --exactOut"
    if recipient is not None:
        command += f" --recipient {recipient}"  # Fixed space before --recipient

    # Use asyncio subprocess functions
    process = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        shell=True,
        cwd=cli_path,
    )

    stdout, stderr = await process.communicate()

    if process.returncode == 0:
        return stdout.decode()
    else:
        print("Command failed with error:")
        print(stderr.decode())
        return None


# Async function to process a single task
async def process_task(task: tuple):
    print(task)
    (
        key,
        risk_asset,
        buy_risk_asset,
        trade_size,
        block_number,
        token_in,
        token_out,
    ) = task
    output = await run_command(
        token_in=token_in,
        token_out=token_out,
        amount=trade_size,
        exact_in=buy_risk_asset,
        block_number=block_number,
    )
    return {
        "timestamp": key,
        "buy_risk_asset": buy_risk_asset,
        "risk_asset": risk_asset,
        "trade_size": trade_size,
        "output": output,
    }


# Async function to run all tasks and collect results
async def run_all_tasks(tasks: list) -> list:
    results = await asyncio.gather(*(process_task(task) for task in tasks))
    return results


# Run the async event loop
if __name__ == "__main__":
    with gzip.open(UNISWAP_TIME_SERIES_PATH, "wt") as f:
        for key, block_data in time_block_dict.items():
            tasks = []
            block_number = block_data["height"]
            print(block_number + "==========")

            for risk_asset in SYMBOL_LIST:
                token_prefix = "W" if risk_asset in ["ETH", "BTC"] else ""
                risk_asset_token = token_prefix + risk_asset
                for buy_risk_asset, trade_size in product(
                    [True, False], TRADE_SIZE_LIST
                ):
                    token_in = TOKEN_DICT[
                        USD_STABLECOIN if buy_risk_asset else risk_asset_token
                    ]
                    token_out = TOKEN_DICT[
                        risk_asset_token if buy_risk_asset else USD_STABLECOIN
                    ]
                    tasks.append(
                        (
                            key,
                            risk_asset_token,
                            buy_risk_asset,
                            trade_size,
                            block_number,
                            token_in,
                            token_out,
                        )
                    )
            results = asyncio.run(run_all_tasks(tasks=tasks))
            for result in results:
                f.write(json.dumps(result) + "\n")


# out_put_list = []
# for key, value in time_block_dict.items():
#     for risk_asset in SYMBOL_LIST:
#         risk_asset = ("W" if risk_asset in ["ETH", "BTC"] else "") + risk_asset
#         for buy_risk_asset in [True, False]:
#             if buy_risk_asset:
#                 token_in = TOKEN_DICT[USD_STABLECOIN]
#                 token_out = TOKEN_DICT[risk_asset]
#             else:
#                 token_in = TOKEN_DICT[risk_asset]
#                 token_out = TOKEN_DICT[USD_STABLECOIN]
#             for s in TRADE_SIZE_LIST:

#                 output = run_command(
#                     token_in=token_in,
#                     token_out=token_out,
#                     amount=1000,
#                     exact_in=buy_risk_asset,
#                     block_number=value["height"],
#                 )
#                 this_dict = {
#                     "timestamp": key,
#                     "buy_risk_asset": buy_risk_asset,
#                     'risk_asset': risk_asset,
#                     'trade_size': s
#                     "output": output,
#                 }
#                 out_put_list.append(this_dict)


# # def parse_output(output: str) -> dict:
# #     # Remove ANSI color codes
# #     output = re.sub(r"\x1b\[\d+m", "", output)

# #     # Split the output into lines
# #     lines = output.strip().split("\n")

# #     # Initialize an empty dictionary to store the parsed data
# #     data = {}

# #     # Initialize a variable to keep track of the current key
# #     current_key = None

# #     # Iterate over each line
# #     for line in lines:
# #         # Split the line into key and value
# #         if ":" in line:
# #             key, value = map(str.strip, line.split(":", 1))
# #             data[key] = value.strip('\n"')
# #             current_key = key
# #         else:
# #             # If the line does not contain a colon, it is a continuation of the previous line
# #             data[current_key] += ("\n" + line.strip()).strip('\n"')

# #     return data


# # # create an aync function to combine run_command and parse_output, and save data to json
# # async def fetch_router(
# #     token_in: str,
# #     token_out: str,
# #     amount: float,
# #     exact_in: bool,
# #     recipient: str,
# #     protocols: str,
# #     block_number: int,
# # ) -> dict | None:
# #     output = run_command(
# #         token_in,
# #         token_out,
# #         amount,
# #         exact_in,
# #         recipient,
# #         protocols,
# #         block_number,
# #     )

# #     if output is not None:
# #         data = parse_output(output)
# #         return data
# #     else:
# #         return None


# # output = run_command(
# #     "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",
# #     "0x1f9840a85d5af5bf1d1762f925bdaddc4201f984",
# #     1000,
# #     True,
# #     "0xAb5801a7D398351b8bE11C439e05C5B3259aeC9B",
# #     "v2,v3",
# #     18485500,
# # )

# # if output is not None:
# #     data = parse_output(output)
# #     for key, value in data.items():
# #         print(f"{key}: {value}")
