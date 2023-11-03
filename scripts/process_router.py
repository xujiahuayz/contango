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
