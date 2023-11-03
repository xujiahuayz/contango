import re
import subprocess

from perp.settings import PROJECT_ROOT

cli_path = PROJECT_ROOT.parent / "smart-order-router"


def run_command(
    token_in: str,
    token_out: str,
    amount: float,
    exact_in: bool,
    recipient: str,
    protocols: str,
    block_number: int,
) -> str | None:
    command = f"./bin/cli quote --tokenIn {token_in} --tokenOut {token_out} --amount {amount} "
    command += "--exactIn " if exact_in else "--exactOut "
    command += (
        f"--recipient {recipient} --protocols {protocols} --blockNumber {block_number}"
    )

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
        cwd=cli_path,
    )
    stdout, stderr = process.communicate()

    if process.returncode == 0:
        return stdout.decode()
    else:
        print("Command failed with error:")
        print(stderr.decode())
        return None


def parse_output(output: str) -> dict[str, str]:
    # Remove ANSI color codes
    output = re.sub(r"\x1b\[\d+m", "", output)

    # Split the output into lines
    lines = output.strip().split("\n")

    # Initialize an empty dictionary to store the parsed data
    data = {}

    # Initialize a variable to keep track of the current key
    current_key = None

    # Iterate over each line
    for line in lines:
        # Split the line into key and value
        if ":" in line:
            key, value = map(str.strip, line.split(":", 1))
            data[key] = value.strip('\n"')
            current_key = key
        else:
            # If the line does not contain a colon, it is a continuation of the previous line
            data[current_key] += ("\n" + line.strip()).strip('\n"')

    return data


output = run_command(
    "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",
    "0x1f9840a85d5af5bf1d1762f925bdaddc4201f984",
    1000,
    False,
    "0xAb5801a7D398351b8bE11C439e05C5B3259aeC9B",
    "v2,v3",
    18485500,
)

if output is not None:
    data = parse_output(output)
    for key, value in data.items():
        print(f"{key}: {value}")
