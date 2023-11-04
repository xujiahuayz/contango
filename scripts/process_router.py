import re

import numpy as np
import pandas as pd
from perp.constants import UNISWAP_TIME_SERIES_PATH
import gzip
import json


def parse_output(output: str) -> dict:
    if output is not None:
        # Remove ANSI color codes
        output = re.sub(r"\x1b\[\d+m", "", output)
        # Split the output into lines
        lines = output.strip().split("\n")
        if len(lines) <= 1:
            return {"Best Route": lines[0]}
    else:
        return {}

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


quotes = []
with gzip.open(UNISWAP_TIME_SERIES_PATH, "rt") as f:
    for line in f:
        result = json.loads(line)
        output = parse_output(output=result["output"])

        result.update(
            {
                "quote": output["Raw Quote Exact In"]
                if "Raw Quote Exact In" in output
                else np.nan
            }
        )
        quotes.append(result)

uniswap_df = pd.DataFrame(quotes)
