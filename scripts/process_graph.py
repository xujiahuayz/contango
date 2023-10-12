import gzip
import json

from perp.constants import AAVE_V3_PARAM_PATH

# # read the jsonl file as a list
with gzip.open(AAVE_V3_PARAM_PATH, "rt") as f:
    reservepara_list = [json.loads(line) for line in f]
