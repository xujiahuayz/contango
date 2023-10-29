import pandas as pd
import requests

from perp.constants import KAIKO_EXCHANGE_PATH

exchanges = requests.get(
    url="https://reference-data-api.kaiko.io/v1/exchanges", timeout=10
).json()["data"]
exchange_df = pd.DataFrame(exchanges)
# save exchange_df to pickle
exchange_df.to_pickle(KAIKO_EXCHANGE_PATH)
