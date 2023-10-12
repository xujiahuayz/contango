import json
from pprint import pprint

import requests


def query_structurer(series: str, spec: str, arg: str = "") -> str:
    # format query arguments
    if arg != "":
        arg = "(" + arg + ")"

    # format query content
    q = series + arg + "{" + spec + "}"
    return q


def graphdata(*q, url: str):
    # pack all subqueries into one big query concatenated with linebreak '\n'
    query = "{" + "\n".join(q) + "}"

    # pretty print out query
    pprint(query)

    r = requests.post(url, json={"query": query})

    response_json = json.loads(r.text)
    return response_json
