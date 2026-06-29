import concurrent.futures
from enum import Enum
from functools import partial

import pandas as pd
import requests

from model.pangu_L1 import fetch_content_pangu_l1_reasoner
from model.pangu_L1 import fetch_content_pangu_l1_chat
from model.pangu_L1 import fetch_content_pangu_l0


class FunctionModel(Enum):
    PANGU_L1_R = partial(fetch_content_pangu_l1_reasoner)
    PANGU_L1_C = partial(fetch_content_pangu_l1_chat)
    PANGU_L0 = partial(fetch_content_pangu_l0)


def single_request(i, model_name, query, proxy):
    try:
        final_result = model_name(query, proxy)
    except Exception as e:
        print(f"question {i} failed. exception: {e}")
        final_result = "Failure"
    return [i, final_result]


def get_model_result_futures(data, model_name, proxy):
    with requests.Session() as session:
        session.verify = False
        MAX_WORKERS = 8

        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [
                executor.submit(single_request, i, model_name, rows["prompt"], proxy) for i, rows in data.iterrows()
            ]

            results = []

            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results.append(result)
            df = pd.DataFrame(results, columns=["Number", "Result"])
    return list(df.sort_values("Number")["Result"])
