import json
import os.path

import pandas as pd
import requests
from tqdm import tqdm

from utils import save_file_txt

RAG_KS_URL = "Change_to_your_rag_ks_url"
repo_id_DICT = {"Chinese": "Change_to_your_chinese_repo_id", "English": "Change_to_your_english_repo_id"}

top_n, min_score = 30, 0.5

RAG_PROXY = {"http:": None, "https": None}


def rag_run_result(query, repo_id, rag_ks_url, page_num=1, page_size=10):
    headers = {
        # 'X-Auth-Token': token,
        "Content-Type": "application/json"
    }
    body = {
        "repo_id": repo_id,
        "content": query,
        "page_num": page_num,
        "page_size": page_size,
    }
    response = requests.post(
        rag_ks_url,
        headers=headers,
        json=body,
        proxies=RAG_PROXY,
        verify=False,
    )

    message = response.json()

    if response.status_code != 200:
        print(f"Failed to get response from CSS: msg={message}")

    return message


def format_search_result(message):
    search_result = ""

    data = pd.DataFrame(message["doc_list"])

    data["score"] = data["score"].astype(float)
    data_sorted = data.sort_values(by="score", ascending=False, ignore_index=True)

    for x, rows in data_sorted.iterrows():
        line = f'[{x}]title: {rows["subtitle"]} \n content:{rows["content"]}\n'
        search_result = search_result + line
        if x >= 5:
            break
    return search_result


def get_rag(data, question_column, save_path):
    messages_list = []
    for i, rows in tqdm(data.iterrows(), total=len(data), desc="Process RAG"):
        try:
            if os.path.exists(save_path + str(i) + ".txt"):
                with open(save_path + str(i) + ".txt", "r", encoding="utf-8-sig") as f:
                    message = json.load(f)
            else:
                message = rag_run_result(rows[question_column], repo_id_DICT["English"], RAG_KS_URL, page_size=top_n)
                save_file_txt(save_path, i, json.dumps(message))
            messages_list.append(format_search_result(message))
        except Exception as e:
            print(e)
            messages_list.append("Failure")
    data["Response"] = messages_list
    return data
