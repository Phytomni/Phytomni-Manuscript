import os.path
import warnings

import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

from llm import FunctionModel
from nl2sql import NL2SQLAgent
from rewrite import Rewriter
from utils import get_token

PROXY = {"http:": None, "https": None}
PROMPT_VERSIONS = ["Optimized", "Raw"]


def eval_nl2sql(data, nlsql_agent):
    sql_list = []
    query_data_list = []
    cell_value_list = []
    judge_res_list = []
    for qix, row in tqdm(data.iterrows(), total=len(data), desc="NL2SQL Eval"):
        sql, query_data = nlsql_agent.call_nl2sql(row["Rewrite_Result"], str(qix))
        sql_list.append(sql)
        query_data_list.append(query_data)
        ans_in_query_data = None
        if query_data == "NA":
            cell_value = "NA"
            ans_in_query_data = False
        else:
            if len(query_data) >= 2:
                cell_value = query_data[1][0]["cell_value"]
            else:
                cell_value = "NA"

        cell_value_list.append(cell_value)
        judge_res_list.append(ans_in_query_data)
    data["insight_sql"] = sql_list
    data["insight_answer"] = query_data_list
    data["ans_in_query_data"] = judge_res_list
    data["cell_value"] = cell_value_list
    return data


def eval_trial(data_name, model, with_rag, prompt_version):
    token = get_token(PROXY)
    if with_rag:
        use_rag = "with-rag"
    else:
        use_rag = "no-rag"

    print(f"Exp for {data_name} - {model.name} - {use_rag} - {prompt_version}")

    rewritten_data = pd.read_excel(f"../output/{data_name}/{prompt_version}/{model.name}-rewrite-{use_rag}.xlsx")

    workspace_id = "Change_to_your_workspace_id"
    subject_id = "Change_to_your_subject_id"
    nl2sql_url = "Change_to_your_nl2sql_url"

    nl2sql_agent = NL2SQLAgent(workspace_id, subject_id, nl2sql_url, proxy=PROXY, token=token)
    nl2sql_data = eval_nl2sql(rewritten_data, nl2sql_agent)
    if not os.path.exists(f"../result/{data_name}/{prompt_version}/"):
        os.makedirs(f"../result/{data_name}/{prompt_version}/")
    nl2sql_data.to_excel(f"../result/{data_name}/{prompt_version}/{model.name}-{use_rag}-eval.xlsx", index=False)


if __name__ == "__main__":
    sheet_names = ["Sheet1"]

    models = (FunctionModel.MODEL,)

    with_rag = False
    for model in models:
        for prompt_version in ["Optimized", "Raw"]:
            for data_name in sheet_names:
                eval_trial(data_name, model, with_rag, prompt_version)
