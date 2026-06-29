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

            if len(query_data) >= 2:
                ans = str(row["Answer"])
                cell_value = str(query_data[1][0]["cell_value"])
                if ans in cell_value or cell_value in ans:
                    ans_in_query_data = True
                else:
                    ans_in_query_data = False
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
    file_name = "../data/PhytoBench-Data.xlsx"
    data = pd.read_excel(file_name, sheet_name=data_name)

    if model.name == "MODEL":
        rewrite_proxy = {"http:": None, "https": None}
        parallel = False
    # else:
    #     rewrite_proxy = {"http:": None, "https": None}
    #     parallel = False

    rewriter = Rewriter(
        data_name=data_name,
        model=model,
        proxy=rewrite_proxy,
        with_rag=with_rag,
        prompt_version=prompt_version,
        parallel=parallel,
    )
    rewritten_data = rewriter.rewrite(data)
    rewritten_data.to_excel(f"../output/{data_name}/{prompt_version}/{model.name}-rewrite-{use_rag}.xlsx")


if __name__ == "__main__":
    sheet_names = ["Sheet1"]

    models = (FunctionModel.MODEL,)

    with_rag = False
    for model in models:
        for prompt_version in ["Optimized", "Raw"]:
            for data_name in sheet_names:
                eval_trial(data_name, model, with_rag, prompt_version)
