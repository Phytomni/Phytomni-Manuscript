import json
import os.path

import pandas as pd
import yaml
from langchain import PromptTemplate
from tqdm import tqdm

from llm import FunctionModel, get_model_result_futures
from rag.rag_run import get_rag

TEMPLATE_FILE_PATH = "prompt_template.yaml"
with open(TEMPLATE_FILE_PATH, "r", encoding="utf-8") as f:
    TEMPLATES = yaml.safe_load(f)


class Rewriter:
    def __init__(
        self,
        data_name: str,
        model: FunctionModel,
        proxy,
        with_rag: bool = False,
        prompt_version: str = "Optimized",
        parallel: bool = False,
    ):
        self.model_name = model.name
        self.model_func = model.value
        self.with_rag = with_rag
        self.data_name = data_name
        self.prompt_version = prompt_version
        if self.with_rag:
            self.template = PromptTemplate.from_template(
                TEMPLATES.get(prompt_version).get("with-rag"), template_format="jinja2"
            )
        else:
            self.template = PromptTemplate.from_template(
                TEMPLATES.get(prompt_version).get("no-rag"), template_format="jinja2"
            )
        if not os.path.exists(f"../output/{self.data_name}/{self.prompt_version}/temp"):
            os.makedirs(f"../output/{self.data_name}/{self.prompt_version}/temp")
        if not os.path.exists(f"../output/{self.data_name}/search_result"):
            os.makedirs(f"../output/{self.data_name}/search_result")
        self.proxy = proxy
        self.parallel = parallel

    def rewrite_chunk(self, data: pd.DataFrame, question_column: str, chunk_id: int) -> pd.DataFrame:
        if self.with_rag:
            data_with_prompt = self.get_prompt_with_rag(data, question_column)
        else:
            data_with_prompt = self.get_prompt_without_rag(data, question_column)
        rewrite_question_list = []
        for rix, row in tqdm(
            data_with_prompt.iterrows(), total=len(data_with_prompt), desc=f"Rewrite - Chunk {str(chunk_id)}"
        ):
            try:
                rewrite_res = self.model_func(row["prompt"], self.proxy)
            except Exception as e:
                print(f"question {rix} failed. exception: {e}")
                rewrite_res = "Failure"
            rewrite_question_list.append(rewrite_res)
        data["Rewrite_Result"] = rewrite_question_list
        return data

    def rewrite_chunk_parallel(self, data: pd.DataFrame, question_column: str = "Question") -> pd.DataFrame:
        if self.with_rag:
            data_with_prompt = self.get_prompt_with_rag(data, question_column)
        else:
            data_with_prompt = self.get_prompt_without_rag(data, question_column)
        data_with_prompt["Rewrite_Result"] = get_model_result_futures(data_with_prompt, self.model_func, self.proxy)
        return data_with_prompt

    def rewrite(self, data: pd.DataFrame, question_column: str = "Question") -> pd.DataFrame:
        chunk_size = 50
        res_list = []
        if self.with_rag:
            use_rag = "with-rag"
        else:
            use_rag = "no-rag"
        chunk_id = 0
        for i in tqdm(
            range(0, len(data), chunk_size), desc=f"Rewrite {len(data)} samples. Chunk Size = {chunk_size}."
        ):
            if self.parallel:
                chunk_res = self.rewrite_chunk_parallel(data.iloc[i : i + chunk_size], question_column)
            else:
                chunk_res = self.rewrite_chunk(data.iloc[i : i + chunk_size], question_column, chunk_id)
            chunk_res.to_excel(
                f"../output/{self.data_name}/{self.prompt_version}/temp/{self.model_name}-rewrite-{use_rag}-{i}~{i + chunk_size}.xlsx",
                index=False,
            )
            res_list.append(chunk_res)
            chunk_id += 1
        rewritten_df = pd.concat(res_list, axis=0)
        rewritten_df.to_excel(
            f"../output/{self.data_name}/{self.prompt_version}/{self.model_name}-rewrite-{use_rag}.xlsx", index=False
        )
        return rewritten_df

    def get_prompt_without_rag(self, data: pd.DataFrame, question_column: str = "Question") -> pd.DataFrame:
        data["prompt"] = data.apply(lambda row: self.template.format(question=row[question_column]), axis=1)
        data.to_excel(f"../output/{self.data_name}/{self.prompt_version}/prompt-no-rag.xlsx", index=False)
        return data

    def get_prompt_with_rag(self, data: pd.DataFrame, question_column: str = "Question") -> pd.DataFrame:
        df = get_rag(data, question_column, f"../output/{self.data_name}/search_result/")
        df["prompt"] = df.apply(
            lambda row: self.template.format(question=row[question_column], rag_info=row["Response"]), axis=1
        )
        df.to_excel(f"../output/{self.data_name}/{self.prompt_version}/prompt-with-rag.xlsx", index=False)
        return df
