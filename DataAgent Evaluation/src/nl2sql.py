import requests
from utils import get_token


class NL2SQLAgent:
    def __init__(self, workspace_id: str, subject_id: str, url: str, proxy, token):
        self.workspace_id = workspace_id
        self.subject_id = subject_id
        self.url = url
        self.proxy = proxy
        self.token = token

    def call_nl2sql(self, message_content: str, question_index: str):
        headers = {"Content-Type": "application/json", "X-Workspace-Id": self.workspace_id, "X-Auth-Token": self.token}

        data = {"message_content": message_content, "dialog_id": question_index, "subject_id": self.subject_id}
        # data = {
        #     "message_content": message_content,
        #     "dialog_id": question_index,
        #     "subject_id": self.subject_id,
        #     "need_insight": True,
        #     "simplify_response": True,
        # }
        try:
            response = requests.post(
                self.url, json=data, headers=headers, verify=False, proxies=self.proxy, timeout=90
            )

            if response.status_code == 200:
                json = response.json()
                query_data = json["query_data"]
                sql = json["sql_text"]
            else:
                sql = response.text
                query_data = "NA"
        except Exception as e:
            print(e)
            sql = str(e)
            query_data = "NA"
        return sql, query_data
