import json
import os
from datetime import datetime

import requests


def get_iam_token(user_name: str, passwd: str, domain_name: str, project_name: str):
    url = "Change_to_your_iam_url"
    body = {
        "auth": {
            "identity": {
                "methods": ["password"],
                "password": {"user": {"name": user_name, "password": passwd, "domain": {"name": domain_name}}},
            },
            "scope": {"project": {"name": project_name}},
        }
    }

    header = {"Content-Type": "application/json"}
    try:
        res = requests.request("POST", url, headers=header, data=json.dumps(body), verify=False)
        response_header = res.headers
        token = response_header["X-Subject-Token"]
        print("Obtain token ", datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3])
    except Exception as e:
        print("Obtain token failed: ", str(e))
        return None
    return token


def save_file_txt(save_path, filename, content):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(save_path + str(filename) + ".txt", "w", encoding="utf-8-sig") as f:
        f.write(content)
        f.close()


def get_token(proxy):
    name_hc_token_name = "Change_to_your_iam_user_name"
    name_hc_token_pwd = "Change_to_your_iam_password"
    name_hc_token_domin = "Change_to_your_iam_domain_name"
    name_hc_token_project_name = "Change_to_your_iam_project_name"
    url = "Change_to_your_iam_url"

    body = {
        "auth": {
            "identity": {
                "methods": ["password"],
                "password": {
                    "user": {
                        "name": name_hc_token_name,
                        "password": name_hc_token_pwd,
                        "domain": {"name": name_hc_token_domin},
                    }
                },
            },
            "scope": {"project": {"name": name_hc_token_project_name}},
        }
    }

    header = {"Content-Type": "application/json"}
    try:
        res = requests.request("POST", url, headers=header, data=json.dumps(body), proxies=proxy, verify=False)
        response_header = res.headers
        X_Auth_Token = response_header["X-Subject-Token"]
        # print(X_Auth_Token)
        # X_Auth_Token = token
        # print("Obtain token ", datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
    except Exception as e:
        print("Obtain token failed: ", str(e))
        return None
    return X_Auth_Token


def get_token_aksk(proxy):
    hw_ak_sk = {
        "access": {"key": "Change_to_your_ak"},
        "secret": {"key": "Change_to_your_sk"},
    }
    iam_path = "Change_to_your_iam_path"
    headers = {"X-Client-Request-Time": "0", "Content-Type": "application/json"}
    data = {
        "auth": {
            "identity": {"methods": ["hw_ak_sk"], "hw_ak_sk": hw_ak_sk},
            "scope": {"project": {"name": "Change_to_your_project_name"}},
        }
    }
    resp = requests.post(iam_path, headers=headers, json=data, verify=False, proxies=proxy)
    print(resp)
    return resp.headers.get("X-Subject-Token")
