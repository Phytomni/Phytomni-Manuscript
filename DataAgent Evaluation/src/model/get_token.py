import json
import requests
from datetime import datetime


def get_token_aksk(proxy):
    hw_ak_sk = {
        "access": {"key": "Change_to_your_ak"},
        "secret": {"key": "Change_to_your_sk"},
    }
    iam_path = "change_to_your_iam_path"
    headers = {
        "X-Client-Request-Time": "0",
        "Content-Type": "application/json",
    }
    data = {
        "auth": {
            "identity": {"methods": ["hw_ak_sk"], "hw_ak_sk": hw_ak_sk},
            "scope": {"project": {"name": "Change_to_your_project_name"}},
        }
    }
    resp = requests.post(
        iam_path,
        headers=headers,
        json=data,
        verify=False,
        proxies=proxy,
    )
    return resp.headers.get("X-Subject-Token")


def get_token():
    name_hc_token_name = "hc_token_name"
    name_hc_token_pwd = "hc_token_pwd"
    name_hc_token_domin = "hc_token_domin"
    name_hc_token_project_name = "hc_token_project_name"

    url = "change_to_your_iam_path"
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
        res = requests.request(
            "POST",
            url,
            headers=header,
            data=json.dumps(body),
            verify=False,
        )
        response_header = res.headers
        token = response_header["X-Subject-Token"]
        X_Auth_Token = token
        print("Obtain token ", datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3])
    except Exception as e:
        print("Obtain token failed: ", str(e))
        return None
    return X_Auth_Token
