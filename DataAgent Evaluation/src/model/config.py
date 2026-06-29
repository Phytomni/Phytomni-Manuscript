import httpx
from openai import OpenAI


def fetch_content(query, proxy):
    client = OpenAI(
        api_key="change_to_your_api_key",
        base_url="change_to_your_base_url",
        http_client=httpx.Client(verify=False),
        # timeout=30,
    )
    client.proxy = proxy

    messages = [{"role": "user", "content": query}]
    response = client.chat.completions.create(
        model="change_to_your_model_name",
        messages=messages,
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    print(fetch_content("Who are you?", {"http:": None, "https": None}))
