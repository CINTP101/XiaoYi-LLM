"""Example backend/client call: python examples/app_api_client.py."""
import os
import uuid
import httpx

endpoint = os.getenv("TCM_API_URL", "http://127.0.0.1:8008/v1/tcm/process")
token = os.getenv("TCM_API_KEY")
headers = {"Authorization": f"Bearer {token}"} if token else {}
state = None
with httpx.Client(timeout=75, headers=headers) as client:
    for text in ["我这阵子觉得嘴巴发干。", "已经四天了，每天两次。", "请总结我的情况。"]:
        response = client.post(endpoint, json={"text":text,"state":state,
                                               "new_session":state is None,"request_id":uuid.uuid4().hex})
        response.raise_for_status()
        result = response.json()["result"]
        print(result["message"])
        state = result["state"]
