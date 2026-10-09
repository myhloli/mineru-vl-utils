"""通过无网络传输检查版本化地址、代理路径及模型发现的一致性。"""

import asyncio

import httpx
import pytest
from PIL import Image

from mineru_vl_utils.vlm_client.http_client import HttpVlmClient


@pytest.mark.parametrize("server_url,prefix", [
    ("http://host:8000", "/v1"), ("http://host:8000/", "/v1"),
    ("http://host:8000/v1", "/v1"), ("http://host:8000/v1/", "/v1"),
    ("http://host:8000/v2", "/v2"), ("http://host:8000/v2/", "/v2"),
    ("https://host/foo/v2", "/foo/v2"), ("https://host/foo/v2/", "/foo/v2"),
    ("https://host/foo/", "/foo/v1"), ("https://host/foo", "/v1"),
])
@pytest.mark.parametrize("discover_model", [False, True])
@pytest.mark.parametrize("from_env", [False, True])
def test_server_url_uses_matching_chat_and_models_paths(monkeypatch, server_url, prefix, discover_model, from_env):
    """显式模型、自动发现及环境变量入口均访问同一代理前缀和版本。"""
    requests = []
    monkeypatch.delenv("MINERU_VL_MODEL_NAME", raising=False)

    def respond(request):
        """记录实际请求路径并返回固定模型和预测结果。"""
        requests.append((request.method, request.url.path))
        if request.method == "GET":
            return httpx.Response(200, json={"data": [{"id": "test"}]})
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": "recognized"}}]})

    def new_client(self):
        """同步请求使用可检查的 MockTransport，完全避免外网访问。"""
        return httpx.Client(transport=httpx.MockTransport(respond))

    async def new_aio_client(self):
        """异步请求使用与同步路径相同的协议模拟。"""
        return httpx.AsyncClient(transport=httpx.MockTransport(respond))

    monkeypatch.setattr(HttpVlmClient, "_new_client", new_client)
    monkeypatch.setattr(HttpVlmClient, "_new_aio_client", new_aio_client)
    if from_env:
        monkeypatch.setenv("MINERU_VL_SERVER", server_url)
    client = HttpVlmClient(server_url=None if from_env else server_url, model_name=None if discover_model else "test")
    image = Image.new("RGB", (4, 4))
    try:
        assert client.predict(image) == "recognized"

        async def aio_request():
            """在同一事件循环内完成预测并释放连接池。"""
            try:
                assert await client.aio_predict(image) == "recognized"
            finally:
                await client.aclose()

        asyncio.run(aio_request())
        assert requests == [("GET", prefix + "/models"), ("POST", prefix + "/chat/completions"),
                            ("POST", prefix + "/chat/completions")]
    finally:
        client._client.close()
