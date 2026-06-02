#!/usr/bin/env python3
"""测试 API 连接"""
import requests
from openai import OpenAI

# 测试 1: 直接用 requests 测试连通性
print("=" * 50)
print("测试 1: requests 直接访问")
print("=" * 50)

base_url = "https://maas-api.ai-yuanjing.com/openapi/compatible-mode/v1"
api_key = "sk-b612fbbaee474a2cb33ebff4cdc6b0e2"

try:
    resp = requests.get(f"{base_url}/models", headers={"Authorization": f"Bearer {api_key}"}, timeout=10)
    print(f"状态码: {resp.status_code}")
    print(f"响应: {resp.text[:500]}")
except Exception as e:
    print(f"requests 错误: {type(e).__name__}: {e}")

# 测试 2: 用 OpenAI SDK
print("\n" + "=" * 50)
print("测试 2: OpenAI SDK")
print("=" * 50)

try:
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )

    completion = client.chat.completions.create(
        model="glm-5",
        messages=[
            {'role': 'user', 'content': '你好，请回复OK'}
        ],
        timeout=30
    )
    print(f"成功! 响应: {completion.choices[0].message.content}")
except Exception as e:
    print(f"OpenAI SDK 错误: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
