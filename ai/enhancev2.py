#!/usr/bin/env python3
"""
AI 增强处理脚本 V2 - 使用 OpenAI SDK
AI Enhancement Script V2 - Using OpenAI SDK
"""
import os
import json
import sys
import re
import time
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict
import argparse
from tqdm import tqdm
import requests

if os.path.exists('.env'):
    import dotenv
    dotenv.load_dotenv()

# 读取模板文件
template = open("template.txt", "r").read()
system = open("system.txt", "r").read()


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="jsonline data file")
    parser.add_argument("--max_workers", type=int, default=1, help="Maximum number of parallel workers")
    return parser.parse_args()


def check_github_code(content: str) -> Dict:
    """提取并验证 GitHub 链接"""
    code_info = {}

    # 1. 优先匹配 github.com/owner/repo 格式
    github_pattern = r"https?://github\.com/([a-zA-Z0-9-_]+)/([a-zA-Z0-9-_\.]+)"
    match = re.search(github_pattern, content)

    if match:
        owner, repo = match.groups()
        # 清理 repo 名称，去掉可能的 .git 后缀或末尾的标点
        repo = repo.rstrip(".git").rstrip(".,)")

        full_url = f"https://github.com/{owner}/{repo}"
        code_info["code_url"] = full_url

        # 尝试调用 GitHub API 获取信息
        github_token = os.environ.get("TOKEN_GITHUB")
        headers = {"Accept": "application/vnd.github.v3+json"}
        if github_token:
            headers["Authorization"] = f"token {github_token}"

        try:
            api_url = f"https://api.github.com/repos/{owner}/{repo}"
            resp = requests.get(api_url, headers=headers, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                code_info["code_stars"] = data.get("stargazers_count", 0)
                code_info["code_last_update"] = data.get("pushed_at", "")[:10]
        except Exception:
            # API 调用失败不影响主流程
            pass
        return code_info

    # 2. 如果没有 github.com，尝试匹配 github.io
    github_io_pattern = r"https?://[a-zA-Z0-9-_]+\.github\.io(?:/[a-zA-Z0-9-_\.]+)*"
    match_io = re.search(github_io_pattern, content)

    if match_io:
        url = match_io.group(0)
        # 清理末尾标点
        url = url.rstrip(".,)")
        code_info["code_url"] = url
        # github.io 不进行 star 和 update 判断

    return code_info


def call_llm_api(client: OpenAI, summary: str, language: str, model_name: str, max_retries: int = 3) -> Dict:
    """
    使用 OpenAI SDK 调用 LLM API

    Args:
        client: OpenAI 客户端实例
        summary: 论文摘要
        language: 语言设置
        model_name: 模型名称
        max_retries: 最大重试次数

    Returns:
        包含 AI 字段的字典
    """
    # 默认返回值
    default_ai_fields = {
        "tldr": "Summary generation failed",
        "motivation": "Motivation analysis unavailable",
        "method": "Method extraction failed",
        "result": "Result analysis unavailable",
        "conclusion": "Conclusion extraction failed"
    }

    # 构建提示词
    user_content = template.format(content=summary)

    # 在系统提示词中添加语言要求
    language_instruction = f"\n\nPlease provide your response in {language}."
    system_with_language = system + language_instruction

    for attempt in range(max_retries):
        try:
            print(f"===> System prompt: {system_with_language}")
            print(f"===> User prompt: {user_content}")
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_with_language},
                    {"role": "user", "content": user_content}
                ],
                # temperature=0.7,
                # max_tokens=2000,
                # response_format={"type": "json_object"}  # 强制返回 JSON
            )

            # 提取响应内容
            content = response.choices[0].message.content
            print("===> response: ", content)

            if not content:
                print(f"Empty response content", file=sys.stderr)
                continue

            # 清理 markdown 代码块格式
            # 匹配 ```json ... ``` 或 ``` ... ``` 格式
            code_block_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', content, re.DOTALL)
            if code_block_match:
                content = code_block_match.group(1).strip()

            # 尝试解析 JSON
            try:
                ai_data = json.loads(content)

                # 确保所有必需字段存在
                for field in default_ai_fields.keys():
                    if field not in ai_data:
                        ai_data[field] = default_ai_fields[field]

                return ai_data

            except json.JSONDecodeError as e:
                print(f"JSON parse error: {e}", file=sys.stderr)
                print(f"Content that failed to parse: {repr(content[:500])}", file=sys.stderr)
                # 尝试提取 JSON 块（支持嵌套结构）
                # 方法1: 查找第一个 { 和最后一个 } 之间的内容
                first_brace = content.find('{')
                last_brace = content.rfind('}')
                if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                    try:
                        json_str = content[first_brace:last_brace + 1]
                        ai_data = json.loads(json_str)
                        print(f"Successfully extracted JSON from position {first_brace} to {last_brace}", file=sys.stderr)
                        for field in default_ai_fields.keys():
                            if field not in ai_data:
                                ai_data[field] = default_ai_fields[field]
                        return ai_data
                    except json.JSONDecodeError as e2:
                        print(f"Failed to parse extracted JSON: {e2}", file=sys.stderr)
                        print(f"Extracted JSON string: {repr(json_str[:500])}", file=sys.stderr)
                continue

        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)
            print(f"Error (attempt {attempt + 1}/{max_retries}): {error_type}: {error_msg[:100]}", file=sys.stderr)

            # 检查是否是速率限制错误
            if "rate" in error_msg.lower() or "429" in error_msg:
                wait_time = 2 ** (attempt + 1)
                print(f"Rate limited, waiting {wait_time}s...", file=sys.stderr)
                time.sleep(wait_time)
                continue

            # 连接错误
            if "connection" in error_msg.lower() or "timeout" in error_msg.lower():
                wait_time = 2 * (attempt + 1)
                print(f"Connection error, waiting {wait_time}s...", file=sys.stderr)
                time.sleep(wait_time)
                continue

            time.sleep(1)
            continue

    # 所有重试都失败，返回默认值
    return default_ai_fields


def process_single_item(client: OpenAI, item: Dict, language: str, model_name: str) -> Dict:
    """
    处理单个数据项

    Args:
        client: OpenAI 客户端实例
        item: 论文数据项
        language: 语言设置
        model_name: 模型名称

    Returns:
        处理后的数据项（包含 AI 字段）
    """
    # 检测代码可用性
    code_info = check_github_code(item.get("summary", ""))
    if code_info:
        item.update(code_info)

    # 调用 LLM API
    ai_result = call_llm_api(
        client=client,
        summary=item.get("summary", ""),
        language=language,
        model_name=model_name
    )

    # 将 AI 结果添加到 item
    item["AI"] = ai_result

    return item


def process_all_items(data: List[Dict], model_name: str, language: str, max_workers: int) -> List[Dict]:
    """并行处理所有数据项"""
    # 获取 API 配置
    api_key = os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL", "")

    if not api_key:
        print("Error: OPENAI_API_KEY not set!", file=sys.stderr)
        # 返回带默认 AI 字段的数据
        for item in data:
            item["AI"] = {
                "tldr": "API key not configured",
                "motivation": "API key not configured",
                "method": "API key not configured",
                "result": "API key not configured",
                "conclusion": "API key not configured"
            }
        return data

    # 创建 OpenAI 客户端
    client = OpenAI(
        api_key=api_key,
        base_url=base_url
    )

    # 使用线程池并行处理
    processed_data = [None] * len(data)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_idx = {
            executor.submit(process_single_item, client, item, language, model_name): idx
            for idx, item in enumerate(data)
        }

        # 使用 tqdm 显示进度
        for future in tqdm(
            as_completed(future_to_idx),
            total=len(data),
            desc="Processing items"
        ):
            idx = future_to_idx[future]
            try:
                result = future.result()
                processed_data[idx] = result
            except Exception as e:
                print(f"Item at index {idx} generated an exception: {e}", file=sys.stderr)
                # 确保数据一致性
                processed_data[idx] = data[idx]
                processed_data[idx]['AI'] = {
                    "tldr": "Processing failed",
                    "motivation": "Processing failed",
                    "method": "Processing failed",
                    "result": "Processing failed",
                    "conclusion": "Processing failed"
                }

    return processed_data


def main():
    args = parse_args()
    model_name = os.environ.get("MODEL_NAME", "glm-5")
    language = os.environ.get("LANGUAGE", "Chinese")

    # 检查并删除目标文件
    target_file = args.data.replace('.jsonl', f'_AI_enhanced_{language}.jsonl')
    if os.path.exists(target_file):
        os.remove(target_file)
        print(f'Removed existing file: {target_file}', file=sys.stderr)

    # 读取数据
    data = []
    with open(args.data, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))

    # 去重
    seen_ids = set()
    unique_data = []
    for item in data:
        if item['id'] not in seen_ids:
            seen_ids.add(item['id'])
            unique_data.append(item)

    data = unique_data
    print(f'Processing {len(data)} items from: {args.data}', file=sys.stderr)

    # 并行处理所有数据
    processed_data = process_all_items(
        data,
        model_name,
        language,
        args.max_workers
    )

    # 保存结果
    with open(target_file, "w", encoding="utf-8") as f:
        for item in processed_data:
            if item is not None:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f'Saved {len([x for x in processed_data if x])} items to: {target_file}', file=sys.stderr)


if __name__ == "__main__":
    main()
