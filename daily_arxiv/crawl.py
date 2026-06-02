#!/usr/bin/env python3
"""
arXiv 论文爬取脚本 - 基于关键词搜索
arXiv paper crawler - keyword-based search

这个脚本独立于 Scrapy，直接使用 arxiv API 获取论文数据。
This script is independent of Scrapy and uses arxiv API directly.
"""
import arxiv
import yaml
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from arxiv import HTTPError


def load_config(config_path: str = None) -> dict:
    """加载配置文件 / Load configuration file"""
    if config_path and Path(config_path).exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    # 尝试查找配置文件
    config_paths = [
        Path(__file__).parent.parent / "config.yaml",  # daily_arxiv/config.yaml
        Path.cwd() / "daily_arxiv" / "config.yaml",
        Path.cwd() / "config.yaml",
    ]

    for path in config_paths:
        if path.exists():
            print(f"Loading config from: {path}")
            with open(path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)

    # 默认配置
    print("No config.yaml found, using default configuration")
    return {
        'arxiv': {
            'keywords': ['ti:"machine learning"'],
            'max_results_per_keyword': 50,
            'max_total_results': 200,
            'sort_by': 'SubmittedDate',
        }
    }


def fetch_papers(config: dict) -> list:
    """使用 arxiv API 获取论文数据 / Fetch papers using arxiv API"""
    arxiv_config = config.get('arxiv', {})

    keywords = arxiv_config.get('keywords', [])
    max_results_per_keyword = arxiv_config.get('max_results_per_keyword', 50)
    max_total_results = arxiv_config.get('max_total_results', 200)
    sort_by_str = arxiv_config.get('sort_by', 'SubmittedDate')

    # 解析排序方式
    sort_by_map = {
        "SubmittedDate": arxiv.SortCriterion.SubmittedDate,
        "Relevance": arxiv.SortCriterion.Relevance,
        "LastUpdatedDate": arxiv.SortCriterion.LastUpdatedDate,
    }
    sort_by = sort_by_map.get(sort_by_str, arxiv.SortCriterion.SubmittedDate)

    print("=" * 60)
    print(f"Starting arxiv API search with {len(keywords)} keywords")
    print(f"Keywords: {keywords}")
    print(f"Max results per keyword: {max_results_per_keyword}")
    print(f"Max total results: {max_total_results}")
    print("=" * 60)

    client = arxiv.Client()
    papers = []
    seen_ids = set()

    for i, keyword in enumerate(keywords):
        if len(papers) >= max_total_results:
            print(f"Reached max_total_results limit ({max_total_results})")
            break

        # 查询间延迟
        if i > 0:
            delay = 5
            print(f"Waiting {delay} seconds before next query (rate limit)...")
            time.sleep(delay)

        print(f"\nExecuting query [{i+1}/{len(keywords)}]: {keyword}")

        # 重试逻辑
        max_retries = 3
        retry_delay = 10

        for attempt in range(max_retries):
            try:
                remaining = max_total_results - len(papers)
                max_results = min(max_results_per_keyword, remaining)

                search = arxiv.Search(
                    query=keyword,
                    max_results=max_results,
                    sort_by=sort_by,
                )

                results = list(client.results(search))
                print(f"Query '{keyword}' returned {len(results)} results")

                for paper in results:
                    # 提取 arxiv ID
                    arxiv_id = paper.entry_id.split('/')[-1]
                    if 'v' in arxiv_id:
                        parts = arxiv_id.rsplit('v', 1)
                        if len(parts) == 2 and parts[1].isdigit():
                            arxiv_id = parts[0]

                    # 去重
                    if arxiv_id in seen_ids:
                        continue

                    seen_ids.add(arxiv_id)

                    papers.append({
                        "id": arxiv_id,
                        "title": paper.title,
                        "authors": [a.name for a in paper.authors],
                        "categories": list(paper.categories),
                        "summary": paper.summary,
                        "comment": paper.comment,
                        "pdf": paper.pdf_url,
                        "abs": paper.entry_id,
                    })

                    print(f"  + [{len(papers)}] {arxiv_id}: {paper.title[:50]}...")

                    if len(papers) >= max_total_results:
                        break

                break  # 成功

            except HTTPError as e:
                status = getattr(e, 'status', 'unknown')
                print(f"HTTPError: status={status}")
                if status == 429:  # Rate limit
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (attempt + 2)
                        print(f"Rate limited. Waiting {wait_time}s before retry...")
                        time.sleep(wait_time)
                    else:
                        print(f"Rate limited after {max_retries} retries")
                else:
                    break
            except Exception as e:
                print(f"Error: {type(e).__name__}: {e}")
                break

    print("=" * 60)
    print(f"Total unique papers fetched: {len(papers)}")
    print("=" * 60)

    return papers


def main():
    parser = argparse.ArgumentParser(description='arXiv keyword-based paper crawler')
    parser.add_argument('--output', '-o', type=str, required=True, help='Output JSONL file path')
    parser.add_argument('--config', '-c', type=str, default=None, help='Config file path')
    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)

    # 获取论文
    papers = fetch_papers(config)

    # 保存结果
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for paper in papers:
            f.write(json.dumps(paper, ensure_ascii=False) + '\n')

    print(f"\nSaved {len(papers)} papers to: {output_path}")


if __name__ == "__main__":
    main()
