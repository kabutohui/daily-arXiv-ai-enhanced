import scrapy
import arxiv
import yaml
import time
from pathlib import Path
from typing import Dict, Any, Set
from arxiv import HTTPError


class ArxivSpider(scrapy.Spider):
    """
    基于关键词搜索的 arXiv 论文爬虫
    Keyword-based arXiv paper spider using arxiv API
    """

    name = "arxiv"
    allowed_domains = ["arxiv.org"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # 加载配置文件 / Load configuration
        self.config = self._load_config()

        # 解析排序方式 / Parse sort criterion
        arxiv_config = self.config.get('arxiv', {})
        sort_by_str = arxiv_config.get('sort_by', 'SubmittedDate')
        sort_by_map = {
            "SubmittedDate": arxiv.SortCriterion.SubmittedDate,
            "Relevance": arxiv.SortCriterion.Relevance,
            "LastUpdatedDate": arxiv.SortCriterion.LastUpdatedDate,
        }
        self.sort_by = sort_by_map.get(sort_by_str, arxiv.SortCriterion.SubmittedDate)

        # 获取关键词列表 / Get keywords list
        self.keywords = arxiv_config.get('keywords', [])
        self.max_results_per_keyword = arxiv_config.get('max_results_per_keyword', 50)
        self.max_total_results = arxiv_config.get('max_total_results', 200)

        # 初始化 arxiv 客户端 / Initialize arxiv client
        self.client = arxiv.Client()

        self.logger.info(f"Loaded {len(self.keywords)} keywords: {self.keywords}")
        self.logger.info(f"Max results per keyword: {self.max_results_per_keyword}")
        self.logger.info(f"Max total results: {self.max_total_results}")

    def _load_config(self) -> Dict[str, Any]:
        """从 config.yaml 加载配置 / Load configuration from config.yaml"""
        # 查找配置文件 / Look for config file
        config_paths = [
            # 当从项目根目录运行时 / When running from project root
            Path.cwd() / "daily_arxiv" / "config.yaml",
            # 当从 daily_arxiv 目录运行时 / When running from daily_arxiv directory
            Path.cwd() / "config.yaml",
            # 相对于 spider 文件的位置 / Relative to spider file
            Path(__file__).parent.parent.parent / "config.yaml",
            Path(__file__).parent.parent / "config.yaml",
        ]

        for config_path in config_paths:
            if config_path.exists():
                self.logger.info(f"Loading config from: {config_path.resolve()}")
                with open(config_path, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f)

        # 未找到配置文件，使用默认配置 / Fallback to default config
        self.logger.warning("No config.yaml found, using default configuration")
        return {
            'arxiv': {
                'keywords': ['ti:"machine learning"'],
                'max_results_per_keyword': 50,
                'max_total_results': 200,
                'sort_by': 'SubmittedDate',
            },
            'llm': {
                'model_name': 'deepseek-chat',
            }
        }

    def start_requests(self):
        """生成初始请求 / Generate initial requests"""
        if not self.keywords:
            self.logger.error("No keywords configured!")
            return

        self.logger.info("=" * 60)
        self.logger.info(f"Starting arxiv API search with {len(self.keywords)} keywords")
        self.logger.info(f"Keywords: {self.keywords}")
        self.logger.info(f"Max results per keyword: {self.max_results_per_keyword}")
        self.logger.info(f"Max total results: {self.max_total_results}")
        self.logger.info("=" * 60)

        # 使用一个虚拟请求来触发数据处理
        # 我们在 parse 方法中直接获取并返回数据
        yield scrapy.Request(
            url="https://arxiv.org/",
            callback=self.parse,
            dont_filter=True,
            meta={'papers_index': 0}  # 用于跟踪处理进度
        )

    def parse(self, response):
        """执行 arxiv API 搜索并产出论文数据 / Execute arxiv API search and yield paper items"""
        seen_ids: Set[str] = set()
        total_yielded = 0

        for i, keyword in enumerate(self.keywords):
            if total_yielded >= self.max_total_results:
                self.logger.info(f"Reached max_total_results limit ({self.max_total_results})")
                break

            # 添加查询间延迟 / Add delay between queries
            if i > 0:
                delay = 5
                self.logger.info(f"Waiting {delay} seconds before next query (rate limit)...")
                time.sleep(delay)

            self.logger.info(f"Executing query [{i+1}/{len(self.keywords)}]: {keyword}")

            # 重试逻辑 / Retry logic
            max_retries = 3
            retry_delay = 10

            for attempt in range(max_retries):
                try:
                    # 计算本次查询的最大结果数
                    remaining = self.max_total_results - total_yielded
                    max_results = min(self.max_results_per_keyword, remaining)

                    # 创建搜索对象
                    search = arxiv.Search(
                        query=keyword,
                        max_results=max_results,
                        sort_by=self.sort_by,
                    )

                    # 执行搜索
                    results = list(self.client.results(search))
                    self.logger.info(f"Query '{keyword}' returned {len(results)} results")

                    for paper in results:
                        # 提取 arxiv ID
                        arxiv_id = paper.entry_id.split('/')[-1]
                        if 'v' in arxiv_id:
                            parts = arxiv_id.rsplit('v', 1)
                            if len(parts) == 2 and parts[1].isdigit():
                                arxiv_id = parts[0]

                        # 去重检查
                        if arxiv_id in seen_ids:
                            self.logger.debug(f"Skipping duplicate: {arxiv_id}")
                            continue

                        seen_ids.add(arxiv_id)

                        self.logger.info(f"Yielding paper: {arxiv_id} - {paper.title[:50]}...")

                        # 产出论文数据 - 使用字典格式
                        yield {
                            "id": arxiv_id,
                            "title": paper.title,
                            "authors": [a.name for a in paper.authors],
                            "categories": list(paper.categories),
                            "summary": paper.summary,
                            "comment": paper.comment,
                            "pdf": paper.pdf_url,
                            "abs": paper.entry_id,
                        }

                        total_yielded += 1
                        if total_yielded >= self.max_total_results:
                            break

                    break  # 成功，跳出重试循环

                except HTTPError as e:
                    status = getattr(e, 'status', 'unknown')
                    self.logger.error(f"HTTPError for query '{keyword}': status={status}")
                    if status == 429:  # Rate limit
                        if attempt < max_retries - 1:
                            wait_time = retry_delay * (attempt + 2)
                            self.logger.warning(f"Rate limited (429). Waiting {wait_time}s before retry...")
                            time.sleep(wait_time)
                        else:
                            self.logger.error(f"Rate limited (429) after {max_retries} retries for query '{keyword}'")
                    else:
                        break
                except Exception as e:
                    self.logger.error(f"Error executing query '{keyword}': {type(e).__name__}: {e}")
                    break

        self.logger.info("=" * 60)
        self.logger.info(f"Total unique papers yielded: {total_yielded}")
        self.logger.info("=" * 60)
