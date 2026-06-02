# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
import arxiv


class DailyArxivPipeline:
    """
    论文数据处理管道 / Paper data processing pipeline

    由于 spider 已经通过 arxiv API 获取了完整的元数据，
    此管道主要做透传处理，仅在数据缺失时调用 API 补充。

    Since the spider already fetches complete metadata via arxiv API,
    this pipeline mainly passes through data, only calling API for fallback.
    """

    def __init__(self):
        self.page_size = 100
        self.client = arxiv.Client(self.page_size)

    def process_item(self, item: dict, spider):
        """
        处理论文数据项 / Process paper data item

        如果 spider 已经提供了完整元数据，直接透传。
        否则通过 arxiv API 获取元数据（向后兼容）。

        If spider already provided complete metadata, pass through.
        Otherwise fetch metadata via arxiv API (backward compatibility).
        """
        # 检查是否已有完整元数据 / Check if complete metadata exists
        if "title" in item and "summary" in item and "authors" in item:
            # 完整数据，直接返回 / Complete data, pass through
            return item

        # 数据不完整，调用 API 获取 / Incomplete data, fetch via API
        spider.logger.info(f"Fetching metadata for {item.get('id', 'unknown')} via API")

        # 确保 pdf 和 abs URL 存在 / Ensure pdf and abs URLs exist
        if "pdf" not in item:
            item["pdf"] = f"https://arxiv.org/pdf/{item['id']}"
        if "abs" not in item:
            item["abs"] = f"https://arxiv.org/abs/{item['id']}"

        try:
            search = arxiv.Search(id_list=[item["id"]])
            paper = next(self.client.results(search))
            item["authors"] = [a.name for a in paper.authors]
            item["title"] = paper.title
            item["categories"] = list(paper.categories)
            item["comment"] = paper.comment
            item["summary"] = paper.summary
        except StopIteration:
            spider.logger.warning(f"Could not fetch metadata for {item['id']}")
        except Exception as e:
            spider.logger.error(f"Error fetching metadata for {item['id']}: {e}")

        return item
