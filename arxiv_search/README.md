# arXiv 最近一年论文检索工具

该工具使用面向对象架构实现 arXiv 论文检索流水线：根据用户输入的 query 自动翻页抓取 arXiv Search 结果，筛选最近一年（默认 365 天）提交的论文，并并发补充详情页、PDF 首页、OCR 兜底、机构识别和大厂标注信息，最终导出 JSON 或 CSV。
### 安装
```pip
requests
beautifulsoup4
PyMuPDF
ocrmac
```
## 模块设计

- `ArxivSearcher`：负责构造 arXiv Search URL、自动翻页、最近 N 天过滤、并发抓取详情页/PDF 信息。
- `PaperParser`：负责解析 arXiv 搜索结果中的标题、作者、摘要、分类、提交日期、abs 链接和 PDF 链接。
- `PDFParser`：负责 PDF 下载、PDF 缓存、首页文本提取，以及文本为空时的 OCR 回退。
- `AffiliationDetector`：负责从详情页/PDF 首页文本中抽取机构线索，并根据大厂字典标注 `is_big_company` 和 `company`。
- `Exporter`：负责输出标准 JSON，并可额外导出 CSV。

## 依赖

```bash
pip install -r LLM/arxiv_search/requirements.txt
```

依赖包括 `requests`、`beautifulsoup4`、`PyMuPDF` 和 `ocrmac`。`ocrmac` 主要用于 PDF 首页无法直接提取文字时的 OCR 兜底；如果运行环境不支持 OCR，可以使用 `--no-ocr`，如果不需要 PDF 机构识别，可以使用 `--no-pdf`。

## 使用方法

自动翻页检索最近一年论文，并输出 JSON：

```bash
python arxiv_recent_papers.py "Query Rewriting"
```

限制最多解析 2 页，并发数设为 8，同时导出 JSON 和 CSV：

```bash
python LLM/arxiv_search/arxiv_recent_papers.py "Query Rewriting" \
  --max-pages 2 \
  --workers 8 \
  --json-output data/papers.json \
  --csv-output data.papers.csv
```

跳过 PDF 下载和 OCR，仅使用 arXiv 详情页识别机构/大厂：

```bash
python LLM/arxiv_search/arxiv_recent_papers.py "Query Rewriting" --no-pdf
```

## 输出格式

```json
{
  "query": "Query Rewriting",
  "papers": [
    {
      "arxiv_id": "2506.01234",
      "title": "LLM Query Rewriting",
      "authors": ["..."],
      "abstract": "...",
      "categories": ["cs.IR", "cs.CL"],
      "submit_date": "2025-06-05",
      "paper_url": "https://arxiv.org/abs/2506.01234",
      "pdf_url": "https://arxiv.org/pdf/2506.01234.pdf",
      "affiliations": ["Google Research", "Tsinghua University"],
      "is_big_company": true,
      "company": ["Google"]
    }
  ]
}
```