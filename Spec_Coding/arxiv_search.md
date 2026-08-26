# 需求
使用python代码实现，用户输入检索query在arxiv.org网页中检索出最近一年的论文，并输出论文标题、作者、摘要、分类，并且标注出大厂的论文

# 环境
需要的库：requests、BeautifulSoup、PyMuPDF、ocrmac等

# 代码实现过程
Step1：解析 arXiv Search URL
已知：输入url（例如：url = "https://arxiv.org/search/?query=Query+Rewriting&searchtype=all&source=header"）
直接请求：
```python
import requests

headers = {
    "User-Agent": "Mozilla/5.0"
}

html = requests.get(url, headers=headers).text
```
Step2：解析论文列表
```python
from bs4 import BeautifulSoup

soup = BeautifulSoup(html, "html.parser")

papers = soup.select("li.arxiv-result")
```
解析：
- 标题：title = paper.select_one("p.title").get_text(strip=True)
- 作者：authors = paper.select_one("p.authors").get_text(strip=True)
- 摘要：abstract = paper.select_one("span.abstract-full").text.strip()
- 论文链接：paper_url = paper.select_one("p.list-title a")["href"]

Step3：过滤最近一年论文
论文提交时间：
```python
import re
date_text = paper.select_one("p.is-size-7").text
```
将date_text 转成 datetime 对象，然后判断是否最近一年

Step4：获取作者机构

大厂字典
```python
BIG_COMPANIES = {
    "Google": "Google",
    "Google DeepMind": "Google",
    "DeepMind": "Google",

    "Microsoft": "Microsoft",
    "Microsoft Research": "Microsoft",

    "Meta": "Meta",
    "Meta AI": "Meta",

    "OpenAI": "OpenAI",

    "Apple": "Apple",

    "Amazon": "Amazon",
    "AWS": "Amazon",

    "NVIDIA": "NVIDIA",

    "Alibaba": "Alibaba",
    "Alibaba DAMO": "Alibaba",

    "ByteDance": "ByteDance",

    "Tencent": "Tencent",

    "Huawei": "Huawei",

    "Baidu": "Baidu",

    "Xiaomi": "Xiaomi",

    "IBM": "IBM",

    "Adobe": "Adobe",

    "Intel": "Intel",

    "Qualcomm": "Qualcomm",

    "Salesforce": "Salesforce",

    "SAP": "SAP"
}
```
1、根据paper_url 进入详情页，查找 Authors 字段是否包含 BIG_COMPANIES中的字段
2、如果没有，则尝试从 pdf中识别机构，识别方法：
- 生成PDF地址
```python
def pdf_url(arxiv_url):

    """
    https://arxiv.org/abs/2501.01234
            ↓
    https://arxiv.org/pdf/2501.01234.pdf
    """

    return arxiv_url.replace(
        "/abs/",
        "/pdf/"
    ) + ".pdf"
```
- 下载PDF
```python
import requests


def download(url, save_path):

    r = requests.get(
        url,
        timeout=60
    )

    with open(save_path, "wb") as f:

        f.write(r.content)
```
- 解析第一页信息
```python
import fitz


def first_page_text(pdf):

    doc = fitz.open(pdf)

    page = doc[0]

    return page.get_text()
```
- 如果第一页解析的信息为空，执行OCR方法
```python
import fitz

doc = fitz.open(pdf_path)

page = doc.load_page(0)

pix = page.get_pixmap(
    dpi=300
)

pix.save("page1.png")
from ocrmac import ocrmac

annotations = ocrmac.OCR("page1.png").recognize()

text = "\n".join(
    x[0]
    for x in annotations
)

print(text)
```

最终输出结构
```python
{
  "arxiv_id":"2506.01234",

  "title":"LLM Query Rewriting",

  "authors":[
      "...",
      "..."
  ],

  "abstract":"...",


  "categories":[
      "cs.CL",
      "cs.IR"
  ],

  "submit_date":"2025-06-05",

  "paper_url":"...",

  "pdf_url":"...",

  "affiliations":[
      "Google Research",
      "Tsinghua University"
  ],

  "is_big_company":true,

  "company":[
      "Google"
  ]
}
```


Step5：最终输出json
```python
{
  "query": "Query Rewriting",
  "papers": [
    {
      "title": "...",
      "authors": [
        "...",
        "..."
      ],
      "abstract": "...",
      "categories": [
        "cs.IR",
        "cs.CL"
      ],
      "submit_date": "2025-10-02",
      "paper_url": "...",
      "affiliations": [
        "Google Research",
        "Tsinghua University"
      ],
      "is_big_company": true,
      "company": [
        "Google"
      ]
    }
  ]
}
```
