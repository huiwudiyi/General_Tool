#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Project : General_Tool
@File    : arxiv_searcher.py
@Author  : zhuzerun
@Date    : 2026-08-04 18:29
@Version : 1.0
@Desc    : 
@Contact : zachary6chu@gmail.com
"""
import os
import json
import time
import datetime
import pandas as pd
from json_repair import repair_json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import quote_plus

import requests

from affiliation_detector import AffiliationDetector
from models import Paper
from paper_parser import PaperParser
from pdf_parser import PDFParser


class ArxivSearcher:
    SEARCH_URL = (
        "https://arxiv.org/search/?query={query}&searchtype=all&source=header"
        "&order=-announced_date_first&size={page_size}&start={start}"
    )

    def __init__(
        self,
        parser: PaperParser | None = None,
        detector: AffiliationDetector | None = None,
        page_size: int = 50,
        max_workers: int = 6,
        timeout: int = 60,
        cache_dir: str | Path = ".arxiv_pdf_cache",
    ):
        self.headers = {"User-Agent": "Mozilla/5.0 (compatible; arxiv-recent-paper-search/2.0)"}
        self.parser = parser or PaperParser()
        pdf_parser = PDFParser(cache_dir=cache_dir, timeout=timeout, headers=self.headers)
        self.detector = detector or AffiliationDetector(pdf_parser=pdf_parser)
        self.page_size = page_size
        self.max_workers = max_workers
        self.timeout = timeout

    def build_search_url(self, query: str, start: int = 0) -> str:
        return self.SEARCH_URL.format(query=quote_plus(query), page_size=self.page_size, start=start)

    def fetch_html(self, url: str) -> str:
        print("开始处理", url)
        try:
            response = requests.get(url, headers=self.headers, timeout=self.timeout)
        except requests.RequestException as e:
            print("url = ", url, "error!!!")
            print("请求出错", e)
            return ""
        response.raise_for_status()
        return response.text

    def iter_recent_candidates(self, query: str, days: int = 90, max_pages: int | None = None):
        cutoff = datetime.utcnow() - timedelta(days=days)
        page_index = 0
        while max_pages is None or page_index < max_pages:
            print("process page_index:", page_index)
            time.sleep(2)
            html = self.fetch_html(self.build_search_url(query, page_index * self.page_size))
            if html == "":
                continue
            result_nodes = self.parser.parse_result_list(html)
            if not result_nodes:
                break
            found_recent_on_page = False
            for node in result_nodes:
                submitted, paper = self.parser.parse_paper(node)
                if submitted is None:
                    continue
                if submitted >= cutoff:
                    found_recent_on_page = True
                    yield paper
            if not found_recent_on_page:
                break
            page_index += 1

    def enrich_paper(self, paper: Paper, use_pdf: bool = True, use_ocr: bool = True) -> Paper:
        detail_html = self.fetch_html(paper.paper_url)
        if detail_html == "":
            return ""
        return self.detector.enrich(paper, detail_html, use_pdf=use_pdf, use_ocr=use_ocr)

    def search(
        self,
        query: str,
        days: int = 90,
        max_pages: int | None = None,
        use_pdf: bool = True,
        use_ocr: bool = True,
    ) -> list[Paper]:
        candidates = list(self.iter_recent_candidates(query, days=days, max_pages=max_pages))
        if not candidates:
            return []
        print("candidates", candidates)
        workers = max(1, self.max_workers)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(self.enrich_paper, paper, use_pdf, use_ocr) for paper in candidates]
            enriched = []
            for future in as_completed(futures):
                if future.result() != "":
                    enriched.append(future.result())
        return sorted(enriched, key=lambda paper: paper.submit_date, reverse=True)