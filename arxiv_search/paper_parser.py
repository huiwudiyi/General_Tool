#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Project : General_Tool
@File    : paper_parser.py
@Author  : zhuzerun
@Date    : 2026-08-04 18:11
@Version : 1.0
@Desc    : 
@Contact : zachary6chu@gmail.com
"""
import os
import json
import datetime
import re
from datetime import datetime
from urllib.parse import urlparse

from bs4 import BeautifulSoup, Tag

from models import Paper


class PaperParser:
    def clean_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    def parse_result_list(self, html: str) -> list[Tag]:
        soup = BeautifulSoup(html, "html.parser")
        return list(soup.select("li.arxiv-result"))

    def parse_authors(self, authors_text: str) -> list[str]:
        authors_text = re.sub(r"^Authors?:", "", authors_text.strip(), flags=re.I)
        return [self.clean_text(author) for author in authors_text.split(",") if self.clean_text(author)]

    def parse_submit_date(self, date_text: str) -> datetime | None:
        match = re.search(r"Submitted\s+([0-9]{1,2}\s+[A-Za-z]+,\s+[0-9]{4})", date_text)
        if not match:
            return None
        return datetime.strptime(match.group(1), "%d %B, %Y")

    def parse_categories(self, paper_node: Tag) -> list[str]:
        categories = [tag.get_text(strip=True) for tag in paper_node.select("span.tag")]
        return [category for category in categories if category]

    def parse_arxiv_id(self, paper_url: str) -> str:
        path = urlparse(paper_url).path.rstrip("/")
        return path.split("/")[-1]

    def build_pdf_url(self, paper_url: str) -> str:
        pdf = paper_url.replace("/abs/", "/pdf/")
        return pdf

    def parse_paper(self, paper_node: Tag) -> tuple[datetime | None, Paper]:
        title_node = paper_node.select_one("p.title")
        authors_node = paper_node.select_one("p.authors")
        abstract_node = paper_node.select_one("span.abstract-full")
        url_node = paper_node.select_one("p.list-title a")
        date_node = paper_node.select_one("p.is-size-7")
        if not all([title_node, authors_node, abstract_node, url_node, date_node]):
            raise ValueError("arXiv result item is missing one or more required fields")

        paper_url = url_node["href"]
        submitted = self.parse_submit_date(date_node.get_text(" "))
        paper = Paper(
            arxiv_id=self.parse_arxiv_id(paper_url),
            title=self.clean_text(title_node.get_text(" ")),
            authors=self.parse_authors(authors_node.get_text(" ")),
            abstract=self.clean_text(abstract_node.get_text(" ").replace("△ Less", "")),
            categories=self.parse_categories(paper_node),
            submit_date=submitted.strftime("%Y-%m-%d") if submitted else "",
            paper_url=paper_url,
            pdf_url=self.build_pdf_url(paper_url),
        )
        return submitted, paper
