#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Project : General_Tool
@File    : affiliation_detector.py
@Author  : zhuzerun
@Date    : 2026-08-04 18:06
@Version : 1.0
@Desc    : 
@Contact : zachary6chu@gmail.com
"""
import os
import json
import datetime
import re

from bs4 import BeautifulSoup

from models import Paper
from pdf_parser import PDFParser


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
    "SAP": "SAP",
}


class AffiliationDetector:
    def __init__(self, pdf_parser: PDFParser | None = None, big_companies: dict[str, str] | None = None):
        self.pdf_parser = pdf_parser
        self.big_companies = big_companies or BIG_COMPANIES
        self.affiliation_markers = ("university", "institute", "research", "laboratory", "lab", "college", "school")

    def clean_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    def find_companies(self, text: str) -> list[str]:
        found = set()
        for keyword, company in self.big_companies.items():
            if re.search(rf"(?<![A-Za-z0-9]){re.escape(keyword)}(?![A-Za-z0-9])", text, re.I):
                found.add(company)
        return sorted(found)

    def extract_affiliations(self, text: str) -> list[str]:
        affiliations = []
        for line in text.splitlines():
            cleaned = self.clean_text(line)
            lower = cleaned.lower()
            if cleaned and (any(marker in lower for marker in self.affiliation_markers) or self.find_companies(cleaned)):
                affiliations.append(cleaned)
        return list(dict.fromkeys(affiliations))

    def detail_page_text(self, html: str) -> str:
        soup = BeautifulSoup(html, "html.parser")
        authors = soup.select_one("div.authors")
        dateline = soup.select_one("div.dateline")
        pieces = [soup.get_text("\n")]
        if authors:
            pieces.append(authors.get_text("\n"))
        if dateline:
            pieces.append(dateline.get_text("\n"))
        return "\n".join(pieces)

    def enrich(self, paper: Paper, detail_html: str, use_pdf: bool = True, use_ocr: bool = True) -> Paper:
        detail_text = self.detail_page_text(detail_html)
        companies = self.find_companies(detail_text)
        affiliations = self.extract_affiliations(detail_text)
        if use_pdf and self.pdf_parser and not companies:
            pdf_text = self.pdf_parser.extract_first_page_text(paper.pdf_url, use_ocr=use_ocr)
            companies = self.find_companies(pdf_text)
            affiliations.extend(self.extract_affiliations(pdf_text))
        paper.affiliations = list(dict.fromkeys(affiliations))
        paper.company = companies
        paper.is_big_company = bool(companies)
        return paper
