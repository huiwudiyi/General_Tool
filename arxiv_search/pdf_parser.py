#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Project : General_Tool
@File    : pdf_parser.py
@Author  : zhuzerun
@Date    : 2026-08-04 18:11
@Version : 1.0
@Desc    : 
@Contact : zachary6chu@gmail.com
"""
import os
import json
import datetime
import pandas as pd
from json_repair import repair_json
import hashlib
from pathlib import Path

import importlib

import requests


class PDFParser:
    def __init__(self, cache_dir: str | Path = ".arxiv_pdf_cache", timeout: int = 60, headers: dict | None = None):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.headers = headers or {"User-Agent": "Mozilla/5.0 (compatible; arxiv-recent-paper-search/2.0)"}

    def cache_path(self, pdf_url: str) -> Path:
        digest = hashlib.sha256(pdf_url.encode("utf-8")).hexdigest()[:16]
        filename = pdf_url.rstrip("/").split("/")[-1] or f"{digest}.pdf"
        if not filename.endswith(".pdf"):
            filename = f"{filename}.pdf"
        return self.cache_dir / f"{digest}-{filename}"

    def download(self, pdf_url: str) -> Path:
        path = self.cache_path(pdf_url)
        if path.exists() and path.stat().st_size > 0:
            return path
        response = requests.get(pdf_url, headers=self.headers, timeout=self.timeout)
        response.raise_for_status()
        path.write_bytes(response.content)
        return path

    def first_page_text(self, pdf_path: str | Path) -> str:
        fitz = importlib.import_module("fitz")
        with fitz.open(pdf_path) as doc:
            if len(doc) == 0:
                return ""
            return doc[0].get_text()

    def ocr_first_page(self, pdf_path: str | Path) -> str:
        fitz = importlib.import_module("fitz")
        ocrmac = importlib.import_module("ocrmac").ocrmac
        pdf_path = Path(pdf_path)
        image_path = pdf_path.with_suffix(".page1.png")
        with fitz.open(pdf_path) as doc:
            if len(doc) == 0:
                return ""
            pix = doc.load_page(0).get_pixmap(dpi=300)
            pix.save(image_path)
        annotations = ocrmac.OCR(str(image_path)).recognize()
        return "\n".join(item[0] for item in annotations)

    def extract_first_page_text(self, pdf_url: str, use_ocr: bool = True) -> str:
        pdf_path = self.download(pdf_url)
        text = self.first_page_text(pdf_path)
        if text.strip() or not use_ocr:
            return text
        return self.ocr_first_page(pdf_path)