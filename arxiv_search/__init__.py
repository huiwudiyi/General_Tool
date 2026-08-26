#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Project : General_Tool
@File    : __init__.py
@Author  : zhuzerun
@Date    : 2026-08-04 18:07
@Version : 1.0
@Desc    : 
@Contact : zachary6chu@gmail.com
"""
"""arXiv recent-paper search package."""

from affiliation_detector import AffiliationDetector
from arxiv_searcher import ArxivSearcher
from exporter import Exporter
from paper_parser import PaperParser
from pdf_parser import PDFParser

__all__ = ["AffiliationDetector", "ArxivSearcher", "Exporter", "PaperParser", "PDFParser"]